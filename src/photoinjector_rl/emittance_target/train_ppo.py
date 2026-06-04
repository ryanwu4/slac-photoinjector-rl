"""
Train Stable-Baselines3 PPO against the PhotoinjectorEnv surrogate-driven env.

On-policy baseline for the photoinjector emittance-minimization task. PPO is
sample-hungrier than the analytic-gradient methods in principle (~5-10x more
transitions, since it estimates the policy gradient zeroth-order rather than
backpropping through the differentiable surrogate), so it needs cheap, high-
throughput rollout collection to stay competitive in wall-clock.

Vec-env layout (`--vec-env`, default `batched`):
    batched  All n_envs advance as ONE (n_envs, 11) surrogate forward on the
             policy device via `SurrogateVecEnv` (wraps DiffPhotoinjectorEnv,
             no_grad). Zero IPC, zero per-env Python stepping, no torch thread
             oversubscription -- collection runs at SHAC/BPTT throughput. This
             is the right layout whenever the env step is cheap (a tiny MLP).
    subproc  Legacy 16-process SubprocVecEnv, CPU surrogate per worker. Only
             worth it when each env step is genuinely expensive (e.g. the real
             Impact-T env in train_ppo_impact.py), not for this surrogate.
    dummy    Single-process DummyVecEnv (no IPC, but still batch-1 stepping).

The subproc/dummy paths pin each worker to 1 torch thread (set_num_threads(1)
+ OMP_NUM_THREADS=1) to avoid the thread-pool contention that, together with
batch-1 CPU inference and pipe serialization, throttled the old default to
~55 env-steps/s.

Usage:
    python -m photoinjector_rl.emittance_target.train_ppo \\
        --ckpt trained/emittance_target_hifi/checkpoints/best-XXX.ckpt \\
        --norm-json processed/emittance_target_hifi_norm.json \\
        --out-dir trained/ppo_emittance_target \\
        --total-timesteps 500000 \\
        --n-envs 16 \\
        --wandb-project photoinjector-rl

Logging covers W&B + TB + CSV; see the `--wandb-*` flags below for the matrix
of logging modes.

Device handling: `--device` controls PPO's policy / value networks AND, for
the `batched` vec-env, the surrogate (one shared GPU forward for all envs).
For the legacy subproc/dummy paths the worker surrogates run on CPU. The diag
/ eval envs in the main process share `--device` with PPO so their inference
matches deployment.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)

from .callbacks import EpisodeMetricsCallback, RolloutDiagnosticCallback
from .env import PhotoinjectorEnv
from .vec_env import SurrogateVecEnv


def _make_env(
    ckpt: str,
    norm_json: str,
    *,
    seed: int,
    device: str,
    terminal_emit_bonus: float,
    distgen_drift_std: float = 0.0,
):
    """Factory closure used by SubprocVecEnv/DummyVecEnv to build one
    Monitor(env). Pins the worker to a single torch thread: with many workers
    each running a tiny batch-1 MLP, all-core OMP/MKL pools per process cause
    catastrophic thread oversubscription -- the dominant cost in the old
    ~55 fps default. 1 thread/worker + process-level parallelism is correct
    here. (DiffPhotoinjectorEnv's batched path sidesteps this entirely.)"""
    def _thunk():
        import torch
        torch.set_num_threads(1)
        env = PhotoinjectorEnv.from_checkpoint(
            ckpt_path=ckpt, norm_json=norm_json, device=device,
            terminal_emit_bonus=terminal_emit_bonus,
            distgen_drift_std=distgen_drift_std,
        )
        env.reset(seed=seed)
        return Monitor(env)
    return _thunk


def build_vec_env(
    ckpt: str,
    norm_json: str,
    *,
    n_envs: int,
    base_seed: int,
    surrogate_device: str,
    terminal_emit_bonus: float,
    vec_env: str = "batched",
    episode_length: int = 64,
    action_scale: float = 0.05,
    distgen_drift_std: float = 0.0,
):
    """Build the training VecEnv.

    `batched` (default) runs all n_envs as a single GPU-batched surrogate
    forward via `SurrogateVecEnv`; wrapped in `VecMonitor` so SB3 still gets
    per-episode reward/length stats. The legacy `subproc`/`dummy` paths give
    each sub-env its own seed (decorrelated initial knob/distgen samples) and
    pin 1 torch thread per worker; they exist for expensive envs (Impact-T),
    not this cheap surrogate.

    `distgen_drift_std` drives a per-step Gaussian random walk on the hidden
    6-D distgen context (cathode jitter); 0.0 keeps it static (default).
    """
    if vec_env == "batched":
        # surrogate_device is "cpu" for the legacy paths; for batched we run
        # the surrogate where the policy lives so the forward is one GPU kernel.
        venv = SurrogateVecEnv(
            n_envs,
            surrogate_ckpt=ckpt,
            norm_json=norm_json,
            device=surrogate_device,
            seed=base_seed,
            episode_length=episode_length,
            action_scale=action_scale,
            distgen_drift_std=distgen_drift_std,
            terminal_emit_bonus=terminal_emit_bonus,
        )
        # VecMonitor populates info["episode"] on done -> ep_rew_mean etc.
        return VecMonitor(venv)

    env_fns = [
        _make_env(
            ckpt, norm_json,
            seed=base_seed + i,
            device=surrogate_device,
            terminal_emit_bonus=terminal_emit_bonus,
            distgen_drift_std=distgen_drift_std,
        )
        for i in range(n_envs)
    ]
    if vec_env == "dummy" or n_envs == 1:
        return DummyVecEnv(env_fns)
    # 'spawn' is the safe start method for CUDA + multiprocessing; we use CPU
    # surrogates anyway but 'spawn' also avoids forking the parent's torch
    # state, which is a frequent source of nondeterminism.
    return SubprocVecEnv(env_fns, start_method="spawn")


def build_single_env(
    ckpt: str,
    norm_json: str,
    *,
    seed: int,
    device: str,
    terminal_emit_bonus: float,
    distgen_drift_std: float = 0.0,
) -> Monitor:
    """Single Monitor(env) for eval / diag rollouts (main process)."""
    env = PhotoinjectorEnv.from_checkpoint(
        ckpt_path=ckpt, norm_json=norm_json, device=device,
        terminal_emit_bonus=terminal_emit_bonus,
        distgen_drift_std=distgen_drift_std,
    )
    env.reset(seed=seed)
    return Monitor(env)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="path to EmittanceMLP checkpoint")
    p.add_argument("--norm-json", required=True, help="path to normalization JSON")
    p.add_argument("--out-dir", default="trained/ppo_emittance_target")

    # PPO hyperparameters.
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=16,
                   help="number of parallel envs (batch dim for the batched "
                        "vec-env; worker processes for subproc).")
    p.add_argument("--vec-env", choices=("batched", "subproc", "dummy"),
                   default="batched",
                   help="rollout-collection layout. 'batched' (default) runs "
                        "all envs as one GPU surrogate forward; 'subproc' is "
                        "the legacy CPU-worker layout (only worth it for "
                        "expensive envs); 'dummy' is single-process batch-1.")
    p.add_argument("--action-scale", type=float, default=0.05,
                   help="Δknob step size (batched vec-env). Matches "
                        "PhotoinjectorEnv / the diff_rl configs.")
    p.add_argument("--episode-length", type=int, default=64,
                   help="steps per episode (batched vec-env).")
    p.add_argument("--distgen-drift-std", type=float, default=0.0,
                   help="per-step Gaussian random-walk std on the hidden 6-D "
                        "distgen context (normalized units), modelling "
                        "cathode jitter. 0 = static context (default). "
                        "Applied to the training vec-env AND the eval/diag "
                        "envs so PPO trains and is evaluated under the same "
                        "drift regime, matching the SHAC/BPTT cfg knob.")
    p.add_argument("--n-steps", type=int, default=256,
                   help="rollout length per env per update")
    p.add_argument("--batch-size", type=int, default=256,
                   help="must divide n_envs * n_steps")
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=float, default=0.0,
                   help="entropy bonus coefficient. 0 = SB3 default for "
                        "continuous control. Bump to 0.01 if exploration "
                        "stalls.")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None,
                   help="early-stop the PPO epoch if approx KL exceeds this. "
                        "Default None = rely on clip-range alone.")
    p.add_argument("--policy", default="MlpPolicy")

    # Eval + checkpoints.
    p.add_argument("--eval-freq", type=int, default=5_000,
                   help="run an eval rollout every N main-process steps; "
                        "SB3 scales internally for vec envs.")
    p.add_argument("--n-eval-episodes", type=int, default=5)
    p.add_argument("--ckpt-freq", type=int, default=20_000)
    p.add_argument("--diag-seeds", type=int, nargs="+",
                   default=[9999, 9998, 9997, 9996],
                   help="list of seeds for diagnostic rollouts.")

    # Reward shaping (kept off by default so per-step reward is the bare
    # objective; bump --terminal-bonus to weight the episode endpoint).
    p.add_argument("--terminal-bonus", type=float, default=0.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto",
                   help="device for the PPO policy. 'auto' picks cuda:0 if "
                        "available else cpu. Surrogate workers always run "
                        "on CPU regardless of this flag.")
    p.add_argument("--smoke", action="store_true")

    # W&B logging flags.
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-mode", default="online",
                   choices=("online", "offline", "disabled"))
    return p


def resolve_device(device: str) -> str:
    """Translate 'auto' -> 'cuda:0' or 'cpu'."""
    import torch
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            f"--device={device} requested but torch.cuda.is_available() is False. "
            "Pass --device cpu to run on CPU."
        )
    if device.startswith("cuda:"):
        idx = int(device.split(":")[1])
        n = torch.cuda.device_count()
        if idx >= n:
            raise SystemExit(
                f"--device={device} requested but only {n} CUDA device(s) visible."
            )
    return device


def train(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval").mkdir(exist_ok=True)
    (out_dir / "tb").mkdir(exist_ok=True)
    (out_dir / "ckpts").mkdir(exist_ok=True)

    if args.smoke:
        # 4096 = one full PPO rollout at 16 envs * 256 steps.
        args.total_timesteps = 4096
        args.eval_freq = 1024
        args.ckpt_freq = 2048
        args.n_eval_episodes = 2

    if (args.n_envs * args.n_steps) % args.batch_size != 0:
        raise SystemExit(
            f"batch_size={args.batch_size} must divide "
            f"n_envs*n_steps={args.n_envs * args.n_steps}"
        )

    policy_device = resolve_device(args.device)
    # batched: surrogate shares the policy device (one GPU forward for all
    # envs). subproc/dummy: CPU surrogate per worker (the legacy layout).
    surrogate_device = policy_device if args.vec_env == "batched" else "cpu"
    print(f"[train_ppo] policy device: {policy_device}  "
          f"vec-env: {args.vec_env}  surrogate: {surrogate_device}")

    train_env = build_vec_env(
        args.ckpt, args.norm_json,
        n_envs=args.n_envs, base_seed=args.seed,
        surrogate_device=surrogate_device,
        terminal_emit_bonus=args.terminal_bonus,
        vec_env=args.vec_env,
        episode_length=args.episode_length,
        action_scale=args.action_scale,
        distgen_drift_std=args.distgen_drift_std,
    )
    eval_env = build_single_env(
        args.ckpt, args.norm_json,
        seed=args.seed + 1000, device=policy_device,
        terminal_emit_bonus=args.terminal_bonus,
        distgen_drift_std=args.distgen_drift_std,
    )
    diag_env = build_single_env(
        args.ckpt, args.norm_json,
        seed=args.diag_seeds[0], device=policy_device,
        terminal_emit_bonus=args.terminal_bonus,
        distgen_drift_std=args.distgen_drift_std,
    )

    # wandb.init MUST run before model.set_logger (it monkey-patches
    # torch.utils.tensorboard.SummaryWriter at init time).
    wandb_run = None
    WandbCallback = None
    if args.wandb_project:
        import wandb
        from wandb.integration.sb3 import WandbCallback as _WandbCallback
        WandbCallback = _WandbCallback

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=vars(args),
            sync_tensorboard=True,
            dir=str(out_dir),
            save_code=False,
        )
        print(f"[train_ppo] wandb run: {wandb_run.url if wandb_run else '(local)'}")

    tb_dir = str(out_dir / "tb")
    model = PPO(
        policy=args.policy,
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        seed=args.seed,
        device=policy_device,
        tensorboard_log=tb_dir,
        verbose=1,
    )

    next_id = get_latest_run_id(tb_dir, "PPO") + 1
    run_dir = os.path.join(tb_dir, f"PPO_{next_id}")
    model.set_logger(configure_logger(
        run_dir, ["stdout", "tensorboard", "csv"],
    ))

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "eval"),
        log_path=str(out_dir / "eval"),
        eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(args.ckpt_freq // max(args.n_envs, 1), 1),
        save_path=str(out_dir / "ckpts"),
        name_prefix="ppo",
    )
    diag_cb = RolloutDiagnosticCallback(
        diag_env=diag_env,
        eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
        out_dir=out_dir / "rollouts",
        diag_seeds=args.diag_seeds,
        wandb_run=wandb_run,
        verbose=1,
    )
    metrics_cb = EpisodeMetricsCallback()
    callbacks: list = [eval_cb, ckpt_cb, diag_cb, metrics_cb]

    if WandbCallback is not None:
        callbacks.append(WandbCallback(
            model_save_freq=max(args.ckpt_freq // max(args.n_envs, 1), 1),
            model_save_path=str(out_dir / "wandb_models"),
            log="all",
            verbose=1,
        ))

    try:
        model.learn(total_timesteps=args.total_timesteps,
                    callback=callbacks,
                    progress_bar=False)
        model.save(out_dir / "ppo_final")
    finally:
        train_env.close()
        if wandb_run is not None:
            wandb_run.finish()

    result = {
        "final_path": str(out_dir / "ppo_final.zip"),
        "best_eval_path": str(out_dir / "eval" / "best_model.zip"),
        "tensorboard": str(out_dir / "tb"),
        "total_timesteps": args.total_timesteps,
    }
    if wandb_run is not None:
        result["wandb_url"] = wandb_run.url if hasattr(wandb_run, "url") else None
    return result


def main() -> None:
    args = build_argparser().parse_args()
    result = train(args)
    print("\nDone.")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
