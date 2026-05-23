"""
Train Stable-Baselines3 PPO against the PhotoinjectorEnv surrogate-driven env.

On-policy ablation baseline mirroring train_sac.py. PPO is sample-hungrier than
SAC in principle (5-10x more transitions), but the surrogate is cheap and PPO
scales naturally with parallel envs -- a SubprocVecEnv with n_envs=16 closes
the wall-clock gap.

Usage:
    python -m photoinjector_rl.emittance_target.train_ppo \\
        --ckpt trained/emittance_target/checkpoints/best-XXX.ckpt \\
        --norm-json processed/emittance_target_norm.json \\
        --out-dir trained/ppo_emittance_target \\
        --total-timesteps 500000 \\
        --n-envs 16 \\
        --wandb-project photoinjector-rl

Logging mirrors train_sac.py exactly (W&B + TB + CSV). See that file's
docstring for the matrix of `--wandb-*` modes.

Device handling: `--device` controls PPO's policy / value networks. The
surrogate inside each SubprocVecEnv worker is always loaded on CPU (sub-ms
inference, sidesteps the CUDA-in-forked-subprocess issue, and saves 16x
GPU-context overhead for a tiny MLP). The diag / eval envs in the main
process share `--device` with PPO so their inference matches deployment.
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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .callbacks import EpisodeMetricsCallback, RolloutDiagnosticCallback
from .env import PhotoinjectorEnv


def _make_env(
    ckpt: str,
    norm_json: str,
    *,
    seed: int,
    device: str,
    terminal_emit_bonus: float,
):
    """Factory closure used by SubprocVecEnv to construct one Monitor(env)."""
    def _thunk():
        env = PhotoinjectorEnv.from_checkpoint(
            ckpt_path=ckpt, norm_json=norm_json, device=device,
            terminal_emit_bonus=terminal_emit_bonus,
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
):
    """Build the training VecEnv. Each sub-env gets its own seed so initial
    knob/distgen samples decorrelate across workers from step 0."""
    env_fns = [
        _make_env(
            ckpt, norm_json,
            seed=base_seed + i,
            device=surrogate_device,
            terminal_emit_bonus=terminal_emit_bonus,
        )
        for i in range(n_envs)
    ]
    if n_envs == 1:
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
) -> Monitor:
    """Single Monitor(env) for eval / diag rollouts (main process)."""
    env = PhotoinjectorEnv.from_checkpoint(
        ckpt_path=ckpt, norm_json=norm_json, device=device,
        terminal_emit_bonus=terminal_emit_bonus,
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
                   help="parallel rollout workers in SubprocVecEnv")
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
                        "SB3 scales internally for vec envs so this stays "
                        "comparable to the SAC value.")
    p.add_argument("--n-eval-episodes", type=int, default=5)
    p.add_argument("--ckpt-freq", type=int, default=20_000)
    p.add_argument("--diag-seeds", type=int, nargs="+",
                   default=[9999, 9998, 9997, 9996],
                   help="list of seeds for diagnostic rollouts; matches the "
                        "SAC defaults so the diag PNGs are directly comparable.")

    # Reward shaping (kept off by default; PPO-vs-SAC comparison should be
    # apples-to-apples, and SAC v3's best result also used 0).
    p.add_argument("--terminal-bonus", type=float, default=0.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto",
                   help="device for the PPO policy. 'auto' picks cuda:0 if "
                        "available else cpu. Surrogate workers always run "
                        "on CPU regardless of this flag.")
    p.add_argument("--smoke", action="store_true")

    # W&B (identical surface to train_sac.py).
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
    print(f"[train_ppo] policy device: {policy_device}  surrogate (workers): cpu")

    train_env = build_vec_env(
        args.ckpt, args.norm_json,
        n_envs=args.n_envs, base_seed=args.seed,
        surrogate_device="cpu",
        terminal_emit_bonus=args.terminal_bonus,
    )
    eval_env = build_single_env(
        args.ckpt, args.norm_json,
        seed=args.seed + 1000, device=policy_device,
        terminal_emit_bonus=args.terminal_bonus,
    )
    diag_env = build_single_env(
        args.ckpt, args.norm_json,
        seed=args.diag_seeds[0], device=policy_device,
        terminal_emit_bonus=args.terminal_bonus,
    )

    # wandb.init MUST run before model.set_logger (it monkey-patches
    # torch.utils.tensorboard.SummaryWriter at init time). See train_sac.py.
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
