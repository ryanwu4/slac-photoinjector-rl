"""
Fine-tune the surrogate-trained PPO policy directly against Impact-T.

Warm-starts from a `--warm-start` zip (default: trained/ppo_v1/eval/best_model.zip),
swaps the env from surrogate -> ImpactPhotoinjectorEnv, and runs a short
on-policy update phase with conservative hyperparameters.

The env reuses the SAME normalization JSON the surrogate trained against,
so reward distribution stays in-distribution and the loaded value function
remains useful.

Usage:
    python -m photoinjector_rl.emittance_target.train_ppo_impact \\
        --warm-start trained/ppo_v1/eval/best_model.zip \\
        --impact-config configs/impact/ImpactT_config.yaml \\
        --distgen-input configs/impact/distgen_template.yaml \\
        --norm-json processed/emittance_target_norm.json \\
        --out-dir trained/ppo_impact_v1 \\
        --total-timesteps 2048

Wall-clock budget: ~45 min at default settings (8 workers * ~10s/Impact-T).
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
from .impact_env import ImpactPhotoinjectorEnv


def _make_env(
    impact_config: str,
    distgen_input: str,
    norm_json: str,
    *,
    seed: int,
    max_steps: int,
    action_scale: float,
    distgen_drift_std: float,
    workdir_root: str | None,
    archive_path: str | None,
):
    """SubprocVecEnv factory. Each worker constructs its own
    ImpactPhotoinjectorEnv from the shared config files."""
    def _thunk():
        env = ImpactPhotoinjectorEnv.from_norm_json(
            norm_json=norm_json,
            impact_config=impact_config,
            distgen_input_file=distgen_input,
            action_scale=action_scale,
            distgen_drift_std=distgen_drift_std,
            max_steps=max_steps,
            workdir_root=workdir_root,
            archive_path=archive_path,
        )
        env.reset(seed=seed)
        return Monitor(env)
    return _thunk


def build_vec_env(args, base_seed: int):
    env_fns = [
        _make_env(
            args.impact_config, args.distgen_input, args.norm_json,
            seed=base_seed + i,
            max_steps=args.max_steps,
            action_scale=args.action_scale,
            distgen_drift_std=args.distgen_drift_std,
            workdir_root=args.workdir_root,
            archive_path=args.archive_path,
        )
        for i in range(args.n_envs)
    ]
    if args.n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method="spawn")


def build_single_env(args, seed: int) -> Monitor:
    env = ImpactPhotoinjectorEnv.from_norm_json(
        norm_json=args.norm_json,
        impact_config=args.impact_config,
        distgen_input_file=args.distgen_input,
        action_scale=args.action_scale,
        distgen_drift_std=args.distgen_drift_std,
        max_steps=args.max_steps,
        workdir_root=args.workdir_root,
        archive_path=args.archive_path,
    )
    env.reset(seed=seed)
    return Monitor(env)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warm-start",
                   default="trained/ppo_v1/eval/best_model.zip",
                   help="path to PPO zip to fine-tune from. Pass 'none' to "
                        "train from scratch (rare; usually you want the "
                        "surrogate-trained policy as the warm start).")
    p.add_argument("--impact-config", required=True,
                   help="path to Impact-T yaml config "
                        "(e.g. configs/impact/ImpactT_config.yaml)")
    p.add_argument("--distgen-input", required=True,
                   help="path to distgen yaml template "
                        "(e.g. configs/impact/distgen_template.yaml)")
    p.add_argument("--norm-json", required=True,
                   help="path to the normalization JSON the surrogate "
                        "trained against (same z-score keeps reward "
                        "distribution in-distribution for the warm-start "
                        "value fn)")
    p.add_argument("--out-dir", default="trained/ppo_impact_emittance_target")
    p.add_argument("--workdir-root", default=None,
                   help="parent dir for per-step Impact-T temp dirs. None = "
                        "system /tmp.")
    p.add_argument("--archive-path", default=None,
                   help="if set, archive every Impact-T run to this dir. "
                        "Off by default (saves disk).")

    # PPO hyperparameters -- tighter defaults than scratch training.
    p.add_argument("--total-timesteps", type=int, default=2048)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--clip-range", type=float, default=0.1)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None)

    # Env knobs.
    p.add_argument("--max-steps", type=int, default=32)
    p.add_argument("--action-scale", type=float, default=0.05)
    p.add_argument("--distgen-drift-std", type=float, default=0.0)

    # Eval + checkpoint + diag (small budgets, each fired callback costs
    # max_steps Impact-T runs).
    p.add_argument("--eval-freq", type=int, default=256,
                   help="env-step count between EvalCallback fires.")
    p.add_argument("--n-eval-episodes", type=int, default=2)
    p.add_argument("--ckpt-freq", type=int, default=512)
    p.add_argument("--diag-seeds", type=int, nargs="+",
                   default=[9999, 9998],
                   help="seeds for diagnostic rollouts; keep short -- each "
                        "fired diag costs len(seeds) * max_steps runs.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu",
                   help="device for the PPO policy. CPU is faster than "
                        "GPU for a small MLP policy with small batches.")
    p.add_argument("--smoke", action="store_true",
                   help="1 PPO update = n_envs * n_steps Impact-T runs.")

    # W&B.
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-mode", default="online",
                   choices=("online", "offline", "disabled"))
    return p


def train(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval").mkdir(exist_ok=True)
    (out_dir / "tb").mkdir(exist_ok=True)
    (out_dir / "ckpts").mkdir(exist_ok=True)

    if args.smoke:
        # One PPO update worth of rollouts (default 256 transitions).
        args.total_timesteps = args.n_envs * args.n_steps
        args.eval_freq = max(args.total_timesteps, 1)
        args.ckpt_freq = max(args.total_timesteps, 1)
        args.n_eval_episodes = 1
        args.diag_seeds = list(args.diag_seeds)[:1]

    if (args.n_envs * args.n_steps) % args.batch_size != 0:
        raise SystemExit(
            f"batch_size={args.batch_size} must divide "
            f"n_envs*n_steps={args.n_envs * args.n_steps}"
        )

    print(f"[train_ppo_impact] device: {args.device}")
    print(f"[train_ppo_impact] warm-start: {args.warm_start}")
    print(f"[train_ppo_impact] total Impact-T runs: ~"
          f"{args.total_timesteps + args.n_eval_episodes * args.max_steps + len(args.diag_seeds) * args.max_steps}")

    train_env = build_vec_env(args, base_seed=args.seed)
    eval_env = build_single_env(args, seed=args.seed + 1000)
    diag_env = build_single_env(args, seed=args.diag_seeds[0])

    # wandb.init() MUST precede SummaryWriter creation (it monkey-patches at
    # init time). Mirror the pattern from train_ppo.py.
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
        print(f"[train_ppo_impact] wandb run: {wandb_run.url if wandb_run else '(local)'}")

    tb_dir = str(out_dir / "tb")

    if args.warm_start and args.warm_start.lower() != "none":
        print(f"[train_ppo_impact] loading PPO from {args.warm_start}")
        model = PPO.load(
            args.warm_start, env=train_env, device=args.device,
            tensorboard_log=tb_dir,
        )
        # Override LR + clip-range schedules from the loaded zip with our
        # conservative fine-tune values. SB3 expects callables for these.
        model.learning_rate = args.learning_rate
        model.lr_schedule = (lambda _: args.learning_rate)
        model.clip_range = (lambda _: args.clip_range)
        # The loaded model's rollout buffer is sized for the warm-start config
        # (n_steps_saved). We want a different n_steps for fine-tuning, so we
        # have to call _setup_model() -- but that ALSO reinitializes the policy
        # network. Snapshot the loaded weights and restore after.
        model.n_steps = args.n_steps
        model.n_epochs = args.n_epochs
        model.batch_size = args.batch_size
        saved_state = {k: v.clone() for k, v in model.policy.state_dict().items()}
        model._setup_model()
        model.policy.load_state_dict(saved_state)
        print(f"[train_ppo_impact] restored {len(saved_state)} param tensors "
              f"after rollout-buffer resize")
    else:
        print("[train_ppo_impact] no warm-start; building fresh PPO")
        model = PPO(
            policy="MlpPolicy",
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
            device=args.device,
            tensorboard_log=tb_dir,
            verbose=1,
        )

    next_id = get_latest_run_id(tb_dir, "PPO") + 1
    run_dir = os.path.join(tb_dir, f"PPO_{next_id}")
    model.set_logger(configure_logger(run_dir, ["stdout", "tensorboard", "csv"]))

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
        name_prefix="ppo_impact",
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
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=True,
        )
        model.save(out_dir / "ppo_impact_final")
    finally:
        train_env.close()
        if wandb_run is not None:
            wandb_run.finish()

    return {
        "final_path": str(out_dir / "ppo_impact_final.zip"),
        "best_eval_path": str(out_dir / "eval" / "best_model.zip"),
        "tensorboard": str(out_dir / "tb"),
        "total_timesteps": args.total_timesteps,
    }


def main() -> None:
    args = build_argparser().parse_args()
    result = train(args)
    print("\nDone.")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
