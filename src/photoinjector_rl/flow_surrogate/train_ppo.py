"""
Stable-Baselines3 PPO against the conditional-flow surrogate (black-box / model-
free baseline for the flow-MBRL comparison). Uses the GPU-batched
``FlowSurrogateVecEnv`` so rollout collection runs at SHAC/BPTT throughput.

PPO hyperparameters mirror `emittance_target.train_ppo` so the flow PPO run is
comparable to the v1 PPO run and to the flow SHAC/BPTT runs (same env-step
budget). Writes SB3's `progress.csv` (read by `compare_algos.py`) and
`ppo_final.zip`. Policy eval is done by the compare script on `FlowBunchEnv`
(no separate gym env needed).

Usage:
    python -m photoinjector_rl.flow_surrogate.train_ppo \
        --flow-ckpt trained/flow_surrogate/checkpoints/best-epoch=493-val_loss=-0.9555.ckpt \
        --norm-json processed/flow_surrogate_norm.json \
        --out-dir logs/ppo_flow --total-timesteps 500000 --seed 0
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.vec_env import VecMonitor

from .moving_shape_cli import CurriculumCallback
from .vec_env import FlowSurrogateVecEnv


def resolve_device(device: str) -> str:
    import torch
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device={device} but CUDA unavailable; use --device cpu.")
    return device


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow-ckpt", required=True, help="ConditionalAffineFlow checkpoint")
    p.add_argument("--norm-json", required=True)
    p.add_argument("--processed", default=None,
                   help="processed .h5 for non-emittance property z-score "
                        "(defaults to <norm-json with _norm.json->.h5>).")
    p.add_argument("--out-dir", default="logs/ppo_flow")
    p.add_argument("--property", default="norm_emit_4d")
    p.add_argument("--reward-mode", default="minimize",
                   choices=["minimize", "maximize", "target"])
    p.add_argument("--target", type=float, default=None)
    p.add_argument("--n-particles", type=int, default=512)
    # Joint aspect+tilt control: if --shape-aspect set, target the (s1,s2) shape.
    p.add_argument("--shape-aspect", type=float, default=None,
                   help="target eigen aspect ratio (>=1); enables ShapeTargetEnv.")
    p.add_argument("--shape-tilt-deg", type=float, default=0.0)
    p.add_argument("--shape-scale", type=float, default=0.3)
    # Moving-target (goal-conditioned) (aspect,tilt) tracking (MovingShapeVecEnv).
    p.add_argument("--moving-shape", action="store_true",
                   help="goal-conditioned moving (aspect,tilt) setpoint tracking.")
    p.add_argument("--curriculum", dest="curriculum", action="store_const", const=True,
                   default=None, help="ramp difficulty (override moving-config).")
    p.add_argument("--no-curriculum", dest="curriculum", action="store_const", const=False,
                   help="full difficulty mix from the start (override moving-config).")
    p.add_argument("--moving-config", default=None,
                   help="YAML/JSON with a `curriculum` block (CurriculumConfig fields).")
    p.add_argument("--r-max", type=float, default=None,
                   help="override curriculum.r_max.")

    # PPO hyperparameters (match emittance_target/train_ppo defaults).
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--action-scale", type=float, default=0.05)
    p.add_argument("--episode-length", type=int, default=64)
    p.add_argument("--distgen-drift-std", type=float, default=0.0)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--policy", default="MlpPolicy")
    p.add_argument("--ckpt-freq", type=int, default=0,
                   help="periodic checkpoint every N steps (0 = off).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true")
    return p


def train(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tb").mkdir(exist_ok=True)

    if args.smoke:
        args.total_timesteps = max(args.n_envs * args.n_steps, 2048)

    if (args.n_envs * args.n_steps) % args.batch_size != 0:
        raise SystemExit(
            f"batch_size={args.batch_size} must divide "
            f"n_envs*n_steps={args.n_envs * args.n_steps}")

    device = resolve_device(args.device)
    print(f"[train_ppo_flow] device={device} property={args.property} "
          f"n_particles={args.n_particles}")

    curriculum = None
    if args.moving_shape:
        from .moving_shape_cli import load_moving_config
        from .moving_shape_env import MovingShapeVecEnv
        from .shape_targets import CurriculumConfig, CurriculumState
        cur_dict = dict(load_moving_config(args.moving_config).get("curriculum", {}))
        if args.r_max is not None:
            cur_dict["r_max"] = float(args.r_max)
        if args.curriculum is not None:           # override config only when explicit
            cur_dict["enabled"] = bool(args.curriculum)
        cfg = CurriculumConfig.from_dict(cur_dict)
        curriculum = CurriculumState(progress=(0.0 if cfg.enabled else 1.0),
                                     enabled=cfg.enabled, config=cfg)
        train_env = VecMonitor(MovingShapeVecEnv(
            args.n_envs,
            flow_ckpt=args.flow_ckpt, norm_json=args.norm_json,
            processed_h5=args.processed, device=device, seed=args.seed,
            curriculum=curriculum, n_particles=args.n_particles,
            episode_length=args.episode_length, action_scale=args.action_scale,
            distgen_drift_std=args.distgen_drift_std,
            scale=args.shape_scale,
        ))
    else:
        train_env = VecMonitor(FlowSurrogateVecEnv(
            args.n_envs,
            flow_ckpt=args.flow_ckpt,
            norm_json=args.norm_json,
            processed_h5=args.processed,
            property=args.property,
            reward_mode=args.reward_mode,
            target=args.target,
            n_particles=args.n_particles,
            device=device,
            seed=args.seed,
            episode_length=args.episode_length,
            action_scale=args.action_scale,
            distgen_drift_std=args.distgen_drift_std,
            shape_aspect=args.shape_aspect,
            shape_tilt_deg=args.shape_tilt_deg,
            shape_scale=args.shape_scale,
        ))

    tb_dir = str(out_dir / "tb")
    model = PPO(
        policy=args.policy, env=train_env,
        learning_rate=args.learning_rate, n_steps=args.n_steps,
        batch_size=args.batch_size, n_epochs=args.n_epochs,
        gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=args.clip_range,
        ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
        seed=args.seed, device=device, tensorboard_log=tb_dir, verbose=1,
    )

    # csv logger so compare_algos can read tb/PPO_N/progress.csv.
    next_id = get_latest_run_id(tb_dir, "PPO") + 1
    run_dir = os.path.join(tb_dir, f"PPO_{next_id}")
    model.set_logger(configure_logger(run_dir, ["stdout", "tensorboard", "csv"]))

    callbacks: list = []
    if curriculum is not None and curriculum.enabled:
        callbacks.append(CurriculumCallback(curriculum, args.total_timesteps))
    if args.ckpt_freq > 0:
        (out_dir / "ckpts").mkdir(exist_ok=True)
        callbacks.append(CheckpointCallback(
            save_freq=max(args.ckpt_freq // max(args.n_envs, 1), 1),
            save_path=str(out_dir / "ckpts"), name_prefix="ppo"))

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks,
                    progress_bar=False)
        model.save(out_dir / "ppo_final")
    finally:
        train_env.close()

    return {"final_path": str(out_dir / "ppo_final.zip"),
            "tensorboard": tb_dir, "total_timesteps": args.total_timesteps}


def main() -> None:
    args = build_argparser().parse_args()
    result = train(args)
    print("\nDone.")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
