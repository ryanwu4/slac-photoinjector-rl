"""
SHAC training entry point for first-order MBRL over the flow surrogate.

Reward = a property computed from the flow's sampled output bunch (default
norm_emit_4d), backpropagated through the bunch to the 5 control knobs. Reuses
`emittance_target.diffrl.SHAC` UNCHANGED; only the env (FlowBunchEnv) differs
from the v1 scalar-MLP setup.

Usage:
    python -m photoinjector_rl.surrogates.flow.train_shac \
        --cfg configs/diff_rl/shac_flow.yaml \
        --flow-ckpt trained/flow_surrogate/checkpoints/best-epoch=493-val_loss=-0.9555.ckpt \
        --norm-json processed/flow_surrogate_norm.json \
        --logdir logs/shac_flow --seed 0 --max-epochs 500
"""
from __future__ import annotations

import argparse
import csv
import os
from functools import partial
from pathlib import Path

import yaml

# diffrl is reused exactly as-is (handwritten for a related project).
from photoinjector_rl.diffrl import SHAC

from .diff_env import FlowBunchEnv
from .moving_shape_cli import (add_moving_shape_args, apply_moving_shape_overrides,
                               build_moving_env_fn, make_progress_hook)
from .shape_env import ShapeTargetEnv


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True, type=str)
    p.add_argument("--flow-ckpt", required=True, type=str,
                   help="ConditionalAffineFlow Lightning checkpoint.")
    p.add_argument("--norm-json", required=True, type=str,
                   help="flow_surrogate processed _norm.json (also locates the "
                        "processed .h5 used to z-score non-emittance properties).")
    p.add_argument("--processed", default=None, type=str,
                   help="processed .h5 for property z-score; defaults to "
                        "<norm-json with _norm.json->.h5>.")
    p.add_argument("--property", default=None, type=str,
                   help="bunch property to optimize (default from cfg / norm_emit_4d).")
    p.add_argument("--reward-mode", default=None, type=str,
                   choices=["minimize", "maximize", "target"])
    p.add_argument("--target", default=None, type=float)
    p.add_argument("--n-particles", default=None, type=int)
    # Joint aspect+tilt control (ShapeTargetEnv): if --shape-aspect is set, the
    # env targets the (s1,s2) shape vector for the given eigen aspect + tilt.
    p.add_argument("--shape-aspect", default=None, type=float,
                   help="target eigen aspect ratio (>=1); enables ShapeTargetEnv.")
    p.add_argument("--shape-tilt-deg", default=0.0, type=float,
                   help="target tilt angle in degrees (with --shape-aspect).")
    p.add_argument("--shape-scale", default=None, type=float,
                   help="shape-target reward scale (O(1)); default from cfg / 0.3.")
    add_moving_shape_args(p)
    p.add_argument("--logdir", default=None, type=str)
    p.add_argument("--seed", default=None, type=int)
    p.add_argument("--device", default=None, type=str)
    p.add_argument("--max-epochs", default=None, type=int)
    p.add_argument("--num-actors", default=None, type=int)
    p.add_argument("--steps-num", default=None, type=int)
    p.add_argument("--action-scale", default=None, type=float)
    p.add_argument("--distgen-drift-std", default=None, type=float)
    p.add_argument("--checkpoint", default=None, type=str,
                   help="Trained policy .pt for --play.")
    p.add_argument("--play", action="store_true")
    return p.parse_args()


def _override(cfg: dict, args: argparse.Namespace) -> dict:
    g, c, de = cfg["params"]["general"], cfg["params"]["config"], cfg["params"]["diff_env"]
    if args.seed is not None:
        g["seed"] = args.seed
    if args.device is not None:
        g["device"] = args.device
    if args.logdir is not None:
        g["logdir"] = args.logdir
    if args.max_epochs is not None:
        c["max_epochs"] = args.max_epochs
    if args.num_actors is not None:
        c["num_actors"] = args.num_actors
    if args.steps_num is not None:
        c["steps_num"] = args.steps_num
    if args.action_scale is not None:
        de["action_scale"] = args.action_scale
    if args.distgen_drift_std is not None:
        de["distgen_drift_std"] = args.distgen_drift_std
    if args.property is not None:
        de["property"] = args.property
    if args.reward_mode is not None:
        de["reward_mode"] = args.reward_mode
    if args.target is not None:
        de["target"] = args.target
    if args.n_particles is not None:
        de["n_particles"] = args.n_particles
    if args.shape_aspect is not None:
        de["shape_aspect"] = args.shape_aspect
        de["shape_tilt_deg"] = args.shape_tilt_deg
        if args.shape_scale is not None:
            de["shape_scale"] = args.shape_scale
    apply_moving_shape_overrides(de, args)
    if args.play:
        g["train"] = False
        g["checkpoint"] = args.checkpoint
    return cfg


def _build_env_fn(cfg: dict, flow_ckpt: str, norm_json: str, processed: str | None):
    """Wrap FlowBunchEnv (or ShapeTargetEnv) so SHAC can call it with its standard
    env kwargs."""
    de = cfg["params"]["diff_env"]
    if de.get("shape_aspect") is not None:
        return partial(
            ShapeTargetEnv,
            flow_ckpt=flow_ckpt, norm_json=norm_json, processed_h5=processed,
            target_aspect=de["shape_aspect"],
            target_tilt_deg=de.get("shape_tilt_deg", 0.0),
            scale=de.get("shape_scale", 0.3),
            n_particles=de.get("n_particles", 512),
            action_scale=de.get("action_scale", 0.05),
            distgen_drift_std=de.get("distgen_drift_std", 0.0),
        )
    return partial(
        FlowBunchEnv,
        flow_ckpt=flow_ckpt,
        norm_json=norm_json,
        processed_h5=processed,
        property=de.get("property", "norm_emit_4d"),
        reward_mode=de.get("reward_mode", "minimize"),
        target=de.get("target", None),
        n_particles=de.get("n_particles", 512),
        action_scale=de.get("action_scale", 0.05),
        distgen_drift_std=de.get("distgen_drift_std", 0.0),
    )


def _attach_csv_hook(algo: SHAC, logdir: str) -> None:
    os.makedirs(logdir, exist_ok=True)
    csv_path = os.path.join(logdir, "learning_curve.csv")
    f = open(csv_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["step", "mean_episode_loss", "wall_time"])

    def hook(step: int, mean_loss: float, wall: float) -> None:
        writer.writerow([step, mean_loss, wall])
        f.flush()

    algo.step_metrics_hook = hook


def main() -> None:
    args = _parse_args()
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    cfg = _override(cfg, args)
    Path(cfg["params"]["general"]["logdir"]).mkdir(parents=True, exist_ok=True)

    de = cfg["params"]["diff_env"]
    curriculum = None
    if de.get("moving_shape"):
        env_fn, curriculum = build_moving_env_fn(
            de, args.flow_ckpt, args.norm_json, args.processed)
    else:
        env_fn = _build_env_fn(cfg, args.flow_ckpt, args.norm_json, args.processed)
    algo = SHAC(cfg, env_fn=env_fn)
    if cfg["params"]["general"]["train"]:
        _attach_csv_hook(algo, cfg["params"]["general"]["logdir"])
        if curriculum is not None:
            algo.step_metrics_hook = make_progress_hook(
                algo.step_metrics_hook, cfg, curriculum, ramp=curriculum.enabled)
        algo.train()
    else:
        algo.play(cfg)


if __name__ == "__main__":
    main()
