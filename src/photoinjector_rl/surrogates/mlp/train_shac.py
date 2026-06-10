"""
SHAC training entry point for the photoinjector surrogate env.

Usage:
    python -m photoinjector_rl.surrogates.mlp.train_shac \
        --cfg configs/diff_rl/shac_photoinjector.yaml \
        --ckpt trained/emittance_target_hifi/checkpoints/best-epoch=191-val_loss=0.0060.ckpt \
        --norm-json processed/emittance_target_hifi_norm.json \
        --logdir logs/shac_smoke --seed 0 --max-epochs 2

The YAML mirrors NVlabs/DiffRL's cfg layout. CLI flags can override the
common values without editing the YAML — useful for sweeping seeds or
matching a fixed env-step budget across algorithms.
"""
from __future__ import annotations

import argparse
import csv
import os
from functools import partial
from pathlib import Path

import yaml

from .diff_env import DiffPhotoinjectorEnv
from photoinjector_rl.diffrl import SHAC


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True, type=str)
    p.add_argument("--ckpt", required=True, type=str,
                   help="EmittanceMLP Lightning checkpoint.")
    p.add_argument("--norm-json", required=True, type=str)
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
    if args.seed is not None:
        cfg["params"]["general"]["seed"] = args.seed
    if args.device is not None:
        cfg["params"]["general"]["device"] = args.device
    if args.logdir is not None:
        cfg["params"]["general"]["logdir"] = args.logdir
    if args.max_epochs is not None:
        cfg["params"]["config"]["max_epochs"] = args.max_epochs
    if args.num_actors is not None:
        cfg["params"]["config"]["num_actors"] = args.num_actors
    if args.steps_num is not None:
        cfg["params"]["config"]["steps_num"] = args.steps_num
    if args.action_scale is not None:
        cfg["params"]["diff_env"]["action_scale"] = args.action_scale
    if args.distgen_drift_std is not None:
        cfg["params"]["diff_env"]["distgen_drift_std"] = args.distgen_drift_std
    if args.play:
        cfg["params"]["general"]["train"] = False
        cfg["params"]["general"]["checkpoint"] = args.checkpoint
    return cfg


def _build_env_fn(cfg: dict, ckpt: str, norm_json: str):
    """Wrap DiffPhotoinjectorEnv so SHAC can call it with its standard kwargs."""
    diff_env_cfg = cfg["params"]["diff_env"]
    return partial(
        DiffPhotoinjectorEnv,
        surrogate_ckpt=ckpt,
        norm_json=norm_json,
        action_scale=diff_env_cfg.get("action_scale", 0.05),
        distgen_drift_std=diff_env_cfg.get("distgen_drift_std", 0.0),
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
    # leaks a file handle on KeyboardInterrupt; harmless for one-shot runs.


def main() -> None:
    args = _parse_args()
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    cfg = _override(cfg, args)
    Path(cfg["params"]["general"]["logdir"]).mkdir(parents=True, exist_ok=True)

    env_fn = _build_env_fn(cfg, args.ckpt, args.norm_json)
    algo = SHAC(cfg, env_fn=env_fn)
    if cfg["params"]["general"]["train"]:
        _attach_csv_hook(algo, cfg["params"]["general"]["logdir"])
        algo.train()
    else:
        algo.play(cfg)


if __name__ == "__main__":
    main()
