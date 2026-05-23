"""
Full-episode BPTT training entry point for the photoinjector surrogate env.

Identical CLI surface to `train_shac.py` (minus critic-specific flags).
"""
from __future__ import annotations

import argparse
import csv
import os
from functools import partial
from pathlib import Path

import yaml

from .diff_env import DiffPhotoinjectorEnv
from .diffrl import BPTT


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True, type=str)
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--norm-json", required=True, type=str)
    p.add_argument("--logdir", default=None, type=str)
    p.add_argument("--seed", default=None, type=int)
    p.add_argument("--device", default=None, type=str)
    p.add_argument("--max-epochs", default=None, type=int)
    p.add_argument("--num-actors", default=None, type=int)
    p.add_argument("--steps-num", default=None, type=int)
    p.add_argument("--action-scale", default=None, type=float)
    p.add_argument("--distgen-drift-std", default=None, type=float)
    p.add_argument("--checkpoint", default=None, type=str)
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
    de = cfg["params"]["diff_env"]
    return partial(
        DiffPhotoinjectorEnv,
        surrogate_ckpt=ckpt,
        norm_json=norm_json,
        action_scale=de.get("action_scale", 0.05),
        distgen_drift_std=de.get("distgen_drift_std", 0.0),
    )


def _attach_csv_hook(algo: BPTT, logdir: str) -> None:
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

    env_fn = _build_env_fn(cfg, args.ckpt, args.norm_json)
    algo = BPTT(cfg, env_fn=env_fn)
    if cfg["params"]["general"]["train"]:
        _attach_csv_hook(algo, cfg["params"]["general"]["logdir"])
        algo.train()
    else:
        algo.play(cfg)


if __name__ == "__main__":
    main()
