"""
Shared CLI / wiring helpers for the moving-target (aspect, tilt) controller, so
`train_shac`, `train_bptt`, and `train_ppo` stay DRY. All logic lives in
flow_surrogate/ (diffrl/ + emittance_target/ untouched).

The curriculum `progress` (0->1) is advanced by the driver each epoch (SHAC/BPTT
via `step_metrics_hook`; PPO via an SB3 callback), biasing the per-episode target
difficulty static -> step -> smooth.
"""
from __future__ import annotations

import argparse
from functools import partial

from stable_baselines3.common.callbacks import BaseCallback


def add_moving_shape_args(p: argparse.ArgumentParser) -> None:
    """Add the goal-conditioned moving (aspect,tilt) target-tracking flags.
    The curriculum CONFIG lives in the YAML `diff_env.curriculum` block; these
    CLI flags override individual fields only when explicitly passed (default
    None -> keep the config-file value)."""
    p.add_argument("--moving-shape", action="store_true",
                   help="goal-conditioned moving (aspect,tilt) setpoint tracking "
                        "(MovingShapeEnv, 9-D obs); ignores --property/--shape-aspect.")
    p.add_argument("--curriculum", dest="curriculum", action="store_const", const=True,
                   default=None, help="ramp difficulty static->step->smooth (override cfg).")
    p.add_argument("--no-curriculum", dest="curriculum", action="store_const", const=False,
                   help="full difficulty mix from the start (override cfg).")
    p.add_argument("--r-max", default=None, type=float,
                   help="override curriculum.r_max (reachable disk radius).")
    p.add_argument("--action-rate-penalty", default=None, type=float,
                   help="lambda for -lambda*||Δaction|| smooth-control penalty (0=off).")


def apply_moving_shape_overrides(de: dict, args: argparse.Namespace) -> None:
    """Stash moving-shape settings into cfg['params']['diff_env'] (SHAC/BPTT path).
    `diff_env.curriculum` is a CurriculumConfig dict; CLI flags override fields."""
    if not getattr(args, "moving_shape", False):
        return
    de["moving_shape"] = True
    cur = dict(de.get("curriculum") or {})              # copy the YAML config block
    if args.curriculum is not None:                     # --curriculum / --no-curriculum
        cur["enabled"] = bool(args.curriculum)
    if getattr(args, "r_max", None) is not None:
        cur["r_max"] = float(args.r_max)
    de["curriculum"] = cur
    if getattr(args, "action_rate_penalty", None) is not None:
        de["action_rate_penalty"] = float(args.action_rate_penalty)
    if getattr(args, "shape_scale", None) is not None:
        de["shape_scale"] = float(args.shape_scale)


def build_moving_env_fn(de: dict, flow_ckpt: str, norm_json: str,
                        processed: str | None):
    """Return (env_fn, curriculum) for SHAC/BPTT. env_fn binds the flow + curriculum
    (which holds the CurriculumConfig); SHAC passes num_envs/device/seed/episode_length."""
    from .moving_shape_env import MovingShapeEnv
    from .shape_targets import CurriculumConfig, CurriculumState

    cfg = CurriculumConfig.from_dict(de.get("curriculum"))
    curriculum = CurriculumState(progress=(0.0 if cfg.enabled else 1.0),
                                 enabled=cfg.enabled, config=cfg)
    env_fn = partial(
        MovingShapeEnv,
        flow_ckpt=flow_ckpt, norm_json=norm_json, processed_h5=processed,
        curriculum=curriculum,
        scale=de.get("shape_scale", 0.3),
        action_rate_penalty=de.get("action_rate_penalty", 0.0),
        n_particles=de.get("n_particles", 512),
        action_scale=de.get("action_scale", 0.05),
        distgen_drift_std=de.get("distgen_drift_std", 0.0),
    )
    return env_fn, curriculum


def make_progress_hook(base_hook, cfg: dict, curriculum, ramp: bool):
    """Wrap an existing step_metrics_hook so it also advances curriculum.progress
    = step_count / total_env_steps each epoch (only when `ramp`)."""
    c, de = cfg["params"]["config"], cfg["params"]["diff_env"]
    steps_num = int(c.get("steps_num", de.get("episode_length", 64)))
    total = max(1, int(c["num_actors"]) * steps_num * int(c["max_epochs"]))

    def hook(step: int, mean_loss: float, wall: float) -> None:
        if ramp and curriculum is not None:
            curriculum.progress = min(1.0, float(step) / total)
        if base_hook is not None:
            base_hook(step, mean_loss, wall)

    return hook


def load_moving_config(path: str | None) -> dict:
    """Load a moving-shape config file (YAML or JSON) that may hold a
    `curriculum` block and/or an `eval_trajectories` block. Returns {} if no path."""
    if not path:
        return {}
    import json
    text = open(path).read()
    if str(path).endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(text) or {}
    return json.loads(text)


class CurriculumCallback(BaseCallback):
    """PPO: advance curriculum.progress = num_timesteps / total_timesteps."""

    def __init__(self, curriculum, total_timesteps: int, ramp: bool = True):
        super().__init__()
        self._curriculum = curriculum
        self._total = max(1, int(total_timesteps))
        self._ramp = bool(ramp)

    def _on_step(self) -> bool:
        if self._ramp and self._curriculum is not None:
            self._curriculum.progress = min(1.0, self.num_timesteps / self._total)
        return True
