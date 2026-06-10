"""
LEGACY differentiable env for the v1 scalar EmittanceMLP surrogate.

Thin subclass of the surrogate-agnostic base
`photoinjector_rl.core.diff_env.DiffPhotoinjectorEnv`: it adds only the
EmittanceMLP checkpoint-loading convenience (`_load_surrogate`). All env
dynamics, gradient management, and the DiffRL API live in the base. The 5-knob /
6-distgen dimensional constants are re-exported here so existing
`from .diff_env import DiffPhotoinjectorEnv, N_KNOB, ...` imports keep working.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from photoinjector_rl.core.diff_env import (  # noqa: F401  (re-exported)
    ACTION_DIM,
    DiffPhotoinjectorEnv as _BaseDiffPhotoinjectorEnv,
    N_DISTGEN,
    N_INPUT,
    N_KNOB,
    OBS_DIM,
)

from .model import EmittanceMLP


class DiffPhotoinjectorEnv(_BaseDiffPhotoinjectorEnv):
    """Base env + the v1 EmittanceMLP checkpoint loader."""

    def _load_surrogate(self, surrogate_ckpt: str | Path | None,
                        norm_json: str | Path | None
                        ) -> tuple[torch.nn.Module, float, float]:
        if surrogate_ckpt is None or norm_json is None:
            raise ValueError(
                "Either pass a pre-loaded `surrogate` (with target_mean / "
                "target_std) or `surrogate_ckpt` + `norm_json` paths.")
        surrogate = EmittanceMLP.load_from_checkpoint(
            str(surrogate_ckpt), map_location=self.device)
        with open(norm_json) as f:
            norm = json.load(f)
        return surrogate, float(norm["target_mean"]), float(norm["target_std"])
