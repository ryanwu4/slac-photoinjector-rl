"""
Gymnasium env that drops the surrogate forward pass for a real Impact-T run.

Subclasses `PhotoinjectorEnv` and only overrides `_forward()`. The rest of the
Gym wiring (action clipping, knob bounds, drift, obs/info packing, reward
shape, reset, max_steps) is inherited unchanged so the policy network
trained on the surrogate stays drop-in-compatible.

Per-step Impact-T runs in a fresh temp workdir; the resulting in-memory
`I.output["particles"][marker]` particle group is consulted for
`norm_emit_4d`, then we apply the SAME log10 + z-score the surrogate
training used (so the warm-started PPO value function stays in-distribution).
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import SETTING_BOUNDS, SETTING_KEYS
from .env import N_DISTGEN, N_KNOB, PhotoinjectorEnv

# Build SETTING_KEYS -> (low, high) arrays once. The order matches the
# canonical 11-D vector used by the surrogate input.
_ALL_LO = np.array([SETTING_BOUNDS[k][0] for k in SETTING_KEYS], dtype=np.float64)
_ALL_HI = np.array([SETTING_BOUNDS[k][1] for k in SETTING_KEYS], dtype=np.float64)

# Constants pinned by configs/sweep/lhs_train.yaml `vocs.constants`. The xopt
# sweep that produced the surrogate's training set merged these into every
# settings dict, so the surrogate effectively learned f(11 knobs | these
# constants). Reproducing them here keeps the env's transition function in
# the same fidelity regime as the surrogate -- 2k particles, 8^3 mesh, etc.
DEFAULT_LHS_CONSTANTS: dict[str, float] = {
    "distgen:total_charge": 500,
    "distgen:n_particle": 2000,
    "header:Nx": 8,
    "header:Ny": 8,
    "header:Nz": 8,
    "numprocs": 1,
    "stop_1:s": 0.95,
}


def _extract_norm_emit_4d(I, marker: str) -> float:
    """Pull norm_emit_4d at `marker` from an Impact object after I.run().

    lume-impact's `I.particles[marker]` may return either a ParticleGroup
    directly or a dict keyed by species — handle both forms.
    """
    pg_or_dict = I.particles[marker]
    if hasattr(pg_or_dict, "norm_emit_4d"):
        pg = pg_or_dict
    else:
        # dict-of-species form: prefer 'electron' if present, else first value.
        if "electron" in pg_or_dict:
            pg = pg_or_dict["electron"]
        else:
            pg = next(iter(pg_or_dict.values()))
    return float(pg.norm_emit_4d)


class ImpactPhotoinjectorEnv(PhotoinjectorEnv):
    """RL env that calls Impact-T per step instead of a trained surrogate.

    Designed for fine-tuning a surrogate-trained PPO policy on the real
    simulator. The action / observation spaces and step semantics match
    `PhotoinjectorEnv` exactly so a saved policy zip can be loaded
    against this env with no adapter layer.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        impact_config: str | Path,
        distgen_input_file: str | Path,
        target_mean: float,
        target_std: float,
        *,
        action_scale: float = 0.05,
        distgen_drift_std: float = 0.0,
        max_steps: int = 32,
        workdir_root: str | Path | None = None,
        archive_path: str | Path | None = None,
        marker: str = "PR10241",
        seed: int | None = None,
        failure_penalty_sigma: float = 5.0,
        constants: dict[str, float] | None = None,
    ):
        # Build the parent with a no-op surrogate placeholder. The placeholder
        # is never called because we override _forward(); we just need
        # PhotoinjectorEnv.__init__ to wire up the Gym spaces and state vars.
        def _noop_surrogate(x):
            return torch.zeros((x.shape[0] if hasattr(x, "shape") else 1, 1))

        super().__init__(
            surrogate=_noop_surrogate,
            target_mean=target_mean,
            target_std=target_std,
            action_scale=action_scale,
            distgen_drift_std=distgen_drift_std,
            max_steps=max_steps,
            seed=None,                # delay reset until after our state is built
            device="cpu",
        )
        self._impact_config = str(impact_config)
        self._distgen_input_file = str(distgen_input_file)
        self._workdir_root = str(workdir_root) if workdir_root else None
        self._archive_path = str(archive_path) if archive_path else None
        self._marker = marker
        # Settings that get merged into every Impact-T call. Default reproduces
        # the lhs_train.yaml `vocs.constants` block so the env matches the
        # fidelity the surrogate was trained against (2k particles, 8^3 mesh).
        self._constants = dict(DEFAULT_LHS_CONSTANTS if constants is None else constants)
        # Penalty z-score returned by _forward() when Impact-T fails. ~5 sigma
        # above the training-set mean log-emit -> roughly an order of magnitude
        # worse, big enough to discourage but not so big it explodes value loss.
        self._failure_penalty = float(failure_penalty_sigma)
        self._failure_count: int = 0

        if seed is not None:
            self.reset(seed=seed)

    # ----- internal helpers -------------------------------------------------

    def _build_settings_dict(self) -> dict[str, float]:
        """Denormalize the current (5 knobs + 6 distgen) state to a flat
        physical-units dict keyed by SETTING_KEYS, then merge in the
        fidelity-pinning constants (particle count, mesh, total_charge,
        etc.) from `self._constants`. This shape is what
        custom_evaluate_impact_with_distgen expects (see data/evaluate.py)."""
        norm = np.concatenate([self._knobs_norm, self._distgen_norm]).astype(np.float64)
        phys = _ALL_LO + norm * (_ALL_HI - _ALL_LO)
        settings = {k: float(v) for k, v in zip(SETTING_KEYS, phys)}
        # Constants last -- xopt's behaviour is to override variables only if
        # there's a collision, but the 11 sampled keys are disjoint from the
        # constants block by design.
        settings.update(self._constants)
        return settings

    def _forward(self) -> float:
        """Run Impact-T at the current state, return z-scored log-emittance.

        On any exception, bump the failure counter and return a large
        positive z-score (reward = -y_norm so this is a strong negative
        reward). The episode is not terminated -- the agent gets to back
        off via the next action.
        """
        from ..data.evaluate import custom_evaluate_impact_with_distgen

        settings = self._build_settings_dict()
        with tempfile.TemporaryDirectory(dir=self._workdir_root) as wd:
            try:
                result = custom_evaluate_impact_with_distgen(
                    settings=settings,
                    distgen_input_file=self._distgen_input_file,
                    impact_config=self._impact_config,
                    workdir=wd,
                    archive_path=self._archive_path,
                    merit_f=lambda I: {
                        "norm_emit_4d": _extract_norm_emit_4d(I, self._marker),
                        "error": False,
                    },
                )
            except Exception:  # noqa: BLE001 — defensive on simulator failure
                self._failure_count += 1
                return self._failure_penalty

        emit_m2 = float(result["norm_emit_4d"])
        if not math.isfinite(emit_m2) or emit_m2 <= 0:
            self._failure_count += 1
            return self._failure_penalty
        return (math.log10(emit_m2) - self._target_mean) / self._target_std

    # ----- info dict augmentation -------------------------------------------

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        info["failure_count"] = self._failure_count
        return info

    # ----- production-path constructor --------------------------------------

    @classmethod
    def from_norm_json(
        cls,
        norm_json: str | Path,
        impact_config: str | Path,
        distgen_input_file: str | Path,
        **kwargs: Any,
    ) -> "ImpactPhotoinjectorEnv":
        """Build using the SAME normalization stats the surrogate trained
        against, so reward distribution matches the surrogate-trained
        policy's expectations. The lhs_train.yaml fidelity constants
        (2k particles, 8^3 mesh, etc.) are applied via the default
        `constants=DEFAULT_LHS_CONSTANTS` argument unless overridden in
        kwargs."""
        with open(norm_json) as f:
            norm = json.load(f)
        return cls(
            impact_config=impact_config,
            distgen_input_file=distgen_input_file,
            target_mean=norm["target_mean"],
            target_std=norm["target_std"],
            **kwargs,
        )
