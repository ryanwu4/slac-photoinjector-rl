"""
Gymnasium environment wrapping the v1 emittance surrogate.

At reset(), the env samples a 6-D distgen state uniformly in normalized [0,1]
and a random initial 5-knob configuration. The agent then controls the 5
Impact-T knobs via small delta-action steps; the 6 distgen params are hidden
context (optionally drifting via a Gaussian random walk to model cathode
jitter). Reward is the negation of the z-scored log-emittance, so an RL
maximizer is equivalent to minimizing norm_emit_4d at PR10241.

This module depends only on the v1 surrogate's `EmittanceMLP` and the
normalization stats produced by `preprocess.py`. The 5-vs-6 split of the
11-D input follows the canonical ordering in `__init__.SETTING_KEYS`.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, SupportsFloat

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from . import SETTING_BOUNDS, SETTING_KEYS

N_KNOB = 5    # SETTING_KEYS[:5] are the 5 controllable Impact knobs
N_DISTGEN = 6 # SETTING_KEYS[5:] are the 6 distgen knobs (hidden context)
N_INPUT = N_KNOB + N_DISTGEN  # 11; must match the surrogate's input_dim

SurrogateFn = Callable[[torch.Tensor], torch.Tensor]


def _denorm_knobs(knobs_norm: np.ndarray) -> np.ndarray:
    """Map 5-D normalized knobs in [0,1] back to physical units."""
    lows = np.array([SETTING_BOUNDS[k][0] for k in SETTING_KEYS[:N_KNOB]],
                    dtype=np.float32)
    highs = np.array([SETTING_BOUNDS[k][1] for k in SETTING_KEYS[:N_KNOB]],
                     dtype=np.float32)
    return lows + knobs_norm * (highs - lows)


class PhotoinjectorEnv(gym.Env):
    """RL env that uses the emittance surrogate as the transition function.

    The agent action is a 5-D Δknob direction in [-1, 1]; the env clips it,
    scales by `action_scale`, adds to the current knob state, and clips the
    result to [0, 1]. The 6-D distgen state is held private and (optionally)
    drifts step-to-step.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        surrogate: SurrogateFn,
        target_mean: float,
        target_std: float,
        *,
        action_scale: float = 0.05,
        distgen_drift_std: float = 0.0,
        max_steps: int = 64,
        seed: int | None = None,
        device: str | torch.device = "cpu",
        terminal_emit_bonus: float = 0.0,
    ):
        super().__init__()
        self._device = torch.device(device)
        # If the surrogate is a torch Module, move it to the requested device
        # so per-step inference doesn't cross host<->device on every call.
        if isinstance(surrogate, torch.nn.Module):
            surrogate = surrogate.to(self._device)
            surrogate.eval()
        self._surrogate = surrogate
        self._target_mean = float(target_mean)
        self._target_std = float(target_std)
        self._action_scale = float(action_scale)
        self._distgen_drift_std = float(distgen_drift_std)
        self._max_steps = int(max_steps)
        self._terminal_emit_bonus = float(terminal_emit_bonus)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_KNOB,), dtype=np.float32,
        )
        # 5 knobs in [0,1] + 1 z-scored log-emit (unbounded above/below).
        self.observation_space = spaces.Box(
            low=np.array([0.0] * N_KNOB + [-np.inf], dtype=np.float32),
            high=np.array([1.0] * N_KNOB + [np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        # State (filled in by reset()).
        self._knobs_norm: np.ndarray = np.zeros(N_KNOB, dtype=np.float32)
        self._distgen_norm: np.ndarray = np.zeros(N_DISTGEN, dtype=np.float32)
        self._step_count: int = 0
        self._last_y_norm: float = 0.0

        if seed is not None:
            self.reset(seed=seed)

    # ----- internal helpers -------------------------------------------------

    def _forward(self) -> float:
        """Return z-scored log-emit from the surrogate at the current state."""
        x = np.concatenate([self._knobs_norm, self._distgen_norm])[None, :]
        x_t = torch.from_numpy(x.astype(np.float32)).to(self._device)
        with torch.no_grad():
            y = self._surrogate(x_t)
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        return float(np.asarray(y).reshape(-1)[0])

    def _physical_emit(self, y_norm: float) -> float:
        """Invert log10 + z-score -> norm_emit_4d in m^2."""
        return float(10.0 ** (y_norm * self._target_std + self._target_mean))

    def _observation(self) -> np.ndarray:
        return np.concatenate(
            [self._knobs_norm, np.array([self._last_y_norm], dtype=np.float32)]
        ).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "emit_m2": self._physical_emit(self._last_y_norm),
            "log_emit_norm": self._last_y_norm,
            "distgen_norm": self._distgen_norm.copy(),
            "knobs_phys": _denorm_knobs(self._knobs_norm),
            "step_count": self._step_count,
        }

    # ----- Gymnasium API ----------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._knobs_norm = self.np_random.uniform(0.0, 1.0, N_KNOB).astype(np.float32)
        self._distgen_norm = self.np_random.uniform(0.0, 1.0, N_DISTGEN).astype(np.float32)
        self._step_count = 0
        self._last_y_norm = self._forward()
        return self._observation(), self._info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (N_KNOB,):
            raise ValueError(f"action shape {action.shape} != ({N_KNOB},)")
        action = np.clip(action, -1.0, 1.0)
        self._knobs_norm = np.clip(
            self._knobs_norm + action * self._action_scale, 0.0, 1.0,
        ).astype(np.float32)

        if self._distgen_drift_std > 0.0:
            drift = self.np_random.normal(
                0.0, self._distgen_drift_std, N_DISTGEN,
            ).astype(np.float32)
            self._distgen_norm = np.clip(
                self._distgen_norm + drift, 0.0, 1.0,
            ).astype(np.float32)

        self._last_y_norm = self._forward()
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self._max_steps

        # Base step reward: minimize z-scored log-emit. On the final step,
        # add `terminal_emit_bonus * (-y_norm)` so the agent learns to value
        # the endpoint over transient transit-through-good-region behaviour.
        # terminal_emit_bonus=0 reproduces the original (1x weight) reward.
        reward = -self._last_y_norm
        if truncated and self._terminal_emit_bonus > 0.0:
            reward += self._terminal_emit_bonus * (-self._last_y_norm)

        return self._observation(), reward, terminated, truncated, self._info()

    # ----- production-path constructor --------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        norm_json: str | Path,
        *,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> "PhotoinjectorEnv":
        """Load an EmittanceMLP checkpoint + normalization JSON and build.

        Surrogate weights are loaded with `map_location=device` so they end up
        on the right hardware without an intermediate CPU copy.
        """
        from .model import EmittanceMLP

        torch_device = torch.device(device)
        model = EmittanceMLP.load_from_checkpoint(
            str(ckpt_path), map_location=torch_device,
        )
        model.eval()
        with open(norm_json) as f:
            norm = json.load(f)

        return cls(
            surrogate=model,
            target_mean=norm["target_mean"],
            target_std=norm["target_std"],
            device=torch_device,
            **kwargs,
        )
