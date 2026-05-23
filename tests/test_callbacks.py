"""
Unit coverage for the custom SB3 callbacks (RolloutDiagnosticCallback,
EpisodeMetricsCallback). No checkpoint required -- a tiny mock policy + mock
surrogate is enough to exercise the plotting and metric-aggregation paths.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from photoinjector_rl.emittance_target.callbacks import (
    EpisodeMetricsCallback,
    RolloutDiagnosticCallback,
    _is_saturated,
)
from photoinjector_rl.emittance_target.env import PhotoinjectorEnv


class _ConstantSurrogate(nn.Module):
    """Tiny nn.Module that returns 0.0 for any input."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros((x.shape[0], 1), dtype=torch.float32) + self.bias


class _DummyPolicy:
    """Mimics an SB3 policy: .predict(obs, deterministic=True) -> (action, _)."""

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def predict(self, obs, deterministic=False):
        del obs, deterministic
        return self.rng.uniform(-1.0, 1.0, 5).astype(np.float32), None


def _make_env(seed: int = 0, max_steps: int = 8) -> PhotoinjectorEnv:
    return PhotoinjectorEnv(
        surrogate=_ConstantSurrogate(),
        target_mean=-10.34,
        target_std=0.17,
        max_steps=max_steps,
        seed=seed,
    )


# ---------- RolloutDiagnosticCallback ----------------------------------------


def test_diag_callback_emits_png_per_render(tmp_path: Path) -> None:
    env = _make_env(max_steps=8)
    cb = RolloutDiagnosticCallback(
        diag_env=env,
        eval_freq=1,
        out_dir=tmp_path,
        diag_seeds=(11, 22, 33),
    )
    cb.model = _DummyPolicy()
    cb.num_timesteps = 42
    out = cb.render_rollout()
    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 0


def test_diag_callback_multi_seed_runs_one_rollout_per_seed(tmp_path: Path) -> None:
    """Internal `_rollout_one` should run a full episode and return arrays."""
    env = _make_env(max_steps=4)
    cb = RolloutDiagnosticCallback(
        diag_env=env, eval_freq=1, out_dir=tmp_path,
        diag_seeds=(1, 2, 3),
    )
    cb.model = _DummyPolicy()
    r = cb._rollout_one(seed=1)
    assert r["knobs"].shape == (5, 5)         # max_steps + 1 = 5, 5 knobs
    assert r["emit"].shape == (5,)
    assert np.all(r["emit"] > 0)


# ---------- EpisodeMetricsCallback -------------------------------------------


def test_is_saturated_detects_bounds() -> None:
    """Helper used by EpisodeMetricsCallback to flag bound-pinning."""
    # Construct a knob vector pinned at lower bound of SOL10111.
    from photoinjector_rl.emittance_target import SETTING_BOUNDS, SETTING_KEYS
    lows = np.array([SETTING_BOUNDS[k][0] for k in SETTING_KEYS[:5]],
                    dtype=np.float32)
    highs = np.array([SETTING_BOUNDS[k][1] for k in SETTING_KEYS[:5]],
                     dtype=np.float32)
    mids = (lows + highs) / 2.0

    assert _is_saturated(lows)            # all at lower bound
    assert _is_saturated(highs)           # all at upper bound
    assert not _is_saturated(mids)        # safely interior


class _FakeLogger:
    """Captures record() calls; no real disk I/O."""

    def __init__(self):
        self.records: dict[str, float] = {}

    def record(self, key, value):
        self.records[key] = float(value)


class _FakeModel:
    """Just enough surface for BaseCallback.logger to find a logger."""

    def __init__(self, logger):
        self.logger = logger


def test_episode_metrics_logs_on_episode_end() -> None:
    """Simulate two env steps + an episode-end info dict; callback should
    log all four scalar metrics exactly once."""
    fake_logger = _FakeLogger()
    cb = EpisodeMetricsCallback()
    cb.model = _FakeModel(fake_logger)  # type: ignore[assignment]

    from photoinjector_rl.emittance_target import SETTING_BOUNDS, SETTING_KEYS
    mid_knobs = np.array(
        [(SETTING_BOUNDS[k][0] + SETTING_BOUNDS[k][1]) / 2
         for k in SETTING_KEYS[:5]],
        dtype=np.float32,
    )

    # Step 1: non-episode-end info, mid knobs, emit_m2=1e-11.
    cb.locals = {
        "infos": [{"emit_m2": 1e-11, "knobs_phys": mid_knobs}],
        "actions": np.array([[0.1, -0.2, 0.3, 0.0, -0.4]], dtype=np.float32),
    }
    cb._on_step()
    assert fake_logger.records == {}  # nothing logged until episode ends

    # Step 2: episode-end info (Monitor adds the "episode" key on done).
    cb.locals = {
        "infos": [{
            "emit_m2": 5e-12, "knobs_phys": mid_knobs,
            "episode": {"r": 12.3, "l": 2},
        }],
        "actions": np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    }
    cb._on_step()

    rec = fake_logger.records
    assert set(rec.keys()) == {
        "rollout/terminal_emit_m2",
        "rollout/min_emit_m2",
        "rollout/action_mean_abs",
        "rollout/saturation_rate",
    }
    assert rec["rollout/terminal_emit_m2"] == 5e-12
    assert rec["rollout/min_emit_m2"] == 5e-12
    # action_mean_abs over the 2 steps: mean(0.2) and mean(0.0) -> 0.1
    assert abs(rec["rollout/action_mean_abs"] - 0.1) < 1e-6
    # Neither step was saturated (mid knobs), so rate is 0.
    assert rec["rollout/saturation_rate"] == 0.0
