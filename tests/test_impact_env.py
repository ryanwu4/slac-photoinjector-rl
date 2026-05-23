"""
Unit coverage for ImpactPhotoinjectorEnv. Heavy parts of the env are mocked:

  - `custom_evaluate_impact_with_distgen` is patched to a fast function that
    just returns a deterministic merit dict, so no Impact-T binary is needed.
  - Failure path is exercised by raising from the patched function.

A regression test that actually runs Impact-T is intentionally NOT here --
that costs ~10 s per step and belongs in a separate, opt-in script.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from unittest.mock import patch

import math
import numpy as np
import pytest

from photoinjector_rl.emittance_target.impact_env import (
    ImpactPhotoinjectorEnv,
    _extract_norm_emit_4d,
)


_FAKE_NORM = {"target_mean": -10.34, "target_std": 0.17}


def _make_env(*, max_steps: int = 4, **kwargs) -> ImpactPhotoinjectorEnv:
    """Construct an env without real config files. The patched evaluate
    function never reads them, so file existence doesn't matter for unit
    tests."""
    return ImpactPhotoinjectorEnv(
        impact_config="/nonexistent/Impact.yaml",
        distgen_input_file="/nonexistent/distgen.yaml",
        target_mean=_FAKE_NORM["target_mean"],
        target_std=_FAKE_NORM["target_std"],
        max_steps=max_steps,
        **kwargs,
    )


def _fake_evaluate(settings, **kwargs):
    """Deterministic fake: emit = product of two knob values, mapped to a
    plausible m² range. Lets us check value flows through correctly."""
    # Build a stable, non-trivial fake emit from settings.
    s = float(settings["SOL10111:solenoid_field_scale"])
    emit_m2 = 1e-11 * (1.0 + 0.1 * abs(s))
    return {"norm_emit_4d": emit_m2, "error": False, "fingerprint": "fake"}


# ---------- _extract_norm_emit_4d helper -------------------------------------


class _FakePG:
    def __init__(self, v):
        self.norm_emit_4d = v


class _FakeImpactDirect:
    """I.particles[marker] returns a ParticleGroup directly."""
    particles = {"PR10241": _FakePG(2.5e-11)}


class _FakeImpactDictOfSpecies:
    """I.particles[marker] returns a {species: ParticleGroup} dict."""
    particles = {"PR10241": {"electron": _FakePG(3.0e-11)}}


def test_extract_handles_direct_particlegroup() -> None:
    assert _extract_norm_emit_4d(_FakeImpactDirect, "PR10241") == 2.5e-11


def test_extract_handles_species_dict() -> None:
    assert _extract_norm_emit_4d(_FakeImpactDictOfSpecies, "PR10241") == 3.0e-11


# ---------- env basics -------------------------------------------------------


def test_action_observation_spaces_match_surrogate_env() -> None:
    env = _make_env()
    assert env.action_space.shape == (5,)
    assert env.observation_space.shape == (6,)


def test_reset_runs_one_impact_call_and_returns_obs(monkeypatch) -> None:
    calls = []

    def fake(settings, **kwargs):
        calls.append(settings)
        return _fake_evaluate(settings, **kwargs)

    monkeypatch.setattr(
        "photoinjector_rl.data.evaluate.custom_evaluate_impact_with_distgen",
        fake,
    )

    env = _make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (6,)
    assert obs.dtype == np.float32
    assert len(calls) == 1
    # Settings dict has 11 SETTING_KEYS + the fidelity-constants block.
    assert "SOL10111:solenoid_field_scale" in calls[0]
    assert "distgen:r_dist:sigma_xy:value" in calls[0]
    assert "header:Nx" in calls[0]
    # Info dict should include the standard keys plus failure_count.
    for k in ("emit_m2", "log_emit_norm", "distgen_norm",
              "knobs_phys", "step_count", "failure_count"):
        assert k in info
    assert info["failure_count"] == 0
    # emit_m2 should round-trip approximately (log10 z-score -> physical).
    expected_emit = 1e-11 * (1.0 + 0.1 * abs(
        float(info["knobs_phys"][0])
    ))
    assert info["emit_m2"] == pytest.approx(expected_emit, rel=1e-6)


def test_step_runs_one_impact_call(monkeypatch) -> None:
    monkeypatch.setattr(
        "photoinjector_rl.data.evaluate.custom_evaluate_impact_with_distgen",
        _fake_evaluate,
    )
    env = _make_env(max_steps=4)
    env.reset(seed=0)
    obs, r, term, trunc, info = env.step(np.zeros(5, dtype=np.float32))
    assert obs.shape == (6,)
    assert isinstance(float(r), float)
    assert term is False
    assert trunc is False
    assert info["failure_count"] == 0


def test_max_steps_truncates(monkeypatch) -> None:
    monkeypatch.setattr(
        "photoinjector_rl.data.evaluate.custom_evaluate_impact_with_distgen",
        _fake_evaluate,
    )
    env = _make_env(max_steps=3)
    env.reset(seed=0)
    for _ in range(2):
        _, _, term, trunc, _ = env.step(np.zeros(5, dtype=np.float32))
        assert (term, trunc) == (False, False)
    _, _, term, trunc, _ = env.step(np.zeros(5, dtype=np.float32))
    assert (term, trunc) == (False, True)


def test_failure_path_returns_penalty_and_counts(monkeypatch) -> None:
    def fake_that_fails(settings, **kwargs):
        raise RuntimeError("Impact-T blew up at extreme knobs")

    monkeypatch.setattr(
        "photoinjector_rl.data.evaluate.custom_evaluate_impact_with_distgen",
        fake_that_fails,
    )

    env = _make_env(max_steps=4, failure_penalty_sigma=5.0)
    obs, info = env.reset(seed=0)
    # Failure path: log_emit_norm should equal the penalty z-score.
    assert info["log_emit_norm"] == pytest.approx(5.0, rel=1e-6)
    assert info["failure_count"] == 1

    obs, r, term, trunc, info = env.step(np.zeros(5, dtype=np.float32))
    # reward = -y_norm = -5.0 on failure
    assert r == pytest.approx(-5.0, rel=1e-6)
    assert info["failure_count"] == 2
    # Episode is NOT terminated on failure -- agent gets to back off.
    assert term is False


def test_non_finite_emit_treated_as_failure(monkeypatch) -> None:
    def fake_nan(settings, **kwargs):
        return {"norm_emit_4d": float("nan"), "error": False, "fingerprint": "x"}

    monkeypatch.setattr(
        "photoinjector_rl.data.evaluate.custom_evaluate_impact_with_distgen",
        fake_nan,
    )
    env = _make_env(failure_penalty_sigma=5.0)
    obs, info = env.reset(seed=0)
    assert info["log_emit_norm"] == pytest.approx(5.0, rel=1e-6)
    assert info["failure_count"] == 1


def test_zero_emit_treated_as_failure(monkeypatch) -> None:
    def fake_zero(settings, **kwargs):
        return {"norm_emit_4d": 0.0, "error": False, "fingerprint": "x"}

    monkeypatch.setattr(
        "photoinjector_rl.data.evaluate.custom_evaluate_impact_with_distgen",
        fake_zero,
    )
    env = _make_env()
    _, info = env.reset(seed=0)
    assert info["failure_count"] == 1


def test_settings_dict_has_correct_keys_and_ranges(monkeypatch) -> None:
    """The dict we pass to evaluate.py must use the exact SETTING_KEYS
    strings (otherwise lume-impact/distgen won't route them) and the
    values must lie in their physical bounds. The fidelity constants from
    DEFAULT_LHS_CONSTANTS must also be merged in so the env matches the
    surrogate's training fidelity."""
    from photoinjector_rl.emittance_target import SETTING_BOUNDS, SETTING_KEYS
    from photoinjector_rl.emittance_target.impact_env import DEFAULT_LHS_CONSTANTS

    captured = {}

    def fake(settings, **kwargs):
        captured.update(settings)
        return _fake_evaluate(settings, **kwargs)

    monkeypatch.setattr(
        "photoinjector_rl.data.evaluate.custom_evaluate_impact_with_distgen",
        fake,
    )

    env = _make_env()
    env.reset(seed=0)
    # All 11 sampled keys present plus all constants.
    assert set(SETTING_KEYS).issubset(captured.keys())
    assert set(DEFAULT_LHS_CONSTANTS.keys()).issubset(captured.keys())
    # Sampled values within their bounds.
    for k in SETTING_KEYS:
        lo, hi = SETTING_BOUNDS[k]
        assert lo - 1e-9 <= captured[k] <= hi + 1e-9, \
            f"{k}={captured[k]} outside [{lo}, {hi}]"
    # Fidelity constants exactly match the lhs_train.yaml block.
    assert captured["distgen:n_particle"] == 2000
    assert captured["header:Nx"] == 8
    assert captured["header:Ny"] == 8
    assert captured["header:Nz"] == 8
    assert captured["distgen:total_charge"] == 500


def test_constants_override(monkeypatch) -> None:
    """User-supplied `constants=` should fully replace the defaults."""
    captured = {}

    def fake(settings, **kwargs):
        captured.update(settings)
        return _fake_evaluate(settings, **kwargs)

    monkeypatch.setattr(
        "photoinjector_rl.data.evaluate.custom_evaluate_impact_with_distgen",
        fake,
    )

    env = _make_env(constants={"header:Nx": 16, "header:Ny": 16, "header:Nz": 16,
                                "distgen:n_particle": 5000})
    env.reset(seed=0)
    assert captured["header:Nx"] == 16
    assert captured["distgen:n_particle"] == 5000
    # Defaults that weren't overridden should NOT be present.
    assert "stop_1:s" not in captured


def test_from_norm_json_loads_stats(tmp_path, monkeypatch) -> None:
    import json
    monkeypatch.setattr(
        "photoinjector_rl.data.evaluate.custom_evaluate_impact_with_distgen",
        _fake_evaluate,
    )
    norm_path = tmp_path / "norm.json"
    norm_path.write_text(json.dumps({
        "target_mean": -10.0, "target_std": 0.2,
        "setting_low": [], "setting_high": [],
    }))
    env = ImpactPhotoinjectorEnv.from_norm_json(
        norm_json=str(norm_path),
        impact_config="/x",
        distgen_input_file="/y",
        max_steps=2,
    )
    assert env._target_mean == -10.0
    assert env._target_std == 0.2


def test_distgen_not_in_observation(monkeypatch) -> None:
    """Privacy invariant: the 6-D distgen state should not leak into obs.
    Identical knobs + different distgen should give same first-5 obs
    components."""
    monkeypatch.setattr(
        "photoinjector_rl.data.evaluate.custom_evaluate_impact_with_distgen",
        _fake_evaluate,
    )
    env_a = _make_env()
    env_b = _make_env()
    obs_a, _ = env_a.reset(seed=0)
    obs_b, _ = env_b.reset(seed=999)
    # If knobs happen to differ, force them equal so the test only varies
    # distgen.
    env_b._knobs_norm = env_a._knobs_norm.copy()
    env_b._last_y_norm = env_b._forward()
    obs_b = env_b._observation()
    np.testing.assert_allclose(obs_a[:5], obs_b[:5])
