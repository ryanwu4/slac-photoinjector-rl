"""
File IO + inference tests for the v1 emittance surrogate.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from photoinjector_rl.emittance_target import N_INPUT, SETTING_BOUNDS, SETTING_KEYS
from photoinjector_rl.emittance_target.dataset import (
    _normalize_settings,
    _normalize_target,
    denormalize_target,
)


# ---------- normalization roundtrip (no checkpoint needed) ------------------


def test_norm_json_loads_and_has_required_keys(norm_json_path: Path) -> None:
    with open(norm_json_path) as f:
        norm = json.load(f)
    for key in ("setting_keys", "setting_low", "setting_high",
                "target_mean", "target_std", "target_transform"):
        assert key in norm, f"missing {key} in norm JSON"
    assert norm["setting_keys"] == SETTING_KEYS
    assert len(norm["setting_low"]) == N_INPUT
    assert len(norm["setting_high"]) == N_INPUT
    assert norm["target_transform"] == "log10"


def test_normalization_roundtrip_target() -> None:
    """log10 + z-score, then invert, gets us back the original."""
    raw = np.array([4.4e-11, 1.0e-10, 2.0e-11], dtype=np.float32)
    t_mean, t_std = -10.34, 0.17
    y_norm = _normalize_target(raw, t_mean, t_std)
    recovered = denormalize_target(torch.tensor(y_norm), t_mean, t_std).numpy()
    np.testing.assert_allclose(recovered, raw, rtol=1e-4)


def test_normalization_settings_to_unit_box() -> None:
    """Lower bound -> 0, upper bound -> 1, midpoint -> 0.5."""
    lows = np.array([SETTING_BOUNDS[k][0] for k in SETTING_KEYS], dtype=np.float32)
    highs = np.array([SETTING_BOUNDS[k][1] for k in SETTING_KEYS], dtype=np.float32)
    mids = (lows + highs) / 2.0
    raw = np.stack([lows, mids, highs])
    out = _normalize_settings(raw, lows, highs)
    np.testing.assert_allclose(out[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[1], 0.5, atol=1e-6)
    np.testing.assert_allclose(out[2], 1.0, atol=1e-6)


# ---------- checkpoint loading + inference -----------------------------------


def test_checkpoint_loads(loaded_model) -> None:
    assert hasattr(loaded_model, "net"), "EmittanceMLP missing .net attribute"
    # input-dim implied by first Linear layer.
    first_linear = loaded_model.net[0]
    assert first_linear.in_features == N_INPUT
    assert first_linear.out_features > 0


def test_inference_shape(loaded_model) -> None:
    x = torch.zeros((4, N_INPUT), dtype=torch.float32)
    with torch.no_grad():
        y = loaded_model(x)
    assert y.shape == (4, 1)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


def test_predict_physical_returns_positive_emittance(loaded_model) -> None:
    """norm_emit_4d is bounded below by zero in m^2."""
    x = torch.zeros((3, N_INPUT), dtype=torch.float32)
    emit = loaded_model.predict_physical(x)
    assert emit.shape == (3, 1)
    assert (emit > 0).all()
    # Physical sanity: emittance for the FACET-II injector should land in
    # the 1e-12 to 1e-9 m^2 range based on training-data stats.
    assert (emit > 1e-13).all() and (emit < 1e-8).all()


def test_inference_regression_fixed_input(loaded_model, loaded_norm) -> None:
    """Same fixed input -> same prediction across reruns.

    Golden: input = all-zeros (corresponds to all-bounds-low knobs +
    all-bounds-low distgen). Locked after the no_moments training run
    completed on 2026-05-12.
    """
    x = torch.zeros((1, N_INPUT), dtype=torch.float32)
    with torch.no_grad():
        y_norm = float(loaded_model(x).item())
    emit_m2 = 10.0 ** (y_norm * loaded_norm["target_std"] + loaded_norm["target_mean"])

    # Loose tolerance because the goldens depend on the specific trained
    # checkpoint; if the user retrains, update these values.
    assert y_norm == pytest.approx(REG_ZERO_INPUT_YNORM, rel=1e-3, abs=1e-3)
    assert emit_m2 == pytest.approx(REG_ZERO_INPUT_EMIT_M2, rel=1e-3)


# Golden values locked 2026-05-12 against
# trained/emittance_target/checkpoints/best-epoch=191-val_loss=0.0060.ckpt
# Replace with fresh numbers if you retrain.
REG_ZERO_INPUT_YNORM = -0.5499303340911865
REG_ZERO_INPUT_EMIT_M2 = 3.599346081384293e-11
