"""
File-IO + inference tests for the conditional-flow surrogate. The checkpoint /
processed-data tests auto-skip when the artifacts are absent (fresh clone, CI
without training), mirroring test_inference.py.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from photoinjector_rl.surrogates.flow import COORD_KEYS, N_INPUT, SETTING_KEYS
from photoinjector_rl.surrogates.flow.dataset import (
    _destandardize_particles,
    _standardize_particles,
)


# ---------- normalization (no checkpoint needed) ----------------------------


def test_particle_standardize_roundtrip() -> None:
    rng = np.random.default_rng(0)
    p = rng.standard_normal((100, 6)).astype(np.float32)
    mean = rng.standard_normal((1, 6)).astype(np.float32)
    std = (rng.random((1, 6)).astype(np.float32) + 0.5)
    z = _standardize_particles(p, mean, std)
    back = _destandardize_particles(z, mean, std)
    np.testing.assert_allclose(back, p, rtol=1e-4, atol=1e-5)


# ---------- processed-data norm JSON (skips if absent) ----------------------


def test_flow_norm_json_keys(flow_norm_json_path: Path) -> None:
    with open(flow_norm_json_path) as f:
        norm = json.load(f)
    for key in ("setting_keys", "setting_low", "setting_high",
                "particle_mean", "particle_std", "P", "n_records"):
        assert key in norm, f"missing {key} in norm JSON"
    assert norm["setting_keys"] == SETTING_KEYS
    assert len(norm["setting_low"]) == N_INPUT
    assert len(norm["setting_high"]) == N_INPUT
    assert len(norm["particle_mean"]) == len(COORD_KEYS)
    assert len(norm["particle_std"]) == len(COORD_KEYS)


# ---------- checkpoint loading + inference (skips if absent) ----------------


def test_flow_checkpoint_loads(loaded_flow) -> None:
    assert loaded_flow.hparams.condition_dim == N_INPUT
    # De-norm buffers must travel with the checkpoint.
    for buf in ("particle_mean", "particle_std", "setting_low",
                "setting_high", "target_mean", "target_std", "mc2"):
        assert hasattr(loaded_flow, buf), f"missing buffer {buf}"
    assert loaded_flow.particle_mean.shape == (len(COORD_KEYS),)


def test_flow_sampled_emittance_physical_band(loaded_flow) -> None:
    """Sampled clouds give norm_emit_4d in the FACET-II injector band."""
    torch.manual_seed(0)
    cond = torch.rand(3, N_INPUT)
    with torch.no_grad():
        emit = loaded_flow.emittance_from_knobs(cond, n=1500)
    assert emit.shape == (3,)
    assert (emit > 1e-13).all() and (emit < 1e-8).all()


def test_flow_forward_finite(loaded_flow) -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        y = loaded_flow(torch.rand(4, N_INPUT))
    assert y.shape == (4, 1)
    assert torch.isfinite(y).all()
