"""
Tests for the PolicyAdapter abstraction added to compare_n_impact.py.

The diffrl-adapter tests hit real saved checkpoints under
`runs/compare_diff/{shac,bptt}/seed_0/final_policy.pt`; they auto-skip when
those files are absent (fresh clone / CI).
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from photoinjector_rl.surrogates.mlp.compare_n_impact import (
    DiffRLAdapter,
    SB3Adapter,
    _load_adapter,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _maybe_path(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"missing {p}; run compare_diff_algos first")
    return p


def test_load_adapter_dispatches_pt_to_diffrl(tmp_path):
    # The dispatch happens before file IO; passing a nonexistent .pt path
    # should still route to DiffRLAdapter and only fail on the torch.load call.
    with pytest.raises((FileNotFoundError, RuntimeError, EOFError)):
        _load_adapter(str(tmp_path / "nope.pt"))


def test_load_adapter_dispatches_zip_to_sb3(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        _load_adapter(str(tmp_path / "nope.zip"))


def test_load_adapter_rejects_unknown_suffix():
    with pytest.raises(ValueError, match="unknown policy format"):
        _load_adapter("/tmp/foo.unknown")


def test_diffrl_adapter_shac_predict():
    p = _maybe_path(REPO_ROOT / "runs" / "compare_diff" / "shac" / "seed_0"
                    / "final_policy.pt")
    adapter = DiffRLAdapter(str(p))
    obs = np.zeros(6, dtype=np.float32)
    action, state = adapter.predict(obs, deterministic=True)
    assert state is None
    assert action.shape == (5,)
    assert action.dtype == np.float32
    # tanh-squashed: must be in [-1, 1].
    assert (action >= -1.0).all() and (action <= 1.0).all()


def test_diffrl_adapter_bptt_predict():
    p = _maybe_path(REPO_ROOT / "runs" / "compare_diff" / "bptt" / "seed_0"
                    / "final_policy.pt")
    adapter = DiffRLAdapter(str(p))
    obs = np.zeros(6, dtype=np.float32)
    action, state = adapter.predict(obs, deterministic=True)
    assert state is None
    assert action.shape == (5,)
    assert (action >= -1.0).all() and (action <= 1.0).all()


def test_diffrl_adapter_deterministic_is_deterministic():
    p = _maybe_path(REPO_ROOT / "runs" / "compare_diff" / "bptt" / "seed_0"
                    / "final_policy.pt")
    adapter = DiffRLAdapter(str(p))
    obs = np.array([0.3, 0.4, 0.5, 0.6, 0.7, -0.1], dtype=np.float32)
    a1, _ = adapter.predict(obs, deterministic=True)
    a2, _ = adapter.predict(obs, deterministic=True)
    np.testing.assert_allclose(a1, a2)


def test_diffrl_adapter_accepts_2d_obs_batch():
    """SB3's `model.predict` accepts either single-env obs or vectorized
    (n_envs, obs_dim) batches. We don't strictly need batched support for
    the multiprocessing pool (it passes one obs at a time) but the path
    should at least not crash.
    """
    p = _maybe_path(REPO_ROOT / "runs" / "compare_diff" / "shac" / "seed_0"
                    / "final_policy.pt")
    adapter = DiffRLAdapter(str(p))
    obs = np.zeros((1, 6), dtype=np.float32)
    action, _ = adapter.predict(obs, deterministic=True)
    # Adapter squeezes batch dim → returns (5,). That matches SB3 single-env
    # predict semantics; the env's step() expects 1-D action.
    assert action.shape == (5,)
