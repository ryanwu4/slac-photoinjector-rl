"""
Shared pytest fixtures for the emittance_target tests.

Tests fall into two tiers:

- **Unit tests** that don't need real model weights — they use small mock
  surrogates returning deterministic outputs. These always run.
- **Regression tests** that load a real trained checkpoint + normalization
  JSON and compare to hardcoded golden values. These auto-skip when the
  artifacts are not present (e.g. fresh clone, CI without training).
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import glob
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_GLOB = "models/emittance_target_hifi/checkpoints/best-*.ckpt"
NORM_JSON_PATH = "data/processed/emittance_target_hifi_norm.json"
FLOW_CHECKPOINT_GLOB = "models/flow_surrogate/checkpoints/best-*.ckpt"
FLOW_NORM_JSON_PATH = "data/processed/flow_surrogate_norm.json"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def checkpoint_path(repo_root: Path) -> Path:
    """Path to the trained EmittanceMLP checkpoint, or skip the test."""
    matches = sorted(glob.glob(str(repo_root / CHECKPOINT_GLOB)))
    if not matches:
        pytest.skip(f"no checkpoint matching {CHECKPOINT_GLOB} -- "
                    "train one with `python -m photoinjector_rl.surrogates.mlp.train`")
    return Path(matches[-1])


@pytest.fixture(scope="session")
def norm_json_path(repo_root: Path) -> Path:
    """Path to the preprocessed normalization JSON, or skip."""
    p = repo_root / NORM_JSON_PATH
    if not p.exists():
        pytest.skip(f"missing {NORM_JSON_PATH} -- "
                    "run `python -m photoinjector_rl.surrogates.mlp.preprocess`")
    return p


@pytest.fixture(scope="session")
def loaded_model(checkpoint_path: Path):
    """Real EmittanceMLP loaded from disk."""
    from photoinjector_rl.surrogates.mlp.model import EmittanceMLP

    model = EmittanceMLP.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
    model.eval()
    return model


@pytest.fixture(scope="session")
def loaded_norm(norm_json_path: Path) -> dict:
    import json
    with open(norm_json_path) as f:
        return json.load(f)


# ------------------------------- flow surrogate ------------------------------


@pytest.fixture(scope="session")
def flow_checkpoint_path(repo_root: Path) -> Path:
    """Path to the trained ConditionalAffineFlow checkpoint, or skip."""
    matches = sorted(glob.glob(str(repo_root / FLOW_CHECKPOINT_GLOB)))
    if not matches:
        pytest.skip(f"no checkpoint matching {FLOW_CHECKPOINT_GLOB} -- "
                    "train one with `python -m photoinjector_rl.surrogates.flow.train`")
    return Path(matches[-1])


@pytest.fixture(scope="session")
def flow_norm_json_path(repo_root: Path) -> Path:
    """Path to the flow's preprocessed normalization JSON, or skip."""
    p = repo_root / FLOW_NORM_JSON_PATH
    if not p.exists():
        pytest.skip(f"missing {FLOW_NORM_JSON_PATH} -- "
                    "run `python -m photoinjector_rl.surrogates.flow.preprocess`")
    return p


@pytest.fixture(scope="session")
def loaded_flow(flow_checkpoint_path: Path):
    """Real ConditionalAffineFlow loaded from disk."""
    from photoinjector_rl.surrogates.flow.model import ConditionalAffineFlow

    model = ConditionalAffineFlow.load_from_checkpoint(
        str(flow_checkpoint_path), map_location="cpu")
    model.eval()
    return model


# ------------------------------- mock surrogates -----------------------------


@pytest.fixture
def constant_surrogate():
    """Mock surrogate that always returns 0.0 (z-scored log-emit)."""
    def fake(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], 1), dtype=torch.float32)
    return fake


@pytest.fixture
def deterministic_surrogate():
    """Mock surrogate: y = mean(x), deterministic in the input vector.

    Useful for tests that want a different prediction depending on knob
    settings without depending on a real checkpoint.
    """
    def fake(x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=1, keepdim=True)
    return fake
