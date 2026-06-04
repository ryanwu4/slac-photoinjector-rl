"""
Dataset + DataModule for the conditional-flow surrogate.

Loads the preprocessed HDF5 fully into memory (lofi is ~0.36 GB at 10k runs x
1500 particles), applies normalization at construction, and serves
(particles_norm (P, 6), cond_norm (11,)) pairs to the trainer.

Normalization:
  - condition (11 knobs): min-max to [0, 1] from the YAML bounds (matches the
    RL env's [0,1] action/context box),
  - particles (6 coords): per-dim StandardScaler (the raw scales span ~10 orders
    of magnitude: z std ~3e-4 m vs pz std ~7e4 eV/c).
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import json

import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def _normalize_settings(raw: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return ((raw - low) / np.maximum(high - low, 1e-12)).astype(np.float32)


def _standardize_particles(p: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((p - mean) / std).astype(np.float32)


def _destandardize_particles(p: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (p * std + mean).astype(np.float32)


def _load_h5_array(h: h5py.File, key: str) -> np.ndarray:
    return np.asarray(h[key][...])  # pyright: ignore[reportIndexIssue]


class FlowDataset(Dataset):
    """In-memory dataset of (particles_norm (P,6), cond_norm (11,)) pairs."""

    def __init__(self, processed_h5: str, norm_json: str | None = None):
        with h5py.File(processed_h5, "r") as h:
            settings = _load_h5_array(h, "settings")        # (N, 11)
            particles = _load_h5_array(h, "particles")      # (N, P, 6)
            norm_emit = _load_h5_array(h, "norm_emit_4d")   # (N,)
            fingerprints = _load_h5_array(h, "fingerprint")

        if norm_json is None:
            norm_json = processed_h5.replace(".h5", "_norm.json")
        with open(norm_json) as f:
            norm = json.load(f)

        self.norm = norm
        self.fingerprints = fingerprints

        low = np.array(norm["setting_low"], dtype=np.float32)
        high = np.array(norm["setting_high"], dtype=np.float32)
        p_mean = np.array(norm["particle_mean"], dtype=np.float32)        # (6,)
        p_std = np.array(norm["particle_std"], dtype=np.float32)          # (6,)

        cond = _normalize_settings(settings, low, high)                  # (N, 11)
        parts = _standardize_particles(
            particles, p_mean[None, None, :], p_std[None, None, :])      # (N, P, 6)

        self.cond = torch.tensor(cond)                                   # (N, 11)
        self.particles = torch.tensor(parts)                             # (N, P, 6)
        self.raw_emit = torch.tensor(norm_emit)                          # (N,)
        self.cond_dim = int(self.cond.shape[1])
        self.P = int(self.particles.shape[1])

    def __len__(self) -> int:
        return self.cond.shape[0]

    def __getitem__(self, idx: int):
        return self.particles[idx], self.cond[idx]


class FlowDataModule(L.LightningDataModule):
    def __init__(
        self,
        processed_h5: str = "processed/flow_surrogate.h5",
        norm_json: str | None = None,
        batch_size: int = 32,
        val_fraction: float = 0.1,
        num_workers: int = 0,
        split_seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.processed_h5 = processed_h5
        self.norm_json = norm_json
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.num_workers = num_workers
        self.split_seed = split_seed
        self.full: FlowDataset | None = None
        self.train: Dataset | None = None
        self.val: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.full is None:
            self.full = FlowDataset(self.processed_h5, self.norm_json)
        n_val = int(round(len(self.full) * self.val_fraction))
        n_train = len(self.full) - n_val
        self.train, self.val = random_split(
            self.full, [n_train, n_val],
            generator=torch.Generator().manual_seed(self.split_seed),
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train is not None, "call setup() first"
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self) -> DataLoader:
        assert self.val is not None, "call setup() first"
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)

    @property
    def norm(self) -> dict:
        assert self.full is not None, "call setup() first"
        return self.full.norm

    @property
    def cond_dim(self) -> int:
        assert self.full is not None, "call setup() first"
        return self.full.cond_dim
