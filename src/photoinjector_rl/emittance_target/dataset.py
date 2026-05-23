"""
Dataset + DataModule for the emittance MLP.

Loads the preprocessed HDF5 fully into memory (it's <1 MB at 10k runs),
applies normalization at construction time, and serves (input_11d, target)
tensors to the trainer.
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


def _normalize_target(raw: np.ndarray, t_mean: float, t_std: float) -> np.ndarray:
    log_target = np.log10(np.clip(raw, 1e-30, None))
    return ((log_target - t_mean) / t_std).astype(np.float32)


def denormalize_target(y_norm: torch.Tensor, t_mean: float, t_std: float) -> torch.Tensor:
    """Invert log10 + z-score: returns norm_emit_4d in m^2."""
    log_target = y_norm * t_std + t_mean
    return 10.0 ** log_target


def _load_h5_array(h: h5py.File, key: str) -> np.ndarray:
    return np.asarray(h[key][...])  # pyright: ignore[reportIndexIssue]


class EmittanceDataset(Dataset):
    """In-memory dataset of (input_11d, target_scalar) pairs.

    All normalization is applied once at construction; tensors are stored as
    contiguous float32 for fast random-access in DataLoader workers.
    """

    def __init__(self, processed_h5: str, norm_json: str | None = None):
        with h5py.File(processed_h5, "r") as h:
            settings = _load_h5_array(h, "settings")
            target = _load_h5_array(h, "target")
            fingerprints = _load_h5_array(h, "fingerprint")

        if norm_json is None:
            norm_json = processed_h5.replace(".h5", "_norm.json")
        with open(norm_json) as f:
            norm = json.load(f)

        self.norm = norm
        self.fingerprints = fingerprints

        s_norm = _normalize_settings(
            settings,
            np.array(norm["setting_low"], dtype=np.float32),
            np.array(norm["setting_high"], dtype=np.float32),
        )
        y_norm = _normalize_target(target, norm["target_mean"], norm["target_std"])

        self.x = torch.tensor(s_norm)              # (N, 11)
        self.y = torch.tensor(y_norm).unsqueeze(1) # (N, 1)
        self.input_dim = int(self.x.shape[1])

        # Keep raw values for de-norm / diagnostics.
        self.raw_target = torch.tensor(target)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class EmittanceDataModule(L.LightningDataModule):
    def __init__(
        self,
        processed_h5: str = "processed/emittance_target.h5",
        norm_json: str | None = None,
        batch_size: int = 256,
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
        self.full: EmittanceDataset | None = None
        self.train: Dataset | None = None
        self.val: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.full is None:
            self.full = EmittanceDataset(self.processed_h5, self.norm_json)
        n_val = int(round(len(self.full) * self.val_fraction))
        n_train = len(self.full) - n_val
        self.train, self.val = random_split(
            self.full, [n_train, n_val],
            generator=torch.Generator().manual_seed(self.split_seed),
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train is not None, "call setup() first"
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0)

    def val_dataloader(self) -> DataLoader:
        assert self.val is not None, "call setup() first"
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0)

    @property
    def norm(self) -> dict:
        assert self.full is not None, "call setup() first"
        return self.full.norm

    @property
    def input_dim(self) -> int:
        assert self.full is not None, "call setup() first"
        return self.full.input_dim
