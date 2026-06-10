"""
Shared preprocessing helpers used by every surrogate's archive ingest.
"""
from __future__ import annotations

import h5py
import numpy as np

from photoinjector_rl.core.settings import SETTING_KEYS


def extract_settings(h5: h5py.File) -> np.ndarray:
    """11 settings in the order locked in core.settings.SETTING_KEYS."""
    attrs = h5["settings"].attrs
    return np.array([float(np.asarray(attrs[k]).item()) for k in SETTING_KEYS],
                    dtype=np.float32)
