"""
Walk archives/{split}/*.h5 -> single processed HDF5 with:
  settings      (N, 11) float32   -- raw XOPT-sampled values
  target        (N,)    float32   -- norm_emit_4d at PR10241 (m^2)
  fingerprint   (N,)    S40       -- archive filename stem (for traceability)

Also writes <out>_norm.json with normalization stats (settings min-max from
the YAML bounds, log10(target) z-score from this batch).

Uses ParticleGroup from openpmd-beamphysics for the emittance read-out.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from beamphysics import ParticleGroup

from . import SETTING_BOUNDS, SETTING_KEYS


def extract_settings(h5: h5py.File) -> np.ndarray:
    """11 settings in the order locked in __init__.SETTING_KEYS."""
    attrs = h5["settings"].attrs
    return np.array([float(np.asarray(attrs[k]).item()) for k in SETTING_KEYS],
                    dtype=np.float32)


def parse_one(path: str, min_terminal_particles: int = 1500) -> dict | None:
    """Parse a single archive; return None on validation failure."""
    try:
        with h5py.File(path, "r") as h:
            if "settings" not in h:
                return None  # un-patched archive (no XOPT settings recorded)

            settings = extract_settings(h)
            tp = ParticleGroup(h5=h["impact/output/particles/PR10241/electron"])

            if len(tp.x) < min_terminal_particles:
                return None

            target = float(np.asarray(tp.norm_emit_4d).item())

            if not np.all(np.isfinite(settings)) or not np.isfinite(target):
                return None

            return {
                "settings": settings,
                "target": np.float32(target),
                "fingerprint": Path(path).stem,
            }
    except (OSError, KeyError):
        return None


def compute_normalization(records: list[dict]) -> dict:
    """Stats for normalizing settings (min-max via bounds) and log10(target)
    (z-score)."""
    setting_low = np.array([SETTING_BOUNDS[k][0] for k in SETTING_KEYS], dtype=np.float32)
    setting_high = np.array([SETTING_BOUNDS[k][1] for k in SETTING_KEYS], dtype=np.float32)

    targets = np.array([r["target"] for r in records])
    log_target = np.log10(targets.clip(min=1e-30))
    target_mean = float(log_target.mean())
    target_std = float(max(log_target.std(), 1e-12))

    return {
        "setting_keys": SETTING_KEYS,
        "setting_low": setting_low.tolist(),
        "setting_high": setting_high.tolist(),
        "target_transform": "log10",
        "target_mean": target_mean,
        "target_std": target_std,
        "n_records": len(records),
    }


def write_processed(records: list[dict], out_h5: str, norm: dict) -> None:
    os.makedirs(os.path.dirname(out_h5) or ".", exist_ok=True)
    settings = np.stack([r["settings"] for r in records])
    targets = np.array([r["target"] for r in records], dtype=np.float32)
    fingerprints = np.array([r["fingerprint"] for r in records], dtype="S40")

    with h5py.File(out_h5, "w") as h:
        h.create_dataset("settings", data=settings, compression="gzip")
        h.create_dataset("target", data=targets, compression="gzip")
        h.create_dataset("fingerprint", data=fingerprints)
        h.attrs["setting_keys"] = SETTING_KEYS
        h.attrs["n_settings"] = len(SETTING_KEYS)
        h.attrs["target_field"] = "norm_emit_4d_m2"

    norm_path = out_h5.replace(".h5", "_norm.json")
    with open(norm_path, "w") as f:
        json.dump(norm, f, indent=2)


def preprocess(archive_glob: str, out_h5: str, *, max_archives: int | None = None,
               min_terminal_particles: int = 1500, verbose: bool = True) -> dict:
    import glob

    files = sorted(glob.glob(archive_glob))
    if max_archives:
        files = files[:max_archives]

    if verbose:
        print(f"Scanning {len(files)} archives from {archive_glob}")

    records: list[dict] = []
    n_skipped = 0
    for i, f in enumerate(files):
        rec = parse_one(f, min_terminal_particles=min_terminal_particles)
        if rec is None:
            n_skipped += 1
            continue
        records.append(rec)
        if verbose and (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)}: kept {len(records)}, skipped {n_skipped}")

    if not records:
        raise RuntimeError(f"No valid archives in {archive_glob}")

    if verbose:
        print(f"Total: kept {len(records)}, skipped {n_skipped}")

    norm = compute_normalization(records)
    write_processed(records, out_h5, norm)
    if verbose:
        print(f"Wrote {out_h5} and {out_h5.replace('.h5', '_norm.json')}")
    return norm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", default="data/archives/train/*.h5",
                        help="glob pattern for input archives")
    parser.add_argument("--out", default="data/processed/emittance_target.h5",
                        help="output HDF5 path")
    parser.add_argument("--max-archives", type=int, default=None,
                        help="cap archive count for smoke testing")
    parser.add_argument("--min-terminal-particles", type=int, default=1500,
                        help="drop runs with fewer terminal particles than this")
    args = parser.parse_args()

    preprocess(
        args.archives, args.out,
        max_archives=args.max_archives,
        min_terminal_particles=args.min_terminal_particles,
    )


if __name__ == "__main__":
    main()
