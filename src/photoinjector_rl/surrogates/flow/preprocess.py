"""
Walk archives/{split}/*.h5 -> single processed HDF5 for the conditional flow:
  settings      (N, 11)   float32  -- raw XOPT-sampled knob values
  particles     (N, P, 6) float32  -- PR10241 output cloud, (x,y,z,px,py,pz),
                                       subsampled to a fixed P per run
  norm_emit_4d  (N,)       float32  -- norm_emit_4d at PR10241 (m^2), kept as a
                                       cheap cross-check / eval baseline
  fingerprint   (N,)       S40      -- archive filename stem (traceability)

Also writes <out>_norm.json with normalization stats:
  - settings min-max from the YAML bounds (same as the v1 MLP surrogate),
  - per-dim particle StandardScaler (mean[6], std[6]) over a capped pooled
    subsample of macroparticles.

Each run is subsampled DOWN to exactly --target-particles macroparticles so the
stored array is dense (no ragged storage). Lofi runs hold ~2000 macroparticles
(the sweep allows loss down to ~1500), so the default P=1500 keeps every run;
runs with fewer than --min-terminal-particles are skipped. Particle coords come
from openpmd-beamphysics ParticleGroup (position [m], momentum [eV/c]).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import h5py
import numpy as np
from beamphysics import ParticleGroup

from . import COORD_KEYS, DEFAULT_P, ELECTRON_MC2_EV, SETTING_BOUNDS, SETTING_KEYS
# Reuse the v1 settings reader so the 11-knob column order stays identical.
from photoinjector_rl.core.preprocess import extract_settings

PR10241_PATH = "impact/output/particles/PR10241/electron"
# Cap on the number of pooled macroparticles used to fit the StandardScaler.
# 2M rows is plenty for a stable 6-D mean/std and bounds hifi (N*P = 200M) RAM.
_SCALER_SUBSAMPLE = 2_000_000


def _stack_coords(pg: ParticleGroup) -> np.ndarray:
    """(n, 6) array in COORD_KEYS order: (x, y, z, px, py, pz)."""
    return np.stack([getattr(pg, k) for k in COORD_KEYS], axis=1).astype(np.float32)


def parse_one(path: str, *, target_particles: int, min_terminal_particles: int,
              rng: np.random.Generator) -> dict | None:
    """Parse a single archive; return None on validation failure.

    Keeps runs with at least ``min_terminal_particles`` terminal macroparticles
    (>= target_particles, enforced by the caller), then subsamples DOWN to
    exactly ``target_particles``.
    """
    try:
        with h5py.File(path, "r") as h:
            if "settings" not in h:
                return None  # un-patched archive (no XOPT settings recorded)

            settings = extract_settings(h)  # (11,) in SETTING_KEYS order
            pg = ParticleGroup(h5=h[PR10241_PATH])

            n = int(pg.n_particle)
            if n < min_terminal_particles:
                return None

            coords = _stack_coords(pg)  # (n, 6)
            # Subsample DOWN to a fixed count (seeded, reproducible).
            idx = rng.choice(n, target_particles, replace=False)
            coords = coords[idx]

            emit = float(np.asarray(pg.norm_emit_4d).item())

            if not (np.all(np.isfinite(settings)) and np.all(np.isfinite(coords))
                    and np.isfinite(emit)):
                return None

            return {
                "settings": settings,
                "particles": coords,                  # (P, 6)
                "norm_emit_4d": np.float32(emit),
                "fingerprint": Path(path).stem,
            }
    except (OSError, KeyError):
        return None


def compute_normalization(records: list[dict], *, target_particles: int,
                          rng: np.random.Generator) -> dict:
    """Settings min-max (from YAML bounds) + per-dim particle StandardScaler."""
    setting_low = np.array([SETTING_BOUNDS[k][0] for k in SETTING_KEYS], dtype=np.float32)
    setting_high = np.array([SETTING_BOUNDS[k][1] for k in SETTING_KEYS], dtype=np.float32)

    # Pool a capped random subset of macroparticles to fit the 6-D scaler.
    n_rec = len(records)
    total = n_rec * target_particles
    if total <= _SCALER_SUBSAMPLE:
        pooled = np.concatenate([r["particles"] for r in records], axis=0)
    else:
        # Sample whole records' worth proportionally, then trim.
        per_rec = max(1, _SCALER_SUBSAMPLE // n_rec)
        chunks = []
        for r in records:
            sel = rng.choice(target_particles, min(per_rec, target_particles),
                             replace=False)
            chunks.append(r["particles"][sel])
        pooled = np.concatenate(chunks, axis=0)

    particle_mean = pooled.mean(axis=0)
    particle_std = np.maximum(pooled.std(axis=0), 1e-12)

    return {
        "setting_keys": SETTING_KEYS,
        "setting_low": setting_low.tolist(),
        "setting_high": setting_high.tolist(),
        "particle_mean": particle_mean.astype(float).tolist(),
        "particle_std": particle_std.astype(float).tolist(),
        "coord_keys": list(COORD_KEYS),
        "electron_mc2_ev": ELECTRON_MC2_EV,
        "P": int(target_particles),
        "n_records": n_rec,
    }


def write_processed(records: list[dict], out_h5: str, norm: dict) -> None:
    os.makedirs(os.path.dirname(out_h5) or ".", exist_ok=True)
    settings = np.stack([r["settings"] for r in records])                # (N, 11)
    particles = np.stack([r["particles"] for r in records])              # (N, P, 6)
    emit = np.array([r["norm_emit_4d"] for r in records], dtype=np.float32)
    fingerprints = np.array([r["fingerprint"] for r in records], dtype="S40")

    P = particles.shape[1]
    with h5py.File(out_h5, "w") as h:
        h.create_dataset("settings", data=settings, compression="gzip")
        h.create_dataset("particles", data=particles, compression="gzip",
                         chunks=(1, P, 6))
        h.create_dataset("norm_emit_4d", data=emit, compression="gzip")
        h.create_dataset("fingerprint", data=fingerprints)
        h.attrs["setting_keys"] = SETTING_KEYS
        h.attrs["n_settings"] = len(SETTING_KEYS)
        h.attrs["P"] = P
        h.attrs["coord_order"] = ",".join(COORD_KEYS)

    norm_path = out_h5.replace(".h5", "_norm.json")
    with open(norm_path, "w") as f:
        json.dump(norm, f, indent=2)


def preprocess(archive_glob: str, out_h5: str, *, max_archives: int | None = None,
               target_particles: int = DEFAULT_P,
               min_terminal_particles: int | None = None,
               seed: int = 0, verbose: bool = True) -> dict:
    if min_terminal_particles is None:
        min_terminal_particles = target_particles
    if min_terminal_particles < target_particles:
        raise ValueError(
            f"min_terminal_particles ({min_terminal_particles}) < target_particles "
            f"({target_particles}); cannot subsample down to a count that isn't met")

    files = sorted(glob.glob(archive_glob))
    if max_archives:
        files = files[:max_archives]

    if verbose:
        print(f"Scanning {len(files)} archives from {archive_glob}")
        print(f"Subsampling each run to P={target_particles} particles")

    rng = np.random.default_rng(seed)
    records: list[dict] = []
    n_skipped = 0
    for i, f in enumerate(files):
        rec = parse_one(f, target_particles=target_particles,
                        min_terminal_particles=min_terminal_particles, rng=rng)
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

    norm = compute_normalization(records, target_particles=target_particles, rng=rng)
    write_processed(records, out_h5, norm)
    if verbose:
        print(f"Wrote {out_h5} ({len(records)}, {target_particles}, 6) "
              f"and {out_h5.replace('.h5', '_norm.json')}")
    return norm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", default="data/archives/train/*.h5",
                        help="glob pattern for input archives")
    parser.add_argument("--out", default="data/processed/flow_surrogate.h5",
                        help="output HDF5 path")
    parser.add_argument("--max-archives", type=int, default=None,
                        help="cap archive count for smoke testing")
    parser.add_argument("--target-particles", type=int, default=DEFAULT_P,
                        help="fixed macroparticle count kept per run (lofi 1500, hifi 20000)")
    parser.add_argument("--min-terminal-particles", type=int, default=None,
                        help="drop runs below this count (defaults to --target-particles)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for subsampling + scaler pooling")
    args = parser.parse_args()

    preprocess(
        args.archives, args.out,
        max_archives=args.max_archives,
        target_particles=args.target_particles,
        min_terminal_particles=args.min_terminal_particles,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
