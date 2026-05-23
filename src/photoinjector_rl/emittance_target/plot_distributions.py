"""
Pre-flight distribution plots: walks archives/{split}/*.h5 and renders a grid
showing the marginal distribution over each XOPT knob (the 11 LHS-sampled
inputs) and the output norm_emit_4d. Dashed vertical lines mark the configured
LHS bounds so the LHS fill is visually obvious.

Run from repo root:
    conda run -n slac-rl python -m photoinjector_rl.emittance_target.plot_distributions \\
        --archives 'archives/train/*.h5' --out plots/emittance_target/distributions.png
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import argparse
import glob
import os
from typing import Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
from beamphysics import ParticleGroup

from . import SETTING_BOUNDS, SETTING_KEYS


def _short_label(key: str) -> str:
    """Compact axis label for a settings key."""
    return key.replace("distgen:", "dg:").replace(":value", "")


def scan_archives(files: Sequence[str], verbose: bool = True) -> dict[str, np.ndarray]:
    """Walk archives, pull out 11 knob values + scalar emittances per run."""
    knob_values: dict[str, list[float]] = {k: [] for k in SETTING_KEYS}
    emit_4d: list[float] = []
    emit_x: list[float] = []
    emit_y: list[float] = []
    n_skipped = 0

    for i, path in enumerate(files):
        try:
            with h5py.File(path, "r") as h:
                if "settings" not in h:
                    n_skipped += 1
                    continue
                attrs = h["settings"].attrs
                for key in SETTING_KEYS:
                    knob_values[key].append(float(np.asarray(attrs[key]).item()))

                tp = ParticleGroup(h5=h["impact/output/particles/PR10241/electron"])
                emit_4d.append(float(np.asarray(tp.norm_emit_4d).item()))
                emit_x.append(float(np.asarray(tp.norm_emit_x).item()))
                emit_y.append(float(np.asarray(tp.norm_emit_y).item()))
        except (OSError, KeyError):
            n_skipped += 1
            continue

        if verbose and (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)} scanned, skipped {n_skipped}")

    if verbose:
        print(f"Scanned {len(files)}, skipped {n_skipped}, kept {len(emit_4d)}")

    return {
        **{k: np.array(v) for k, v in knob_values.items()},
        "norm_emit_4d": np.array(emit_4d),
        "norm_emit_x": np.array(emit_x),
        "norm_emit_y": np.array(emit_y),
    }


def plot_grid(data: dict[str, np.ndarray], out_png: str, bins: int = 50) -> None:
    n_panels = len(SETTING_KEYS) + 3  # 11 knobs + 3 emittances
    n_cols = 4
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    # Panels 0..10: knobs
    for i, key in enumerate(SETTING_KEYS):
        ax = axes[i]
        vals = data[key]
        ax.hist(vals, bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
        lo, hi = SETTING_BOUNDS[key]
        ax.axvline(lo, color="crimson", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(hi, color="crimson", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(_short_label(key), fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("count" if i % n_cols == 0 else "")
        ax.tick_params(labelsize=8)

    # Panels 11..13: emittances (log-x for emit_4d since it's lognormal-ish)
    for j, (key, scale, color) in enumerate([
        ("norm_emit_4d", "log", "darkorange"),
        ("norm_emit_x", "log", "seagreen"),
        ("norm_emit_y", "log", "purple"),
    ]):
        ax = axes[len(SETTING_KEYS) + j]
        vals = data[key]
        vals_pos = vals[vals > 0]
        if scale == "log" and len(vals_pos) > 0:
            bin_edges = np.logspace(np.log10(vals_pos.min()),
                                    np.log10(vals_pos.max()), bins)
            ax.hist(vals_pos, bins=bin_edges, color=color, edgecolor="white",
                    linewidth=0.3)
            ax.set_xscale("log")
        else:
            ax.hist(vals, bins=bins, color=color, edgecolor="white", linewidth=0.3)
        unit = "m^2" if "4d" in key else "m"
        ax.set_title(f"{key} ({unit}, N={len(vals)})", fontsize=9)
        ax.tick_params(labelsize=8)

    # Hide any unused axes.
    for k in range(len(SETTING_KEYS) + 3, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        f"LHS sweep coverage  |  {len(data['norm_emit_4d'])} runs  |  "
        "red dashes = configured bounds",
        fontsize=11, y=1.00,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", default="archives/train/*.h5")
    parser.add_argument("--out", default="plots/emittance_target/distributions.png")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--max-archives", type=int, default=None)
    args = parser.parse_args()

    files = sorted(glob.glob(args.archives))
    if args.max_archives:
        files = files[: args.max_archives]
    if not files:
        raise SystemExit(f"No archives matched {args.archives}")

    data = scan_archives(files)
    plot_grid(data, args.out, bins=args.bins)


if __name__ == "__main__":
    main()
