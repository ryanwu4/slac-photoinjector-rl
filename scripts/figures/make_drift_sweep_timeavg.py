#!/usr/bin/env python
"""
Report-ready version of logs/eval_drift_sweep_timeavg/timeavg_vs_drift.png:
clean-trained PPO / SHAC / BPTT policies re-evaluated under a sweep of
eval-time distgen jitter (no retraining), TIME-AVERAGED metric.

x = eval distgen_drift_std (normalized, per step); y = time-averaged
norm_emit_4d over the episode (m^2, log scale). Per algorithm:
  * line + markers = mean over seeds of the per-(algo,seed) median-over-rollouts,
  * shaded band    = mean +/- stderr (ddof=1) over the 3 seeds,
  * faint dots     = the per-seed medians.

Aggregation mirrors emittance_target/eval_drift_sweep_timeavg.py::_plot_timeavg
exactly (reads the same summary.csv); only the styling differs (seaborn
whitegrid + large fonts, unified report colors, capitalized algo names).

Run (from repo root, slac-rl env):
  PYTHONPATH=$PWD/src \
  /home/rwu4/miniconda3/envs/slac-rl/bin/python \
      figures/report/make_drift_sweep_timeavg.py
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO / "results/eval_drift_sweep_timeavg/summary.csv"
OUT_DIR = REPO / "figures/report"

# Unified report color scheme: PPO blue, SHAC orange, BPTT green.
COLORS = {"ppo": "#1f77b4", "shac": "#ff7f0e", "bptt": "#2ca02c"}
LABELS = {"ppo": "PPO", "shac": "SHAC", "bptt": "BPTT"}
ALGOS = ["ppo", "shac", "bptt"]      # legend order (matches learning-curve fig)


def load_rows(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "algo": r["algo"],
                "drift": float(r["drift"]),
                "med": float(r["timeavg_emit_median_m2"]),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--out-name", default="drift_sweep_timeavg")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    sns.set_theme(context="talk", style="whitegrid", font_scale=1.0)
    plt.rcParams.update({
        "axes.edgecolor": "#444444",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
    })

    rows = load_rows(Path(args.csv))
    drift_levels = sorted({r["drift"] for r in rows})

    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    for algo in ALGOS:
        color = COLORS[algo]
        xs, means, errs = [], [], []
        for d in drift_levels:
            meds = [r["med"] for r in rows
                    if r["algo"] == algo and r["drift"] == d
                    and not math.isnan(r["med"])]
            if not meds:
                continue
            m = np.asarray(meds)
            xs.append(d)
            means.append(float(m.mean()))
            errs.append(float(m.std(ddof=1) / math.sqrt(len(m))) if len(m) >= 2 else 0.0)
            ax.scatter([d] * len(m), m, color=color, alpha=0.35, s=26, zorder=2,
                       linewidths=0)
        if not xs:
            continue
        xs_a, m_a, e_a = np.asarray(xs), np.asarray(means), np.asarray(errs)
        ax.plot(xs_a, m_a, "-o", color=color, lw=2.4, ms=8,
                label=LABELS[algo], zorder=3)
        ax.fill_between(xs_a, m_a - e_a, m_a + e_a, color=color, alpha=0.18,
                        linewidth=0, zorder=1)

    ax.set_yscale("log")
    ax.set_xlabel("Distgen Drift Std. (normalized)")
    ax.set_ylabel("Time-Averaged $\\epsilon_{4D}$ (m$^2$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="trained @ drift = 0", frameon=True, framealpha=0.9)
    fig.tight_layout()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    stub = Path(args.out_dir) / args.out_name
    fig.savefig(f"{stub}.png", dpi=300, bbox_inches="tight")
    if not args.no_pdf:
        fig.savefig(f"{stub}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stub}.png" + ("" if args.no_pdf else " (+ .pdf)"))
    for algo in ALGOS:
        hi = [r["med"] for r in rows if r["algo"] == algo and r["drift"] == max(drift_levels)]
        lo = [r["med"] for r in rows if r["algo"] == algo and r["drift"] == 0.0]
        if hi and lo:
            print(f"  {LABELS[algo]:>5}: drift0 {np.mean(lo):.3e} -> "
                  f"drift{max(drift_levels)} {np.mean(hi):.3e} m^2")


if __name__ == "__main__":
    main()
