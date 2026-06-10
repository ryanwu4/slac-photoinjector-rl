#!/usr/bin/env python
"""
Report-ready learning-curve figure (env-steps panel only) for the surrogate-side
PPO / SHAC / BPTT benchmark on the scalar-emittance task.

This reproduces the LEFT panel of logs/compare_diff_hifi/compare.png in the report
style (seaborn "talk" context, large clear fonts), with two changes vs the
original:
  * the "random-policy expectation" dashed line is removed,
  * algorithm names are capitalized in the legend (PPO / SHAC / BPTT).

Data parity: it imports the *exact* ingestion + aggregation routines used by
emittance_target/compare_diff_algos.py, so the curves are computed identically:
  * PPO     -> tb/PPO_*/progress.csv : (total_timesteps, rollout/ep_rew_mean)
  * SHAC/BPTT -> learning_curve.csv  : (step, -mean_episode_loss)   [loss negated]
  * per-algo mean +/- stderr (ddof=1) interpolated onto a common 200-pt grid.

Run (from repo root, slac-rl env):
  PYTHONPATH=$PWD/src \
  /home/rwu4/miniconda3/envs/slac-rl/bin/python \
      figures/report/make_learning_curves_envsteps.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from photoinjector_rl.surrogates.mlp.compare_diff_algos import (  # noqa: E402
    _aggregate,
    _read_diffrl_csv,
    _read_ppo_progress,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / "runs/compare_diff_hifi"
OUT_DIR = REPO / "figures/report"

ALGOS = ["ppo", "shac", "bptt"]
SEEDS = [0, 1, 2]
# Conventional mapping (matches the original compare.png + its histogram panel).
COLORS = {"ppo": "#1f77b4", "shac": "#ff7f0e", "bptt": "#2ca02c"}
LABELS = {"ppo": "PPO", "shac": "SHAC", "bptt": "BPTT"}


def load_curves(run_root: Path) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Per-algo list of (env_steps, episode_return) curves, one per seed."""
    per_algo: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for algo in ALGOS:
        curves = []
        for seed in SEEDS:
            run_dir = run_root / algo / f"seed_{seed}"
            if algo == "ppo":
                pr = _read_ppo_progress(run_dir)
                if pr is None:
                    continue
                steps, returns, _walls = pr
                curves.append((steps, returns))
            else:
                csv_path = run_dir / "learning_curve.csv"
                if not csv_path.exists():
                    continue
                steps, losses, _walls = _read_diffrl_csv(csv_path)
                curves.append((steps, -losses))   # loss -> return
        if curves:
            per_algo[algo] = curves
    return per_algo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--out-name", default="learning_curves_envsteps",
                    help="output file stem (without extension)")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    # Larger fonts: this panel is displayed at a reduced size, so oversize the
    # text now to keep it legible after downscaling.
    sns.set_theme(context="talk", style="whitegrid", font_scale=1.35)
    plt.rcParams.update({
        "axes.edgecolor": "#444444",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
    })

    per_algo = load_curves(Path(args.run))

    # Common env-step grid spanning all seeds/algos (same recipe as the original).
    all_xs = [xs for curves in per_algo.values() for xs, _ in curves if xs.size]
    x_max = float(max(arr.max() for arr in all_xs)) if all_xs else 500_000.0
    grid = np.linspace(1, x_max, 200)

    fig, ax = plt.subplots(figsize=(9.5, 7))
    for algo in ALGOS:
        curves = per_algo.get(algo)
        if not curves:
            continue
        mean, stderr = _aggregate(curves, grid)
        ax.plot(grid, mean, label=LABELS[algo], color=COLORS[algo], lw=2.4)
        ax.fill_between(grid, mean - stderr, mean + stderr,
                        color=COLORS[algo], alpha=0.2, linewidth=0)

    ax.set_xlabel("env steps")
    ax.set_ylabel("mean episode return\n(sum of $-y_{\\mathrm{norm}}$ over episode)")
    # Fewer x ticks so the wide 6-digit labels don't crowd at the larger font.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    stub = Path(args.out_dir) / args.out_name
    fig.savefig(f"{stub}.png", dpi=300, bbox_inches="tight")
    if not args.no_pdf:
        fig.savefig(f"{stub}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {stub}.png" + ("" if args.no_pdf else " (+ .pdf)"))
    for algo in ALGOS:
        curves = per_algo.get(algo, [])
        finals = [ys[np.argmax(xs)] for xs, ys in curves]
        if finals:
            print(f"  {LABELS[algo]:>5}: {len(curves)} seeds, "
                  f"final return ~ {np.mean(finals):.1f}")


if __name__ == "__main__":
    main()
