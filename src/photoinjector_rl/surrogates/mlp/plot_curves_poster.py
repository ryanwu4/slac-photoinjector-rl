"""
Poster-sized remake of the two learning-curve panels from
`compare_diff_algos.py` (env-steps and wall-clock), with large text sized for
a printed poster.

This reads the SAME on-disk learning-curve data the comparison already wrote
(SHAC/BPTT `learning_curve.csv`, PPO `tb/PPO_*/progress.csv`) and re-plots only
panels A and B — no retraining or re-eval. Aggregation (mean +- stderr across
seeds) matches the original exactly.

Usage:
    python -m photoinjector_rl.surrogates.mlp.plot_curves_poster \
        --in-dir logs/compare_diff_hifi --algos ppo,shac,bptt --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


# --- data ingest (mirrors compare_diff_algos.py) ----------------------------


def _read_diffrl_csv(path: Path):
    steps, losses, walls = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            losses.append(float(row["mean_episode_loss"]))
            walls.append(float(row["wall_time"]))
    return np.array(steps), np.array(losses), np.array(walls)


def _read_ppo_progress(run_dir: Path):
    matches = sorted((run_dir / "tb").glob("PPO_*/progress.csv"))
    if not matches:
        return None
    steps, rewards, walls = [], [], []
    with open(matches[-1]) as f:
        for row in csv.DictReader(f):
            rew = row.get("rollout/ep_rew_mean", "")
            if "time/total_timesteps" not in row or not rew:
                continue
            try:
                steps.append(int(float(row["time/total_timesteps"])))
                rewards.append(float(rew))
                walls.append(float(row.get("time/time_elapsed", "0") or 0.0))
            except ValueError:
                continue
    if not steps:
        return None
    return np.array(steps), np.array(rewards), np.array(walls)


def _aggregate(curves, grid):
    """Interp each (xs, ys) onto `grid`; return mean, stderr across seeds."""
    rows = []
    for xs, ys in curves:
        order = np.argsort(xs)
        rows.append(np.interp(grid, xs[order], ys[order]))
    mat = np.stack(rows)
    n = len(rows)
    stderr = (mat.std(axis=0, ddof=1) / math.sqrt(n) if n >= 2
              else np.zeros_like(mat[0]))
    return mat.mean(axis=0), stderr


def _load(in_dir: Path, algos, seeds):
    per_algo = {a: [] for a in algos}
    for algo in algos:
        for seed in seeds:
            run_dir = in_dir / algo / f"seed_{seed}"
            entry = {"seed": seed}
            if algo == "ppo":
                pr = _read_ppo_progress(run_dir)
                if pr is not None:
                    steps, returns, walls = pr
                    entry["lc"] = (steps, returns)
                    entry["lc_wall"] = (walls, returns)
            else:
                csv_path = run_dir / "learning_curve.csv"
                if csv_path.exists():
                    steps, losses, walls = _read_diffrl_csv(csv_path)
                    entry["lc"] = (steps, -losses)
                    entry["lc_wall"] = (walls, -losses)
            per_algo[algo].append(entry)
    return per_algo


# --- plotting ---------------------------------------------------------------

# Poster typography. These are tuned so that, when each PNG is scaled to fill a
# ~23 cm poster column, the labels read comfortably from a meter away.
POSTER_RC = {
    "font.size": 26,
    "axes.titlesize": 30,
    "axes.labelsize": 28,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
    "lines.linewidth": 3.0,
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
}

# Match the original default color cycle: ppo=C0(blue), shac=C1(orange),
# bptt=C2(green). Pin explicitly so order of plotting can't shuffle them.
ALGO_COLOR = {"ppo": "C0", "shac": "C1", "bptt": "C2"}


def _plot_env_steps(ax, per_algo):
    all_xs = [r["lc"][0] for runs in per_algo.values() for r in runs
              if "lc" in r]
    x_max = float(max(a.max() for a in all_xs)) if all_xs else 500_000.0
    grid = np.linspace(1, x_max, 200)
    for algo, runs in per_algo.items():
        curves = [r["lc"] for r in runs if "lc" in r]
        if not curves:
            continue
        mean, stderr = _aggregate(curves, grid)
        ax.plot(grid, mean, label=algo, color=ALGO_COLOR[algo])
        ax.fill_between(grid, mean - stderr, mean + stderr, alpha=0.2,
                        color=ALGO_COLOR[algo])
    ax.set_xlabel("env steps")
    ax.set_ylabel("mean episode return")
    ax.set_title("Learning curves vs env-steps")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))


def _plot_wallclock(ax, per_algo):
    all_walls = [r["lc_wall"][0] for runs in per_algo.values() for r in runs
                 if "lc_wall" in r]
    w_max = float(max(a.max() for a in all_walls)) if all_walls else 1.0
    wgrid = np.linspace(0.1, w_max, 200)
    final_walls = {}
    for algo, runs in per_algo.items():
        curves = [r["lc_wall"] for r in runs if "lc_wall" in r]
        if not curves:
            continue
        mean, stderr = _aggregate(curves, wgrid)
        ax.plot(wgrid, mean, label=algo, color=ALGO_COLOR[algo])
        ax.fill_between(wgrid, mean - stderr, mean + stderr, alpha=0.2,
                        color=ALGO_COLOR[algo])
        final_walls[algo] = [float(c[0][-1]) for c in curves]
    ax.set_xlabel("training wall-clock (s)")
    ax.set_ylabel("mean episode return")
    ax.set_title("Learning curves vs wall-clock")
    ax.legend(loc="lower right", framealpha=0.9)
    if final_walls:
        lines = [f"{algo}: {np.mean(w):.0f}s (mean of {len(w)} seeds)"
                 for algo, w in final_walls.items()]
        ax.text(0.03, 0.03, "\n".join(lines), transform=ax.transAxes,
                fontsize=20, va="bottom",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="grey"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="runs/compare_diff_hifi")
    p.add_argument("--out-dir", default=None,
                   help="defaults to <in-dir>/poster")
    p.add_argument("--algos", default="ppo,shac,bptt")
    p.add_argument("--seeds", default="0,1,2")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "poster"
    out_dir.mkdir(parents=True, exist_ok=True)
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    per_algo = _load(in_dir, algos, seeds)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Presentation-quality seaborn base, with our poster font sizes layered on
    # top via `rc` so set_theme can't shrink them back down.
    sns.set_theme(style="whitegrid", context="poster", font="sans-serif",
                  rc=POSTER_RC)

    # Standalone panel A.
    fig, ax = plt.subplots(figsize=(12, 6))
    _plot_env_steps(ax, per_algo)
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(out_dir / "curve_env_steps.png")
    fig.savefig(out_dir / "curve_env_steps.pdf")
    plt.close(fig)

    # Standalone panel B.
    fig, ax = plt.subplots(figsize=(9, 7))
    _plot_wallclock(ax, per_algo)
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(out_dir / "curve_wallclock.png")
    fig.savefig(out_dir / "curve_wallclock.pdf")
    plt.close(fig)

    # Combined two-up (spans two poster columns nicely).
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    _plot_env_steps(axes[0], per_algo)
    _plot_wallclock(axes[1], per_algo)
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(out_dir / "curves_two_up.png")
    fig.savefig(out_dir / "curves_two_up.pdf")
    plt.close(fig)

    for f in ("curve_env_steps", "curve_wallclock", "curves_two_up"):
        print(f"[poster] wrote {out_dir / (f + '.png')}  (+ .pdf)")


if __name__ == "__main__":
    main()
