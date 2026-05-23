"""
Paired same-seed comparison of two trained policies (typically SAC vs PPO).

Runs N rollouts of each policy on the same `n_samples` distgen seeds, then
reports paired statistics:

    - median terminal_emit / initial_emit ratio per algo
    - fraction of seeds where each algo improved emit > 5%
    - per-seed head-to-head win rate (does algo A's terminal beat B's?)
    - mean transit-vs-settle gap (min_emit vs terminal_emit) per algo
    - rough wall-clock for each algo's rollouts

Also writes a 4-panel PNG:
    1. histograms of terminal_emit per algo (log x)
    2. paired scatter terminal_emit_A vs terminal_emit_B (color = winner)
    3. histogram of log10(ratio_A / ratio_B)  -- < 0 = algo A wins
    4. CDF of ratio = terminal / initial per algo

Usage:
    python -m photoinjector_rl.emittance_target.compare_algos \\
        --algo-a sac --policy-a trained/sac_v3/eval/best_model.zip \\
        --algo-b ppo --policy-b trained/ppo_v1/eval/best_model.zip \\
        --ckpt trained/emittance_target/checkpoints/best-XXX.ckpt \\
        --norm-json processed/emittance_target_norm.json \\
        --out trained/compare_sac_vs_ppo.png \\
        --n-samples 200 --device cuda:0
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .env import PhotoinjectorEnv
from .policy_scatter import _ALGOS, run_rollouts


def _stats(data: dict) -> dict:
    """Per-algo summary statistics."""
    init = data["initial_emit"]
    term = data["terminal_emit"]
    minimum = data["min_emit"]
    ratio = term / np.maximum(init, 1e-30)
    return {
        "median_ratio": float(np.median(ratio)),
        "mean_ratio": float(np.mean(ratio)),
        "pct_improved_5": float(np.mean(ratio < 0.95) * 100),
        "pct_improved_10": float(np.mean(ratio < 0.90) * 100),
        "mean_transit_gap_m2": float(np.mean(term - minimum)),
        "mean_terminal_m2": float(np.mean(term)),
        "mean_min_m2": float(np.mean(minimum)),
    }


def _print_table(stats_a: dict, stats_b: dict,
                 name_a: str, name_b: str,
                 win_rate_a: float, wall_a: float, wall_b: float) -> None:
    rows = [
        ("median terminal/initial ratio", "median_ratio", "{:.3f}"),
        ("mean   terminal/initial ratio", "mean_ratio",   "{:.3f}"),
        ("% seeds improved >5%",           "pct_improved_5",  "{:.1f}%"),
        ("% seeds improved >10%",          "pct_improved_10", "{:.1f}%"),
        ("mean terminal emit (m²)",        "mean_terminal_m2", "{:.3e}"),
        ("mean min emit (m²)",             "mean_min_m2",      "{:.3e}"),
        ("mean transit gap = term-min (m²)", "mean_transit_gap_m2", "{:.3e}"),
    ]
    width = max(len(r[0]) for r in rows)
    print(f"\n{'metric'.ljust(width)}  | {name_a:>14s}  | {name_b:>14s}")
    print("-" * (width + 2 + 16 * 2 + 2))
    for label, key, fmt in rows:
        va = fmt.format(stats_a[key])
        vb = fmt.format(stats_b[key])
        print(f"{label.ljust(width)}  | {va:>14s}  | {vb:>14s}")
    print("-" * (width + 2 + 16 * 2 + 2))
    print(f"{name_a} beats {name_b} (terminal emit) on {win_rate_a:.1f}% of seeds")
    print(f"wall-clock for rollouts: {name_a}={wall_a:.1f}s, {name_b}={wall_b:.1f}s")


def _plot_paired(data_a: dict, data_b: dict,
                 name_a: str, name_b: str,
                 out_png: str) -> None:
    term_a = data_a["terminal_emit"]
    term_b = data_b["terminal_emit"]
    init = data_a["initial_emit"]  # same seeds → same initial values
    ratio_a = term_a / np.maximum(init, 1e-30)
    ratio_b = term_b / np.maximum(init, 1e-30)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) terminal-emit histograms
    ax = axes[0, 0]
    pos_a = term_a[term_a > 0]
    pos_b = term_b[term_b > 0]
    lo = min(pos_a.min(), pos_b.min())
    hi = max(pos_a.max(), pos_b.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 40)
    ax.hist(pos_a, bins=bins, alpha=0.55, label=name_a, color="C0",
            edgecolor="white", linewidth=0.3)
    ax.hist(pos_b, bins=bins, alpha=0.55, label=name_b, color="C1",
            edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("terminal emit (m²)")
    ax.set_ylabel("count")
    ax.set_title("terminal-emit distributions")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # (0,1) paired scatter terminal_a vs terminal_b
    ax = axes[0, 1]
    a_wins = term_a < term_b
    ax.scatter(term_b[a_wins], term_a[a_wins], s=18, alpha=0.7,
               color="C0", label=f"{name_a} wins")
    ax.scatter(term_b[~a_wins], term_a[~a_wins], s=18, alpha=0.7,
               color="C1", label=f"{name_b} wins")
    lo_s = min(term_a.min(), term_b.min()) * 0.9
    hi_s = max(term_a.max(), term_b.max()) * 1.1
    ax.plot([lo_s, hi_s], [lo_s, hi_s], "k--", linewidth=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo_s, hi_s)
    ax.set_ylim(lo_s, hi_s)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{name_b} terminal emit (m²)")
    ax.set_ylabel(f"{name_a} terminal emit (m²)")
    ax.set_title(f"paired terminal emit  ({name_a} wins below diagonal)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # (1,0) histogram of log10(ratio_a / ratio_b)
    ax = axes[1, 0]
    log_diff = np.log10(np.maximum(ratio_a, 1e-30)) - np.log10(np.maximum(ratio_b, 1e-30))
    ax.hist(log_diff, bins=40, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"log10(ratio_{name_a} / ratio_{name_b})   < 0 = {name_a} wins")
    ax.set_ylabel("count")
    ax.set_title("paired log-ratio of improvement")
    ax.grid(True, linestyle=":", alpha=0.4)

    # (1,1) ratio CDFs
    ax = axes[1, 1]
    for arr, name, color in [(ratio_a, name_a, "C0"), (ratio_b, name_b, "C1")]:
        sorted_r = np.sort(arr)
        y = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        ax.plot(sorted_r, y, label=name, color=color, linewidth=1.6)
    ax.axvline(1.0, color="black", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("terminal / initial emit ratio")
    ax.set_ylabel("CDF")
    ax.set_title("improvement ratio (lower-left = better)")
    ax.legend()
    ax.set_xlim(0, max(2.0, float(max(ratio_a.max(), ratio_b.max()))))
    ax.grid(True, linestyle=":", alpha=0.4)

    n = len(term_a)
    a_win_rate = float(np.mean(a_wins) * 100)
    fig.suptitle(
        f"{name_a} vs {name_b}   |   N={n} paired seeds   "
        f"|   {name_a} wins terminal-emit on {a_win_rate:.1f}% of seeds",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo-a", default="sac", choices=sorted(_ALGOS.keys()))
    p.add_argument("--algo-b", default="ppo", choices=sorted(_ALGOS.keys()))
    p.add_argument("--policy-a", required=True)
    p.add_argument("--policy-b", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--norm-json", required=True)
    p.add_argument("--out", default="trained/compare.png")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    env = PhotoinjectorEnv.from_checkpoint(
        ckpt_path=args.ckpt, norm_json=args.norm_json, device=args.device,
    )

    def _run(algo: str, policy_path: str) -> tuple[dict, float]:
        print(f"\n[{algo.upper()}] loading {policy_path}")
        policy = _ALGOS[algo].load(policy_path, env=env, device=args.device)
        t0 = time.time()
        data = run_rollouts(policy, env, args.n_samples, base_seed=args.seed)
        dt = time.time() - t0
        print(f"[{algo.upper()}] {args.n_samples} rollouts in {dt:.1f}s")
        return data, dt

    # base_seed is the same for both → same seed sequence → paired rollouts.
    data_a, wall_a = _run(args.algo_a, args.policy_a)
    data_b, wall_b = _run(args.algo_b, args.policy_b)

    # Sanity: seeds should match exactly between runs.
    if not np.array_equal(data_a["seeds"], data_b["seeds"]):
        raise SystemExit("seed mismatch between A and B rollouts (bug?)")
    # And initial emits should match (same seed → same distgen → same init).
    if not np.allclose(data_a["initial_emit"], data_b["initial_emit"], rtol=1e-6):
        raise SystemExit("initial_emit mismatch — env reset is not deterministic")

    name_a = args.algo_a.upper()
    name_b = args.algo_b.upper()

    stats_a = _stats(data_a)
    stats_b = _stats(data_b)
    a_wins = float(np.mean(data_a["terminal_emit"] < data_b["terminal_emit"]) * 100)
    _print_table(stats_a, stats_b, name_a, name_b, a_wins, wall_a, wall_b)

    _plot_paired(data_a, data_b, name_a, name_b, args.out)

    # Dump combined npz for downstream re-analysis.
    npz_path = Path(args.out).with_suffix(".npz")
    np.savez(npz_path,
             seeds=data_a["seeds"],
             initial_emit=data_a["initial_emit"],
             terminal_emit_a=data_a["terminal_emit"],
             terminal_emit_b=data_b["terminal_emit"],
             min_emit_a=data_a["min_emit"],
             min_emit_b=data_b["min_emit"],
             distgen_phys=data_a["distgen_phys"],
             algo_a=np.array(args.algo_a),
             algo_b=np.array(args.algo_b))
    print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
