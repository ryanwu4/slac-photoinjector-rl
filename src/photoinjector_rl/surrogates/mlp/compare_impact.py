"""
Paired same-seed rollout collector on the Impact-T env.

Runs N rollouts of each of two PPO policies (typically the surrogate-trained
warm start vs the Impact-T-fine-tuned policy) on `ImpactPhotoinjectorEnv`
using the SAME N seeds. Uses a multiprocessing.Pool with `--n-workers`
processes so the Impact-T calls (~10 s each at lhs_train fidelity) run in
parallel.

Outputs:
    <out-stem>.npz   -- raw paired data (seeds + per-policy emits + per-step trajectories)
    <out-stem>.png   -- 4-panel paired figure
    stdout           -- per-metric table + head-to-head win rate

Usage:
    python -m photoinjector_rl.surrogates.mlp.compare_impact \\
        --policy-a trained/ppo_emittance_target/eval/best_model.zip \\
        --policy-b trained/ppo_impact_v1/ppo_impact_final.zip \\
        --label-a 'PPO (surrogate-only)' \\
        --label-b 'PPO (impact-finetune)' \\
        --impact-config configs/impact/ImpactT_config.yaml \\
        --distgen-input configs/impact/distgen_template.yaml \\
        --norm-json processed/emittance_target_hifi_norm.json \\
        --n-samples 30 --n-workers 8 --max-steps 32 \\
        --out trained/compare_impact

For >=2 policies or to mix PPO (.zip) with SHAC/BPTT (.pt) checkpoints, use
compare_n_impact.py instead (its PolicyAdapter dispatches by file suffix).
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Worker-local globals -- set by `_worker_init` once per pool worker, reused
# across all rollouts that worker handles. Avoids rebuilding the env and
# reloading the PPO zip per rollout.
_G_ENV = None
_G_POLICY = None


def _worker_init(policy_path, impact_config, distgen_input, norm_json, max_steps):
    """One-time setup per pool worker: build env + load policy."""
    global _G_ENV, _G_POLICY
    # Import inside the worker so the imports happen in the subprocess
    # (matters under 'spawn' start method).
    from stable_baselines3 import PPO

    from photoinjector_rl.impact.impact_env import ImpactPhotoinjectorEnv

    _G_ENV = ImpactPhotoinjectorEnv.from_norm_json(
        norm_json=norm_json,
        impact_config=impact_config,
        distgen_input_file=distgen_input,
        max_steps=max_steps,
    )
    _G_POLICY = PPO.load(policy_path, device="cpu")


def _run_one(seed: int) -> dict:
    """Run one full rollout at the given seed. Returns trajectory dict."""
    assert _G_ENV is not None and _G_POLICY is not None
    obs, info = _G_ENV.reset(seed=int(seed))
    init_emit = float(info["emit_m2"])
    distgen_norm = info["distgen_norm"].copy()
    emits = [init_emit]
    knobs = [info["knobs_phys"].copy()]
    done = False
    while not done:
        action, _ = _G_POLICY.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = _G_ENV.step(action)
        emits.append(float(info["emit_m2"]))
        knobs.append(info["knobs_phys"].copy())
        done = bool(term or trunc)
    return {
        "seed": int(seed),
        "init_emit": init_emit,
        "terminal_emit": float(emits[-1]),
        "min_emit": float(np.min(emits)),
        "emits": np.asarray(emits, dtype=np.float64),
        "knobs": np.asarray(knobs, dtype=np.float64),
        "distgen_norm": distgen_norm,
        "failure_count": int(info.get("failure_count", 0)),
    }


def collect(policy_path, seeds, args) -> list[dict]:
    """Spawn a pool of workers and collect rollouts in parallel."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.n_workers,
        initializer=_worker_init,
        initargs=(policy_path, args.impact_config, args.distgen_input,
                  args.norm_json, args.max_steps),
    ) as pool:
        t0 = time.time()
        results = []
        for r in pool.imap_unordered(_run_one, seeds.tolist()):
            results.append(r)
            n_done = len(results)
            if n_done % max(1, len(seeds) // 10) == 0 or n_done == len(seeds):
                rate = n_done / (time.time() - t0)
                print(f"  {n_done}/{len(seeds)} rollouts  "
                      f"({rate:.2f} rollouts/s, "
                      f"latest term_emit={r['terminal_emit']:.3e})")
        # imap_unordered returns out-of-order; sort by seed to align with input.
        results.sort(key=lambda x: x["seed"])
    return results


def _stack(rs: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rs])


def _stats(rs: list[dict]) -> dict:
    init = _stack(rs, "init_emit")
    term = _stack(rs, "terminal_emit")
    mn = _stack(rs, "min_emit")
    ratio = term / np.maximum(init, 1e-30)
    return {
        "median_ratio": float(np.median(ratio)),
        "mean_ratio": float(np.mean(ratio)),
        "pct_improved_5":  float(np.mean(ratio < 0.95) * 100),
        "pct_improved_10": float(np.mean(ratio < 0.90) * 100),
        "mean_terminal_m2": float(np.mean(term)),
        "mean_min_m2":      float(np.mean(mn)),
        "mean_transit_gap_m2": float(np.mean(term - mn)),
        "failures": int(sum(r["failure_count"] for r in rs)),
    }


def _print_table(s_a, s_b, name_a, name_b, win_rate_a, wall_a, wall_b):
    rows = [
        ("median terminal/initial ratio", "median_ratio", "{:.3f}"),
        ("mean   terminal/initial ratio", "mean_ratio",   "{:.3f}"),
        ("% seeds improved >5%",            "pct_improved_5",  "{:.1f}%"),
        ("% seeds improved >10%",           "pct_improved_10", "{:.1f}%"),
        ("mean terminal emit (m²)",         "mean_terminal_m2", "{:.3e}"),
        ("mean min emit (m²)",              "mean_min_m2",      "{:.3e}"),
        ("mean transit gap = term-min (m²)", "mean_transit_gap_m2", "{:.3e}"),
        ("Impact-T failures (count)",       "failures",         "{}"),
    ]
    width = max(len(r[0]) for r in rows)
    print(f"\n{'metric'.ljust(width)}  | {name_a:>22s}  | {name_b:>22s}")
    print("-" * (width + 2 + 24 * 2 + 2))
    for label, key, fmt in rows:
        va = fmt.format(s_a[key])
        vb = fmt.format(s_b[key])
        print(f"{label.ljust(width)}  | {va:>22s}  | {vb:>22s}")
    print("-" * (width + 2 + 24 * 2 + 2))
    print(f"{name_a} beats {name_b} on terminal emit: {win_rate_a:.1f}% of seeds")
    print(f"wall-clock for rollouts: {name_a}={wall_a:.1f}s, {name_b}={wall_b:.1f}s")


def _plot(rs_a, rs_b, name_a, name_b, out_png):
    init_a = _stack(rs_a, "init_emit")
    term_a = _stack(rs_a, "terminal_emit")
    term_b = _stack(rs_b, "terminal_emit")
    ratio_a = term_a / np.maximum(init_a, 1e-30)
    ratio_b = term_b / np.maximum(_stack(rs_b, "init_emit"), 1e-30)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) terminal-emit histograms
    ax = axes[0, 0]
    pos_a = term_a[term_a > 0]
    pos_b = term_b[term_b > 0]
    lo = min(pos_a.min(), pos_b.min())
    hi = max(pos_a.max(), pos_b.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 30)
    ax.hist(pos_a, bins=bins, alpha=0.55, label=name_a, color="C0",
            edgecolor="white", linewidth=0.3)
    ax.hist(pos_b, bins=bins, alpha=0.55, label=name_b, color="C1",
            edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("terminal emit (m²)")
    ax.set_ylabel("count")
    ax.set_title("terminal-emit distributions  (Impact-T)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # (0,1) paired scatter terminal_a vs terminal_b
    ax = axes[0, 1]
    a_wins = term_a < term_b
    ax.scatter(term_b[a_wins], term_a[a_wins], s=30, alpha=0.7,
               color="C0", label=f"{name_a} wins")
    ax.scatter(term_b[~a_wins], term_a[~a_wins], s=30, alpha=0.7,
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

    # (1,0) per-step mean emit trajectories
    ax = axes[1, 0]
    traj_a = np.stack([r["emits"] for r in rs_a])  # (N, T+1)
    traj_b = np.stack([r["emits"] for r in rs_b])
    T = traj_a.shape[1]
    xs = np.arange(T)
    for traj, name, c in [(traj_a, name_a, "C0"), (traj_b, name_b, "C1")]:
        med = np.median(traj, axis=0)
        q25 = np.quantile(traj, 0.25, axis=0)
        q75 = np.quantile(traj, 0.75, axis=0)
        ax.plot(xs, med, color=c, linewidth=1.6, label=f"{name} (median)")
        ax.fill_between(xs, q25, q75, color=c, alpha=0.18,
                        label=f"{name} (IQR)")
    ax.set_yscale("log")
    ax.set_xlabel("env step")
    ax.set_ylabel("norm_emit_4d (m²)")
    ax.set_title("per-step emit  (median + IQR over seeds)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

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
    ax.legend(fontsize=8)
    ax.set_xlim(0, max(2.0, float(max(ratio_a.max(), ratio_b.max()))))
    ax.grid(True, linestyle=":", alpha=0.4)

    n = len(term_a)
    a_win_rate = float(np.mean(a_wins) * 100)
    fig.suptitle(
        f"Impact-T paired rollouts   |   N={n} same-seed pairs   "
        f"|   {name_a} wins {a_win_rate:.0f}%",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-a", required=True)
    p.add_argument("--policy-b", required=True)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--impact-config", required=True)
    p.add_argument("--distgen-input", required=True)
    p.add_argument("--norm-json", required=True)
    p.add_argument("--out", default="results/compare_impact",
                   help="stem path; appended with .png and .npz")
    p.add_argument("--n-samples", type=int, default=30)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=12345,
                   help="seed for the seed-list generator (deterministic)")
    args = p.parse_args()

    # Generate N seeds once -- shared across both policies for paired rollouts.
    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, 1_000_000_000, size=args.n_samples).astype(np.int64)
    print(f"running {args.n_samples} paired rollouts (n_workers={args.n_workers})")
    print(f"seed-list base_seed={args.seed}; first 5 seeds: {seeds[:5].tolist()}")

    print(f"\n[{args.label_a}] collecting rollouts...")
    t0 = time.time()
    rs_a = collect(args.policy_a, seeds, args)
    wall_a = time.time() - t0

    print(f"\n[{args.label_b}] collecting rollouts...")
    t0 = time.time()
    rs_b = collect(args.policy_b, seeds, args)
    wall_b = time.time() - t0

    # Sanity: paired seeds + (approx) paired init emits (env reset is
    # deterministic for fixed seed).
    seeds_a = _stack(rs_a, "seed")
    seeds_b = _stack(rs_b, "seed")
    assert np.array_equal(seeds_a, seeds_b), "seeds diverged between collections"

    s_a = _stats(rs_a)
    s_b = _stats(rs_b)
    a_wins = float(np.mean(_stack(rs_a, "terminal_emit") <
                           _stack(rs_b, "terminal_emit")) * 100)
    _print_table(s_a, s_b, args.label_a, args.label_b, a_wins, wall_a, wall_b)

    _plot(rs_a, rs_b, args.label_a, args.label_b, str(args.out) + ".png")

    npz_path = str(args.out) + ".npz"
    np.savez(
        npz_path,
        seeds=seeds_a,
        label_a=np.array(args.label_a),
        label_b=np.array(args.label_b),
        init_emit=_stack(rs_a, "init_emit"),
        terminal_a=_stack(rs_a, "terminal_emit"),
        terminal_b=_stack(rs_b, "terminal_emit"),
        min_a=_stack(rs_a, "min_emit"),
        min_b=_stack(rs_b, "min_emit"),
        emits_a=np.stack([r["emits"] for r in rs_a]),
        emits_b=np.stack([r["emits"] for r in rs_b]),
        knobs_a=np.stack([r["knobs"] for r in rs_a]),
        knobs_b=np.stack([r["knobs"] for r in rs_b]),
        distgen_norm=np.stack([r["distgen_norm"] for r in rs_a]),
        failure_counts_a=np.array([r["failure_count"] for r in rs_a]),
        failure_counts_b=np.array([r["failure_count"] for r in rs_b]),
    )
    print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
