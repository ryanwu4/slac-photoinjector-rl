"""
Policy-vs-distgen scatter. For a trained SAC policy, runs N rollouts at random
distgen seeds and plots:

    6 panels (one per hidden distgen knob): physical-unit knob value (x) vs
    terminal norm_emit_4d (y, log scale). Color encodes the improvement ratio
    `terminal_emit / initial_emit` -- darker = bigger reduction.

    + 1 histogram of terminal emits
    + 1 scatter of (initial_emit, terminal_emit) for the same trajectories

Tells you which cathode regimes the policy actually solves and which it
struggles with.

Usage:
    python -m photoinjector_rl.emittance_target.policy_scatter \\
        --policy trained/sac_emittance_target/eval/best_model.zip \\
        --ckpt trained/emittance_target/checkpoints/best-XXX.ckpt \\
        --norm-json processed/emittance_target_norm.json \\
        --out plots/policy_scatter.png \\
        --n-samples 200
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC

from . import SETTING_BOUNDS, SETTING_KEYS
from .env import PhotoinjectorEnv

_ALGOS = {"sac": SAC, "ppo": PPO}

_DISTGEN_KEYS = SETTING_KEYS[5:]
_DISTGEN_LO = np.array([SETTING_BOUNDS[k][0] for k in _DISTGEN_KEYS], dtype=np.float64)
_DISTGEN_HI = np.array([SETTING_BOUNDS[k][1] for k in _DISTGEN_KEYS], dtype=np.float64)


def _short(key: str) -> str:
    return key.replace("distgen:", "").replace(":value", "")


def run_rollouts(
    policy,
    env: PhotoinjectorEnv,
    n_samples: int,
    base_seed: int = 0,
) -> dict:
    """Run `n_samples` rollouts at random seeds; collect distgen + emit stats."""
    rng = np.random.default_rng(base_seed)
    distgen_phys_all = []
    initial_emits = []
    terminal_emits = []
    min_emits = []
    seeds_used = []

    for i in range(n_samples):
        seed = int(rng.integers(0, 1_000_000_000))
        seeds_used.append(seed)
        obs, info = env.reset(seed=seed)
        distgen_norm = info["distgen_norm"].astype(np.float64)
        distgen_phys = _DISTGEN_LO + distgen_norm * (_DISTGEN_HI - _DISTGEN_LO)

        initial_emit = float(info["emit_m2"])
        ep_emits = [initial_emit]
        done = False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            ep_emits.append(float(info["emit_m2"]))
            done = bool(term or trunc)

        distgen_phys_all.append(distgen_phys)
        initial_emits.append(initial_emit)
        terminal_emits.append(ep_emits[-1])
        min_emits.append(min(ep_emits))

        if (i + 1) % max(1, n_samples // 10) == 0:
            print(f"  {i + 1}/{n_samples} rollouts done")

    return {
        "distgen_phys": np.array(distgen_phys_all),    # (N, 6)
        "initial_emit": np.array(initial_emits),       # (N,)
        "terminal_emit": np.array(terminal_emits),     # (N,)
        "min_emit": np.array(min_emits),               # (N,)
        "seeds": np.array(seeds_used, dtype=np.int64), # (N,)
    }


def plot_scatter(data: dict, out_png: str) -> None:
    distgen = data["distgen_phys"]
    term = data["terminal_emit"]
    init = data["initial_emit"]
    ratio = term / np.maximum(init, 1e-30)   # < 1 = improvement, > 1 = regression

    fig = plt.figure(figsize=(16, 9))
    # 2 rows x 4 cols: 6 distgen panels + 1 hist + 1 init-vs-term scatter.
    grid = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.45,
                            left=0.06, right=0.93, top=0.90, bottom=0.08)

    sc = None
    # 6 distgen-knob panels.
    for i, key in enumerate(_DISTGEN_KEYS):
        ax = fig.add_subplot(grid[i // 4, i % 4])
        sc = ax.scatter(
            distgen[:, i], term, c=ratio,
            cmap="RdYlGn_r", vmin=0.3, vmax=1.7,
            s=18, alpha=0.75, edgecolor="none",
        )
        ax.set_xlabel(_short(key), fontsize=10)
        ax.set_ylabel("terminal emit (m²)", fontsize=9)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.tick_params(labelsize=8)
    assert sc is not None  # always populated when _DISTGEN_KEYS is non-empty

    # Histogram of terminal emit.
    ax_hist = fig.add_subplot(grid[1, 2])
    pos = term[term > 0]
    if len(pos) > 0:
        bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 40)
        ax_hist.hist(pos, bins=bins, color="steelblue", edgecolor="white",
                     linewidth=0.4)
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("terminal emit (m²)", fontsize=9)
    ax_hist.set_ylabel("count", fontsize=8)
    ax_hist.set_title("terminal emit distribution", fontsize=9)
    ax_hist.grid(True, which="both", linestyle=":", alpha=0.4)
    ax_hist.tick_params(labelsize=8)

    # Init vs terminal scatter.
    ax_it = fig.add_subplot(grid[1, 3])
    ax_it.scatter(init, term, s=14, alpha=0.7, color="purple", edgecolor="none")
    lo = float(min(init.min(), term.min())) * 0.9
    hi = float(max(init.max(), term.max())) * 1.1
    ax_it.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.6)
    ax_it.set_xlabel("initial emit (m²)", fontsize=9)
    ax_it.set_ylabel("terminal emit (m²)", fontsize=8)
    ax_it.set_xscale("log")
    ax_it.set_yscale("log")
    ax_it.set_xlim(lo, hi)
    ax_it.set_ylim(lo, hi)
    ax_it.set_aspect("equal", adjustable="box")
    ax_it.set_title(f"init vs terminal  (med ratio={np.median(ratio):.2f})",
                    fontsize=9)
    ax_it.grid(True, which="both", linestyle=":", alpha=0.4)
    ax_it.tick_params(labelsize=8)

    cax = fig.add_axes((0.945, 0.15, 0.012, 0.7))
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("terminal / initial   (<1 better, >1 worse)", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    n = len(term)
    pct_improve = float(np.mean(ratio < 0.95) * 100)
    fig.suptitle(
        f"Policy performance vs hidden distgen state   |   N={n}\n"
        f"median improvement = {np.median(ratio):.2f}×   "
        f"|   policy improved (>5%) on {pct_improve:.1f}% of seeds",
        fontsize=11,
    )

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True,
                   help="path to SB3 policy zip (e.g. best_model.zip)")
    p.add_argument("--algo", default="sac", choices=sorted(_ALGOS.keys()),
                   help="which SB3 algorithm produced the zip "
                        "(default 'sac' for backward compat)")
    p.add_argument("--ckpt", required=True,
                   help="path to EmittanceMLP surrogate checkpoint")
    p.add_argument("--norm-json", required=True,
                   help="path to processed/emittance_target_norm.json")
    p.add_argument("--out", default="plots/policy_scatter.png")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu",
                   help="surrogate device; use cuda:N if available")
    args = p.parse_args()

    print(f"loading {args.algo.upper()} policy: {args.policy}")
    env = PhotoinjectorEnv.from_checkpoint(
        ckpt_path=args.ckpt, norm_json=args.norm_json, device=args.device,
    )
    policy = _ALGOS[args.algo].load(args.policy, env=env, device=args.device)

    print(f"running {args.n_samples} rollouts (seed={args.seed})...")
    data = run_rollouts(policy, env, args.n_samples, base_seed=args.seed)
    plot_scatter(data, args.out)

    # Also dump raw data as npz alongside the PNG.
    npz_path = Path(args.out).with_suffix(".npz")
    np.savez(npz_path,
             distgen_phys=data["distgen_phys"],
             distgen_keys=np.array(_DISTGEN_KEYS),
             initial_emit=data["initial_emit"],
             terminal_emit=data["terminal_emit"],
             min_emit=data["min_emit"],
             seeds=data["seeds"])
    print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
