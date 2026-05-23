"""
N-way paired comparison of PPO policies on the Impact-T env.

Generalization of compare_impact.py to >=2 policies. Useful for 3-way
ablations like "surrogate-only" vs "impact-from-scratch" vs
"surrogate-pretrain + impact-fine-tune" -- the second of which answers
the question of whether the surrogate pretraining is actually load-bearing.

Each `--policy LABEL=PATH` argument adds one policy. All policies are
evaluated on the SAME N seeds (so head-to-head per-seed comparisons are
meaningful).

Usage:
    python -m photoinjector_rl.emittance_target.compare_n_impact \\
        --policy surrogate_only=trained/ppo_v1/eval/best_model.zip \\
        --policy impact_scratch=trained/ppo_impact_v1_BROKEN_scratch/ppo_impact_final.zip \\
        --policy impact_finetune=trained/ppo_impact_v1/ppo_impact_final.zip \\
        --impact-config configs/impact/ImpactT_config.yaml \\
        --distgen-input configs/impact/distgen_template.yaml \\
        --norm-json processed/emittance_target_norm.json \\
        --n-samples 30 --n-workers 8 --max-steps 32 \\
        --out trained/compare_three
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

from .compare_impact import _stack, _stats


# ---------------------------------------------------------------------------
# Policy adapters — let the worker treat SB3 zips and diffrl .pt checkpoints
# uniformly via a single `predict(obs, deterministic=True) -> (action, _state)`
# contract that mirrors SB3.
# ---------------------------------------------------------------------------


class PolicyAdapter:
    def predict(self, obs, deterministic: bool = True):
        raise NotImplementedError


class SB3Adapter(PolicyAdapter):
    def __init__(self, path: str):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")

    def predict(self, obs, deterministic: bool = True):
        return self.model.predict(obs, deterministic=deterministic)


class DiffRLAdapter(PolicyAdapter):
    """Loads SHAC/BPTT checkpoints saved by `diffrl.{SHAC,BPTT}.save()`.

    SHAC saves `[actor, critic, target_critic, obs_rms, ret_rms]`;
    BPTT saves `[actor, obs_rms]`. We pick out obs_rms by `isinstance` so
    both layouts work without a branch.
    """

    def __init__(self, path: str):
        import torch
        from .diffrl.utils import RunningMeanStd
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        self.actor = ckpt[0].to("cpu").eval()
        self.obs_rms = next(
            (x for x in ckpt if isinstance(x, RunningMeanStd)), None
        )
        if self.obs_rms is not None:
            self.obs_rms = self.obs_rms.to("cpu")
        self._torch = torch

    def predict(self, obs, deterministic: bool = True):
        torch = self._torch
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(obs, dtype=np.float32))
            if x.ndim == 1:
                x = x.unsqueeze(0)
            if self.obs_rms is not None:
                x = self.obs_rms.normalize(x)
            a = self.actor(x, deterministic=deterministic)
            a = torch.tanh(a).squeeze(0).cpu().numpy().astype(np.float32)
        return a, None


def _load_adapter(path: str) -> PolicyAdapter:
    p = str(path)
    if p.endswith(".zip"):
        return SB3Adapter(p)
    if p.endswith(".pt"):
        return DiffRLAdapter(p)
    raise ValueError(f"unknown policy format (need .zip or .pt): {p}")


# Worker-local state. The worker holds ONE env (Impact-T config doesn't vary
# across policies) and a dict of preloaded policy adapters keyed by label, so
# each task can dispatch by label without reloading anything.
_G_ENV = None
_G_POLICIES: dict | None = None


def _load_constants_yaml(path):
    """Pull `vocs.constants` from a sweep YAML (e.g. lhs_train_hifi.yaml).
    Returns None if path is falsy so callers can fall back to env defaults."""
    if not path:
        return None
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    consts = cfg.get("vocs", {}).get("constants")
    if not consts:
        raise ValueError(f"{path}: no vocs.constants block found")
    return dict(consts)


def _worker_init_multi(policy_paths_by_label, impact_config, distgen_input,
                       norm_json, max_steps, constants):
    """One-time setup per pool worker: build env + load all policy adapters."""
    global _G_ENV, _G_POLICIES
    from .impact_env import ImpactPhotoinjectorEnv

    env_kwargs = dict(
        norm_json=norm_json,
        impact_config=impact_config,
        distgen_input_file=distgen_input,
        max_steps=max_steps,
    )
    if constants is not None:
        env_kwargs["constants"] = constants
    _G_ENV = ImpactPhotoinjectorEnv.from_norm_json(**env_kwargs)
    _G_POLICIES = {
        label: _load_adapter(path)
        for label, path in policy_paths_by_label.items()
    }


def _run_task(task: tuple[str, int]) -> dict:
    """Run one (label, seed) rollout using the pre-loaded policy."""
    label, seed = task
    assert _G_ENV is not None and _G_POLICIES is not None
    policy = _G_POLICIES[label]

    obs, info = _G_ENV.reset(seed=int(seed))
    init_emit = float(info["emit_m2"])
    distgen_norm = info["distgen_norm"].copy()
    emits = [init_emit]
    knobs = [info["knobs_phys"].copy()]
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = _G_ENV.step(action)
        emits.append(float(info["emit_m2"]))
        knobs.append(info["knobs_phys"].copy())
        done = bool(term or trunc)
    return {
        "label": label,
        "seed": int(seed),
        "init_emit": init_emit,
        "terminal_emit": float(emits[-1]),
        "min_emit": float(np.min(emits)),
        "emits": np.asarray(emits, dtype=np.float64),
        "knobs": np.asarray(knobs, dtype=np.float64),
        "distgen_norm": distgen_norm,
        "failure_count": int(info.get("failure_count", 0)),
    }


def collect_all(policy_paths_by_label, seeds, args) -> dict[str, list[dict]]:
    """Run all (policy, seed) rollouts in one shared pool. Returns dict
    label -> list-of-results sorted by seed."""
    labels = list(policy_paths_by_label.keys())
    tasks = [(label, int(s)) for label in labels for s in seeds]

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.n_workers,
        initializer=_worker_init_multi,
        initargs=(policy_paths_by_label, args.impact_config, args.distgen_input,
                  args.norm_json, args.max_steps,
                  getattr(args, "_constants", None)),
    ) as pool:
        t0 = time.time()
        all_results = []
        report_every = max(1, len(tasks) // 20)
        for r in pool.imap_unordered(_run_task, tasks):
            all_results.append(r)
            n_done = len(all_results)
            if n_done % report_every == 0 or n_done == len(tasks):
                rate = n_done / (time.time() - t0)
                print(f"  {n_done}/{len(tasks)} rollouts "
                      f"({rate:.2f} rollouts/s, "
                      f"latest [{r['label']}] term_emit={r['terminal_emit']:.3e})")

    # Partition by label and sort by seed (so all policies share row order).
    by_label: dict[str, list[dict]] = {label: [] for label in labels}
    for r in all_results:
        by_label[r["label"]].append(r)
    for label in labels:
        by_label[label].sort(key=lambda x: x["seed"])
    return by_label


def _parse_policy_arg(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"--policy must be LABEL=PATH, got {s!r}"
        )
    label, path = s.split("=", 1)
    return label.strip(), path.strip()


def _print_n_table(labels, stats, wins_matrix, walls):
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
    col_w = 22
    width = max(len(r[0]) for r in rows)
    header = "  | ".join([" " * width] + [f"{lbl:>{col_w}s}" for lbl in labels])
    print("\n" + header)
    print("-" * (width + (col_w + 4) * len(labels)))
    for label, key, fmt in rows:
        values = [fmt.format(s[key]) for s in stats]
        print(f"{label.ljust(width)}  | " +
              "  | ".join(f"{v:>{col_w}s}" for v in values))
    print("-" * (width + (col_w + 4) * len(labels)))

    print("\nhead-to-head terminal-emit wins (row vs col, % of seeds):")
    print(" " * 20 + "  ".join(f"{lbl:>20s}" for lbl in labels))
    for i, lbl in enumerate(labels):
        cells = []
        for j in range(len(labels)):
            if i == j:
                cells.append("        --")
            else:
                cells.append(f"{wins_matrix[i, j]*100:7.1f}%")
        print(f"{lbl:>20s}  " + "  ".join(f"{c:>20s}" for c in cells))

    print("\nwall-clock for rollouts:")
    for lbl, w in zip(labels, walls):
        print(f"  {lbl:>20s}: {w:.1f}s")


def _plot_n(rs_list, labels, out_png):
    n = len(labels)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(n, 4)))[:n]

    # Same 4-panel layout as compare_impact, generalized to N policies.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    init = _stack(rs_list[0], "init_emit")  # same for all (same seeds)
    terms = [_stack(rs, "terminal_emit") for rs in rs_list]
    ratios = [t / np.maximum(init, 1e-30) for t in terms]

    # (0,0) terminal-emit histograms.
    ax = axes[0, 0]
    pos_all = np.concatenate([t[t > 0] for t in terms])
    lo, hi = pos_all.min(), pos_all.max()
    bins = np.logspace(np.log10(lo), np.log10(hi), 30)
    for t, lbl, c in zip(terms, labels, colors):
        ax.hist(t[t > 0], bins=bins, alpha=0.5, label=lbl, color=c,
                edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("terminal emit (m²)")
    ax.set_ylabel("count")
    ax.set_title("terminal-emit distributions  (Impact-T)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # (0,1) paired per-seed strip: y = terminal emit, x = seed-index, color = policy.
    ax = axes[0, 1]
    order = np.argsort(init)   # sort by init emit for visual coherence
    for t, lbl, c in zip(terms, labels, colors):
        ax.scatter(np.arange(len(t)), t[order], s=30, alpha=0.7,
                   label=lbl, color=c, edgecolor="none")
    # init_emit envelope.
    ax.plot(np.arange(len(init)), init[order], "k--", linewidth=0.8,
            alpha=0.5, label="initial emit")
    ax.set_yscale("log")
    ax.set_xlabel("seed index (sorted by init emit)")
    ax.set_ylabel("emit (m²)")
    ax.set_title("per-seed terminal emit  (lower = better)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # (1,0) per-step trajectory median + IQR.
    ax = axes[1, 0]
    for rs, lbl, c in zip(rs_list, labels, colors):
        traj = np.stack([r["emits"] for r in rs])
        T = traj.shape[1]
        xs = np.arange(T)
        med = np.median(traj, axis=0)
        q25 = np.quantile(traj, 0.25, axis=0)
        q75 = np.quantile(traj, 0.75, axis=0)
        ax.plot(xs, med, color=c, linewidth=1.6, label=lbl)
        ax.fill_between(xs, q25, q75, color=c, alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("env step")
    ax.set_ylabel("norm_emit_4d (m²)")
    ax.set_title("per-step emit  (median + IQR over seeds)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # (1,1) ratio CDFs.
    ax = axes[1, 1]
    for r, lbl, c in zip(ratios, labels, colors):
        sorted_r = np.sort(r)
        y = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        ax.plot(sorted_r, y, label=lbl, color=c, linewidth=1.6)
    ax.axvline(1.0, color="black", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("terminal / initial emit ratio")
    ax.set_ylabel("CDF")
    ax.set_title("improvement ratio CDF  (lower-left = better)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, max(2.0, float(np.max([r.max() for r in ratios]))))
    ax.grid(True, linestyle=":", alpha=0.4)

    n_samples = len(init)
    fig.suptitle(
        f"Impact-T paired rollouts   |   N={n_samples} same-seed pairs   |   "
        f"{n}-way: " + " vs ".join(labels),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", action="append", required=True,
                   type=_parse_policy_arg,
                   help="LABEL=PATH (repeat for each policy to compare)")
    p.add_argument("--impact-config", required=True)
    p.add_argument("--distgen-input", required=True)
    p.add_argument("--norm-json", required=True)
    p.add_argument("--out", default="trained/compare_n_impact")
    p.add_argument("--n-samples", type=int, default=30)
    p.add_argument("--n-workers", type=int, default=0,
                   help="0 = auto (min(cpu_count, total_tasks))")
    p.add_argument("--max-steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=12345,
                   help="seed for the seed-list generator (matches "
                        "compare_impact.py default so npz files are paired)")
    p.add_argument("--constants-yaml", default=None,
                   help="sweep YAML whose vocs.constants block pins the "
                        "Impact-T fidelity (mesh, n_particle, total_charge). "
                        "Omit to use impact_env.DEFAULT_LHS_CONSTANTS (lo-fi).")
    args = p.parse_args()
    # Resolve once in the parent so workers don't each re-parse the YAML.
    args._constants = _load_constants_yaml(args.constants_yaml)
    if args._constants is not None:
        print(f"[compare_n_impact] pinning fidelity from "
              f"{args.constants_yaml}: {args._constants}")

    if len(args.policy) < 2:
        raise SystemExit("Need at least 2 --policy entries")

    labels = [lbl for lbl, _ in args.policy]
    paths = [pth for _, pth in args.policy]
    if len(set(labels)) != len(labels):
        raise SystemExit("Duplicate --policy labels")

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, 1_000_000_000, size=args.n_samples).astype(np.int64)
    total_tasks = args.n_samples * len(labels)
    if args.n_workers <= 0:
        args.n_workers = min(os.cpu_count() or 8, total_tasks)
    print(f"running {args.n_samples} paired rollouts × {len(labels)} policies "
          f"(n_workers={args.n_workers}, max_steps={args.max_steps})")
    print(f"policies: {labels}")
    print(f"seed-list base_seed={args.seed}; first 5 seeds: {seeds[:5].tolist()}")

    policy_dict = dict(zip(labels, paths))
    print(f"\ncollecting {args.n_samples * len(labels)} rollouts via one pool "
          f"of {args.n_workers} workers (each worker holds all {len(labels)} "
          f"policies + 1 Impact env)")
    t_total = time.time()
    by_label = collect_all(policy_dict, seeds, args)
    total_wall = time.time() - t_total
    print(f"all rollouts done in {total_wall:.1f}s")

    # Per-policy walls are not separately measured in shared-pool mode; report
    # the aggregate and let max() of per-rollout timing approximate per-policy.
    rs_list = [by_label[lbl] for lbl in labels]
    walls = [total_wall / len(labels)] * len(labels)  # rough split

    # Sanity: seeds aligned.
    for rs in rs_list[1:]:
        assert np.array_equal(_stack(rs_list[0], "seed"), _stack(rs, "seed")), \
            "seed mismatch between collections"

    stats = [_stats(rs) for rs in rs_list]
    n = len(labels)
    wins = np.zeros((n, n), dtype=np.float64)
    terms = [_stack(rs, "terminal_emit") for rs in rs_list]
    for i in range(n):
        for j in range(n):
            if i != j:
                wins[i, j] = float(np.mean(terms[i] < terms[j]))

    _print_n_table(labels, stats, wins, walls)
    _plot_n(rs_list, labels, str(args.out) + ".png")

    npz_path = str(args.out) + ".npz"
    save_dict = {
        "labels": np.array(labels),
        "seeds": _stack(rs_list[0], "seed"),
        "init_emit": _stack(rs_list[0], "init_emit"),
        "distgen_norm": np.stack([r["distgen_norm"] for r in rs_list[0]]),
    }
    for lbl, rs in zip(labels, rs_list):
        save_dict[f"terminal_{lbl}"] = _stack(rs, "terminal_emit")
        save_dict[f"min_{lbl}"] = _stack(rs, "min_emit")
        save_dict[f"emits_{lbl}"] = np.stack([r["emits"] for r in rs])
        save_dict[f"knobs_{lbl}"] = np.stack([r["knobs"] for r in rs])
        save_dict[f"failures_{lbl}"] = np.array([r["failure_count"] for r in rs])
    np.savez(npz_path, **save_dict)
    print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
