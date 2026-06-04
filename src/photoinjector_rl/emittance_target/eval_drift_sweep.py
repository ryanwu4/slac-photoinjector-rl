"""
Eval-only distgen-drift robustness sweep.

Loads the CLEAN-trained (distgen_drift_std = 0) PPO / SHAC / BPTT policies
produced by `compare_diff_algos.py` (default root: logs/compare_diff_hifi) and
re-runs the deterministic terminal-emittance eval under a sweep of nonzero
drift levels, WITHOUT retraining. This isolates the *test-time* robustness to
cathode jitter of policies that never saw jitter during training -- the
complement to `compare_hifi_drift.sh`, which trains AND evals under drift.

Each trained run's cfg.yaml records distgen_drift_std = 0.0. This script
ignores that for eval and forces each requested drift level instead; for
SHAC/BPTT the eval env still inherits action_scale + episode_length from the
run's cfg.yaml so it otherwise matches how the policy was trained.

It reuses the very same eval helpers `compare_diff_algos.py` used to build the
baseline summary (`_eval_diffrl_policy` for SHAC/BPTT, `_eval_ppo_policy` for
PPO), so the drift = 0.0 column reproduces logs/compare_diff_hifi/summary.csv
as a built-in sanity check.

Usage:
    python -m photoinjector_rl.emittance_target.eval_drift_sweep \
        --ckpt trained/emittance_target_hifi/checkpoints/best-epoch=191-val_loss=0.0060.ckpt \
        --norm-json processed/emittance_target_hifi_norm.json \
        --runs-root logs/compare_diff_hifi \
        --out-dir logs/eval_drift_sweep \
        --drift-levels 0.0,0.02,0.05,0.1 \
        --algos ppo,shac,bptt --seeds 0,1,2 --eval-rollouts 256 --device cuda:0

Outputs (in --out-dir; the source run root is NOT modified):
    summary.csv      one row per (algo, seed, drift): terminal-emit stats (m^2)
    drift_sweep.png  median terminal emit vs drift level, mean +- stderr / seeds
    stats.json       full structured dump + _config
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .compare_diff_algos import (
    _eval_diffrl_policy,
    _eval_ppo_policy,
    _load_env_kwargs,
)


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _find_diffrl_policy(run_dir: Path) -> Path | None:
    """final_policy.pt if present, else best_policy.pt (matches compare)."""
    for name in ("final_policy.pt", "best_policy.pt"):
        p = run_dir / name
        if p.exists():
            return p
    return None


def _eval_ppo_sameenv(run_dir: Path, ns: SimpleNamespace, n_rollouts: int,
                      drift: float) -> np.ndarray:
    """Terminal emit (m^2) for a PPO policy on the SAME batched
    DiffPhotoinjectorEnv (seed=12345) used for SHAC/BPTT — fair, common-random-
    numbers comparison. PPO is a Gaussian policy: deterministic=True gives the
    mean action, which the env clips to [-1, 1] (matching SurrogateVecEnv at
    train time, NOT a tanh squash). PPO trained with action_scale=0.05,
    episode_length=64."""
    import torch

    from stable_baselines3 import PPO

    from .diff_env import DiffPhotoinjectorEnv

    candidates = [run_dir / "ppo_final.zip", run_dir / "eval" / "best_model.zip"]
    policy_zip = next((c for c in candidates if c.exists()), None)
    if policy_zip is None:
        raise FileNotFoundError(f"No PPO zip in {run_dir}")
    model = PPO.load(str(policy_zip), device=ns.device)

    ep_len = 64
    env = DiffPhotoinjectorEnv(
        num_envs=n_rollouts, device=ns.device, seed=12345,
        episode_length=ep_len, stochastic_init=True, no_grad=True,
        surrogate_ckpt=ns.ckpt, norm_json=ns.norm_json,
        action_scale=0.05, distgen_drift_std=float(drift),
    )
    obs = env.reset()
    info: dict = {}
    with torch.no_grad():
        for _ in range(ep_len):
            a, _ = model.predict(obs.detach().cpu().numpy(), deterministic=True)
            a_t = torch.from_numpy(np.clip(a, -1.0, 1.0).astype(np.float32)).to(ns.device)
            obs, _r, _d, info = env.step(a_t)
    return env.physical_emit(info["obs_before_reset"][:, -1]).cpu().numpy()


def _eval_one(algo: str, run_dir: Path, drift: float, ns: SimpleNamespace,
              n_rollouts: int) -> np.ndarray | None:
    """Eval one trained policy under a single drift level. Returns terminal
    emit (m^2) per rollout, or None if the policy artifact is missing."""
    if algo == "ppo":
        if getattr(ns, "ppo_eval_env", "gym") == "diff":
            return _eval_ppo_sameenv(run_dir, ns, n_rollouts, drift)
        # _eval_ppo_policy reads ns.distgen_drift_std for the eval env.
        ns.distgen_drift_std = float(drift)
        return _eval_ppo_policy(run_dir, ns, n_rollouts)

    policy_pt = _find_diffrl_policy(run_dir)
    if policy_pt is None:
        return None
    # Inherit action_scale / episode_length from the run's cfg.yaml, but
    # OVERRIDE the recorded distgen_drift_std (0.0) with the swept level.
    env_kwargs = dict(_load_env_kwargs(run_dir))
    env_kwargs["distgen_drift_std"] = float(drift)
    return _eval_diffrl_policy(policy_pt, ns, n_rollouts, env_kwargs)


def _plot_sweep(rows: list[dict], drift_levels: list[float],
                out_path: Path, env_note: str = "") -> None:
    import matplotlib.pyplot as plt

    algos = sorted({r["algo"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("tab10")

    for ai, algo in enumerate(algos):
        color = cmap(ai)
        means, stderrs, xs = [], [], []
        for d in drift_levels:
            medians = [r["terminal_emit_median_m2"] for r in rows
                       if r["algo"] == algo and r["drift"] == d
                       and not math.isnan(r["terminal_emit_median_m2"])]
            if not medians:
                continue
            xs.append(d)
            m = np.asarray(medians, dtype=float)
            means.append(float(m.mean()))
            # unbiased stderr across seeds; 0 when a single seed.
            stderrs.append(float(m.std(ddof=1) / math.sqrt(len(m)))
                           if len(m) >= 2 else 0.0)
            # faint per-seed markers
            ax.scatter([d] * len(m), m, color=color, alpha=0.35, s=18,
                       zorder=2)
        if not xs:
            continue
        xs_a = np.asarray(xs); mean_a = np.asarray(means)
        err_a = np.asarray(stderrs)
        ax.plot(xs_a, mean_a, "-o", color=color, label=algo, zorder=3)
        ax.fill_between(xs_a, mean_a - err_a, mean_a + err_a,
                        color=color, alpha=0.18, zorder=1)

    ax.set_yscale("log")
    ax.set_xlabel("eval distgen_drift_std (normalized, per step)")
    ax.set_ylabel("terminal norm_emit_4d (m²)")
    ax.set_title("Clean-trained policies evaluated under distgen jitter\n"
                 "(median terminal emittance; mean ± stderr over seeds, "
                 "dots = per-seed)"
                 + (f"\n{env_note}" if env_note else ""))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="algo (trained @ drift=0)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[drift-sweep] wrote {out_path}")


SUMMARY_FIELDS = [
    "algo", "seed", "drift", "n_rollouts",
    "terminal_emit_median_m2", "terminal_emit_p10_m2", "terminal_emit_p90_m2",
    "run_dir",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True,
                   help="EmittanceMLP checkpoint (the surrogate the policies "
                        "were trained against).")
    p.add_argument("--norm-json", required=True)
    p.add_argument("--runs-root", default="logs/compare_diff_hifi",
                   help="root holding <algo>/seed_<n>/ trained-policy dirs.")
    p.add_argument("--out-dir", default="logs/eval_drift_sweep")
    p.add_argument("--algos", default="ppo,shac,bptt")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--drift-levels", default="0.0,0.02,0.05,0.1",
                   help="comma-separated eval distgen_drift_std values. "
                        "Include 0.0 to reproduce the clean baseline.")
    p.add_argument("--eval-rollouts", type=int, default=256)
    p.add_argument("--ppo-eval-env", choices=("gym", "diff"), default="gym",
                   help="env PPO is evaluated on. 'gym' (default) = "
                        "PhotoinjectorEnv (own numpy RNG, per-rollout seed); "
                        "'diff' = the SAME batched DiffPhotoinjectorEnv "
                        "(seed=12345) used for SHAC/BPTT, so all algos share the "
                        "identical hidden-context realization (fair common-"
                        "random-numbers comparison; also batched, faster). "
                        "SHAC/BPTT unaffected. NOTE: 'diff' breaks the exact "
                        "drift=0 baseline reproduction (different RNG path).")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_root = Path(args.runs_root)
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    seeds = _parse_ints(args.seeds)
    drift_levels = _parse_floats(args.drift_levels)

    # Lightweight stand-in for compare_diff_algos' argparse Namespace; the
    # eval helpers only touch these four fields.
    ns = SimpleNamespace(ckpt=args.ckpt, norm_json=args.norm_json,
                         device=args.device, distgen_drift_std=0.0,
                         ppo_eval_env=args.ppo_eval_env)
    env_note = ("all algos on DiffPhotoinjectorEnv (shared seed=12345, "
                "common random numbers)" if args.ppo_eval_env == "diff"
                else "PPO on PhotoinjectorEnv (separate RNG); SHAC/BPTT on "
                     "DiffPhotoinjectorEnv")

    rows: list[dict] = []
    for algo in algos:
        for seed in seeds:
            run_dir = runs_root / algo / f"seed_{seed}"
            if not run_dir.exists():
                print(f"[drift-sweep] skip {algo} seed={seed}: "
                      f"{run_dir} not found")
                continue
            for drift in drift_levels:
                try:
                    term = _eval_one(algo, run_dir, drift, ns,
                                     args.eval_rollouts)
                except Exception as e:  # noqa: BLE001 - log and continue sweep
                    print(f"[drift-sweep] FAILED {algo} seed={seed} "
                          f"drift={drift}: {e}")
                    term = None
                if term is None:
                    rows.append({
                        "algo": algo, "seed": seed, "drift": drift,
                        "n_rollouts": 0,
                        "terminal_emit_median_m2": float("nan"),
                        "terminal_emit_p10_m2": float("nan"),
                        "terminal_emit_p90_m2": float("nan"),
                        "run_dir": str(run_dir),
                    })
                    continue
                med = float(np.median(term))
                rows.append({
                    "algo": algo, "seed": seed, "drift": drift,
                    "n_rollouts": int(term.size),
                    "terminal_emit_median_m2": med,
                    "terminal_emit_p10_m2": float(np.quantile(term, 0.1)),
                    "terminal_emit_p90_m2": float(np.quantile(term, 0.9)),
                    "run_dir": str(run_dir),
                })
                print(f"[drift-sweep] {algo} seed={seed} drift={drift:<5} "
                      f"median={med:.3e} m²")

    # --- summary.csv ---------------------------------------------------------
    out_csv = out_dir / "summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[drift-sweep] wrote {out_csv} ({len(rows)} rows)")

    # --- console pivot: mean of per-seed medians, per (algo, drift) ----------
    print("\n  median terminal emit (m²), mean over seeds:")
    header = "    algo   " + "".join(f"{d:>12}" for d in drift_levels)
    print(header)
    for algo in algos:
        cells = []
        for d in drift_levels:
            meds = [r["terminal_emit_median_m2"] for r in rows
                    if r["algo"] == algo and r["drift"] == d
                    and not math.isnan(r["terminal_emit_median_m2"])]
            cells.append(f"{np.mean(meds):>12.3e}" if meds else f"{'--':>12}")
        print(f"    {algo:<6} " + "".join(cells))

    # --- plot ----------------------------------------------------------------
    try:
        _plot_sweep(rows, drift_levels, out_dir / "drift_sweep.png",
                    env_note=env_note)
    except Exception as e:  # noqa: BLE001
        print(f"[drift-sweep] plot failed: {e}")

    # --- stats.json ----------------------------------------------------------
    stats = {
        "_config": {
            "mode": "eval_only_drift_sweep",
            "note": "policies trained at distgen_drift_std=0; only eval drifts",
            "runs_root": str(runs_root),
            "drift_levels": drift_levels,
            "algos": algos,
            "seeds": seeds,
            "eval_rollouts": int(args.eval_rollouts),
            "ppo_eval_env": args.ppo_eval_env,
            "ckpt": args.ckpt,
            "norm_json": args.norm_json,
        },
        "rows": rows,
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[drift-sweep] wrote {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
