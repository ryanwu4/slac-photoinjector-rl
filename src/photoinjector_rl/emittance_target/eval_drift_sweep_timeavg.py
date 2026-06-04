"""
Eval-only distgen-drift sweep, scored by TIME-AVERAGED episode emittance.

Same setup as `eval_drift_sweep.py` (load the clean-trained PPO/SHAC/BPTT
policies from logs/compare_diff_hifi/, re-eval under a sweep of jitter levels,
no retraining), but the per-rollout metric is the mean of the physical
norm_emit_4d over ALL episode steps rather than just the terminal step.

Why time-average: the per-step reward is -y_norm, so an episode's return is
exactly the (negated) sum of per-step emittance -- the time-average is its
physical analogue. Unlike the terminal value (a single snapshot at the, under
drift, randomized final step), it folds in the transit (how fast the policy
drives random init -> optimum) AND the settled behaviour, so it separates algos
that reach the same endpoint at different speeds / steady-state tightness.

Collecting the trajectory also gives, for free, the mean emit(t) curve, plotted
at a few representative drift levels so the time behaviour is visible.

The eval envs are built identically to compare_diff_algos' helpers (same fixed
seeds: diffrl 12345 batched, PPO 12345+i per rollout) so terminal_mean here
cross-checks the terminal-emit sweep.

Usage:
    python -m photoinjector_rl.emittance_target.eval_drift_sweep_timeavg \
        --ckpt trained/emittance_target_hifi/checkpoints/best-epoch=191-val_loss=0.0060.ckpt \
        --norm-json processed/emittance_target_hifi_norm.json \
        --runs-root logs/compare_diff_hifi --out-dir logs/eval_drift_sweep_timeavg \
        --drift-levels 0.0,0.1,0.2,0.4,0.7,1.0 \
        --algos ppo,shac,bptt --seeds 0,1,2 --eval-rollouts 256 --device cuda:0

Outputs (in --out-dir; source run root is NOT modified):
    summary.csv          one row per (algo, seed, drift): time-avg emit stats (m^2)
    timeavg_vs_drift.png median time-avg emit vs drift, mean +- stderr / seeds
    emit_trajectories.png mean emit(t) per algo at representative drift levels
    stats.json           full structured dump (+ mean emit(t) curves) + _config
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .compare_diff_algos import _load_env_kwargs


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _find_diffrl_policy(run_dir: Path) -> Path | None:
    for name in ("final_policy.pt", "best_policy.pt"):
        p = run_dir / name
        if p.exists():
            return p
    return None


def _traj_diffrl(policy_pt: Path, ns: SimpleNamespace, n_rollouts: int,
                 env_kwargs: dict) -> np.ndarray:
    """Per-step physical emit (m^2) for a SHAC/BPTT policy under drift.

    Returns array (ep_len, n_rollouts). Mirrors compare_diff_algos
    `_eval_diffrl_policy` env construction (seed=12345, batched); the post-step
    emit for EVERY env (before any auto-reset) is read from
    info["obs_before_reset"][:, -1], so the last column equals that helper's
    terminal value.
    """
    from .diff_env import DiffPhotoinjectorEnv
    from .diffrl.utils import RunningMeanStd

    ckpt = torch.load(str(policy_pt), weights_only=False,
                      map_location=ns.device)
    actor = ckpt[0].to(ns.device).eval()
    obs_rms = next((x for x in ckpt if isinstance(x, RunningMeanStd)), None)
    if obs_rms is not None:
        obs_rms = obs_rms.to(ns.device)

    ep_len = int(env_kwargs.get("episode_length", 64))
    env = DiffPhotoinjectorEnv(
        num_envs=n_rollouts, device=ns.device, seed=12345,
        episode_length=ep_len, stochastic_init=True, no_grad=True,
        surrogate_ckpt=ns.ckpt, norm_json=ns.norm_json,
        action_scale=float(env_kwargs.get("action_scale", 0.05)),
        distgen_drift_std=float(env_kwargs.get("distgen_drift_std", 0.0)),
    )
    obs = env.reset()
    emits = np.empty((ep_len, n_rollouts), dtype=np.float64)
    with torch.no_grad():
        for t in range(ep_len):
            o = obs_rms.normalize(obs) if obs_rms is not None else obs
            a = actor(o, deterministic=True)
            obs, _r, _d, info = env.step(torch.tanh(a))
            y = info["obs_before_reset"][:, -1]
            emits[t] = env.physical_emit(y).cpu().numpy()
    return emits


def _traj_ppo(run_dir: Path, ns: SimpleNamespace, n_rollouts: int) -> np.ndarray:
    """Per-step physical emit (m^2) for a PPO policy under drift.

    Returns array (max_steps, n_rollouts). Mirrors compare_diff_algos
    `_eval_ppo_policy` (PhotoinjectorEnv, seed=12345+i per rollout, max_steps=64,
    terminated never -> every episode is exactly max_steps long); info["emit_m2"]
    is the post-step emit, so the last row equals that helper's terminal value.
    """
    from stable_baselines3 import PPO

    from .env import PhotoinjectorEnv

    candidates = [run_dir / "ppo_final.zip",
                  run_dir / "eval" / "best_model.zip"]
    policy_zip = next((c for c in candidates if c.exists()), None)
    if policy_zip is None:
        raise FileNotFoundError(f"No PPO zip in {run_dir}")

    model = PPO.load(str(policy_zip), device=ns.device)
    max_steps = 64
    env = PhotoinjectorEnv.from_checkpoint(
        ckpt_path=ns.ckpt, norm_json=ns.norm_json, device=ns.device,
        max_steps=max_steps, distgen_drift_std=float(ns.distgen_drift_std),
    )
    emits = np.empty((max_steps, n_rollouts), dtype=np.float64)
    for i in range(n_rollouts):
        obs, _info = env.reset(seed=12345 + i)
        done = False
        t = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            emits[t, i] = float(info["emit_m2"])
            t += 1
            done = bool(term or trunc)
    return emits


def _ppo_actfn(run_dir: Path, ns: SimpleNamespace):
    """Load a PPO zip and return obs(torch)->action(torch) for the SAME-ENV path.

    PPO is a Gaussian policy; deterministic=True gives the mean action, which
    the env clips to [-1, 1] (matching how SurrogateVecEnv fed it in training,
    `np.clip(actions, -1, 1)` — NOT a tanh squash like the diff-RL actors).
    """
    from stable_baselines3 import PPO

    candidates = [run_dir / "ppo_final.zip", run_dir / "eval" / "best_model.zip"]
    policy_zip = next((c for c in candidates if c.exists()), None)
    if policy_zip is None:
        raise FileNotFoundError(f"No PPO zip in {run_dir}")
    model = PPO.load(str(policy_zip), device=ns.device)

    def act_fn(obs: torch.Tensor) -> torch.Tensor:
        a, _ = model.predict(obs.detach().cpu().numpy(), deterministic=True)
        return torch.from_numpy(np.clip(a, -1.0, 1.0).astype(np.float32)).to(ns.device)
    return act_fn


def _traj_ppo_sameenv(run_dir: Path, ns: SimpleNamespace, n_rollouts: int,
                      drift: float) -> np.ndarray:
    """Per-step emit (m^2) for a PPO policy on the SAME batched
    DiffPhotoinjectorEnv (seed=12345) used for SHAC/BPTT, so all algos share the
    identical hidden-context realization. PPO was trained on this env (via
    SurrogateVecEnv) with action_scale=0.05, episode_length=64.
    """
    from .diff_env import DiffPhotoinjectorEnv

    act_fn = _ppo_actfn(run_dir, ns)
    ep_len = 64
    env = DiffPhotoinjectorEnv(
        num_envs=n_rollouts, device=ns.device, seed=12345,
        episode_length=ep_len, stochastic_init=True, no_grad=True,
        surrogate_ckpt=ns.ckpt, norm_json=ns.norm_json,
        action_scale=0.05, distgen_drift_std=float(drift),
    )
    obs = env.reset()
    emits = np.empty((ep_len, n_rollouts), dtype=np.float64)
    with torch.no_grad():
        for t in range(ep_len):
            obs, _r, _d, info = env.step(act_fn(obs))
            emits[t] = env.physical_emit(info["obs_before_reset"][:, -1]).cpu().numpy()
    return emits


def _eval_traj(algo: str, run_dir: Path, drift: float, ns: SimpleNamespace,
               n_rollouts: int) -> np.ndarray | None:
    if algo == "ppo":
        if getattr(ns, "ppo_eval_env", "gym") == "diff":
            return _traj_ppo_sameenv(run_dir, ns, n_rollouts, drift)
        ns.distgen_drift_std = float(drift)
        return _traj_ppo(run_dir, ns, n_rollouts)
    policy_pt = _find_diffrl_policy(run_dir)
    if policy_pt is None:
        return None
    env_kwargs = dict(_load_env_kwargs(run_dir))
    env_kwargs["distgen_drift_std"] = float(drift)
    return _traj_diffrl(policy_pt, ns, n_rollouts, env_kwargs)


# --- plotting ---------------------------------------------------------------


def _plot_timeavg(rows: list[dict], drift_levels: list[float],
                  out_path: Path, env_note: str = "") -> None:
    import matplotlib.pyplot as plt

    algos = sorted({r["algo"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("tab10")
    for ai, algo in enumerate(algos):
        color = cmap(ai)
        xs, means, errs = [], [], []
        for d in drift_levels:
            meds = [r["timeavg_emit_median_m2"] for r in rows
                    if r["algo"] == algo and r["drift"] == d
                    and not math.isnan(r["timeavg_emit_median_m2"])]
            if not meds:
                continue
            xs.append(d)
            m = np.asarray(meds)
            means.append(float(m.mean()))
            errs.append(float(m.std(ddof=1) / math.sqrt(len(m)))
                        if len(m) >= 2 else 0.0)
            ax.scatter([d] * len(m), m, color=color, alpha=0.35, s=18, zorder=2)
        if not xs:
            continue
        xs_a, m_a, e_a = np.asarray(xs), np.asarray(means), np.asarray(errs)
        ax.plot(xs_a, m_a, "-o", color=color, label=algo, zorder=3)
        ax.fill_between(xs_a, m_a - e_a, m_a + e_a, color=color, alpha=0.18,
                        zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("eval distgen_drift_std (normalized, per step)")
    ax.set_ylabel("time-averaged norm_emit_4d over episode (m²)")
    ax.set_title("Clean-trained policies under distgen jitter — TIME-AVERAGED\n"
                 "(per-rollout episode mean; median over rollouts, "
                 "mean ± stderr over seeds, dots = per-seed)"
                 + (f"\n{env_note}" if env_note else ""))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="algo (trained @ drift=0)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[timeavg-sweep] wrote {out_path}")


def _plot_trajectories(traj_curves: dict, drift_levels: list[float],
                       out_path: Path, env_note: str = "") -> None:
    """mean emit(t) per algo, one panel per representative drift level.

    traj_curves[(algo, drift)] = (ep_len,) mean-over-rollouts-and-seeds curve.
    """
    import matplotlib.pyplot as plt

    # Up to 3 representative levels: first, middle, last.
    if len(drift_levels) <= 3:
        sel = list(drift_levels)
    else:
        sel = [drift_levels[0],
               drift_levels[len(drift_levels) // 2],
               drift_levels[-1]]
    algos = sorted({a for (a, _d) in traj_curves})
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, len(sel), figsize=(6 * len(sel), 5.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for j, d in enumerate(sel):
        ax = axes[j]
        for ai, algo in enumerate(algos):
            curve = traj_curves.get((algo, d))
            if curve is None:
                continue
            t = np.arange(curve.shape[0])
            ax.plot(t, curve, color=cmap(ai), lw=1.4, label=algo)
        ax.set_yscale("log")
        ax.set_xlabel("env step")
        ax.set_title(f"drift_std = {d}")
        ax.grid(True, which="both", alpha=0.3)
        if j == 0:
            ax.set_ylabel("mean norm_emit_4d (m²)\n(over rollouts & seeds)")
            ax.legend(title="algo")
    fig.suptitle("Emittance trajectory over the episode "
                 "(transit → settle), by eval jitter level"
                 + (f"\n{env_note}" if env_note else ""))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[timeavg-sweep] wrote {out_path}")


SUMMARY_FIELDS = [
    "algo", "seed", "drift", "n_rollouts",
    "timeavg_emit_median_m2", "timeavg_emit_p10_m2", "timeavg_emit_p90_m2",
    "terminal_emit_median_m2", "run_dir",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--norm-json", required=True)
    p.add_argument("--runs-root", default="logs/compare_diff_hifi")
    p.add_argument("--out-dir", default="logs/eval_drift_sweep_timeavg")
    p.add_argument("--algos", default="ppo,shac,bptt")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--drift-levels", default="0.0,0.1,0.2,0.4,0.7,1.0")
    p.add_argument("--eval-rollouts", type=int, default=256)
    p.add_argument("--ppo-eval-env", choices=("gym", "diff"), default="gym",
                   help="env PPO is evaluated on. 'gym' (default) = "
                        "PhotoinjectorEnv (its own numpy RNG, per-rollout seed "
                        "12345+i); 'diff' = the SAME batched DiffPhotoinjectorEnv "
                        "(seed=12345) used for SHAC/BPTT, so all algos see the "
                        "identical hidden-context realization — a fair, "
                        "common-random-numbers comparison (also batched, so far "
                        "faster). SHAC/BPTT are unaffected by this flag.")
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

    ns = SimpleNamespace(ckpt=args.ckpt, norm_json=args.norm_json,
                         device=args.device, distgen_drift_std=0.0,
                         ppo_eval_env=args.ppo_eval_env)
    env_note = ("all algos on DiffPhotoinjectorEnv (shared seed=12345, "
                "common random numbers)" if args.ppo_eval_env == "diff"
                else "PPO on PhotoinjectorEnv (separate RNG); SHAC/BPTT on "
                     "DiffPhotoinjectorEnv")

    rows: list[dict] = []
    # accumulate mean-emit(t) per (algo, drift) summed over seeds for plotting
    traj_sum: dict[tuple, np.ndarray] = {}
    traj_cnt: dict[tuple, int] = {}

    for algo in algos:
        for seed in seeds:
            run_dir = runs_root / algo / f"seed_{seed}"
            if not run_dir.exists():
                print(f"[timeavg-sweep] skip {algo} seed={seed}: not found")
                continue
            for drift in drift_levels:
                try:
                    traj = _eval_traj(algo, run_dir, drift, ns,
                                      args.eval_rollouts)  # (T, n_rollouts)
                except Exception as e:  # noqa: BLE001
                    print(f"[timeavg-sweep] FAILED {algo} seed={seed} "
                          f"drift={drift}: {e}")
                    traj = None
                if traj is None:
                    rows.append({
                        "algo": algo, "seed": seed, "drift": drift,
                        "n_rollouts": 0,
                        "timeavg_emit_median_m2": float("nan"),
                        "timeavg_emit_p10_m2": float("nan"),
                        "timeavg_emit_p90_m2": float("nan"),
                        "terminal_emit_median_m2": float("nan"),
                        "run_dir": str(run_dir),
                    })
                    continue
                # per-rollout time-average over the episode
                ta = traj.mean(axis=0)              # (n_rollouts,)
                term = traj[-1]                     # (n_rollouts,)
                rows.append({
                    "algo": algo, "seed": seed, "drift": drift,
                    "n_rollouts": int(ta.size),
                    "timeavg_emit_median_m2": float(np.median(ta)),
                    "timeavg_emit_p10_m2": float(np.quantile(ta, 0.1)),
                    "timeavg_emit_p90_m2": float(np.quantile(ta, 0.9)),
                    "terminal_emit_median_m2": float(np.median(term)),
                    "run_dir": str(run_dir),
                })
                key = (algo, drift)
                mean_traj = traj.mean(axis=1)       # (T,)
                if key not in traj_sum:
                    traj_sum[key] = np.zeros_like(mean_traj)
                    traj_cnt[key] = 0
                traj_sum[key] += mean_traj
                traj_cnt[key] += 1
                print(f"[timeavg-sweep] {algo} seed={seed} drift={drift:<5} "
                      f"time-avg median={np.median(ta):.3e}  "
                      f"terminal median={np.median(term):.3e} m²")

    # --- summary.csv ---------------------------------------------------------
    out_csv = out_dir / "summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[timeavg-sweep] wrote {out_csv} ({len(rows)} rows)")

    # --- console pivot -------------------------------------------------------
    print("\n  TIME-AVERAGED emit (m²), median over rollouts, mean over seeds:")
    print("    algo   " + "".join(f"{('d='+str(d)):>12}" for d in drift_levels))
    for algo in algos:
        cells = []
        for d in drift_levels:
            v = [r["timeavg_emit_median_m2"] for r in rows
                 if r["algo"] == algo and r["drift"] == d
                 and not math.isnan(r["timeavg_emit_median_m2"])]
            cells.append(f"{np.mean(v):>12.3e}" if v else f"{'--':>12}")
        print(f"    {algo:<6} " + "".join(cells))

    # --- plots ---------------------------------------------------------------
    traj_curves = {k: traj_sum[k] / traj_cnt[k] for k in traj_sum}
    try:
        _plot_timeavg(rows, drift_levels, out_dir / "timeavg_vs_drift.png",
                      env_note=env_note)
    except Exception as e:  # noqa: BLE001
        print(f"[timeavg-sweep] timeavg plot failed: {e}")
    try:
        _plot_trajectories(traj_curves, drift_levels,
                           out_dir / "emit_trajectories.png", env_note=env_note)
    except Exception as e:  # noqa: BLE001
        print(f"[timeavg-sweep] trajectory plot failed: {e}")

    # --- stats.json ----------------------------------------------------------
    stats = {
        "_config": {
            "mode": "eval_only_drift_sweep_timeavg",
            "metric": "mean of physical norm_emit_4d over all episode steps",
            "runs_root": str(runs_root),
            "drift_levels": drift_levels,
            "algos": algos, "seeds": seeds,
            "eval_rollouts": int(args.eval_rollouts),
            "ppo_eval_env": args.ppo_eval_env,
            "ckpt": args.ckpt, "norm_json": args.norm_json,
        },
        "rows": rows,
        "mean_emit_trajectory": {
            f"{a}|{d}": traj_curves[(a, d)].tolist() for (a, d) in traj_curves
        },
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[timeavg-sweep] wrote {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
