"""
Head-to-head comparison of PPO (SB3), SHAC, and BPTT on the v1 photoinjector
surrogate env.

Usage:
    python -m photoinjector_rl.emittance_target.compare_diff_algos \
        --ckpt trained/emittance_target/checkpoints/best-epoch=191-val_loss=0.0060.ckpt \
        --norm-json processed/emittance_target_norm.json \
        --out-dir logs/compare_diff \
        --seeds 0,1,2 --budget 500000 --algos ppo,shac,bptt

For each (algo, seed) combination this script:
  1. Launches the algorithm's training script as a subprocess at a budget
     matched in env-steps. For SHAC/BPTT this means setting --max-epochs
     to ceil(budget / (num_actors * steps_num)).
  2. After training, loads the final policy and runs `--eval-rollouts`
     deterministic 64-step rollouts on a fresh DiffPhotoinjectorEnv to
     collect terminal emittance (in m²).
  3. Ingests each algo's learning-curve CSV (SHAC/BPTT) or SB3 progress.csv
     (PPO) and aligns them on the env-step axis.

Outputs in --out-dir:
    learning_curves.png       Episode return vs env-steps, mean ± stderr per algo.
    terminal_emit_hist.png    Distribution of terminal emit (m²) per algo.
    summary.csv               One row per (algo, seed): final return + terminal stats.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs" / "diff_rl"
DEFAULT_SHAC_CFG = CONFIG_DIR / "shac_photoinjector.yaml"
DEFAULT_BPTT_CFG = CONFIG_DIR / "bptt_photoinjector.yaml"


# --- subprocess launchers ---------------------------------------------------


def _max_epochs_for_budget(cfg_path: Path, budget: int) -> int:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    num_actors = int(cfg["params"]["config"]["num_actors"])
    steps_num = int(cfg["params"]["config"]["steps_num"])
    return max(1, math.ceil(budget / (num_actors * steps_num)))


def _run_shac(args, seed: int, run_dir: Path) -> None:
    max_epochs = _max_epochs_for_budget(DEFAULT_SHAC_CFG, args.budget)
    cmd = [
        sys.executable, "-m", "photoinjector_rl.emittance_target.train_shac",
        "--cfg", str(DEFAULT_SHAC_CFG),
        "--ckpt", args.ckpt, "--norm-json", args.norm_json,
        "--logdir", str(run_dir), "--seed", str(seed),
        "--max-epochs", str(max_epochs),
        "--device", args.device,
    ]
    print(f"[compare] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _run_bptt(args, seed: int, run_dir: Path) -> None:
    max_epochs = _max_epochs_for_budget(DEFAULT_BPTT_CFG, args.budget)
    cmd = [
        sys.executable, "-m", "photoinjector_rl.emittance_target.train_bptt",
        "--cfg", str(DEFAULT_BPTT_CFG),
        "--ckpt", args.ckpt, "--norm-json", args.norm_json,
        "--logdir", str(run_dir), "--seed", str(seed),
        "--max-epochs", str(max_epochs),
        "--device", args.device,
    ]
    print(f"[compare] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _run_ppo(args, seed: int, run_dir: Path) -> None:
    cmd = [
        sys.executable, "-m", "photoinjector_rl.emittance_target.train_ppo",
        "--ckpt", args.ckpt, "--norm-json", args.norm_json,
        "--out-dir", str(run_dir), "--seed", str(seed),
        "--total-timesteps", str(args.budget),
    ]
    print(f"[compare] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


# --- learning-curve ingest --------------------------------------------------


def _read_diffrl_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps, losses, walls = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["mean_episode_loss"]))
            walls.append(float(row["wall_time"]))
    return np.array(steps), np.array(losses), np.array(walls)


def _read_ppo_progress(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """SB3 progress.csv → (steps, ep_rew_mean, wall_seconds).

    `rollout/ep_rew_mean` is the mean per-episode sum of rewards — same metric
    as SHAC/BPTT's `-mean_episode_loss`. `time/time_elapsed` is cumulative
    training wall-clock since `model.learn()` was called.
    """
    tb_dir = run_dir / "tb"
    if not tb_dir.exists():
        return None
    matches = sorted(tb_dir.glob("PPO_*/progress.csv"))
    if not matches:
        return None
    csv_path = matches[-1]
    steps, rewards, walls = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "time/total_timesteps" not in row:
                continue
            rew = row.get("rollout/ep_rew_mean", "")
            if not rew:
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


# --- post-training deterministic eval --------------------------------------


def _eval_diffrl_policy(policy_pt: Path, args, num_rollouts: int,
                        env_kwargs: dict) -> np.ndarray:
    """Load SHAC/BPTT saved policy and roll out on DiffPhotoinjectorEnv.

    SHAC saves `[actor, critic, target_critic, obs_rms, ret_rms]`; BPTT saves
    `[actor, obs_rms]`. We pick out obs_rms by type rather than position so
    both formats work without a branch.

    `env_kwargs` carries action_scale / distgen_drift_std / episode_length so
    that the eval env matches what the policy was trained on.

    Returns terminal emit (m²) per rollout, shape (num_rollouts,).
    """
    from .diff_env import DiffPhotoinjectorEnv
    from .diffrl.utils import RunningMeanStd

    ckpt = torch.load(str(policy_pt), weights_only=False, map_location=args.device)
    actor = ckpt[0].to(args.device).eval()
    obs_rms = next((x for x in ckpt if isinstance(x, RunningMeanStd)), None)
    if obs_rms is not None:
        obs_rms = obs_rms.to(args.device)

    ep_len = int(env_kwargs.get("episode_length", 64))
    env = DiffPhotoinjectorEnv(
        num_envs=num_rollouts, device=args.device, seed=12345,
        episode_length=ep_len, stochastic_init=True, no_grad=True,
        surrogate_ckpt=args.ckpt, norm_json=args.norm_json,
        action_scale=float(env_kwargs.get("action_scale", 0.05)),
        distgen_drift_std=float(env_kwargs.get("distgen_drift_std", 0.0)),
    )
    obs = env.reset()
    with torch.no_grad():
        info: dict = {}
        for _ in range(ep_len):
            o = obs_rms.normalize(obs) if obs_rms is not None else obs
            a = actor(o, deterministic=True)
            obs, _r, _d, info = env.step(torch.tanh(a))
        # NOTE: after the final step the env auto-resets (step_count hit
        # episode_length), so env._last_y_norm is the fresh random init's
        # y_norm, NOT the policy-driven terminal. Pull the pre-reset value
        # from info["obs_before_reset"][:, -1] instead.
        terminal_y = info["obs_before_reset"][:, -1].detach()
        terminal_m2 = env.physical_emit(terminal_y).cpu().numpy()
    return terminal_m2


def _collect_rollouts(policy_pt: Path, args, env_kwargs: dict,
                      n_rollouts: int = 8) -> dict:
    """Run `n_rollouts` deterministic rollouts and return per-step traces.

    Returns:
      knobs:  (T+1, n_rollouts, 5) knob trajectory in [0,1].
      emit:   (T+1, n_rollouts)    physical norm_emit_4d in m².
    """
    from .diff_env import DiffPhotoinjectorEnv
    from .diffrl.utils import RunningMeanStd

    ckpt = torch.load(str(policy_pt), weights_only=False,
                      map_location=args.device)
    actor = ckpt[0].to(args.device).eval()
    obs_rms = next((x for x in ckpt if isinstance(x, RunningMeanStd)), None)
    if obs_rms is not None:
        obs_rms = obs_rms.to(args.device)

    ep_len = int(env_kwargs.get("episode_length", 64))
    env = DiffPhotoinjectorEnv(
        num_envs=n_rollouts, device=args.device, seed=2024,
        episode_length=ep_len, stochastic_init=True, no_grad=True,
        surrogate_ckpt=args.ckpt, norm_json=args.norm_json,
        action_scale=float(env_kwargs.get("action_scale", 0.05)),
        distgen_drift_std=float(env_kwargs.get("distgen_drift_std", 0.0)),
    )
    obs = env.reset()
    knobs = [env._knobs.detach().cpu().numpy().copy()]
    emit = [env.physical_emit(env._last_y_norm).detach().cpu().numpy().copy()]
    with torch.no_grad():
        for t in range(ep_len):
            o = obs_rms.normalize(obs) if obs_rms is not None else obs
            a = actor(o, deterministic=True)
            obs, _r, _d, info = env.step(torch.tanh(a))
            # Read pre-reset state from info; on the final step env resets
            # and env._knobs is the fresh-init state.
            if t < ep_len - 1:
                knobs.append(env._knobs.detach().cpu().numpy().copy())
                emit.append(env.physical_emit(env._last_y_norm)
                            .detach().cpu().numpy().copy())
            else:
                pre = info["obs_before_reset"]
                knobs.append(pre[:, :5].cpu().numpy().copy())
                emit.append(env.physical_emit(pre[:, -1]).cpu().numpy().copy())
    return {"knobs": np.stack(knobs), "emit": np.stack(emit)}


def _plot_sample_rollouts(algo: str, rollouts: dict, out_path: Path) -> None:
    """6-row plot: emit (log-y) on top + 5 knob trajectories below."""
    import matplotlib.pyplot as plt

    knobs = rollouts["knobs"]      # (T+1, N, 5)
    emit = rollouts["emit"]        # (T+1, N)
    t = np.arange(emit.shape[0])
    n = emit.shape[1]
    fig, axes = plt.subplots(6, 1, figsize=(8, 11), sharex=True)
    cmap = plt.get_cmap("viridis", n)

    for i in range(n):
        axes[0].plot(t, emit[:, i], color=cmap(i), lw=1.2, alpha=0.9,
                     label=f"rollout {i}")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("norm_emit_4d\n(m²)")
    axes[0].set_title(f"{algo.upper()} — {n} deterministic rollouts "
                      f"(stochastic distgen init)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")

    knob_names = ["SOL10111", "CQ10121", "SQ10122", "GUNF scale", "GUNF phase"]
    for k in range(5):
        ax = axes[k + 1]
        for i in range(n):
            ax.plot(t, knobs[:, i, k], color=cmap(i), lw=1.0, alpha=0.85)
        ax.set_ylabel(knob_names[k] + "\n(normalized)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("env step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def _load_env_kwargs(run_dir: Path) -> dict:
    """Pull diff_env kwargs from the saved cfg.yaml so eval matches training."""
    cfg_path = run_dir / "cfg.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("params", {}).get("diff_env", {})


def _eval_ppo_policy(run_dir: Path, args, num_rollouts: int = 256) -> np.ndarray:
    """Load SB3 PPO zip and roll out on the existing PhotoinjectorEnv."""
    from stable_baselines3 import PPO

    from .env import PhotoinjectorEnv

    candidates = [run_dir / "ppo_final.zip",
                  run_dir / "eval" / "best_model.zip"]
    policy_zip = next((c for c in candidates if c.exists()), None)
    if policy_zip is None:
        raise FileNotFoundError(f"No PPO zip in {run_dir}")

    model = PPO.load(str(policy_zip), device=args.device)
    env = PhotoinjectorEnv.from_checkpoint(
        ckpt_path=args.ckpt, norm_json=args.norm_json,
        device=args.device, max_steps=64,
    )
    emits = np.empty(num_rollouts, dtype=np.float32)
    for i in range(num_rollouts):
        obs, _info = env.reset(seed=12345 + i)
        done = False
        info = _info
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            done = bool(term or trunc)
        emits[i] = info["emit_m2"]
    return emits


# --- plotting ---------------------------------------------------------------


def _aggregate(curves: list[tuple[np.ndarray, np.ndarray]],
               grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interp each (xs, ys) onto `grid`; return mean, stderr across seeds.

    Uses unbiased std (ddof=1) for stderr; falls back to ddof=0 when n=1.
    """
    rows = []
    for xs, ys in curves:
        order = np.argsort(xs)
        rows.append(np.interp(grid, xs[order], ys[order]))
    mat = np.stack(rows)
    n = len(rows)
    if n >= 2:
        stderr = mat.std(axis=0, ddof=1) / math.sqrt(n)
    else:
        stderr = np.zeros_like(mat[0])
    return mat.mean(axis=0), stderr


def _plot_results(per_algo: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    # Panel A: learning curves vs env-steps.
    # Panel B: learning curves vs wall-clock (training time, seconds).
    # Panel C: post-training terminal-emit histogram (m², log x).
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- Panel A: env-steps -------------------------------------------------
    all_xs: list[np.ndarray] = []
    for runs in per_algo.values():
        for r in runs:
            lc = r.get("learning_curve")
            if lc is not None and lc[0].size > 0:
                all_xs.append(lc[0])
    x_max = (float(max(arr.max() for arr in all_xs)) if all_xs else 500_000.0)
    grid = np.linspace(1, x_max, 200)
    for algo, runs in per_algo.items():
        curves = [r["learning_curve"] for r in runs
                  if r.get("learning_curve") is not None]
        if not curves:
            continue
        mean, stderr = _aggregate(curves, grid)
        axes[0].plot(grid, mean, label=algo)
        axes[0].fill_between(grid, mean - stderr, mean + stderr, alpha=0.2)
    axes[0].set_xlabel("env steps")
    axes[0].set_ylabel("mean episode return\n(sum of -y_norm over episode)")
    axes[0].set_title("Learning curves vs env-steps")
    axes[0].axhline(0.0, color="grey", lw=0.5, ls="--",
                    label="random-policy expectation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ---- Panel B: wall-clock ------------------------------------------------
    all_walls: list[np.ndarray] = []
    for runs in per_algo.values():
        for r in runs:
            lc = r.get("learning_curve_wall")
            if lc is not None and lc[0].size > 0:
                all_walls.append(lc[0])
    w_max = float(max(arr.max() for arr in all_walls)) if all_walls else 1.0
    wgrid = np.linspace(0.1, w_max, 200)
    final_walls: dict[str, list[float]] = {}
    for algo, runs in per_algo.items():
        curves = [r["learning_curve_wall"] for r in runs
                  if r.get("learning_curve_wall") is not None]
        if not curves:
            continue
        mean, stderr = _aggregate(curves, wgrid)
        axes[1].plot(wgrid, mean, label=algo)
        axes[1].fill_between(wgrid, mean - stderr, mean + stderr, alpha=0.2)
        final_walls[algo] = [float(c[0][-1]) for c in curves]
    axes[1].set_xlabel("training wall-clock (s)")
    axes[1].set_ylabel("mean episode return")
    axes[1].set_title("Learning curves vs wall-clock")
    axes[1].axhline(0.0, color="grey", lw=0.5, ls="--")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    # Annotate per-algo total training time in the panel corner.
    if final_walls:
        lines = [f"{algo}: {np.mean(walls):.0f}s (mean of {len(walls)} seeds)"
                 for algo, walls in final_walls.items()]
        axes[1].text(0.02, 0.02, "\n".join(lines),
                     transform=axes[1].transAxes, fontsize=9,
                     verticalalignment="bottom",
                     bbox=dict(facecolor="white", alpha=0.75,
                               edgecolor="grey"))

    bins = np.logspace(-11, -10, 40)
    n_per_seed = 0
    for algo, runs in per_algo.items():
        terms = [r["terminal_emit"] for r in runs
                 if r.get("terminal_emit") is not None]
        if not terms:
            continue
        all_term = np.concatenate(terms)
        if all_term.size == 0:
            continue
        n_per_seed = max(n_per_seed, max(t.size for t in terms))
        axes[2].hist(all_term, bins=bins, alpha=0.5, label=algo)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("terminal norm_emit_4d (m²)")
    axes[2].set_ylabel("rollouts")
    axes[2].set_title(
        f"Deterministic-eval terminal emittance ({n_per_seed} rollouts/seed)"
    )
    if axes[2].get_legend_handles_labels()[0]:
        axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "compare.png", dpi=140)
    print(f"[compare] wrote {out_dir / 'compare.png'}")


SUMMARY_FIELDS = [
    "algo", "seed", "n_rollouts",
    "terminal_emit_median_m2", "terminal_emit_p10_m2", "terminal_emit_p90_m2",
    "run_dir",
]


def _write_summary(per_algo: dict, out_dir: Path) -> None:
    rows = []
    for algo, runs in per_algo.items():
        for r in runs:
            t = r.get("terminal_emit")
            rows.append({
                "algo": algo,
                "seed": r["seed"],
                "n_rollouts": int(t.size) if t is not None else 0,
                "terminal_emit_median_m2": float(np.median(t)) if t is not None else float("nan"),
                "terminal_emit_p10_m2": float(np.quantile(t, 0.1)) if t is not None else float("nan"),
                "terminal_emit_p90_m2": float(np.quantile(t, 0.9)) if t is not None else float("nan"),
                "run_dir": r["run_dir"],
            })
    out_csv = out_dir / "summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[compare] wrote {out_csv} ({len(rows)} rows)")


# --- entry point ------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--norm-json", required=True)
    p.add_argument("--out-dir", required=True, type=str)
    p.add_argument("--seeds", default="0,1,2",
                   help="comma-separated list")
    p.add_argument("--budget", type=int, default=50_000)
    p.add_argument("--algos", default="ppo,shac,bptt",
                   help="comma-separated; subset of ppo,shac,bptt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval-rollouts", type=int, default=256)
    p.add_argument("--skip-train", action="store_true",
                   help="If runs already exist, skip subprocess training "
                        "and just aggregate.")
    return p.parse_args()


def _parse_seeds(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seeds)
    algos = [a.strip() for a in args.algos.split(",")]

    per_algo: dict[str, list[dict]] = {a: [] for a in algos}

    failures: list[dict] = []
    for algo in algos:
        for seed in seeds:
            run_dir = out_dir / algo / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            done_marker = run_dir / ".done"
            failed_marker = run_dir / ".failed"
            if not args.skip_train and not done_marker.exists():
                try:
                    if algo == "ppo":
                        _run_ppo(args, seed, run_dir)
                    elif algo == "shac":
                        _run_shac(args, seed, run_dir)
                    elif algo == "bptt":
                        _run_bptt(args, seed, run_dir)
                    else:
                        raise ValueError(f"unknown algo {algo}")
                    done_marker.touch()
                    if failed_marker.exists():
                        failed_marker.unlink()
                except subprocess.CalledProcessError as e:
                    print(f"[compare] FAILED {algo} seed={seed}: rc={e.returncode}",
                          file=sys.stderr, flush=True)
                    failed_marker.write_text(str(e))
                    failures.append({"algo": algo, "seed": seed,
                                     "returncode": e.returncode})
                    per_algo[algo].append({"seed": seed,
                                           "run_dir": str(run_dir),
                                           "terminal_emit": None})
                    continue

            entry: dict = {"seed": seed, "run_dir": str(run_dir)}

            # learning curve — common metric is mean episode return (sum of
            # rewards = -sum(y_norm) over an episode). PPO reports it natively
            # as rollout/ep_rew_mean; SHAC/BPTT report -ep_rew (loss), so we
            # negate. Capture both env-steps and wall-clock x-axes.
            if algo == "ppo":
                pr = _read_ppo_progress(run_dir)
                if pr is not None:
                    steps, returns, walls = pr
                    entry["learning_curve"] = (steps, returns)
                    entry["learning_curve_wall"] = (walls, returns)
            else:
                csv_path = run_dir / "learning_curve.csv"
                if csv_path.exists():
                    steps, losses, walls = _read_diffrl_csv(csv_path)
                    entry["learning_curve"] = (steps, -losses)
                    entry["learning_curve_wall"] = (walls, -losses)

            # final policy eval
            try:
                if algo == "ppo":
                    term = _eval_ppo_policy(run_dir, args, args.eval_rollouts)
                else:
                    policy_pt = run_dir / "final_policy.pt"
                    if not policy_pt.exists():
                        # SHAC saves "best_policy.pt" when best_loss improves
                        policy_pt = run_dir / "best_policy.pt"
                    env_kwargs = _load_env_kwargs(run_dir)
                    term = _eval_diffrl_policy(policy_pt, args,
                                               args.eval_rollouts,
                                               env_kwargs)
                entry["terminal_emit"] = term
                print(f"[compare] {algo} seed={seed}: terminal emit "
                      f"median={np.median(term):.3e} m²")
            except Exception as e:
                print(f"[compare] eval failed for {algo} seed={seed}: {e}")
                entry["terminal_emit"] = None

            per_algo[algo].append(entry)

    _plot_results(per_algo, out_dir)
    _write_summary(per_algo, out_dir)

    # Per-algo sample-rollout plots for the diffrl algorithms. Pick the
    # seed with the best median terminal emit so the plot reflects the
    # algo's best policy, not a noisy one.
    for algo in ("shac", "bptt"):
        if algo not in per_algo:
            continue
        runs = [r for r in per_algo[algo]
                if r.get("terminal_emit") is not None]
        if not runs:
            continue
        best = min(runs, key=lambda r: float(np.median(r["terminal_emit"])))
        run_dir = Path(best["run_dir"])
        policy_pt = run_dir / "final_policy.pt"
        if not policy_pt.exists():
            policy_pt = run_dir / "best_policy.pt"
        if not policy_pt.exists():
            print(f"[compare] no policy.pt under {run_dir}; skipping rollouts")
            continue
        try:
            rollouts = _collect_rollouts(policy_pt, args,
                                         _load_env_kwargs(run_dir),
                                         n_rollouts=8)
            _plot_sample_rollouts(algo, rollouts,
                                  out_dir / f"rollouts_{algo}.png")
        except Exception as e:
            print(f"[compare] rollout plot failed for {algo}: {e}",
                  file=sys.stderr)

    # JSON dump of all aggregated stats for downstream analysis.
    stats: dict = {}
    for algo, runs in per_algo.items():
        stats[algo] = []
        for r in runs:
            t = r.get("terminal_emit")
            stats[algo].append({
                "seed": r["seed"],
                "run_dir": r["run_dir"],
                "terminal_emit_median_m2": float(np.median(t)) if t is not None else None,
            })
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[compare] wrote {out_dir / 'stats.json'}")

    if failures:
        with open(out_dir / "failures.json", "w") as f:
            json.dump(failures, f, indent=2)
        print(f"[compare] {len(failures)} training run(s) failed; "
              f"see {out_dir / 'failures.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
