"""
Head-to-head comparison of PPO (SB3), SHAC, and BPTT on the conditional-FLOW
surrogate, optimizing a computed bunch property (default norm_emit_4d). The flow
analog of `emittance_target.compare_diff_algos`: same budget-matched protocol,
same metrics, but every algo trains against the flow (FlowBunchEnv / its batched
SB3 wrapper) and every policy is evaluated on a fresh FlowBunchEnv.

Usage:
    python -m photoinjector_rl.flow_surrogate.compare_algos \
        --flow-ckpt trained/flow_surrogate/checkpoints/best-epoch=493-val_loss=-0.9555.ckpt \
        --norm-json processed/flow_surrogate_norm.json \
        --out-dir logs/compare_flow \
        --seeds 0,1,2 --budget 500000 --algos ppo,shac,bptt

For each (algo, seed): budget-matched training subprocess (SHAC/BPTT epochs =
ceil(budget/(num_actors*steps_num)); PPO total-timesteps = budget), then
`--eval-rollouts` deterministic rollouts on a fresh FlowBunchEnv for terminal
property, plus learning-curve ingest. Outputs compare.png / summary.csv /
stats.json in --out-dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs" / "diff_rl"
DEFAULT_SHAC_CFG = CONFIG_DIR / "shac_flow.yaml"
DEFAULT_BPTT_CFG = CONFIG_DIR / "bptt_flow.yaml"


# --- subprocess launchers ---------------------------------------------------

def _max_epochs_for_budget(cfg_path: Path, budget: int) -> int:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    num_actors = int(cfg["params"]["config"]["num_actors"])
    steps_num = int(cfg["params"]["config"]["steps_num"])
    return max(1, math.ceil(budget / (num_actors * steps_num)))


def _common_flow_args(args) -> list[str]:
    out = ["--flow-ckpt", args.flow_ckpt, "--norm-json", args.norm_json,
           "--property", args.property, "--n-particles", str(args.n_particles),
           "--reward-mode", args.reward_mode, "--device", args.device]
    if args.processed:
        out += ["--processed", args.processed]
    if args.target is not None:
        out += ["--target", str(args.target)]
    return out


def _run_diffrl(module: str, cfg: Path, args, seed: int, run_dir: Path) -> None:
    max_epochs = _max_epochs_for_budget(cfg, args.budget)
    cmd = [sys.executable, "-m", f"photoinjector_rl.flow_surrogate.{module}",
           "--cfg", str(cfg), "--logdir", str(run_dir), "--seed", str(seed),
           "--max-epochs", str(max_epochs),
           "--distgen-drift-std", str(args.distgen_drift_std)] + _common_flow_args(args)
    print(f"[compare] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _run_ppo(args, seed: int, run_dir: Path) -> None:
    cmd = [sys.executable, "-m", "photoinjector_rl.flow_surrogate.train_ppo",
           "--out-dir", str(run_dir), "--seed", str(seed),
           "--total-timesteps", str(args.budget),
           "--distgen-drift-std", str(args.distgen_drift_std)] + _common_flow_args(args)
    print(f"[compare] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


# --- learning-curve ingest --------------------------------------------------

def _read_diffrl_csv(path: Path):
    steps, losses, walls = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            losses.append(float(row["mean_episode_loss"]))
            walls.append(float(row["wall_time"]))
    return np.array(steps), np.array(losses), np.array(walls)


def _read_ppo_progress(run_dir: Path):
    tb_dir = run_dir / "tb"
    matches = sorted(tb_dir.glob("PPO_*/progress.csv")) if tb_dir.exists() else []
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
    return (np.array(steps), np.array(rewards), np.array(walls)) if steps else None


# --- deterministic eval on a fresh FlowBunchEnv -----------------------------

def _make_eval_env(args, env_kwargs: dict, num_rollouts: int, seed: int):
    from .diff_env import FlowBunchEnv
    ep_len = int(env_kwargs.get("episode_length", args.episode_length))
    return FlowBunchEnv(
        num_envs=num_rollouts, device=args.device, seed=seed,
        episode_length=ep_len, stochastic_init=True, no_grad=True,
        flow_ckpt=args.flow_ckpt, norm_json=args.norm_json,
        processed_h5=args.processed, property=args.property,
        reward_mode=args.reward_mode, target=args.target,
        n_particles=args.n_particles,
        action_scale=float(env_kwargs.get("action_scale", 0.05)),
        distgen_drift_std=float(env_kwargs.get("distgen_drift_std",
                                               args.distgen_drift_std)),
    ), ep_len


def _rollout_terminal(env, ep_len: int, action_fn) -> np.ndarray:
    """Roll out `action_fn` deterministically; return terminal property (B,).

    `action_fn(obs_torch) -> action_torch` ([-1,1]^5 on the env device). The
    terminal value is read from info["obs_before_reset"] (the env auto-resets on
    the final step), inverted to physical units via env.physical_emit.
    """
    obs = env.reset()
    info: dict = {}
    with torch.no_grad():
        for _ in range(ep_len):
            obs, _r, _d, info = env.step(action_fn(obs))
        terminal_y = info["obs_before_reset"][:, -1].detach()
        return env.physical_emit(terminal_y).cpu().numpy()


def _load_diffrl_actor(policy_pt: Path, device: str):
    from photoinjector_rl.emittance_target.diffrl.utils import RunningMeanStd
    ckpt = torch.load(str(policy_pt), weights_only=False, map_location=device)
    actor = ckpt[0].to(device).eval()
    obs_rms = next((x for x in ckpt if isinstance(x, RunningMeanStd)), None)
    if obs_rms is not None:
        obs_rms = obs_rms.to(device)
    return actor, obs_rms


def _eval_diffrl(policy_pt: Path, args, num_rollouts: int,
                 env_kwargs: dict) -> np.ndarray:
    actor, obs_rms = _load_diffrl_actor(policy_pt, args.device)
    env, ep_len = _make_eval_env(args, env_kwargs, num_rollouts, seed=12345)

    def action_fn(obs):
        o = obs_rms.normalize(obs) if obs_rms is not None else obs
        return torch.tanh(actor(o, deterministic=True))

    return _rollout_terminal(env, ep_len, action_fn)


def _eval_ppo(run_dir: Path, args, num_rollouts: int,
              env_kwargs: dict) -> np.ndarray:
    from stable_baselines3 import PPO
    zip_path = run_dir / "ppo_final.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"No PPO zip in {run_dir}")
    model = PPO.load(str(zip_path), device=args.device)
    env, ep_len = _make_eval_env(args, env_kwargs, num_rollouts, seed=12345)

    def action_fn(obs):
        obs_np = obs.detach().cpu().numpy()
        act_np, _ = model.predict(obs_np, deterministic=True)
        return torch.from_numpy(np.asarray(act_np, dtype=np.float32)).to(args.device)

    return _rollout_terminal(env, ep_len, action_fn)


def _load_env_kwargs(run_dir: Path) -> dict:
    cfg_path = run_dir / "cfg.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("params", {}).get("diff_env", {})


# --- plotting / summary -----------------------------------------------------

def _aggregate(curves, grid):
    rows = []
    for xs, ys in curves:
        order = np.argsort(xs)
        rows.append(np.interp(grid, xs[order], ys[order]))
    mat = np.stack(rows)
    n = len(rows)
    stderr = mat.std(axis=0, ddof=1) / math.sqrt(n) if n >= 2 else np.zeros_like(mat[0])
    return mat.mean(axis=0), stderr


def _plot_results(per_algo: dict, out_dir: Path, prop: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    all_xs = [r["learning_curve"][0] for runs in per_algo.values() for r in runs
              if r.get("learning_curve") is not None and r["learning_curve"][0].size]
    x_max = float(max(a.max() for a in all_xs)) if all_xs else 5e5
    grid = np.linspace(1, x_max, 200)
    for algo, runs in per_algo.items():
        curves = [r["learning_curve"] for r in runs if r.get("learning_curve") is not None]
        if not curves:
            continue
        mean, stderr = _aggregate(curves, grid)
        axes[0].plot(grid, mean, label=algo)
        axes[0].fill_between(grid, mean - stderr, mean + stderr, alpha=0.2)
    axes[0].set_xlabel("env steps")
    axes[0].set_ylabel("mean episode return (sum of -y_norm)")
    axes[0].set_title("Learning curves vs env-steps")
    axes[0].axhline(0.0, color="grey", lw=0.5, ls="--")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    all_walls = [r["learning_curve_wall"][0] for runs in per_algo.values() for r in runs
                 if r.get("learning_curve_wall") is not None and r["learning_curve_wall"][0].size]
    w_max = float(max(a.max() for a in all_walls)) if all_walls else 1.0
    wgrid = np.linspace(0.1, w_max, 200)
    for algo, runs in per_algo.items():
        curves = [r["learning_curve_wall"] for r in runs if r.get("learning_curve_wall") is not None]
        if not curves:
            continue
        mean, stderr = _aggregate(curves, wgrid)
        axes[1].plot(wgrid, mean, label=algo)
        axes[1].fill_between(wgrid, mean - stderr, mean + stderr, alpha=0.2)
    axes[1].set_xlabel("training wall-clock (s)")
    axes[1].set_ylabel("mean episode return")
    axes[1].set_title("Learning curves vs wall-clock")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # terminal-property histogram (log x, range adapted to the data)
    terms_all = np.concatenate(
        [r["terminal_emit"] for runs in per_algo.values() for r in runs
         if r.get("terminal_emit") is not None]
        or [np.array([1.0])])
    lo, hi = float(terms_all.min()), float(terms_all.max())
    lo = max(lo, 1e-30)
    bins = np.logspace(np.log10(lo) - 0.05, np.log10(hi) + 0.05, 40)
    n_per_seed = 0
    for algo, runs in per_algo.items():
        terms = [r["terminal_emit"] for r in runs if r.get("terminal_emit") is not None]
        if not terms:
            continue
        n_per_seed = max(n_per_seed, max(t.size for t in terms))
        axes[2].hist(np.concatenate(terms), bins=bins, alpha=0.5, label=algo)
    axes[2].set_xscale("log")
    axes[2].set_xlabel(f"terminal {prop}")
    axes[2].set_ylabel("rollouts")
    axes[2].set_title(f"Deterministic-eval terminal {prop} ({n_per_seed} rollouts/seed)")
    if axes[2].get_legend_handles_labels()[0]:
        axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "compare.png", dpi=140)
    plt.close(fig)
    print(f"[compare] wrote {out_dir / 'compare.png'}")


SUMMARY_FIELDS = ["algo", "seed", "n_rollouts", "terminal_median",
                  "terminal_p10", "terminal_p90", "run_dir"]


def _write_summary(per_algo: dict, out_dir: Path) -> None:
    rows = []
    for algo, runs in per_algo.items():
        for r in runs:
            t = r.get("terminal_emit")
            rows.append({
                "algo": algo, "seed": r["seed"],
                "n_rollouts": int(t.size) if t is not None else 0,
                "terminal_median": float(np.median(t)) if t is not None else float("nan"),
                "terminal_p10": float(np.quantile(t, 0.1)) if t is not None else float("nan"),
                "terminal_p90": float(np.quantile(t, 0.9)) if t is not None else float("nan"),
                "run_dir": r["run_dir"],
            })
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader(); w.writerows(rows)
    print(f"[compare] wrote {out_dir / 'summary.csv'} ({len(rows)} rows)")


# --- entry point ------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--flow-ckpt", required=True)
    p.add_argument("--norm-json", required=True)
    p.add_argument("--processed", default=None,
                   help="processed .h5 for non-emittance property z-score "
                        "(defaults to <norm-json with _norm.json->.h5>).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--budget", type=int, default=500_000)
    p.add_argument("--algos", default="ppo,shac,bptt")
    p.add_argument("--property", default="norm_emit_4d")
    p.add_argument("--reward-mode", default="minimize",
                   choices=["minimize", "maximize", "target"])
    p.add_argument("--target", type=float, default=None)
    p.add_argument("--n-particles", type=int, default=512)
    p.add_argument("--episode-length", type=int, default=64)
    p.add_argument("--distgen-drift-std", type=float, default=0.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval-rollouts", type=int, default=256)
    p.add_argument("--diffrl-policy", choices=["best", "final"], default="best",
                   help="which saved SHAC/BPTT checkpoint to evaluate for the "
                        "terminal-property comparison. 'best' (default) uses "
                        "best_policy.pt (saved at the lowest training loss) so a "
                        "late-training collapse does not penalize the reported "
                        "metric; 'final' uses final_policy.pt (end-of-training). "
                        "PPO always uses ppo_final.zip (stable; no separate best "
                        "checkpoint is saved).")
    p.add_argument("--skip-train", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.processed is None:
        args.processed = args.norm_json.replace("_norm.json", ".h5")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    algos = [a.strip() for a in args.algos.split(",")]
    per_algo: dict[str, list[dict]] = {a: [] for a in algos}
    failures: list[dict] = []

    for algo in algos:
        for seed in seeds:
            run_dir = out_dir / algo / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            done_marker = run_dir / ".done"
            if not args.skip_train and not done_marker.exists():
                try:
                    if algo == "ppo":
                        _run_ppo(args, seed, run_dir)
                    elif algo == "shac":
                        _run_diffrl("train_shac", DEFAULT_SHAC_CFG, args, seed, run_dir)
                    elif algo == "bptt":
                        _run_diffrl("train_bptt", DEFAULT_BPTT_CFG, args, seed, run_dir)
                    else:
                        raise ValueError(f"unknown algo {algo}")
                    done_marker.touch()
                except subprocess.CalledProcessError as e:
                    print(f"[compare] FAILED {algo} seed={seed}: rc={e.returncode}",
                          file=sys.stderr, flush=True)
                    (run_dir / ".failed").write_text(str(e))
                    failures.append({"algo": algo, "seed": seed, "rc": e.returncode})
                    per_algo[algo].append({"seed": seed, "run_dir": str(run_dir),
                                           "terminal_emit": None})
                    continue

            entry: dict = {"seed": seed, "run_dir": str(run_dir)}
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

            try:
                if algo == "ppo":
                    term = _eval_ppo(run_dir, args, args.eval_rollouts,
                                     _load_env_kwargs(run_dir))
                else:
                    primary, fallback = (("best_policy.pt", "final_policy.pt")
                                         if args.diffrl_policy == "best"
                                         else ("final_policy.pt", "best_policy.pt"))
                    policy_pt = run_dir / primary
                    if not policy_pt.exists():
                        policy_pt = run_dir / fallback
                    term = _eval_diffrl(policy_pt, args, args.eval_rollouts,
                                        _load_env_kwargs(run_dir))
                entry["terminal_emit"] = term
                print(f"[compare] {algo} seed={seed}: terminal {args.property} "
                      f"median={np.median(term):.3e}")
            except Exception as e:
                print(f"[compare] eval failed for {algo} seed={seed}: {e}")
                entry["terminal_emit"] = None
            per_algo[algo].append(entry)

    _plot_results(per_algo, out_dir, args.property)
    _write_summary(per_algo, out_dir)

    stats: dict = {"_config": {"property": args.property,
                               "reward_mode": args.reward_mode,
                               "n_particles": args.n_particles,
                               "distgen_drift_std": float(args.distgen_drift_std),
                               "budget": int(args.budget), "seeds": seeds,
                               "eval_rollouts": int(args.eval_rollouts),
                               "diffrl_policy": args.diffrl_policy,
                               "flow_ckpt": args.flow_ckpt}}
    for algo, runs in per_algo.items():
        stats[algo] = [{"seed": r["seed"], "run_dir": r["run_dir"],
                        "terminal_median": (float(np.median(r["terminal_emit"]))
                                            if r.get("terminal_emit") is not None else None)}
                       for r in runs]
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[compare] wrote {out_dir / 'stats.json'}")
    if failures:
        with open(out_dir / "failures.json", "w") as f:
            json.dump(failures, f, indent=2)
        print(f"[compare] {len(failures)} run(s) failed; see failures.json",
              file=sys.stderr)


if __name__ == "__main__":
    main()
