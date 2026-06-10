"""
Tracking-aware evaluation for the moving-target (aspect, tilt) controller.

Rolls trained policies on HELD-OUT, deterministic setpoint trajectories
(a (aspect,tilt) step staircase, a continuous tilt rotation, an aspect ramp),
records per-step achieved (aspect(t), tilt(t)) vs commanded, and reports tracking
metrics (s-space RMSE, aspect RMSE, circular tilt MAE; per-segment settling for
the staircase). Plots achieved vs commanded for each policy.

The eval env uses `MovingShapeEnv(fixed_target_traj=..., episode_length=T+slack)`
so the held-out schedule plays out with NO auto-reset inside the window. We read
the achieved shape from the env's per-step `_s1_cur/_s2_cur` (the response to the
setpoint the policy saw in the obs).

Usage:
    python -m photoinjector_rl.surrogates.flow.eval_tracking \
        --flow-ckpt <ckpt> --norm-json <norm.json> \
        --shac logs/move_shac/seed0 --bptt logs/move_bptt/seed0 --ppo logs/move_ppo \
        --out figures/tracking.png
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import properties
from .moving_shape_cli import load_moving_config
from .moving_shape_env import MovingShapeEnv
from .shape_targets import build_eval_trajectories


def _build_action_fn(algo: str, run_dir: Path, device: str, diffrl_policy: str = "best"):
    """Deterministic action fn fn(obs_torch)->action_torch for a saved policy."""
    run_dir = Path(run_dir)
    if algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(str(run_dir / "ppo_final.zip"), device=device)

        def fn(obs):
            act_np, _ = model.predict(obs.detach().cpu().numpy(), deterministic=True)
            return torch.from_numpy(np.asarray(act_np, dtype=np.float32)).to(device)
        return fn

    from .compare_algos import _load_diffrl_actor
    primary, fallback = (("best_policy.pt", "final_policy.pt") if diffrl_policy == "best"
                         else ("final_policy.pt", "best_policy.pt"))
    pt = run_dir / primary
    if not pt.exists():
        pt = run_dir / fallback
    actor, obs_rms = _load_diffrl_actor(pt, device)

    def fn(obs):
        o = obs_rms.normalize(obs) if obs_rms is not None else obs
        return torch.tanh(actor(o, deterministic=True))
    return fn


def _rollout_tracking(env: MovingShapeEnv, T: int, action_fn):
    """Return (target_s (T,B,2), achieved_s (T,B,2)) numpy arrays."""
    obs = env.reset()
    tgt, ach = [], []
    with torch.no_grad():
        for _ in range(T):
            tgt.append(obs[:, 7:9].detach().cpu().numpy().copy())     # setpoint policy sees
            obs, _r, _d, _info = env.step(action_fn(obs))
            ach.append(torch.stack([env._s1_cur, env._s2_cur], dim=-1)
                       .detach().cpu().numpy().copy())                # response
    return np.stack(tgt), np.stack(ach)                               # (T,B,2)


def _tilt_circular_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Tilt error with 180° period (ellipse orientation is mod 180°)."""
    return (a - b + 90.0) % 180.0 - 90.0


def _metrics(tgt: np.ndarray, ach: np.ndarray, traj_name: str,
             settle_skip: int = 8) -> dict:
    """Tracking metrics. The PRIMARY, robust metric is the (s1,s2)-space RMSE; the
    `_settled` variants exclude the first `settle_skip` steps (the rollout starts
    from RANDOM knobs, so early steps are settling-from-init, not tracking).
    Aspect error is reported in LOG space because aspect=√((1+r)/(1-r)) blows up as
    r→1, which makes a plain aspect RMSE heavy-tailed / non-robust."""
    s_err = np.linalg.norm(ach - tgt, axis=-1)                        # (T,B)
    asp_t, tilt_t = properties.s_to_aspect_tilt(torch.as_tensor(tgt[..., 0]),
                                                torch.as_tensor(tgt[..., 1]))
    asp_a, tilt_a = properties.s_to_aspect_tilt(torch.as_tensor(ach[..., 0]),
                                                torch.as_tensor(ach[..., 1]))
    asp_t, tilt_t = asp_t.numpy(), tilt_t.numpy()
    asp_a, tilt_a = asp_a.numpy(), tilt_a.numpy()
    tilt_d = _tilt_circular_diff_deg(tilt_a, tilt_t)
    sl = slice(settle_skip, None)                                     # settled window
    out = {
        "settle_skip": int(settle_skip),
        "s_rmse": float(np.sqrt((s_err ** 2).mean())),                 # all steps
        "s_rmse_settled": float(np.sqrt((s_err[sl] ** 2).mean())),     # primary, robust
        "tilt_mae_deg": float(np.abs(tilt_d).mean()),
        "tilt_mae_settled_deg": float(np.abs(tilt_d[sl]).mean()),
        "aspect_logmae_settled": float(np.median(np.abs(            # blow-up-safe
            np.log10(asp_a[sl]) - np.log10(asp_t[sl])))),
        "aspect_rmse_raw": float(np.sqrt(((asp_a - asp_t) ** 2).mean())),  # heavy-tailed
    }
    if traj_name == "staircase":
        # per-segment settling: steps after each setpoint change until median
        # s-error stays < tol for the rest of the segment.
        tol = 0.08
        med = np.median(s_err, axis=1)                                # (T,)
        changes = [0] + [t for t in range(1, len(tgt))
                         if np.linalg.norm(tgt[t, 0] - tgt[t - 1, 0]) > 1e-6]
        settle = []
        for k, c in enumerate(changes):
            end = changes[k + 1] if k + 1 < len(changes) else len(med)
            seg = med[c:end]
            within = np.where(seg < tol)[0]
            settle.append(int(within[0]) if within.size else -1)
        out["settle_steps_per_segment"] = settle
        out["steady_state_s_err_per_segment"] = [
            float(med[(changes[k + 1] if k + 1 < len(changes) else len(med)) - 1])
            for k in range(len(changes))]
    return out


def _plot(results: dict, trajectories: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(trajectories.keys())
    fig, axes = plt.subplots(2, len(names), figsize=(6 * len(names), 8), squeeze=False)
    for j, name in enumerate(names):
        tgt = trajectories[name]
        ts = np.arange(tgt.shape[0])
        asp_c, tilt_c = properties.s_to_aspect_tilt(
            torch.as_tensor(tgt[:, 0]), torch.as_tensor(tgt[:, 1]))
        axes[0][j].plot(ts, asp_c.numpy(), "k--", lw=2, label="commanded")
        axes[1][j].plot(ts, tilt_c.numpy(), "k--", lw=2, label="commanded")
        for algo, per_traj in results.items():
            ach = per_traj[name]["achieved"]                          # (T,B,2)
            asp_a, tilt_a = properties.s_to_aspect_tilt(
                torch.as_tensor(ach[..., 0]), torch.as_tensor(ach[..., 1]))
            asp_a, tilt_a = asp_a.numpy(), tilt_a.numpy()
            am, alo, ahi = (np.median(asp_a, 1), np.quantile(asp_a, .25, 1),
                            np.quantile(asp_a, .75, 1))
            tm = np.median(tilt_a, 1)
            axes[0][j].plot(ts, am, label=algo)
            axes[0][j].fill_between(ts, alo, ahi, alpha=0.2)
            axes[1][j].plot(ts, tm, label=algo)
        axes[0][j].set_title(f"{name}: aspect"); axes[0][j].set_ylabel("eigen aspect")
        axes[1][j].set_title(f"{name}: tilt"); axes[1][j].set_ylabel("tilt (deg)")
        for ax in (axes[0][j], axes[1][j]):
            ax.set_xlabel("step"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[eval_tracking] wrote {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow-ckpt", required=True)
    p.add_argument("--norm-json", required=True)
    p.add_argument("--processed", default=None)
    p.add_argument("--shac", default=None, help="run dir with best_policy.pt")
    p.add_argument("--bptt", default=None, help="run dir with best_policy.pt")
    p.add_argument("--ppo", default=None, help="run dir with ppo_final.zip")
    p.add_argument("--diffrl-policy", default="best", choices=["best", "final"])
    p.add_argument("--traj-config", default=None,
                   help="YAML/JSON with an `eval_trajectories` block (held-out "
                        "schedules); defaults to the built-in staircase/rotation/ramp.")
    p.add_argument("--episode-length", type=int, default=64)
    p.add_argument("--n-rollouts", type=int, default=64)
    p.add_argument("--n-particles", type=int, default=512)
    p.add_argument("--action-scale", type=float, default=0.05)
    p.add_argument("--scale", type=float, default=0.3, help="reward scale (env build).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="figures/tracking.png")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    T = args.episode_length
    spec = load_moving_config(args.traj_config).get("eval_trajectories") or None
    trajectories = build_eval_trajectories(T, spec)
    algos = {k: v for k, v in (("shac", args.shac), ("bptt", args.bptt),
                               ("ppo", args.ppo)) if v}
    if not algos:
        raise SystemExit("pass at least one of --shac/--bptt/--ppo")

    results: dict = {a: {} for a in algos}
    summary: dict = {a: {} for a in algos}
    for algo, run_dir in algos.items():
        action_fn = _build_action_fn(algo, Path(run_dir), args.device, args.diffrl_policy)
        for name, traj in trajectories.items():
            env = MovingShapeEnv(
                num_envs=args.n_rollouts, device=args.device, seed=12345,
                episode_length=T + 4, no_grad=True, flow_ckpt=args.flow_ckpt,
                norm_json=args.norm_json, processed_h5=args.processed,
                n_particles=args.n_particles, action_scale=args.action_scale,
                scale=args.scale, fixed_target_traj=traj)
            tgt, ach = _rollout_tracking(env, T, action_fn)
            results[algo][name] = {"achieved": ach}
            summary[algo][name] = _metrics(tgt, ach, name)
            m = summary[algo][name]
            print(f"[{algo:4s}/{name:13s}] s_rmse(settled)={m['s_rmse_settled']:.3f} "
                  f"(all={m['s_rmse']:.3f}) tilt_mae(settled)={m['tilt_mae_settled_deg']:.1f}deg "
                  f"aspect_logMAE={m['aspect_logmae_settled']:.3f}"
                  + (f" settle={m.get('settle_steps_per_segment')}"
                     if name == "staircase" else ""))

    _plot(results, trajectories, Path(args.out))
    out_json = Path(args.out).with_suffix(".json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval_tracking] wrote {out_json}")


if __name__ == "__main__":
    main()
