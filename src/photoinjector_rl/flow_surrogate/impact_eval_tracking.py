"""
Evaluate a trained moving-target (aspect,tilt) controller on the REAL Impact-T
simulation (sim-to-sim transfer from the flow surrogate).

A moving-target rollout is inherently SEQUENTIAL: each step's knobs depend on the
policy reacting to the previous bunch, so this runs ~T+1 Impact-T jobs back-to-back
(~9 s each at lofi -> ~10 min for T=64). The closed loop mirrors `MovingShapeEnv`
exactly: obs = [knobs(5), s1_cur, s2_cur, s1*(t), s2*(t)] (9-D); at decision step t
the policy sees target traj[t] and acts to reach it; the achieved shape is read from
the real PR10241 bunch.

Reuses (imports only; edits nothing in emittance_target/, diffrl/, data/):
  - data.evaluate.custom_evaluate_impact_with_distgen (+ a custom merit returning coords)
  - emittance_target SETTING_KEYS/SETTING_BOUNDS + impact_env.DEFAULT_LHS_CONSTANTS
  - flow_surrogate.properties._s1/_s2/s_to_aspect_tilt (identical shape formulas)
  - flow_surrogate.eval_tracking._build_action_fn/_metrics, shape_targets.build_eval_trajectories

Usage:
  python -m photoinjector_rl.flow_surrogate.impact_eval_tracking \
    --flow-ckpt <ckpt> --norm-json processed/flow_surrogate_norm.json \
    --shac logs/move_shac/seed0 --which-traj tilt_rotation --fidelity lofi \
    --parity-check --out-dir figures
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import argparse
import json
import os
import shutil
import sysconfig
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from photoinjector_rl.emittance_target import SETTING_BOUNDS, SETTING_KEYS
from photoinjector_rl.emittance_target.impact_env import DEFAULT_LHS_CONSTANTS

from . import properties
from .eval_tracking import _build_action_fn, _metrics
from .shape_targets import build_eval_trajectories

COORD_KEYS = ("x", "y", "z", "px", "py", "pz")
MARKER = "PR10241"
IMPACT_CFG = "configs/impact/ImpactT_config.yaml"
DISTGEN_INPUT = "configs/impact/distgen_template.yaml"
_LO = np.array([SETTING_BOUNDS[k][0] for k in SETTING_KEYS], dtype=np.float64)
_HI = np.array([SETTING_BOUNDS[k][1] for k in SETTING_KEYS], dtype=np.float64)


def _ensure_impact_binary() -> None:
    """Export IMPACTT_BIN from the active conda env if unset (running
    `<env>/bin/python -m ...` does not put `<env>/bin` on PATH)."""
    if os.environ.get("IMPACTT_BIN") or shutil.which("ImpactTexe"):
        return
    cand = Path(sysconfig.get_path("scripts")) / "ImpactTexe"
    if cand.exists() and os.access(cand, os.X_OK):
        os.environ["IMPACTT_BIN"] = str(cand)


_ensure_impact_binary()


# ----- Impact-T run + shape extraction --------------------------------------

def _shape_merit(I):
    """custom merit: return the PR10241 bunch coords (n,6) + norm_emit_4d."""
    pg = I.particles[MARKER]
    if not hasattr(pg, "x"):                                   # dict-of-species form
        pg = pg.get("electron", next(iter(pg.values())))
    coords = np.stack([getattr(pg, k) for k in COORD_KEYS], axis=1).astype(np.float64)
    return {"coords": coords, "norm_emit_4d": float(pg.norm_emit_4d), "error": False}


def _run_impact(norm11: np.ndarray, constants: dict, workdir_root=None,
                archive_path=None):
    """Run Impact-T at the given 11-D normalized knob+distgen vector. Returns the
    PR10241 coords (n,6) or None on failure."""
    from photoinjector_rl.data.evaluate import custom_evaluate_impact_with_distgen

    phys = _LO + np.asarray(norm11, dtype=np.float64) * (_HI - _LO)
    settings = {k: float(v) for k, v in zip(SETTING_KEYS, phys)}
    settings.update(constants)
    try:
        with tempfile.TemporaryDirectory(dir=workdir_root) as wd:
            out = custom_evaluate_impact_with_distgen(
                settings=settings, distgen_input_file=DISTGEN_INPUT,
                impact_config=IMPACT_CFG, workdir=wd, archive_path=archive_path,
                merit_f=_shape_merit)
        c = out["coords"]
        if not np.isfinite(c).all() or len(c) < 10:
            return None
        return c
    except Exception:                                          # noqa: BLE001
        return None


def _coords_to_s1s2(coords: np.ndarray) -> tuple[float, float]:
    """(n,6) coords -> (s1, s2) via the exact training formulas (properties)."""
    t = torch.from_numpy(np.asarray(coords, dtype=np.float32))[None]   # (1,n,6)
    return float(properties._s1(t)[0]), float(properties._s2(t)[0])


def _load_constants(fidelity: str, constants_yaml: str | None) -> dict:
    if constants_yaml:
        import yaml
        cfg = yaml.safe_load(open(constants_yaml))
        # accept either a bare dict or an xopt-style {..: {constants: {...}}}
        for v in ([cfg] + list(cfg.values()) if isinstance(cfg, dict) else []):
            if isinstance(v, dict) and "constants" in v:
                return dict(v["constants"])
        return dict(cfg)
    if fidelity == "hifi":
        import yaml
        cfg = yaml.safe_load(open("configs/sweep/lhs_train_hifi.yaml"))
        for v in [cfg] + list(cfg.values()):
            if isinstance(v, dict) and "constants" in v:
                return dict(v["constants"])
    return dict(DEFAULT_LHS_CONSTANTS)


# ----- sequential closed-loop rollout ---------------------------------------

def rollout_impact_tracking(action_fn, traj: np.ndarray, T: int, *, start: float,
                            distgen: float, action_scale: float, constants: dict,
                            out_stem: Path, workdir_root=None) -> dict:
    """Roll the policy on real Impact-T against the fixed target trajectory `traj`
    (T,2). obs/indexing identical to MovingShapeEnv. Checkpoints every step."""
    dg = np.full(6, float(distgen), dtype=np.float64)
    knobs = np.full(5, float(start), dtype=np.float64)
    target = np.zeros((T, 2)); achieved = np.full((T, 2), np.nan)
    knob_hist = np.full((T, 5), np.nan); failures = 0

    def shape_at(kn):
        c = _run_impact(np.concatenate([kn, dg]), constants, workdir_root)
        if c is None:
            return None
        return _coords_to_s1s2(c)

    s = shape_at(knobs)                                        # reset bunch
    if s is None:
        raise RuntimeError("initial Impact-T run failed at start knobs")
    s1, s2 = s
    for t in range(T):
        obs = torch.tensor([[*knobs, s1, s2, traj[t, 0], traj[t, 1]]],
                           dtype=torch.float32)                # (1,9) — MovingShapeEnv layout
        a = action_fn(obs).detach().cpu().numpy()[0]           # (5,) in [-1,1]
        knobs = np.clip(knobs + a * action_scale, 0.0, 1.0)
        s = shape_at(knobs)
        if s is None:                                          # hold shape, count failure
            failures += 1
        else:
            s1, s2 = s
        target[t] = traj[t]; achieved[t] = (s1, s2); knob_hist[t] = knobs
        asp, tilt = properties.s_to_aspect_tilt(torch.tensor(s1), torch.tensor(s2))
        casp, ctilt = properties.s_to_aspect_tilt(torch.tensor(traj[t, 0]),
                                                  torch.tensor(traj[t, 1]))
        print(f"  step {t + 1:2d}/{T}: aspect={float(asp):.2f} tilt={float(tilt):+5.0f}°"
              f"  (cmd aspect={float(casp):.2f} tilt={float(ctilt):+5.0f}°)"
              f"{'  [FAIL]' if s is None else ''}", flush=True)
        np.savez(str(out_stem) + ".partial.npz", target=target, achieved=achieved,
                 knobs=knob_hist, failures=failures, step=t + 1)
    return {"target": target, "achieved": achieved, "knobs": knob_hist,
            "failures": int(failures)}


# ----- surrogate-vs-Impact parity pre-check ---------------------------------

def parity_check(flow_ckpt, norm_json, constants, n_points, distgen, n_particles,
                 out_png, seed=0):
    """At n_points random knob settings (fixed distgen), compare flow-predicted
    (s1,s2) to real Impact-T (s1,s2). Saves a scatter; returns per-axis MAE."""
    from .model import ConditionalAffineFlow
    flow = ConditionalAffineFlow.load_from_checkpoint(str(flow_ckpt), map_location="cpu").eval()
    rng = np.random.default_rng(seed)
    dg = np.full(6, float(distgen))
    f_s, i_s = [], []
    for j in range(n_points):
        kn = rng.uniform(0, 1, 5)
        x = torch.tensor(np.concatenate([kn, dg]), dtype=torch.float32)[None]
        with torch.no_grad():
            p = flow.sample_physical(x, int(n_particles))
        f_s.append((float(properties._s1(p)[0]), float(properties._s2(p)[0])))
        c = _run_impact(np.concatenate([kn, dg]), constants)
        i_s.append((np.nan, np.nan) if c is None else _coords_to_s1s2(c))
        print(f"  parity {j + 1}/{n_points}: flow s=({f_s[-1][0]:+.3f},{f_s[-1][1]:+.3f})"
              f"  impact s=({i_s[-1][0]:+.3f},{i_s[-1][1]:+.3f})", flush=True)
    f_s, i_s = np.array(f_s), np.array(i_s)
    m = np.isfinite(i_s).all(1)
    mae = np.abs(f_s[m] - i_s[m]).mean(0) if m.any() else np.array([np.nan, np.nan])
    _parity_plot(f_s[m], i_s[m], mae, Path(out_png))
    return {"s1_mae": float(mae[0]), "s2_mae": float(mae[1]), "n_ok": int(m.sum())}


def _parity_plot(flow_s, impact_s, mae, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for k, ax in enumerate(axes):
        ax.scatter(flow_s[:, k], impact_s[:, k], s=40)
        lim = 0.9
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        ax.set_xlabel(f"flow s{k + 1}"); ax.set_ylabel(f"Impact-T s{k + 1}")
        ax.set_title(f"s{k + 1}  (MAE={mae[k]:.3f})"); ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Flow surrogate vs real Impact-T shape parity")
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130); plt.close(fig)
    print(f"[impact_eval] wrote {out_png}")


# ----- surrogate prediction overlay -----------------------------------------

def _surrogate_track(action_fn, flow_ckpt, norm_json, traj, T, start, distgen):
    """Run the SAME policy on the flow surrogate (deterministic start) for overlay."""
    from .moving_shape_env import MovingShapeEnv
    env = MovingShapeEnv(num_envs=1, device="cpu", seed=0, episode_length=T + 4,
                         stochastic_init=False, no_grad=True, flow_ckpt=flow_ckpt,
                         norm_json=norm_json, n_particles=1500, fixed_target_traj=traj)
    obs = env.reset()
    ach = []
    with torch.no_grad():
        for _ in range(T):
            obs, _r, _d, _i = env.step(action_fn(obs))
            ach.append([float(env._s1_cur[0]), float(env._s2_cur[0])])
    return np.array(ach)


# ----- plotting -------------------------------------------------------------

def _track_plot(traj, impact_ach, surr_ach, out_png: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = np.arange(len(traj))
    casp, ctilt = properties.s_to_aspect_tilt(torch.as_tensor(traj[:, 0]),
                                              torch.as_tensor(traj[:, 1]))
    iasp, itilt = properties.s_to_aspect_tilt(torch.as_tensor(impact_ach[:, 0]),
                                              torch.as_tensor(impact_ach[:, 1]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for k, (ax, cmd, imp, lab) in enumerate([
            (axes[0], casp.numpy(), iasp.numpy(), "eigen aspect"),
            (axes[1], ctilt.numpy(), itilt.numpy(), "tilt (deg)")]):
        ax.plot(ts, cmd, "k--", lw=2, label="commanded")
        ax.plot(ts, imp, "-", color="tab:red", lw=2, label="Impact-T (real)")
        if surr_ach is not None:
            sasp, stilt = properties.s_to_aspect_tilt(torch.as_tensor(surr_ach[:, 0]),
                                                      torch.as_tensor(surr_ach[:, 1]))
            ax.plot(ts, (sasp if k == 0 else stilt).numpy(), ":", color="tab:blue",
                    lw=2, label="flow surrogate")
        ax.set_xlabel("step"); ax.set_ylabel(lab); ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle(title); fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"[impact_eval] wrote {out_png}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow-ckpt", required=True)
    p.add_argument("--norm-json", required=True)
    p.add_argument("--shac", default=None)
    p.add_argument("--bptt", default=None)
    p.add_argument("--ppo", default=None)
    p.add_argument("--diffrl-policy", default="best", choices=["best", "final"])
    p.add_argument("--traj-config", default="configs/diff_rl/moving_eval.yaml")
    p.add_argument("--which-traj", default="tilt_rotation",
                   help="held-out schedule name, or 'all'.")
    p.add_argument("--episode-length", type=int, default=64)
    p.add_argument("--fidelity", default="lofi", choices=["lofi", "hifi"])
    p.add_argument("--constants-yaml", default=None)
    p.add_argument("--start-knobs", type=float, default=0.5)
    p.add_argument("--distgen", type=float, default=0.5)
    p.add_argument("--action-scale", type=float, default=0.05)
    p.add_argument("--n-particles", type=int, default=1500, help="flow sampling (overlay/parity).")
    p.add_argument("--workdir-root", default=None)
    p.add_argument("--archive-path", default=None)
    p.add_argument("--parity-check", action="store_true")
    p.add_argument("--parity-points", type=int, default=6)
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--log-dir", default="logs/impact_eval")
    return p.parse_args()


def main() -> None:
    _ensure_impact_binary()
    args = _parse_args()
    algo, run_dir = next(((a, d) for a, d in (("shac", args.shac), ("bptt", args.bptt),
                                              ("ppo", args.ppo)) if d), (None, None))
    if algo is None:
        raise SystemExit("pass one of --shac/--bptt/--ppo")
    T = args.episode_length
    constants = _load_constants(args.fidelity, args.constants_yaml)
    print(f"[impact_eval] algo={algo} fidelity={args.fidelity} "
          f"n_particle={constants.get('distgen:n_particle')} mesh={constants.get('header:Nx')}")
    out_dir, log_dir = Path(args.out_dir), Path(args.log_dir)

    if args.parity_check:
        print("[impact_eval] parity pre-check (flow vs Impact-T)...")
        pj = parity_check(args.flow_ckpt, args.norm_json, constants, args.parity_points,
                          args.distgen, args.n_particles,
                          out_dir / f"impact_parity_{args.fidelity}.png")
        print(f"[impact_eval] parity MAE s1={pj['s1_mae']:.3f} s2={pj['s2_mae']:.3f} "
              f"(n_ok={pj['n_ok']})")

    action_fn = _build_action_fn(algo, Path(run_dir), "cpu", args.diffrl_policy)
    spec = None
    try:
        import yaml
        spec = yaml.safe_load(open(args.traj_config)).get("eval_trajectories")
    except Exception:                                          # noqa: BLE001
        pass
    all_traj = build_eval_trajectories(T, spec)
    names = list(all_traj) if args.which_traj == "all" else [args.which_traj]

    for name in names:
        traj = all_traj[name]
        stem = log_dir / f"{algo}_{name}_{args.fidelity}" / "rollout"
        stem.parent.mkdir(parents=True, exist_ok=True)
        print(f"[impact_eval] === {algo} / {name} on Impact-T ({T} steps, sequential) ===")
        t0 = time.time()
        r = rollout_impact_tracking(
            action_fn, traj, T, start=args.start_knobs, distgen=args.distgen,
            action_scale=args.action_scale, constants=constants, out_stem=stem,
            workdir_root=args.workdir_root)
        print(f"[impact_eval] rollout done in {(time.time() - t0) / 60:.1f} min, "
              f"{r['failures']} Impact failures")
        surr = _surrogate_track(action_fn, args.flow_ckpt, args.norm_json, traj, T,
                                args.start_knobs, args.distgen)
        np.savez(str(stem) + ".npz", target=r["target"], achieved=r["achieved"],
                 knobs=r["knobs"], surrogate=surr, failures=r["failures"])
        m_imp = _metrics(r["target"][:, None, :], r["achieved"][:, None, :], name)
        m_sur = _metrics(r["target"][:, None, :], surr[:, None, :], name)
        metrics = {"impact": m_imp, "surrogate": m_sur, "failures": r["failures"]}
        out_json = out_dir / f"impact_tracking_{algo}_{name}_{args.fidelity}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        json.dump(metrics, open(out_json, "w"), indent=2)
        _track_plot(traj, r["achieved"], surr,
                    out_dir / f"impact_tracking_{algo}_{name}_{args.fidelity}.png",
                    f"{algo.upper()} on Impact-T ({args.fidelity}) — {name}")
        print(f"[impact_eval] {name}: Impact tilt_mae={m_imp['tilt_mae_settled_deg']:.1f}° "
              f"s_rmse(settled)={m_imp['s_rmse_settled']:.3f} | "
              f"surrogate tilt_mae={m_sur['tilt_mae_settled_deg']:.1f}°")


if __name__ == "__main__":
    main()
