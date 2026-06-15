"""
Animated GIFs of the moving-target controller: for each held-out test case
(staircase / tilt_rotation / aspect_ramp), roll out SHAC/BPTT/PPO from an
IDENTICAL deterministic start and a SHARED latent z (so the three clouds differ
only by what each policy does to the knobs), sample the output bunch each step,
and animate:
  - a large x–y panel (the transverse phase space the controller is shaping), and
  - the full 6D corner plot (x,y,z,px,py,pz),
with the three controllers' particles overlaid and color-coded, plus the
commanded (aspect,tilt) shown as a dashed reference ellipse. One GIF per case.

Usage:
    python -m photoinjector_rl.surrogates.flow.animate_tracking \
        --flow-ckpt <ckpt> --norm-json <norm.json> \
        --shac logs/move_shac/seed0 --bptt logs/move_bptt/seed0 --ppo logs/move_ppo \
        --traj-config configs/diff_rl/moving_eval.yaml --out-dir figures/move_anim
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from . import properties
from .eval_tracking import _build_action_fn
from .moving_shape_cli import load_moving_config
from .moving_shape_env import MovingShapeEnv
from .shape_targets import build_eval_trajectories

COORD_LABELS = ["x [mm]", "y [mm]", "z [mm]", "px [keV/c]", "py [keV/c]", "Δpz [keV/c]"]
COORD_SCALE = np.array([1e3, 1e3, 1e3, 1e-3, 1e-3, 1e-3])   # m->mm, eV->keV
COLORS = {"shac": "tab:blue", "bptt": "tab:orange", "ppo": "tab:green"}
# brighter, higher-contrast variants for the black-background slide version
COLORS_DARK = {"shac": "#4ea1ff", "bptt": "#ffa64d", "ppo": "#5fd75f"}
# the 5 controllable knobs (SETTING_KEYS[:5]) — short labels for the history panels
KNOB_LABELS = ["SOL10111", "CQ10121", "SQ10122", "GUNF amp", "GUNF phase"]
Z_SEED = 12321                                               # shared latent across frames+algos


def _rollout_particles(flow_ckpt, norm_json, processed, traj, T, n_vis,
                       action_fn, device, action_scale):
    """Roll one controller from a deterministic start; per step sample the bunch
    at the post-action knobs (shared latent z). Returns (parts (T,n_vis,6),
    target_s (T,2), achieved_s (T,2), knobs (T,5) normalized in [0,1])."""
    env = MovingShapeEnv(
        num_envs=1, device=device, seed=0, episode_length=T + 4,
        stochastic_init=False, no_grad=True, flow_ckpt=flow_ckpt,
        norm_json=norm_json, processed_h5=processed, n_particles=64,
        action_scale=action_scale, fixed_target_traj=traj)
    obs = env.reset()
    parts, tgt, ach, knobs = [], [], [], []
    with torch.no_grad():
        for _ in range(T):
            tgt.append(obs[:, 7:9].cpu().numpy()[0].copy())
            obs, _r, _d, _i = env.step(action_fn(obs))
            ach.append(np.array([float(env._s1_cur[0]), float(env._s2_cur[0])]))
            knobs.append(env._knobs[0].cpu().numpy().copy())   # post-action knobs [0,1]^5
            torch.manual_seed(Z_SEED)                       # identical latent every frame/algo
            x = torch.cat([env._knobs, env._distgen], dim=-1)
            p = env._flow.sample_physical(x, n_vis)[0].cpu().numpy()   # (n_vis, 6)
            parts.append(p)
    return np.stack(parts), np.stack(tgt), np.stack(ach), np.stack(knobs)


def _limits(per_algo_parts: dict, lo=1.0, hi=99.0):
    """Fixed per-coordinate display limits (centered on the global mean) from all
    controllers/frames, via robust percentiles. Returns (means(6,), lims list)."""
    allp = np.concatenate([v.reshape(-1, 6) for v in per_algo_parts.values()], 0)
    means = allp.mean(0)
    disp = (allp - means) * COORD_SCALE
    lims = [(np.percentile(disp[:, k], lo), np.percentile(disp[:, k], hi))
            for k in range(6)]
    # pad 8%
    lims = [(c - 0.08 * (d - c), d + 0.08 * (d - c)) for c, d in lims]
    return means, lims


def _ellipse_xy(aspect, tilt_deg, rms_size):
    """Boundary points of the commanded ellipse (2-σ), oriented by tilt, ratio=aspect."""
    th = np.deg2rad(tilt_deg)
    a_maj = 2.0 * rms_size * np.sqrt(aspect)                # 2-sigma major (RMS*2)
    a_min = 2.0 * rms_size / np.sqrt(aspect)
    t = np.linspace(0, 2 * np.pi, 100)
    ex, ey = a_maj * np.cos(t), a_min * np.sin(t)
    xr = ex * np.cos(th) - ey * np.sin(th)
    yr = ex * np.sin(th) + ey * np.cos(th)
    return xr, yr                                           # in mm (already scaled by caller)


def _animate_case(name, per_algo, per_algo_knobs, tgt, T, means, lims, out_path,
                  fps, stride, dark=False):
    import contextlib

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    # styling: white default vs. high-contrast black-background slide version
    colors = COLORS_DARK if dark else COLORS
    ref_color = "white" if dark else "k"          # commanded ellipse / legend
    if dark:
        FS = dict(corner_lab=12, corner_tick=9, xy_lab=17, xy_title=18, xy_leg=15,
                  knob_lab=15, knob_title=17, knob_tick=12, suptitle=23,
                  sc_xy=7.0, sc_corner=2.5, a_xy=0.35, a_corner=0.30,
                  hist_lw=2.0, knob_lw=3.0, knob_ms=8, ell_lw=3.0, ref_ms=11,
                  grid_a=0.22)
    else:
        FS = dict(corner_lab=7, corner_tick=6, xy_lab=10, xy_title=11, xy_leg=9,
                  knob_lab=8, knob_title=10, knob_tick=6, suptitle=13,
                  sc_xy=3.0, sc_corner=1.0, a_xy=0.22, a_corner=0.18,
                  hist_lw=1.0, knob_lw=1.5, knob_ms=4, ell_lw=1.5, ref_ms=6,
                  grid_a=0.3)
    style_ctx = (plt.style.context("dark_background") if dark
                 else contextlib.nullcontext())
    style_ctx.__enter__()

    algos = list(per_algo.keys())
    frames = list(range(0, T, stride))
    nk = len(KNOB_LABELS)
    gs_top = 0.88 if dark else 0.93     # leave room for the larger suptitle/x–y title
    fig = plt.figure(figsize=(20, 9))
    # left: 6x6 corner + big x–y (phase space); right: 5 stacked knob-history panels
    gsL = fig.add_gridspec(6, 6, left=0.035, right=0.60, top=gs_top, bottom=0.06,
                           hspace=0.08, wspace=0.08)
    ax = {(i, j): fig.add_subplot(gsL[i, j]) for i in range(6) for j in range(i + 1)}
    ax_xy = fig.add_subplot(gsL[0:3, 3:6])                  # big x–y in the free upper-right
    gsR = fig.add_gridspec(nk, 1, left=0.69, right=0.985, top=gs_top, bottom=0.06,
                           hspace=0.45)
    ax_knob = [fig.add_subplot(gsR[k, 0]) for k in range(nk)]

    # commanded aspect/tilt per frame (from target shape vector)
    asp_c, tilt_c = properties.s_to_aspect_tilt(torch.as_tensor(tgt[:, 0]),
                                                torch.as_tensor(tgt[:, 1]))
    asp_c, tilt_c = asp_c.numpy(), tilt_c.numpy()
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=colors[a],
                          label=a.upper()) for a in algos]
    handles.append(plt.Line2D([0], [0], ls="--", color=ref_color, label="commanded"))

    def draw(frame_idx):
        t = frames[frame_idx]
        for a in ax.values():
            a.cla()
        ax_xy.cla()
        # corner: lower triangle scatter + diagonal hist
        for i in range(6):
            for j in range(i + 1):
                a = ax[(i, j)]
                if i == j:
                    for alg in algos:
                        d = (per_algo[alg][t][:, i] - means[i]) * COORD_SCALE[i]
                        a.hist(d, bins=40, range=lims[i], histtype="step",
                               color=colors[alg], lw=FS["hist_lw"], density=True)
                    a.set_xlim(*lims[i]); a.set_yticks([])
                else:
                    for alg in algos:
                        xd = (per_algo[alg][t][:, j] - means[j]) * COORD_SCALE[j]
                        yd = (per_algo[alg][t][:, i] - means[i]) * COORD_SCALE[i]
                        a.scatter(xd, yd, s=FS["sc_corner"], c=colors[alg],
                                  alpha=FS["a_corner"], rasterized=True, linewidths=0)
                    a.set_xlim(*lims[j]); a.set_ylim(*lims[i])
                if j == 0:
                    a.set_ylabel(COORD_LABELS[i], fontsize=FS["corner_lab"])
                else:
                    a.set_yticklabels([])
                if i == 5:
                    a.set_xlabel(COORD_LABELS[j], fontsize=FS["corner_lab"])
                else:
                    a.set_xticklabels([])
                a.tick_params(labelsize=FS["corner_tick"])
        # big x–y (equal aspect so tilt is visually true), centered
        rms_sizes = []
        for alg in algos:
            xd = (per_algo[alg][t][:, 0] - means[0]) * COORD_SCALE[0]
            yd = (per_algo[alg][t][:, 1] - means[1]) * COORD_SCALE[1]
            ax_xy.scatter(xd, yd, s=FS["sc_xy"], c=colors[alg], alpha=FS["a_xy"],
                          rasterized=True, linewidths=0)
            rms_sizes.append(np.sqrt(0.5 * (xd.var() + yd.var())))
        ex, ey = _ellipse_xy(asp_c[t], tilt_c[t], float(np.median(rms_sizes)))
        ax_xy.plot(ex, ey, "--", color=ref_color, lw=FS["ell_lw"])
        m = max(abs(lims[0][0]), abs(lims[0][1]), abs(lims[1][0]), abs(lims[1][1]))
        ax_xy.set_xlim(-m, m); ax_xy.set_ylim(-m, m); ax_xy.set_aspect("equal")
        ax_xy.set_xlabel(COORD_LABELS[0], fontsize=FS["xy_lab"])
        ax_xy.set_ylabel(COORD_LABELS[1], fontsize=FS["xy_lab"])
        ax_xy.tick_params(labelsize=FS["xy_lab"] - 4)
        ax_xy.set_title(f"x–y transverse phase space   step {t}/{T - 1}\n"
                        f"commanded: aspect={asp_c[t]:.2f}, tilt={tilt_c[t]:+.0f}°",
                        fontsize=FS["xy_title"])
        ax_xy.legend(handles=handles, fontsize=FS["xy_leg"], loc="upper right")
        ax_xy.grid(True, alpha=FS["grid_a"])
        # control-knob histories (right column): one panel per knob, lines grow to t
        ts = np.arange(t + 1)
        for k, axk in enumerate(ax_knob):
            axk.cla()
            for alg in algos:
                kh = per_algo_knobs[alg][:t + 1, k]
                axk.plot(ts, kh, color=colors[alg], lw=FS["knob_lw"])
                axk.plot(t, kh[-1], "o", color=colors[alg], ms=FS["knob_ms"])
            axk.set_xlim(0, T - 1); axk.set_ylim(-0.03, 1.03)
            axk.set_ylabel(KNOB_LABELS[k], fontsize=FS["knob_lab"])
            axk.tick_params(labelsize=FS["knob_tick"])
            axk.grid(True, alpha=FS["grid_a"])
            if k == 0:
                axk.set_title("control knobs (normalized 0–1)", fontsize=FS["knob_title"])
            if k == len(ax_knob) - 1:
                axk.set_xlabel("step", fontsize=FS["knob_lab"])
            else:
                axk.set_xticklabels([])
        fig.suptitle(f"Moving-target controller — {name}  (SHAC/BPTT/PPO overlaid)",
                     fontsize=FS["suptitle"], y=0.995)

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"facecolor": "black"} if dark else {}
    anim.save(str(out_path), writer=PillowWriter(fps=fps), dpi=90,
              savefig_kwargs=save_kwargs)
    plt.close(fig)
    style_ctx.__exit__(None, None, None)
    print(f"[animate_tracking] wrote {out_path}  ({len(frames)} frames)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow-ckpt", required=True)
    p.add_argument("--norm-json", required=True)
    p.add_argument("--processed", default=None)
    p.add_argument("--shac", default=None)
    p.add_argument("--bptt", default=None)
    p.add_argument("--ppo", default=None)
    p.add_argument("--diffrl-policy", default="best", choices=["best", "final"])
    p.add_argument("--traj-config", default=None)
    p.add_argument("--episode-length", type=int, default=64)
    p.add_argument("--n-vis", type=int, default=1000, help="particles per controller per frame.")
    p.add_argument("--action-scale", type=float, default=0.05)
    p.add_argument("--stride", type=int, default=2, help="animate every Nth step.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default="figures/move_anim")
    p.add_argument("--dark", action="store_true",
                   help="black-background, large-font slideshow styling.")
    p.add_argument("--only-case", default=None,
                   help="render just this trajectory case (e.g. staircase).")
    p.add_argument("--suffix", default="",
                   help="appended to output filename: move_<name><suffix>.gif")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    T = args.episode_length
    spec = load_moving_config(args.traj_config).get("eval_trajectories") or None
    trajectories = build_eval_trajectories(T, spec)
    if args.only_case:
        if args.only_case not in trajectories:
            raise SystemExit(f"--only-case {args.only_case!r} not in "
                             f"{list(trajectories)}")
        trajectories = {args.only_case: trajectories[args.only_case]}
    runs = {k: v for k, v in (("shac", args.shac), ("bptt", args.bptt),
                              ("ppo", args.ppo)) if v}
    if not runs:
        raise SystemExit("pass at least one of --shac/--bptt/--ppo")
    action_fns = {a: _build_action_fn(a, Path(d), args.device, args.diffrl_policy)
                  for a, d in runs.items()}

    out_dir = Path(args.out_dir)
    for name, traj in trajectories.items():
        per_algo, per_algo_knobs, tgt = {}, {}, None
        for a in runs:
            parts, tgt, _ach, knobs = _rollout_particles(
                args.flow_ckpt, args.norm_json, args.processed, traj, T,
                args.n_vis, action_fns[a], args.device, args.action_scale)
            per_algo[a] = parts
            per_algo_knobs[a] = knobs
        means, lims = _limits(per_algo)
        _animate_case(name, per_algo, per_algo_knobs, tgt, T, means, lims,
                      out_dir / f"move_{name}{args.suffix}.gif", args.fps,
                      args.stride, dark=args.dark)


if __name__ == "__main__":
    main()
