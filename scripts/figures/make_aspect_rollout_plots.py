#!/usr/bin/env python
"""
Roll out the trained aspect-ratio-target policies in logs/compare_aspec_target2/
(property=aspect_ratio, reward_mode=target, target=2.0) inside the flow surrogate
env, then render full 6D phase-space CORNER plots of the resulting terminal beams.

The agent controls 5 Impact knobs to drive the beam's projected sigma_x/sigma_y
toward 2.0 (wide-in-x). For each policy (PPO / SHAC / BPTT, seed_0) we:
  - build a FlowBunchEnv with R=3 envs (3 distinct reset contexts),
  - capture the INITIAL reset beam (flow-sampled at the reset knobs/distgen),
  - step the deterministic policy for 63 steps (stop BEFORE step 64 = auto-reset),
  - sample the full-fidelity TERMINAL beam from the settled (knobs, distgen),
  - compute achieved aspect = sigma_x/sigma_y per rollout.

Plotting reuses the openPMD-beamphysics (LUME-Impact) display convention from
figures/phase_space/make_phase_space_plots.py: real space x,y,z in mm; slopes
xp=px/pz, yp=py/pz in mrad; longitudinal delta = pz/<pz> - 1 in %; a full 6x6
lower-triangle corner plot (15 hist2d projections + 1D diagonals), x-y equal aspect.

This script ONLY reads model/dataset/policy files and writes PNGs into
figures/aspect_rollouts/. It modifies no existing source or config files.
"""
import os
import glob

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from beamphysics import ParticleGroup

from photoinjector_rl.surrogates.flow.diff_env import FlowBunchEnv
from photoinjector_rl.surrogates.flow.compare_algos import _load_diffrl_actor

# ---------------------------------------------------------------- config
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(REPO, "runs", "compare_aspec_target2")
OUT = os.path.join(REPO, "figures", "aspect_rollouts")
CKPT = sorted(glob.glob(os.path.join(REPO, "models", "flow_surrogate",
                                     "checkpoints", "best-*.ckpt")))[-1]
NORM_JSON = os.path.join(REPO, "data", "processed", "flow_surrogate_norm.json")
PROC_H5 = os.path.join(REPO, "data", "processed", "flow_surrogate.h5")
DEVICE = "cuda:0"

TARGET = 2.0
PROPERTY = "aspect_ratio"
REWARD_MODE = "target"
N_ROLLOUTS = 3          # R distinct reset contexts per policy
N_PARTICLES_ENV = 2000  # particles used inside the env reward
N_PARTICLES_FULL = 4000 # full-fidelity beam for plotting
EP_STEPS = 63           # stop BEFORE step 64 (auto-reset)
SEED = 7                # reset seed -> 3 distinct distgen contexts

PPO_ZIP = os.path.join(RUN, "ppo", "seed_0", "ppo_final.zip")
SHAC_PT = os.path.join(RUN, "shac", "seed_0", "best_policy.pt")
BPTT_PT = os.path.join(RUN, "bptt", "seed_0", "best_policy.pt")

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 180, "font.size": 10})


# ---------------------------------------------------------------- env + rollout
def make_env():
    return FlowBunchEnv(
        num_envs=N_ROLLOUTS, device=DEVICE, seed=SEED, episode_length=64,
        no_grad=True, flow_ckpt=CKPT, norm_json=NORM_JSON, processed_h5=PROC_H5,
        property=PROPERTY, reward_mode=REWARD_MODE, target=TARGET,
        n_particles=N_PARTICLES_ENV,
    )


def _sample_full(env):
    """Full-fidelity beam at the env's CURRENT (knobs, distgen): (R, n, 6) numpy."""
    x = torch.cat([env._knobs, env._distgen], dim=-1)
    parts = env._flow.sample_physical(x, N_PARTICLES_FULL)
    return parts.detach().cpu().numpy()


def _aspect(parts):
    """sigma_x / sigma_y per rollout. parts: (R, n, 6)."""
    return parts[:, :, 0].std(axis=1) / parts[:, :, 1].std(axis=1)


def _ppo_action_fn():
    from stable_baselines3 import PPO
    model = PPO.load(PPO_ZIP, device=DEVICE)

    def fn(obs):
        obs_np = obs.detach().cpu().numpy()
        act_np, _ = model.predict(obs_np, deterministic=True)  # already in [-1,1]
        return torch.from_numpy(np.asarray(act_np, dtype=np.float32)).to(DEVICE)

    return fn


def _diffrl_action_fn(policy_pt):
    actor, obs_rms = _load_diffrl_actor(policy_pt, DEVICE)

    def fn(obs):
        o = obs_rms.normalize(obs) if obs_rms is not None else obs
        return torch.tanh(actor(o, deterministic=True))

    return fn


def rollout(action_fn):
    """Returns (initial_beam, terminal_beam) each (R, n, 6) numpy arrays."""
    env = make_env()
    obs = env.reset()
    initial = _sample_full(env)  # beam at reset, before stepping
    with torch.no_grad():
        for _ in range(EP_STEPS):
            obs, _r, _d, _i = env.step(action_fn(obs))
        assert int(env._step_count.min()) == EP_STEPS, \
            f"unexpected step_count {env._step_count.cpu().numpy()}"
        terminal = _sample_full(env)  # settled beam, no reset yet
    return initial, terminal


# ---------------------------------------------------------------- ParticleGroup
def make_pg(arr6):
    """Build a beamphysics.ParticleGroup from a (n, 6) physical array
    (x,y,z [m]; px,py,pz [eV/c]). Charge/weight are nominal (500 pC)."""
    n = arr6.shape[0]
    data = dict(
        x=arr6[:, 0], y=arr6[:, 1], z=arr6[:, 2],
        px=arr6[:, 3], py=arr6[:, 4], pz=arr6[:, 5],
        t=np.zeros(n), status=np.ones(n, dtype=int),
        weight=np.full(n, 500e-12 / n), species="electron",
    )
    return ParticleGroup(data=data)


# ---------------------------------------------------------------- display coords
def coords(P):
    """openPMD-beamphysics display coords, nice units (matches phase_space script)."""
    pz0 = P["mean_pz"]
    return {
        "x":  (P.x * 1e3, "x [mm]"),
        "y":  (P.y * 1e3, "y [mm]"),
        "z":  ((P.z - P["mean_z"]) * 1e3, "z [mm]"),
        "xp": (P.xp * 1e3, r"$x'$ [mrad]"),
        "yp": (P.yp * 1e3, r"$y'$ [mrad]"),
        "dp": ((P.pz / pz0 - 1.0) * 100.0, r"$\delta_{p_z}$ [%]"),
    }


def hist2d(ax, xv, yv, bins=60):
    ax.hist2d(xv, yv, bins=bins, cmap="viridis", norm=LogNorm(), rasterized=True)


# ---------------------------------------------------------------- corner plot
def corner_plot(P, title, fname):
    """Full 6x6 lower-triangle corner: 15 hist2d projections + 1D diagonals."""
    c = coords(P)
    order = ["x", "xp", "y", "yp", "z", "dp"]
    n = len(order)
    fig, axes = plt.subplots(n, n, figsize=(13, 13))
    for r in range(n):
        for col in range(n):
            ax = axes[r, col]
            if col > r:
                ax.axis("off")
                continue
            kx, ky = order[col], order[r]
            xv, xl = c[kx]
            if r == col:
                ax.hist(xv, bins=60, color="steelblue", alpha=0.85)
                ax.set_yticks([])
            else:
                yv, yl = c[ky]
                hist2d(ax, xv, yv)
                if kx == "x" and ky == "y":
                    ax.set_aspect("equal", "box")
            if r == n - 1:
                ax.set_xlabel(xl)
            else:
                ax.set_xticklabels([])
            if col == 0 and r != 0:
                ax.set_ylabel(c[ky][1])
            elif not (r == col):
                ax.set_yticklabels([])
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = os.path.join(OUT, fname)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- before/after
def before_after_panels(Pi, Pf, asp_i, asp_f, algo, fname):
    """3x2 grid: x-y, x-xp, y-yp for INITIAL (top) vs TERMINAL (bottom)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.6))
    rows = [("INITIAL (reset)", Pi, asp_i), ("TERMINAL (settled)", Pf, asp_f)]
    # common x-y limits for fair before/after shape comparison
    lim = max(max(abs(P.x).max(), abs(P.y).max()) for _, P, _ in rows) * 1e3 * 1.05
    for ridx, (rlab, P, asp) in enumerate(rows):
        c = coords(P)
        ax = axes[ridx, 0]
        hist2d(ax, c["x"][0], c["y"][0])
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", "box")
        ax.set_xlabel(c["x"][1]); ax.set_ylabel(c["y"][1])
        ax.set_title(rf"{rlab}  x-y   $\sigma_x/\sigma_y$={asp:.2f}")
        ax.text(0.04, 0.96,
                rf"$\sigma_x$={P['sigma_x']*1e3:.2f} mm" "\n"
                rf"$\sigma_y$={P['sigma_y']*1e3:.2f} mm",
                transform=ax.transAxes, va="top", ha="left", color="w", fontsize=9,
                bbox=dict(boxstyle="round", fc="black", alpha=0.45))

        ax = axes[ridx, 1]
        hist2d(ax, c["x"][0], c["xp"][0])
        ax.set_xlabel(c["x"][1]); ax.set_ylabel(c["xp"][1])
        ax.set_title(r"x-x'")

        ax = axes[ridx, 2]
        hist2d(ax, c["y"][0], c["yp"][0])
        ax.set_xlabel(c["y"][1]); ax.set_ylabel(c["yp"][1])
        ax.set_title(r"y-y'")
    fig.suptitle(
        f"{algo}  before/after  (aspect_ratio -> target {TARGET:.1f})   "
        rf"$\sigma_x/\sigma_y$: {asp_i:.2f} -> {asp_f:.2f}",
        fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = os.path.join(OUT, fname)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- x-y comparison
def xy_comparison(entries, fname):
    """entries: list of (label, ParticleGroup, aspect). One panel each,
    equal aspect, common scale."""
    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.2))
    lim = max(max(abs(P.x).max(), abs(P.y).max()) for _, P, _ in entries) * 1e3 * 1.05
    for ax, (lab, P, asp) in zip(axes, entries):
        c = coords(P)
        hist2d(ax, c["x"][0], c["y"][0])
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", "box")
        ax.set_xlabel(c["x"][1]); ax.set_ylabel(c["y"][1])
        ax.set_title(rf"{lab}" "\n" rf"$\sigma_x/\sigma_y$={asp:.2f}", fontsize=11)
    fig.suptitle(
        f"Transverse real-space x-y (equal aspect, common scale)   "
        f"aspect target = {TARGET:.1f}",
        fontsize=13, y=1.01)
    fig.tight_layout()
    out = os.path.join(OUT, fname)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"flow ckpt : {CKPT}")
    print(f"property={PROPERTY}  reward_mode={REWARD_MODE}  target={TARGET}")
    print(f"R={N_ROLLOUTS} rollouts/policy, {EP_STEPS} steps, "
          f"{N_PARTICLES_FULL} particles/beam\n")

    policies = [
        ("PPO", _ppo_action_fn),
        ("SHAC", lambda: _diffrl_action_fn(SHAC_PT)),
        ("BPTT", lambda: _diffrl_action_fn(BPTT_PT)),
    ]

    saved = []
    results = {}            # algo -> dict(initial, terminal, asp_i, asp_f, rep)
    xy_entries = []
    init_added = False

    for algo, fn_factory in policies:
        action_fn = fn_factory()
        initial, terminal = rollout(action_fn)
        asp_i = _aspect(initial)
        asp_f = _aspect(terminal)
        # representative rollout = closest achieved aspect to the target
        rep = int(np.argmin(np.abs(asp_f - TARGET)))
        results[algo] = dict(initial=initial, terminal=terminal,
                             asp_i=asp_i, asp_f=asp_f, rep=rep)

        print(f"=== {algo} ===")
        for r in range(N_ROLLOUTS):
            star = "  <-- representative" if r == rep else ""
            print(f"  rollout {r}: initial sx/sy={asp_i[r]:.3f}  "
                  f"terminal sx/sy={asp_f[r]:.3f}{star}")
        print(f"  median terminal sx/sy = {np.median(asp_f):.3f}\n")

        # 1) per-policy corner plot of the representative terminal beam
        Pf = make_pg(terminal[rep])
        title = (f"{algo}  terminal beam (rep rollout {rep})   "
                 rf"$\sigma_x/\sigma_y$={asp_f[rep]:.2f}  "
                 rf"$\sigma_x$={Pf['sigma_x']*1e3:.2f} mm  "
                 rf"$\sigma_y$={Pf['sigma_y']*1e3:.2f} mm   "
                 rf"E={Pf['mean_energy']/1e6:.2f} MeV   "
                 rf"target $\sigma_x/\sigma_y$={TARGET:.1f}")
        saved.append(corner_plot(Pf, title, f"corner_terminal_{algo.lower()}.png"))

        # collect x-y comparison entries (terminal); add the initial once
        if not init_added:
            Pi0 = make_pg(initial[rep])
            xy_entries.append((f"INITIAL (reset, {algo} ctx)", Pi0, asp_i[rep]))
            init_added = True
        xy_entries.append((f"{algo} terminal", Pf, asp_f[rep]))

    # 2) before/after for ONE rollout (BPTT, representative)
    rb = results["BPTT"]
    rep = rb["rep"]
    Pi = make_pg(rb["initial"][rep])
    Pf = make_pg(rb["terminal"][rep])
    saved.append(before_after_panels(
        Pi, Pf, rb["asp_i"][rep], rb["asp_f"][rep], "BPTT", "before_after_bptt.png"))

    # 3) x-y real-space comparison across the 3 policies + initial
    saved.append(xy_comparison(xy_entries, "xy_terminal_comparison.png"))

    print("=== Saved figures ===")
    for s in saved:
        print(s)


if __name__ == "__main__":
    main()
