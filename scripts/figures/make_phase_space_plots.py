#!/usr/bin/env python
"""
Publication-quality 6D phase-space "slice" plots for three example beams from the
FACET-II photoinjector dataset (PR10241 screen readout), following the
openPMD-beamphysics (LUME / LUME-Impact) ParticleGroup plotting convention.

Plotting standard (from `beamphysics.ParticleGroup` introspection, authoritative/offline):
  - Real space x, y, z in [m] -> displayed in mm.
  - Transverse slopes xp = px/pz, yp = py/pz in [rad] -> displayed in mrad.
  - Longitudinal energy spread shown as delta_pz = pz/<pz> - 1 (fractional, %),
    and z in mm (bunch length).
  - 2D marginals via density (hist2d) since ~2000 particles is too many for clean
    scatter; 1D marginals as histograms on the diagonal of the corner plot.
  - x-y real space drawn with equal aspect so the transverse ellipse shape is faithful
    (this is the panel that reveals the sigma_x/sigma_y aspect ratio).

This script ONLY reads dataset files and writes PNGs into figures/phase_space/.
"""
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from beamphysics import ParticleGroup

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(REPO, "data", "processed", "flow_surrogate.h5")
OUT = os.path.join(REPO, "figures", "phase_space")
ARCHIVE_DIRS = [
    os.path.join(REPO, "data", "archives", "train"),
]
SCREEN = "impact/output/particles/PR10241/electron"
SETTING_KEYS = [
    "SOL10111:solenoid_field_scale", "CQ10121:b1_gradient", "SQ10122:b1_gradient",
    "GUNF:rf_field_scale", "GUNF:theta0_deg", "distgen:r_dist:sigma_xy:value",
    "distgen:r_dist:truncation_radius:value", "distgen:start:MTE:value",
    "distgen:t_dist:sigma_t:value", "distgen:transforms:s1:scale",
    "distgen:transforms:r1:angle:value",
]
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 180, "font.size": 10})


# ---------------------------------------------------------------- selection
def select_beams():
    with h5py.File(PROC, "r") as h:
        part = h["particles"][:]          # (N,1500,6) x,y,z,px,py,pz
        sett = h["settings"][:]
        fp = h["fingerprint"][:]
        ne4 = h["norm_emit_4d"][:]
    sx = part[:, :, 0].std(axis=1)
    sy = part[:, :, 1].std(axis=1)
    aspect = sx / sy
    mpz = part[:, :, 5].mean(axis=1)
    emed = np.median(mpz)
    eband = np.abs(mpz - emed) < 0.04 * emed   # common energy band -> shape comparison

    def pick(target, lo, hi):
        mask = eband & (aspect >= lo) & (aspect <= hi)
        idxs = np.where(mask)[0]
        return idxs[np.argmin(np.abs(aspect[idxs] - target))]

    sel = {
        "round":  pick(1.00, 0.85, 1.15),
        "wide-x": pick(5.30, 4.00, 8.00),   # near 97th percentile
        "wide-y": pick(0.20, 0.13, 0.25),   # near 3rd percentile
    }
    info = {}
    for lab, i in sel.items():
        info[lab] = dict(
            idx=int(i), fp=fp[i].decode(), aspect=float(aspect[i]),
            sx=float(sx[i]), sy=float(sy[i]), pz=float(mpz[i]), ne4=float(ne4[i]),
            knobs={k: float(sett[i, j]) for j, k in enumerate(SETTING_KEYS)},
        )
    return info, part


# ---------------------------------------------------------------- loading
def load_pg(rec, part_proc):
    """Load full-fidelity ParticleGroup from archive; fall back to processed arrays."""
    fp = rec["fp"]
    for d in ARCHIVE_DIRS:
        path = os.path.join(d, fp + ".h5")
        if os.path.isfile(path):
            with h5py.File(path, "r") as h:
                P = ParticleGroup(h5=h[SCREEN])
            return P, path
    # fallback: reconstruct from processed (1500,6)
    arr = part_proc[rec["idx"]]
    n = arr.shape[0]
    data = dict(
        x=arr[:, 0], y=arr[:, 1], z=arr[:, 2],
        px=arr[:, 3], py=arr[:, 4], pz=arr[:, 5],
        t=np.zeros(n), status=np.ones(n, dtype=int),
        weight=np.full(n, 500e-12 / n), species="electron",
    )
    return ParticleGroup(data=data), "(reconstructed from processed)"


# ---------------------------------------------------------------- derived coords
def coords(P):
    """Standard openPMD-beamphysics display coords, scaled to nice units."""
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
    ax.hist2d(xv, yv, bins=bins, cmap="viridis",
              norm=LogNorm(), rasterized=True)


# ---------------------------------------------------------------- corner plot
def corner_plot(P, rec, label, fname):
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
            if r == col:                      # diagonal: 1D marginal
                ax.hist(xv, bins=60, color="steelblue", alpha=0.85)
                ax.set_yticks([])
            else:                              # lower triangle: 2D density
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
            else:
                if not (r == col):
                    ax.set_yticklabels([])
    fig.suptitle(
        f"{label}  |  {rec['fp'][:12]}…   "
        rf"$\sigma_x/\sigma_y$={rec['aspect']:.2f}   "
        rf"$\sigma_x$={P['sigma_x']*1e3:.2f} mm  $\sigma_y$={P['sigma_y']*1e3:.2f} mm   "
        f"E={P['mean_energy']/1e6:.2f} MeV  Q={P.charge*1e12:.0f} pC",
        fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = os.path.join(OUT, fname)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- canonical panel
def canonical_panel(P, rec, label, fname):
    c = coords(P)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))

    # 1) transverse real space x-y (equal aspect -> faithful ellipse shape)
    ax = axes[0]
    hist2d(ax, *[c["x"][0], c["y"][0]])
    ax.set_xlabel(c["x"][1]); ax.set_ylabel(c["y"][1])
    ax.set_aspect("equal", "box")
    ax.set_title("transverse  x-y")
    ax.text(0.04, 0.96,
            rf"$\sigma_x$={P['sigma_x']*1e3:.2f} mm" "\n"
            rf"$\sigma_y$={P['sigma_y']*1e3:.2f} mm" "\n"
            rf"$\sigma_x/\sigma_y$={rec['aspect']:.2f}",
            transform=ax.transAxes, va="top", ha="left", color="w", fontsize=9,
            bbox=dict(boxstyle="round", fc="black", alpha=0.45))

    # 2) horizontal phase space x-xp
    ax = axes[1]
    hist2d(ax, c["x"][0], c["xp"][0])
    ax.set_xlabel(c["x"][1]); ax.set_ylabel(c["xp"][1])
    ax.set_title("horizontal  x-x'")
    ax.text(0.04, 0.96, rf"$\varepsilon_{{n,x}}$={P['norm_emit_x']*1e6:.2f} $\mu$m",
            transform=ax.transAxes, va="top", color="w", fontsize=9,
            bbox=dict(boxstyle="round", fc="black", alpha=0.45))

    # 3) vertical phase space y-yp
    ax = axes[2]
    hist2d(ax, c["y"][0], c["yp"][0])
    ax.set_xlabel(c["y"][1]); ax.set_ylabel(c["yp"][1])
    ax.set_title("vertical  y-y'")
    ax.text(0.04, 0.96, rf"$\varepsilon_{{n,y}}$={P['norm_emit_y']*1e6:.2f} $\mu$m",
            transform=ax.transAxes, va="top", color="w", fontsize=9,
            bbox=dict(boxstyle="round", fc="black", alpha=0.45))

    # 4) longitudinal z-delta
    ax = axes[3]
    hist2d(ax, c["z"][0], c["dp"][0])
    ax.set_xlabel(c["z"][1]); ax.set_ylabel(c["dp"][1])
    ax.set_title(r"longitudinal  z-$\delta$")

    fig.suptitle(
        f"{label}  |  {rec['fp'][:12]}…   "
        rf"$\sigma_x/\sigma_y$={rec['aspect']:.2f}   E={P['mean_energy']/1e6:.2f} MeV   "
        f"Q={P.charge*1e12:.0f} pC   "
        f"CQ10121={rec['knobs']['CQ10121:b1_gradient']:+.3f}  "
        f"SQ10122={rec['knobs']['SQ10122:b1_gradient']:+.3f}",
        fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT, fname)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- combined x-y
def xy_comparison(pgs, recs, labels, fname):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    # common limits for fair shape comparison
    lim = max(max(abs(P.x).max(), abs(P.y).max()) for P in pgs) * 1e3 * 1.05
    for ax, P, rec, lab in zip(axes, pgs, recs, labels):
        c = coords(P)
        hist2d(ax, c["x"][0], c["y"][0])
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", "box")
        ax.set_xlabel(c["x"][1]); ax.set_ylabel(c["y"][1])
        ax.set_title(rf"{lab}   $\sigma_x/\sigma_y$={rec['aspect']:.2f}" "\n"
                     f"CQ={rec['knobs']['CQ10121:b1_gradient']:+.3f}  "
                     f"SQ={rec['knobs']['SQ10122:b1_gradient']:+.3f}",
                     fontsize=11)
    fig.suptitle("Transverse real-space shape comparison (x-y, equal aspect, common scale)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    out = os.path.join(OUT, fname)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    info, part_proc = select_beams()
    labels = {"round": "Round", "wide-x": "Wide-x", "wide-y": "Wide-y"}
    saved = []
    pgs, recs, labs = [], [], []
    print("=== Beam selection ===")
    for key in ["round", "wide-x", "wide-y"]:
        rec = info[key]
        P, src = load_pg(rec, part_proc)
        rec["src"] = src
        rec["nfull"] = P.n_particle
        rec["nex"] = P["norm_emit_x"]
        rec["ney"] = P["norm_emit_y"]
        rec["charge"] = P.charge
        rec["energy"] = P["mean_energy"]
        print(f"{key:7s} fp={rec['fp']} aspect={rec['aspect']:.3f} "
              f"sx={P['sigma_x']*1e3:.3f}mm sy={P['sigma_y']*1e3:.3f}mm "
              f"CQ={rec['knobs']['CQ10121:b1_gradient']:+.4f} "
              f"SQ={rec['knobs']['SQ10122:b1_gradient']:+.4f} "
              f"ne_x={P['norm_emit_x']*1e6:.2f}um ne_y={P['norm_emit_y']*1e6:.2f}um "
              f"Q={P.charge*1e12:.0f}pC E={P['mean_energy']/1e6:.3f}MeV n={P.n_particle} src={os.path.basename(src)}")
        tag = key.replace("-", "")
        saved.append(corner_plot(P, rec, labels[key], f"corner_{tag}.png"))
        saved.append(canonical_panel(P, rec, labels[key], f"canonical_{tag}.png"))
        pgs.append(P); recs.append(rec); labs.append(labels[key])
    saved.append(xy_comparison(pgs, recs, labs, "xy_comparison.png"))
    print("\n=== Saved figures ===")
    for s in saved:
        print(s)


if __name__ == "__main__":
    main()
