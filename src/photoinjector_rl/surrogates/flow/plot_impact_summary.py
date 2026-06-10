"""
Summary figure for the moving-target (aspect,tilt) controller evaluated on REAL
Impact-T, with lofi and hifi overlaid (the "fidelity-intersect" view). Mirrors the
flow-surrogate `move_tracking.png`: a 2x3 grid (rows = eigen aspect / tilt deg;
cols = the three held-out schedules), commanded dashed-black, one achieved curve
per (controller, fidelity): color = controller (SHAC/BPTT/PPO), linestyle = fidelity
(lofi solid, hifi dotted). Same-color lofi/hifi curves nearly coincide -> the
controller is fidelity-robust.

Reads logs/impact_eval/<algo>_<traj>_<fid>/rollout.npz (target, achieved).

Usage:
    python -m photoinjector_rl.surrogates.flow.plot_impact_summary \
        --log-dir logs/impact_eval --out figures/impact_tracking_summary.png
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import torch

from . import properties

COLORS = {"shac": "tab:blue", "bptt": "tab:orange", "ppo": "tab:green"}
STYLE = {"lofi": "-", "hifi": ":"}
TRAJS = ["staircase", "tilt_rotation", "aspect_ramp"]
ALGOS = ["shac", "bptt", "ppo"]


def _load(log_dir: str) -> dict:
    data = {}
    for f in glob.glob(str(Path(log_dir) / "*" / "rollout.npz")):
        m = re.search(r"/(shac|bptt|ppo)_(staircase|tilt_rotation|aspect_ramp)_"
                      r"(lofi|hifi)/rollout\.npz$", f)
        if m:
            data[m.groups()] = np.load(f)
    return data


def _aspect_tilt(arr: np.ndarray):
    a, t = properties.s_to_aspect_tilt(torch.as_tensor(arr[:, 0]),
                                       torch.as_tensor(arr[:, 1]))
    return a.numpy(), t.numpy()


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.lines as ml
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", default="runs/impact_eval")
    ap.add_argument("--out", default="figures/impact_tracking_summary.png")
    args = ap.parse_args()

    data = _load(args.log_dir)
    if not data:
        raise SystemExit(f"no rollout.npz found under {args.log_dir}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    for j, traj in enumerate(TRAJS):
        key = next(((a, traj, fd) for a in ALGOS for fd in ("lofi", "hifi")
                    if (a, traj, fd) in data), None)
        if key is None:
            continue
        tgt = data[key]["target"]
        ts = np.arange(len(tgt))
        casp, ctilt = _aspect_tilt(tgt)
        axes[0, j].plot(ts, casp, "k--", lw=2.5, zorder=5)
        axes[1, j].plot(ts, ctilt, "k--", lw=2.5, zorder=5)
        for a in ALGOS:
            for fd in ("lofi", "hifi"):
                if (a, traj, fd) not in data:
                    continue
                asp, tilt = _aspect_tilt(data[(a, traj, fd)]["achieved"])
                axes[0, j].plot(ts, asp, STYLE[fd], color=COLORS[a], lw=1.6, alpha=0.9)
                axes[1, j].plot(ts, tilt, STYLE[fd], color=COLORS[a], lw=1.6, alpha=0.9)
        axes[0, j].set_title(traj, fontsize=12)
        for ax in (axes[0, j], axes[1, j]):
            ax.grid(True, alpha=0.3)
        axes[1, j].set_xlabel("step")
    axes[0, 0].set_ylabel("eigen aspect")
    axes[1, 0].set_ylabel("tilt (deg)")

    handles = [ml.Line2D([], [], color=COLORS[a], lw=2.5, label=a.upper()) for a in ALGOS]
    handles += [ml.Line2D([], [], color="grey", ls="-", lw=1.8, label="lofi"),
                ml.Line2D([], [], color="grey", ls=":", lw=1.8, label="hifi"),
                ml.Line2D([], [], color="k", ls="--", lw=2.0, label="commanded")]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=11,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Moving-target (aspect,tilt) controller on real Impact-T — "
                 "lofi vs hifi (trained on the lofi flow surrogate)", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"[plot_impact_summary] wrote {args.out}")


if __name__ == "__main__":
    main()
