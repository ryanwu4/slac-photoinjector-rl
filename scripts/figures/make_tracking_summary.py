#!/usr/bin/env python
"""
Report-ready version of figures/impact_tracking_summary.png:
the moving-target (aspect, tilt) controller evaluated on REAL Impact-T, lofi vs
hifi (trained on the lofi flow surrogate).

2x3 grid: rows = eigen aspect / tilt [deg], cols = the three held-out schedules
(staircase, tilt_rotation, aspect_ramp). For each panel:
  * commanded setpoint  -> black dashed,
  * one achieved curve per (controller, fidelity): color = controller
    (SHAC/BPTT/PPO), linestyle = fidelity (lofi solid, hifi dotted).
Same-color lofi/hifi curves nearly coincide -> the controller is fidelity-robust.

Data: logs/impact_eval/<algo>_<traj>_<fid>/rollout.npz (target + achieved are
(T, 2) Stokes shape vectors, converted to (aspect, tilt) via properties).
Computation mirrors flow_surrogate/plot_impact_summary.py exactly; only the
styling differs (seaborn whitegrid + large fonts, capitalized algo names).

Run (from repo root, slac-rl env):
  PYTHONPATH=$PWD/src \
  /home/rwu4/miniconda3/envs/slac-rl/bin/python \
      figures/report/make_tracking_summary.py
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as ml  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402

from photoinjector_rl.surrogates.flow import properties  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LOG = REPO / "runs/impact_eval"
OUT_DIR = REPO / "figures/report"

# Color = controller, linestyle = fidelity. Mapping matches the report
# learning-curve figures: PPO blue, SHAC orange, BPTT green.
COLORS = {"ppo": "#1f77b4", "shac": "#ff7f0e", "bptt": "#2ca02c"}
STYLE = {"lofi": "-", "hifi": ":"}
TRAJS = ["tilt_rotation", "aspect_ramp", "staircase"]
TRAJ_TITLES = {"staircase": "Staircase",
               "tilt_rotation": "Tilt Rotation",
               "aspect_ramp": "Aspect Ramp"}
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", default=str(DEFAULT_LOG))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--out-name", default="tracking_summary")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    sns.set_theme(context="talk", style="whitegrid", font_scale=1.0)
    plt.rcParams.update({
        "axes.titleweight": "bold",
        "axes.edgecolor": "#444444",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
    })

    data = _load(args.log_dir)
    if not data:
        raise SystemExit(f"no rollout.npz found under {args.log_dir}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    for j, traj in enumerate(TRAJS):
        key = next(((a, traj, fd) for a in ALGOS for fd in ("lofi", "hifi")
                    if (a, traj, fd) in data), None)
        if key is None:
            continue
        tgt = data[key]["target"]
        ts = np.arange(len(tgt))
        casp, ctilt = _aspect_tilt(tgt)
        axes[0, j].plot(ts, casp, "k--", lw=2.8, zorder=5)
        axes[1, j].plot(ts, ctilt, "k--", lw=2.8, zorder=5)
        for a in ALGOS:
            for fd in ("lofi", "hifi"):
                if (a, traj, fd) not in data:
                    continue
                asp, tilt = _aspect_tilt(data[(a, traj, fd)]["achieved"])
                axes[0, j].plot(ts, asp, STYLE[fd], color=COLORS[a], lw=2.2, alpha=0.9)
                axes[1, j].plot(ts, tilt, STYLE[fd], color=COLORS[a], lw=2.2, alpha=0.9)
        axes[0, j].set_title(TRAJ_TITLES[traj])
        for ax in (axes[0, j], axes[1, j]):
            ax.grid(True, alpha=0.3)
        axes[1, j].set_xlabel("step")
    axes[0, 0].set_ylabel("Eigen Aspect Ratio")
    axes[1, 0].set_ylabel("Tilt (deg)")

    handles = [ml.Line2D([], [], color=COLORS[a], lw=3.0, label=a.upper()) for a in ALGOS]
    handles += [ml.Line2D([], [], color="k", ls="--", lw=2.4, label="commanded")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    stub = Path(args.out_dir) / args.out_name
    fig.savefig(f"{stub}.png", dpi=300, bbox_inches="tight")
    if not args.no_pdf:
        fig.savefig(f"{stub}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stub}.png" + ("" if args.no_pdf else " (+ .pdf)")
          + f"   ({len(data)}/18 rollouts loaded)")


if __name__ == "__main__":
    main()
