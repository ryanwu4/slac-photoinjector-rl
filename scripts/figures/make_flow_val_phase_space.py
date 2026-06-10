#!/usr/bin/env python
"""
Report-ready true-vs-predicted phase-space overlays for the conditional-flow
surrogate, on held-out VALIDATION conditions.

For each of the first N held-out val conditions we sample a (P, 6) cloud from the
trained flow and overlay it on the Impact-T truth cloud, exactly reproducing the
held-out evaluation used in `flow_surrogate/train.py`:

  * same FlowDataModule split (split_seed=42, val_fraction=0.1)  -> identical val set
  * same best checkpoint loaded with ConditionalAffineFlow.load_from_checkpoint
  * same Sliced-Wasserstein distance + 2D/4D/6D emittance %err helpers
    (imported from train.py, so the metrics are bit-for-bit the same routine)

Each figure is a 2x2 grid of the four phase-space slices:
  (x,y)   (x,px)
  (y,py)  (z,pz)

Styling: seaborn "talk" context (large, clear fonts) + whitegrid, blue = Impact-T
truth, red = flow surrogate. The legend is drawn once (top-left panel). Scatter is
rasterized so the PDF stays small while axes/text remain vector.

Units (raw ParticleGroup coords are positions [m], momenta [eV/c]):
  x, y, z -> mm (x1e3);  px, py -> eV/c (x1);  pz -> MeV/c (x1e-6).

Run (from repo root, with the slac-rl env):
  PYTHONPATH=$PWD/src \
  /home/rwu4/miniconda3/envs/slac-rl/bin/python \
      figures/report/make_flow_val_phase_space.py --n-samples 4 --seed 0
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402

from photoinjector_rl.surrogates.flow.dataset import (  # noqa: E402
    FlowDataModule,
    _destandardize_particles,
)
from photoinjector_rl.surrogates.flow.model import ConditionalAffineFlow  # noqa: E402
from photoinjector_rl.surrogates.flow.train import (  # noqa: E402
    _emit_pct_errs,
    sliced_wasserstein_distance,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = REPO / "models/flow_surrogate/checkpoints/best-epoch=493-val_loss=-0.9555.ckpt"
DEFAULT_PROC = REPO / "data/processed/flow_surrogate.h5"
DEFAULT_NORM = REPO / "data/processed/flow_surrogate_norm.json"
OUT_DIR = REPO / "figures/report"

# Raw particle column order in the dataset is [x, y, z, px, py, pz]
# (positions [m], momenta [eV/c]). Plane index pairs match flow_surrogate/train.py.
PLANES = [
    (0, 1, "x  [mm]", "y  [mm]", 1e3, 1e3),
    (0, 3, "x  [mm]", "px  [eV/c]", 1e3, 1.0),
    (1, 4, "y  [mm]", "py  [eV/c]", 1e3, 1.0),
    (2, 5, "z  [mm]", "pz  [MeV/c]", 1e3, 1e-6),
]

C_TRUTH = "#2c6fbb"      # Impact-T (truth)  -- clear blue
C_FLOW = "#c0392b"       # flow surrogate    -- clear red


def collect_val_samples(model, dm, n_samples, seed):
    """Return a list of per-sample dicts (true cloud, flow cloud, metrics)."""
    model.eval()
    full = dm.full
    assert full is not None and dm.val is not None
    val_idx = list(dm.val.indices)[:n_samples]

    p_mean = np.array(full.norm["particle_mean"], dtype=np.float32)
    p_std = np.array(full.norm["particle_std"], dtype=np.float32)

    out = []
    # Seed once so the whole batch of figures is reproducible; sampling order
    # then matches the val-index order.
    torch.manual_seed(seed)
    with torch.no_grad():
        for rank, idx in enumerate(val_idx):
            cond = full.cond[idx:idx + 1]                      # (1, 11)
            true_phys = _destandardize_particles(
                full.particles[idx].numpy(), p_mean[None, :], p_std[None, :])
            pred_phys = model.sample_physical(cond, full.P)[0].cpu().numpy()

            sw = sliced_wasserstein_distance(pred_phys, true_phys)
            errs = _emit_pct_errs(true_phys, pred_phys)
            out.append({
                "rank": rank,
                "val_index": int(idx),
                "true": true_phys,
                "pred": pred_phys,
                "sw": sw,
                "errs": errs,
            })
    return out


def plot_sample(sample, out_stub, save_pdf=True):
    true_phys, pred_phys = sample["true"], sample["pred"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    for k, (ax, (i, j, xl, yl, sx, sy)) in enumerate(zip(axes.flat, PLANES)):
        ax.scatter(true_phys[:, i] * sx, true_phys[:, j] * sy,
                   s=6, alpha=0.45, color=C_TRUTH, linewidths=0,
                   label="Impact-T", rasterized=True)
        ax.scatter(pred_phys[:, i] * sx, pred_phys[:, j] * sy,
                   s=6, alpha=0.45, color=C_FLOW, linewidths=0,
                   label="flow surrogate", rasterized=True)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        if k == 0:                       # legend drawn once, on the (x, y) panel
            leg = ax.legend(loc="upper left", framealpha=0.9, markerscale=3,
                            handletextpad=0.3, borderpad=0.4)
            for lh in leg.legend_handles:
                lh.set_alpha(1.0)

    fig.tight_layout()

    png = f"{out_stub}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(f"{out_stub}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--processed", default=str(DEFAULT_PROC))
    ap.add_argument("--norm-json", default=str(DEFAULT_NORM))
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    sns.set_theme(context="talk", style="whitegrid", font_scale=1.0)
    plt.rcParams.update({
        "axes.titleweight": "bold",
        "axes.edgecolor": "#444444",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,      # editable text in the PDF
        "figure.facecolor": "white",
    })

    dm = FlowDataModule(
        processed_h5=args.processed,
        norm_json=args.norm_json,
        val_fraction=0.1,
        split_seed=42,
    )
    dm.setup()
    model = ConditionalAffineFlow.load_from_checkpoint(args.ckpt, map_location="cpu")

    samples = collect_val_samples(model, dm, args.n_samples, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{'rank':>4} {'val_idx':>8} {'SW(6D)':>10} {'2D%':>7} {'4D%':>7} {'6D%':>7}")
    for s in samples:
        e = s["errs"]
        print(f"{s['rank']:>4} {s['val_index']:>8} {s['sw']:>10.1f} "
              f"{e['x_xp']:>7.2f} {e['fourd']:>7.2f} {e['sixd']:>7.2f}")
        stub = os.path.join(args.out_dir, f"flow_val_phase_space_sample{s['rank']}")
        png = plot_sample(s, stub, save_pdf=not args.no_pdf)
        print(f"      wrote {png}" + ("" if args.no_pdf else " (+ .pdf)"))

    # Aggregate over the rendered samples (sanity check vs final_metrics.json).
    arr_sw = np.array([s["sw"] for s in samples])
    arr_4d = np.array([s["errs"]["fourd"] for s in samples])
    arr_6d = np.array([s["errs"]["sixd"] for s in samples])
    print(f"\nmean over {len(samples)} rendered samples: "
          f"SW={arr_sw.mean():.1f}  4D%={arr_4d.mean():.2f}  6D%={arr_6d.mean():.2f}")


if __name__ == "__main__":
    main()
