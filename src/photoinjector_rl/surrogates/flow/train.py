"""
Training entry point for the conditional affine-coupling flow surrogate.

Loads the preprocessed HDF5 + normalization JSON, builds a ConditionalAffineFlow,
trains with Lightning (NLL + emittance aux losses, grad-clipped), and saves a
checkpoint + final-metrics JSON + a true-vs-predicted phase-space overlay.

Held-out evaluation: for a handful of val conditions, sample a cloud and report
the Sliced-Wasserstein distance and 2D/4D/6D emittance % error vs the Impact-T
truth.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import lightning as L
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402
from lightning.pytorch.loggers import CSVLogger  # noqa: E402

from .dataset import FlowDataModule, _destandardize_particles  # noqa: E402
from .model import ConditionalAffineFlow, compute_emittance_torch  # noqa: E402


def sliced_wasserstein_distance(p1: np.ndarray, p2: np.ndarray,
                                n_projections: int = 100, seed: int = 0) -> float:
    """SW distance between two 6D clouds (mean over random 1D projections)."""
    rng = np.random.default_rng(seed)
    dists = []
    for _ in range(n_projections):
        theta = rng.standard_normal(6)
        theta /= np.linalg.norm(theta)
        a = np.sort(p1 @ theta)
        b = np.sort(p2 @ theta)
        dists.append(np.mean(np.abs(a - b)))
    return float(np.mean(dists))


def _emit_pct_errs(true_phys: np.ndarray, pred_phys: np.ndarray) -> dict[str, float]:
    """2D/4D/6D emittance percent error between two raw (P,6) clouds."""
    et = compute_emittance_torch(torch.tensor(true_phys).unsqueeze(0))
    ep = compute_emittance_torch(torch.tensor(pred_phys).unsqueeze(0))
    eps = 1e-20
    out = {}
    for key in ("x_xp", "y_yp", "z_delta", "fourd", "sixd"):
        out[key] = float(torch.abs((ep[key] - et[key]) / (et[key] + eps)).item() * 100.0)
    return out


def evaluate_flow(model: ConditionalAffineFlow, dm: FlowDataModule,
                  n_samples: int = 5) -> dict:
    """Sample clouds for held-out val conditions; SW + emittance %err."""
    model.eval()
    full = dm.full
    assert full is not None and dm.val is not None
    val_idx = list(dm.val.indices)[:n_samples]  # type: ignore[attr-defined]

    p_mean = np.array(full.norm["particle_mean"], dtype=np.float32)
    p_std = np.array(full.norm["particle_std"], dtype=np.float32)

    sw_list, e4d_list, e6d_list = [], [], []
    first = None
    with torch.no_grad():
        for k, idx in enumerate(val_idx):
            cond = full.cond[idx:idx + 1]                      # (1, 11)
            P = full.P
            true_phys = _destandardize_particles(
                full.particles[idx].numpy(), p_mean[None, :], p_std[None, :])
            pred_phys = model.sample_physical(cond, P)[0].cpu().numpy()

            sw = sliced_wasserstein_distance(pred_phys, true_phys)
            errs = _emit_pct_errs(true_phys, pred_phys)
            sw_list.append(sw)
            e4d_list.append(errs["fourd"])
            e6d_list.append(errs["sixd"])
            if k == 0:
                first = (true_phys, pred_phys, sw, errs)

    return {
        "val_sw_mean": float(np.mean(sw_list)),
        "val_emit_4d_pct_err": float(np.mean(e4d_list)),
        "val_emit_6d_pct_err": float(np.mean(e6d_list)),
        "_first": first,
    }


def plot_phase_space_overlay(first: tuple, out_png: str) -> None:
    """2x2 true-vs-pred phase planes + a beam-matrix %err heatmap."""
    true_phys, pred_phys, sw, errs = first
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    planes = [
        (0, 1, "x [mm]", "y [mm]", 1e3, 1e3),
        (0, 3, "x [mm]", "px [eV/c]", 1e3, 1.0),
        (1, 4, "y [mm]", "py [eV/c]", 1e3, 1.0),
        (2, 5, "z [mm]", "pz [MeV/c]", 1e3, 1e-6),
    ]
    for ax, (i, j, xl, yl, sx, sy) in zip(axes.flat[:4], planes):
        ax.scatter(true_phys[:, i] * sx, true_phys[:, j] * sy, s=2, alpha=0.4,
                   color="steelblue", label="Impact-T")
        ax.scatter(pred_phys[:, i] * sx, pred_phys[:, j] * sy, s=2, alpha=0.4,
                   color="firebrick", label="flow")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Beam-matrix percent-error heatmap (canonical (x,px,y,py,z,pz) order).
    perm = [0, 3, 1, 4, 2, 5]
    labels = ["x", "px", "y", "py", "z", "pz"]
    cov_t = np.cov(true_phys[:, perm].T)
    cov_p = np.cov(pred_phys[:, perm].T)
    err = 100.0 * np.abs((cov_p - cov_t) / (np.abs(cov_t) + 1e-30))
    ax = axes.flat[4]
    im = ax.imshow(err, cmap="Reds", vmin=0, vmax=100)
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_title("beam-matrix % error")
    fig.colorbar(im, ax=ax, fraction=0.046)

    axes.flat[5].axis("off")
    fig.suptitle(f"Val sample: SW={sw:.3e}  4D emit %err={errs['fourd']:.1f}  "
                 f"6D %err={errs['sixd']:.1f}", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed", default="data/processed/flow_surrogate.h5")
    p.add_argument("--norm-json", default=None, help="defaults to <processed>_norm.json")
    p.add_argument("--out-dir", default="models/flow_surrogate")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--n-layers", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-aux-particles", type=int, default=None,
                   help="particles sampled for the aux emittance loss (default: batch P)")
    p.add_argument("--w-emit-2d", type=float, default=1.0)
    p.add_argument("--w-emit-4d", type=float, default=0.5)
    p.add_argument("--w-emit-6d", type=float, default=0.1)
    p.add_argument("--w-beam", type=float, default=0.0,
                   help="beam-matrix SMAPE weight; off by default (slow + can destabilize)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--n-eval-samples", type=int, default=5)
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices", default="auto")
    return p


def train(args: argparse.Namespace) -> dict:
    dm = FlowDataModule(
        processed_h5=args.processed,
        norm_json=args.norm_json,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        split_seed=args.split_seed,
    )
    dm.setup()
    norm = dm.norm

    # Target z-score (for the forward() diff_env drop-in) from the TRAIN split.
    train_idx = list(dm.train.indices)  # type: ignore[attr-defined]
    log_emit_train = np.log10(np.clip(dm.full.raw_emit.numpy()[train_idx], 1e-30, None))  # type: ignore[union-attr]
    target_mean = float(log_emit_train.mean())
    target_std = float(max(log_emit_train.std(), 1e-12))

    model = ConditionalAffineFlow(
        condition_dim=dm.cond_dim,
        hidden_dim=args.hidden,
        n_layers=args.n_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        w_emit_2d=args.w_emit_2d,
        w_emit_4d=args.w_emit_4d,
        w_emit_6d=args.w_emit_6d,
        w_beam=args.w_beam,
        n_aux_particles=args.n_aux_particles,
        particle_mean=norm["particle_mean"],
        particle_std=norm["particle_std"],
        setting_low=norm["setting_low"],
        setting_high=norm["setting_high"],
        target_mean=target_mean,
        target_std=target_std,
        electron_mc2_ev=norm.get("electron_mc2_ev", 0.51099895e6),
    )
    print(f"[train] cond_dim={dm.cond_dim} P={dm.full.P} "  # type: ignore[union-attr]
          f"target_mean={target_mean:.4f} target_std={target_std:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    logger = CSVLogger(save_dir=args.out_dir, name="csv_logs")
    ckpt_cb = ModelCheckpoint(
        dirpath=Path(args.out_dir) / "checkpoints",
        filename="best-{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=1, save_last=True,
    )
    es_cb = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min",
                          min_delta=1e-5)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        callbacks=[ckpt_cb, es_cb],
        gradient_clip_val=args.grad_clip,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    trainer.fit(model, datamodule=dm)

    metrics: dict[str, object] = {
        k: float(v.item()) for k, v in trainer.callback_metrics.items()
        if isinstance(v, torch.Tensor)
    }
    metrics["cond_dim"] = dm.cond_dim
    metrics["P"] = dm.full.P  # type: ignore[union-attr]
    metrics["n_records"] = norm["n_records"]
    metrics["target_mean"] = target_mean
    metrics["target_std"] = target_std
    metrics["best_checkpoint"] = str(ckpt_cb.best_model_path)
    best_score = ckpt_cb.best_model_score
    metrics["best_val_loss"] = float(best_score.item()) if best_score is not None else None

    # Post-training held-out eval + overlay plot.
    best_model = ConditionalAffineFlow.load_from_checkpoint(
        ckpt_cb.best_model_path, map_location="cpu")
    ev = evaluate_flow(best_model, dm, n_samples=args.n_eval_samples)
    first = ev.pop("_first")
    metrics.update(ev)
    if first is not None:
        plot_path = str(Path(args.out_dir) / "val_phase_space_overlay.png")
        plot_phase_space_overlay(first, plot_path)
        metrics["val_phase_space_plot"] = plot_path

    with open(Path(args.out_dir) / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


def main() -> None:
    args = build_argparser().parse_args()
    metrics = train(args)
    print("\nFinal metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
