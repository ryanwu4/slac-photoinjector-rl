"""
End-to-end training entry point. Loads the preprocessed HDF5 + normalization
JSON, builds an EmittanceMLP, trains with Lightning, and saves a checkpoint
plus a final-metrics JSON.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .dataset import EmittanceDataModule
from .model import EmittanceMLP


def predict_val_physical(
    model: EmittanceMLP, dm: EmittanceDataModule
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over the val DataLoader and return (true, pred) in m^2."""
    model.eval()
    device = next(model.parameters()).device
    preds_norm: list[torch.Tensor] = []
    truths_norm: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in dm.val_dataloader():
            preds_norm.append(model(x.to(device)))
            truths_norm.append(y)
    pred = torch.cat(preds_norm).squeeze(-1).cpu().numpy()
    true = torch.cat(truths_norm).squeeze(-1).cpu().numpy()
    # Invert log10 + z-score.
    pred_phys = 10.0 ** (pred * model.target_std + model.target_mean)
    true_phys = 10.0 ** (true * model.target_std + model.target_mean)
    return true_phys, pred_phys


def regression_diagnostics(true: np.ndarray, pred: np.ndarray) -> dict:
    """R^2, MAPE (%), MAE (physical units, m^2)."""
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # MAPE: guard against zero truths (shouldn't happen here, but be safe).
    safe_true = np.where(np.abs(true) < 1e-30, np.nan, true)
    mape = float(np.nanmean(np.abs((pred - true) / safe_true)) * 100.0)
    mae = float(np.mean(np.abs(pred - true)))
    return {"val_r2": r2, "val_mape_pct": mape, "val_mae_phys_m2": mae}


def plot_pred_vs_true(
    true: np.ndarray, pred: np.ndarray, diag: dict, out_png: str
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    lo = float(min(true.min(), pred.min())) * 0.9
    hi = float(max(true.max(), pred.max())) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6, label="y = x")
    ax.scatter(true, pred, s=8, alpha=0.5, color="steelblue", edgecolor="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True  norm_emit_4d  (m²)")
    ax.set_ylabel("Predicted  norm_emit_4d  (m²)")
    ax.set_title(
        f"Validation  |  N = {len(true)}\n"
        f"R² = {diag['val_r2']:.4f}    "
        f"MAPE = {diag['val_mape_pct']:.2f}%    "
        f"MAE = {diag['val_mae_phys_m2']:.2e} m²",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed", default="data/processed/emittance_target.h5")
    p.add_argument("--norm-json", default=None,
                   help="defaults to <processed>_norm.json")
    p.add_argument("--out-dir", default="models/emittance_target")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20,
                   help="early-stopping patience on val_loss")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128, 128])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices", default="auto")
    return p


def train(args: argparse.Namespace) -> dict:
    dm = EmittanceDataModule(
        processed_h5=args.processed,
        norm_json=args.norm_json,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        split_seed=args.split_seed,
    )
    dm.setup()
    norm = dm.norm

    model = EmittanceMLP(
        input_dim=dm.input_dim,
        hidden_dims=tuple(args.hidden),
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        target_mean=norm["target_mean"],
        target_std=norm["target_std"],
    )
    print(f"[train] input_dim={dm.input_dim}")

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
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    trainer.fit(model, datamodule=dm)

    metrics: dict[str, object] = {
        k: float(v.item()) for k, v in trainer.callback_metrics.items()
        if isinstance(v, torch.Tensor)
    }
    metrics["input_dim"] = dm.input_dim
    metrics["best_checkpoint"] = str(ckpt_cb.best_model_path)
    best_score = ckpt_cb.best_model_score
    metrics["best_val_loss"] = float(best_score.item()) if best_score is not None else None

    # Post-training: reload the best-val checkpoint and produce the
    # pred-vs-true scatter (in physical units) + R^2 / MAPE.
    best_model = EmittanceMLP.load_from_checkpoint(ckpt_cb.best_model_path)
    true_phys, pred_phys = predict_val_physical(best_model, dm)
    diag = regression_diagnostics(true_phys, pred_phys)
    metrics.update(diag)

    scatter_path = str(Path(args.out_dir) / "val_pred_vs_true.png")
    plot_pred_vs_true(true_phys, pred_phys, diag, scatter_path)
    metrics["val_pred_vs_true_plot"] = scatter_path

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
