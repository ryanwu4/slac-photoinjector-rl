"""
Lightning MLP that maps the 11 XOPT-sampled knob values to scalar
log10(norm_emit_4d) (normalized). Standard regression-with-MSE; physical-unit
metrics (MAE in m^2, MAPE) are logged for human readability.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import lightning as L
import torch
from torch import nn


class EmittanceMLP(L.LightningModule):
    def __init__(
        self,
        input_dim: int = 11,
        hidden_dims: tuple[int, ...] = (128, 128, 128),
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.lr = lr
        self.weight_decay = weight_decay
        # Stored for de-normalization of logged metrics.
        self.target_mean = target_mean
        self.target_std = target_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _physical_mae(self, y_hat_norm: torch.Tensor, y_norm: torch.Tensor) -> torch.Tensor:
        """MAE in physical units (m^2). Inverts log10 + z-score."""
        y_hat_log = y_hat_norm * self.target_std + self.target_mean
        y_log = y_norm * self.target_std + self.target_mean
        y_hat_phys = 10.0 ** y_hat_log
        y_phys = 10.0 ** y_log
        return torch.mean(torch.abs(y_hat_phys - y_phys))

    def _log_metrics(self, y_hat: torch.Tensor, y: torch.Tensor, prefix: str,
                     batch_size: int) -> torch.Tensor:
        loss = nn.functional.mse_loss(y_hat, y)
        mae_norm = torch.mean(torch.abs(y_hat - y))
        mae_phys = self._physical_mae(y_hat, y)
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=batch_size)
        self.log(f"{prefix}_mae_log_norm", mae_norm, on_step=False, on_epoch=True,
                 batch_size=batch_size)
        self.log(f"{prefix}_mae_phys_m2", mae_phys, on_step=False, on_epoch=True,
                 prog_bar=(prefix == "val"), batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        x, y = batch
        y_hat = self(x)
        return self._log_metrics(y_hat, y, "train", batch_size=x.shape[0])

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        x, y = batch
        y_hat = self(x)
        return self._log_metrics(y_hat, y, "val", batch_size=x.shape[0])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)

    def predict_physical(self, x: torch.Tensor) -> torch.Tensor:
        """Inference helper: returns norm_emit_4d in m^2 (un-normalized)."""
        self.eval()
        with torch.no_grad():
            y_norm = self(x)
        y_log = y_norm * self.target_std + self.target_mean
        return 10.0 ** y_log
