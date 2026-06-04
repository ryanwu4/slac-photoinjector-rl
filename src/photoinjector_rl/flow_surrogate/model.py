"""
Conditional affine-coupling normalizing flow (Lightning module).

Transforms a base N(0, I) in 6-D into the PR10241 output-bunch density,
conditioned on the 11 normalized knobs. Trained by per-particle
negative-log-likelihood plus optional emittance / beam-matrix matching terms.

The flow + coupling math is ported (unmodified in spirit) from
``accelerator_flow_model/train_norm_flow_conditional.py``; the new pieces are
the 11-knob conditioning, the self-contained (de)normalization buffers, and the
differentiable ``knobs -> emittance`` path used by first-order MBRL.

GRADIENT CONTRACT (for MBRL): ``sample`` draws z ~ N(0, I) ONCE as a constant
and applies a deterministic, condition-parameterized transform -- the
reparameterization trick. Gradients therefore flow from any function of the
sampled particles (e.g. emittance) back to the condition (the knobs). Never wrap
``sample`` / ``emittance_from_knobs`` / ``forward`` in ``torch.no_grad`` and
never ``.detach`` the condition if you need that gradient.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import math

import lightning as L
import numpy as np
import torch
from torch import nn

_LOG_2PI = math.log(2.0 * math.pi)


# ============================================================================
# Differentiable beam statistics (ported, assume (B, N, 6) in (x,y,z,px,py,pz))
# ============================================================================

def compute_emittance_torch(particles: torch.Tensor) -> dict[str, torch.Tensor]:
    """2D/4D/6D emittances (geometric, = sqrt(det Sigma)) for a batch of bunches.

    particles: (B, N, 6) in (x, y, z, px, py, pz) order. Returns a dict with
    'x_xp', 'y_yp', 'z_delta', 'fourd', 'sixd', each shape (B,).
    """
    means = particles.mean(dim=1, keepdim=True)
    centered = particles - means
    cov = torch.bmm(centered.transpose(1, 2), centered) / (particles.shape[1] - 1)

    emittances: dict[str, torch.Tensor] = {}
    cov_x_px = cov[:, [0, 3]][:, :, [0, 3]]
    emittances["x_xp"] = torch.sqrt(torch.abs(torch.linalg.det(cov_x_px)) + 1e-16)
    cov_y_py = cov[:, [1, 4]][:, :, [1, 4]]
    emittances["y_yp"] = torch.sqrt(torch.abs(torch.linalg.det(cov_y_py)) + 1e-16)
    cov_z_pz = cov[:, [2, 5]][:, :, [2, 5]]
    emittances["z_delta"] = torch.sqrt(torch.abs(torch.linalg.det(cov_z_pz)) + 1e-16)
    cov_4d = cov[:, [0, 3, 1, 4]][:, :, [0, 3, 1, 4]]
    emittances["fourd"] = torch.sqrt(torch.abs(torch.linalg.det(cov_4d)) + 1e-16)
    cov_6d = cov[:, [0, 3, 1, 4, 2, 5]][:, :, [0, 3, 1, 4, 2, 5]]
    emittances["sixd"] = torch.sqrt(torch.abs(torch.linalg.det(cov_6d)) + 1e-16)
    return emittances


def compute_beam_matrix_torch(particles: torch.Tensor) -> torch.Tensor:
    """(B, 6, 6) covariance matrices in (x, px, y, py, z, pz) order."""
    perm = [0, 3, 1, 4, 2, 5]
    particles_reordered = particles[:, :, perm]
    cov_list = [torch.cov(particles_reordered[i].T) for i in range(particles.shape[0])]
    return torch.stack(cov_list)


# ============================================================================
# Affine coupling layer + flow stack (ported)
# ============================================================================

class ConditionalCouplingLayer(nn.Module):
    """Single affine coupling layer with conditioning and alternating masking.

    Operates on an ALREADY-ENCODED condition of width ``cond_dim`` (the encoder
    is owned by the parent module so it runs once per batch, not once per layer).
    """

    def __init__(self, dim: int, cond_dim: int, mlp_dim: int = 64,
                 reverse_mask: bool = False):
        super().__init__()
        self.dim = dim
        self.d = dim // 2
        self.reverse_mask = reverse_mask
        self.net = nn.Sequential(
            nn.Linear(self.d + cond_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, (dim - self.d) * 2),
        )

    def _split(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.reverse_mask:
            return z[:, self.d:], z[:, :self.d]
        return z[:, :self.d], z[:, self.d:]

    def _params(self, z1: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(torch.cat([z1, cond], dim=1))
        scale = params[:, :(self.dim - self.d)]
        shift = params[:, (self.dim - self.d):]
        # Bounded scale keeps exp() from overflowing (the flow's NaN guard).
        scale = torch.tanh(scale) * 0.5
        return scale, shift

    def _join(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # 'a' is the transformed half, 'b' the untouched half.
        if self.reverse_mask:
            return torch.cat([a, b], dim=1)
        return torch.cat([b, a], dim=1)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z1, z2 = self._split(z)
        scale, shift = self._params(z1, cond)
        z2_new = z2 * torch.exp(scale) + shift
        return self._join(z2_new, z1), scale.sum(dim=1)

    def inverse(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z1, z2 = self._split(z)
        scale, shift = self._params(z1, cond)
        z2_new = (z2 - shift) * torch.exp(-scale)
        return self._join(z2_new, z1), -scale.sum(dim=1)


class ConditionalAffineFlow(L.LightningModule):
    """Conditional affine-coupling flow: N(0, I) <-> normalized output bunch."""

    def __init__(
        self,
        condition_dim: int = 11,
        latent_dim: int = 6,
        hidden_dim: int = 128,
        n_layers: int = 16,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        w_emit_2d: float = 1.0,
        w_emit_4d: float = 0.5,
        w_emit_6d: float = 0.1,
        w_beam: float = 0.0,  # off by default: slow (per-bunch torch.cov loop) and can destabilize early training
        beam_reg: float = 1e-2,
        nll_dim_norm: float = 6.0,
        n_aux_particles: int | None = None,
        forward_n_particles: int = 512,
        # (De)normalization stats -- stored as buffers so the differentiable
        # knobs->emittance path is self-contained and travels with the ckpt.
        particle_mean: list[float] | None = None,
        particle_std: list[float] | None = None,
        setting_low: list[float] | None = None,
        setting_high: list[float] | None = None,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        electron_mc2_ev: float = 0.51099895e6,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim

        self.condition_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.flows = nn.ModuleList([
            ConditionalCouplingLayer(latent_dim, hidden_dim, reverse_mask=(i % 2 == 1))
            for i in range(n_layers)
        ])

        self.register_buffer("base_mean", torch.zeros(latent_dim))
        self.register_buffer("base_std", torch.ones(latent_dim))

        pm = torch.zeros(latent_dim) if particle_mean is None else torch.tensor(particle_mean)
        ps = torch.ones(latent_dim) if particle_std is None else torch.tensor(particle_std)
        sl = torch.zeros(condition_dim) if setting_low is None else torch.tensor(setting_low)
        sh = torch.ones(condition_dim) if setting_high is None else torch.tensor(setting_high)
        self.register_buffer("particle_mean", pm.float())
        self.register_buffer("particle_std", ps.float())
        self.register_buffer("setting_low", sl.float())
        self.register_buffer("setting_high", sh.float())
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_std", torch.tensor(float(target_std)))
        self.register_buffer("mc2", torch.tensor(float(electron_mc2_ev)))

    # ----- flow primitives --------------------------------------------------

    def _encode(self, cond_norm: torch.Tensor) -> torch.Tensor:
        return self.condition_net(cond_norm)

    def _forward_enc(self, z: torch.Tensor, cond_enc: torch.Tensor):
        """base -> data, with a pre-encoded condition."""
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for flow in self.flows:
            z, ld = flow(z, cond_enc)
            log_det = log_det + ld
        return z, log_det

    def _inverse_enc(self, x: torch.Tensor, cond_enc: torch.Tensor):
        """data -> base, with a pre-encoded condition."""
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for flow in reversed(self.flows):
            x, ld = flow.inverse(x, cond_enc)
            log_det = log_det + ld
        return x, log_det

    def forward_flow(self, z: torch.Tensor, cond_norm: torch.Tensor):
        """base -> data. cond_norm: (M, condition_dim)."""
        return self._forward_enc(z, self._encode(cond_norm))

    def inverse(self, x: torch.Tensor, cond_norm: torch.Tensor):
        """data -> base. cond_norm: (M, condition_dim)."""
        return self._inverse_enc(x, self._encode(cond_norm))

    def log_prob(self, x: torch.Tensor, cond_norm: torch.Tensor) -> torch.Tensor:
        """log p(x | cond) for normalized particles x: (M, 6)."""
        z, log_det = self.inverse(x, cond_norm)
        log_prob_base = (-0.5 * (z ** 2) - 0.5 * _LOG_2PI).sum(dim=1)
        return log_prob_base + log_det

    # ----- sampling (reparameterized, differentiable wrt cond) --------------

    def sample(self, cond_norm: torch.Tensor, n: int) -> torch.Tensor:
        """Sample n particles per condition. Returns (B, n, 6), NORMALIZED.

        Differentiable w.r.t. ``cond_norm`` via the reparameterization trick:
        the only stochastic node is ``z`` (a constant w.r.t. autograd); the
        transform is deterministic given the (gradient-carrying) condition.
        """
        b = cond_norm.shape[0]
        cond_enc = self._encode(cond_norm)                       # (B, hidden)
        cond_flat = cond_enc.unsqueeze(1).expand(b, n, -1).reshape(b * n, -1)
        z = torch.randn(b * n, self.latent_dim,
                        device=cond_norm.device, dtype=cond_norm.dtype)
        x_flat, _ = self._forward_enc(z, cond_flat)
        return x_flat.reshape(b, n, self.latent_dim)

    def _destandardize(self, parts_norm: torch.Tensor) -> torch.Tensor:
        return parts_norm * self.particle_std + self.particle_mean

    def sample_physical(self, cond_norm: torch.Tensor, n: int) -> torch.Tensor:
        """Sample n particles per condition in raw units (m, eV/c). (B, n, 6)."""
        return self._destandardize(self.sample(cond_norm, n))

    # ----- differentiable knobs -> emittance (MBRL hook) --------------------

    def emittance_from_knobs(self, x_norm: torch.Tensor, n: int | None = None) -> torch.Tensor:
        """norm_emit_4d (m^2) per condition, differentiable w.r.t. x_norm (B, 11).

        x_norm is the 11-D knob vector in [0, 1] (the same box the RL env feeds,
        cat(knobs, distgen)). Path: sample -> de-standardize -> 4D geometric
        emittance -> / (mc^2)^2, reproducing ParticleGroup.norm_emit_4d.
        """
        if n is None:
            n = self.hparams.n_aux_particles or 1500
        parts = self._destandardize(self.sample(x_norm, n))      # (B, n, 6)
        emit_geo_4d = compute_emittance_torch(parts)["fourd"]    # (B,)
        return emit_geo_4d / (self.mc2 ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """diff_env drop-in: x (B, 11) -> (B, 1) normalized log10(norm_emit_4d).

        Matches the EmittanceMLP contract used by DiffPhotoinjectorEnv (output is
        z-scored log-emit; reward = -y_norm). Stochastic across calls (resampled
        z); raise ``forward_n_particles`` to cut variance for RL use.
        """
        emit = self.emittance_from_knobs(x, n=self.hparams.forward_n_particles)
        log_emit = torch.log10(emit.clamp_min(1e-30))
        y_norm = (log_emit - self.target_mean) / self.target_std
        return y_norm.unsqueeze(-1)

    # ----- training ---------------------------------------------------------

    def _step(self, batch, prefix: str) -> torch.Tensor:
        particles_norm, cond_norm = batch          # (B, P, 6), (B, 11)
        b, p = particles_norm.shape[0], particles_norm.shape[1]

        # NLL: encode the B conditions ONCE, broadcast over particles.
        cond_enc = self._encode(cond_norm)                            # (B, hidden)
        cond_flat = cond_enc.unsqueeze(1).expand(b, p, -1).reshape(b * p, -1)
        parts_flat = particles_norm.reshape(b * p, self.latent_dim)
        z, log_det = self._inverse_enc(parts_flat, cond_flat)
        log_prob_base = (-0.5 * (z ** 2) - 0.5 * _LOG_2PI).sum(dim=1)
        nll = -(log_prob_base + log_det).mean() / self.hparams.nll_dim_norm

        loss = nll
        logs = {f"{prefix}_nll": nll}

        hp = self.hparams
        use_aux = hp.w_emit_2d > 0 or hp.w_emit_4d > 0 or hp.w_emit_6d > 0 or hp.w_beam > 0
        if use_aux:
            n_aux = hp.n_aux_particles or p
            x_pred = self.sample(cond_norm, n_aux)                     # (B, n_aux, 6)
            em_pred = compute_emittance_torch(x_pred)
            em_true = compute_emittance_torch(particles_norm)
            eps = 1e-20

            if hp.w_emit_2d > 0:
                l2d = (
                    torch.abs((em_pred["x_xp"] - em_true["x_xp"]) / (em_true["x_xp"] + eps)).mean()
                    + torch.abs((em_pred["y_yp"] - em_true["y_yp"]) / (em_true["y_yp"] + eps)).mean()
                    + torch.abs((em_pred["z_delta"] - em_true["z_delta"]) / (em_true["z_delta"] + eps)).mean()
                ) / 3.0
                loss = loss + hp.w_emit_2d * l2d
                logs[f"{prefix}_emit_2d"] = l2d
            if hp.w_emit_4d > 0:
                l4d = torch.abs((em_pred["fourd"] - em_true["fourd"]) / (em_true["fourd"] + eps)).mean()
                loss = loss + hp.w_emit_4d * l4d
                logs[f"{prefix}_emit_4d"] = l4d
            if hp.w_emit_6d > 0:
                l6d = torch.abs((em_pred["sixd"] - em_true["sixd"]) / (em_true["sixd"] + eps)).mean()
                loss = loss + hp.w_emit_6d * l6d
                logs[f"{prefix}_emit_6d"] = l6d
            if hp.w_beam > 0:
                cov_pred = compute_beam_matrix_torch(x_pred)
                cov_true = compute_beam_matrix_torch(particles_norm)
                beam = (2 * torch.abs(cov_pred - cov_true)
                        / (torch.abs(cov_pred) + torch.abs(cov_true) + hp.beam_reg)).mean()
                loss = loss + hp.w_beam * beam
                logs[f"{prefix}_beam"] = beam

        logs[f"{prefix}_loss"] = loss
        self.log_dict(logs, on_step=False, on_epoch=True,
                      prog_bar=(prefix == "val"), batch_size=b)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                               weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}
