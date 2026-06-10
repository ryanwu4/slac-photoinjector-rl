"""
Differentiable output-bunch properties + a reward specification for MBRL.

Every property maps a sampled cloud ``parts (B, n, 6)`` in PHYSICAL units
(x,y,z [m]; px,py,pz [eV/c], the COORD_KEYS order) to a per-env scalar ``(B,)``,
using only differentiable torch ops so the reward gradient backpropagates through
the particles to the flow's condition (the knobs). All formulas verified against
openpmd-beamphysics ParticleGroup (ratios 1.0):
  norm_emit_x = x_xp / mc^2,  norm_emit_4d = fourd / mc^2^2,
  sigma_* = std(coord),  mean_energy = mean(sqrt(p^2 + mc^2^2)),
  energy_spread = std(E)/mean(E).

`RewardSpec` turns a property into the env's ``y_norm`` (reward = -y_norm), with a
transform (log10 / identity) and a z-score so the reward scale matches the v1
emittance setup; `minimize` reproduces the baseline reward exactly for norm_emit_4d.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import h5py
import numpy as np
import torch

from . import ELECTRON_MC2_EV
from .model import compute_emittance_torch

MC2 = ELECTRON_MC2_EV


# ----- property functions: (parts (B,n,6) physical) -> (B,) ------------------

def _norm_emit_4d(parts: torch.Tensor) -> torch.Tensor:
    return compute_emittance_torch(parts)["fourd"] / (MC2 ** 2)


def _norm_emit_x(parts: torch.Tensor) -> torch.Tensor:
    return compute_emittance_torch(parts)["x_xp"] / MC2


def _norm_emit_y(parts: torch.Tensor) -> torch.Tensor:
    return compute_emittance_torch(parts)["y_yp"] / MC2


def _sigma_x(parts: torch.Tensor) -> torch.Tensor:
    return parts[:, :, 0].std(dim=1, unbiased=False)


def _sigma_y(parts: torch.Tensor) -> torch.Tensor:
    return parts[:, :, 1].std(dim=1, unbiased=False)


def _sigma_z(parts: torch.Tensor) -> torch.Tensor:
    return parts[:, :, 2].std(dim=1, unbiased=False)


def _aspect_ratio(parts: torch.Tensor) -> torch.Tensor:
    """Projected transverse aspect ratio sigma_x / sigma_y (lab frame)."""
    sx = parts[:, :, 0].std(dim=1, unbiased=False)
    sy = parts[:, :, 1].std(dim=1, unbiased=False)
    return sx / sy.clamp_min(1e-30)


def _particle_energy(parts: torch.Tensor) -> torch.Tensor:
    """Per-particle total energy [eV]: sqrt(px^2+py^2+pz^2 + mc^2^2)."""
    p2 = parts[:, :, 3] ** 2 + parts[:, :, 4] ** 2 + parts[:, :, 5] ** 2
    return torch.sqrt(p2 + MC2 ** 2)


def _mean_energy(parts: torch.Tensor) -> torch.Tensor:
    return _particle_energy(parts).mean(dim=1)


def _energy_spread(parts: torch.Tensor) -> torch.Tensor:
    e = _particle_energy(parts)
    return e.std(dim=1, unbiased=False) / e.mean(dim=1)


# ----- transverse SHAPE: aspect + tilt via the normalized 2nd-moment vector ---
# All differentiable (centered means/vars/cov are smooth in the particle coords),
# so SHAC/BPTT can backprop the reward through them.

def _cov_xy_components(parts: torch.Tensor):
    """Population (ddof=0) var_x, var_y, cov_xy of the (x,y) cloud. Each (B,)."""
    xc = parts[:, :, 0] - parts[:, :, 0].mean(dim=1, keepdim=True)
    yc = parts[:, :, 1] - parts[:, :, 1].mean(dim=1, keepdim=True)
    var_x = (xc * xc).mean(dim=1)
    var_y = (yc * yc).mean(dim=1)
    cov_xy = (xc * yc).mean(dim=1)
    return var_x, var_y, cov_xy


def _s1(parts: torch.Tensor) -> torch.Tensor:
    """Normalized x/y elongation (σx²−σy²)/(σx²+σy²) ∈ [−1,1] (NOT positive)."""
    vx, vy, _ = _cov_xy_components(parts)
    return (vx - vy) / (vx + vy).clamp_min(1e-30)


def _s2(parts: torch.Tensor) -> torch.Tensor:
    """Normalized x–y coupling 2·cov_xy/(σx²+σy²) ∈ [−1,1] (NOT positive)."""
    vx, vy, cxy = _cov_xy_components(parts)
    return 2.0 * cxy / (vx + vy).clamp_min(1e-30)


def _eigen_aspect(parts: torch.Tensor) -> torch.Tensor:
    """Rotation-invariant eigen aspect ratio σ_major/σ_minor = sqrt(λmax/λmin)
    of the 2×2 (x,y) covariance. ≥ 1 (round=1)."""
    vx, vy, cxy = _cov_xy_components(parts)
    tr = vx + vy
    det = vx * vy - cxy * cxy
    disc = torch.sqrt((tr * tr - 4.0 * det).clamp_min(0.0) + 1e-30)
    lam_max = 0.5 * (tr + disc)
    lam_min = (0.5 * (tr - disc)).clamp_min(1e-30)
    return torch.sqrt(lam_max / lam_min)


def _tilt_angle_deg(parts: torch.Tensor) -> torch.Tensor:
    """Major-axis tilt angle ½·atan2(2·cov_xy, σx²−σy²) in degrees ∈ (−90,90].
    Reporting only (circular; ill-defined for round beams)."""
    vx, vy, cxy = _cov_xy_components(parts)
    return 0.5 * torch.atan2(2.0 * cxy, vx - vy) * (180.0 / math.pi)


# name -> (fn, default transform). transform chooses the space the z-score /
# target live in: log10 for positive, right-skewed quantities; identity else.
PROPERTY_REGISTRY: dict[str, tuple[Callable[[torch.Tensor], torch.Tensor], str]] = {
    "norm_emit_4d": (_norm_emit_4d, "log10"),
    "norm_emit_x": (_norm_emit_x, "log10"),
    "norm_emit_y": (_norm_emit_y, "log10"),
    "sigma_x": (_sigma_x, "log10"),
    "sigma_y": (_sigma_y, "log10"),
    "sigma_z": (_sigma_z, "log10"),
    # Projected x/y spot-size ratio; log10 so round=1->0 and r vs 1/r are
    # symmetric (target-mode: |log10(r)-log10(target)|/std). Lever = CQ10121
    # (normal quad), with SQ10122 (skew) coupling. Intended use: reward-mode target.
    "aspect_ratio": (_aspect_ratio, "log10"),
    # Rotation-invariant true shape (eigenvalue ratio). Solenoid-dominated lever.
    "eigen_aspect": (_eigen_aspect, "log10"),
    "mean_energy": (_mean_energy, "identity"),
    "energy_spread": (_energy_spread, "log10"),
}
# s1/s2/tilt are NOT registered: s1/s2 can be negative (the registry + its
# positivity tests assume positive scalars), and tilt is circular. They are used
# directly by the joint shape-target path (ShapeTargetSpec / ShapeTargetEnv).


# ----- reward spec -----------------------------------------------------------

@dataclass
class RewardSpec:
    """Maps a bunch property to ``y_norm`` (env reward = -y_norm).

    mode:
      - "minimize": y_norm = (t(p) - mean) / std            (reward grows as p falls)
      - "maximize": y_norm = -(t(p) - mean) / std           (reward grows as p rises)
      - "target":   y_norm = |t(p) - t(target)| / std       (reward grows as p -> target)
    where t = log10 or identity. `mean`/`std` z-score t(p) to an O(1) reward.
    """

    name: str
    property_fn: Callable[[torch.Tensor], torch.Tensor]
    transform: str  # "log10" | "identity"
    mean: float
    std: float
    mode: str = "minimize"
    target: float | None = None

    def _t(self, p: torch.Tensor) -> torch.Tensor:
        if self.transform == "log10":
            return torch.log10(p.clamp_min(1e-30))
        return p

    def value(self, parts: torch.Tensor) -> torch.Tensor:
        """Physical property value (B,)."""
        return self.property_fn(parts)

    def normalize(self, p: torch.Tensor) -> torch.Tensor:
        """Property -> y_norm (B,). Differentiable."""
        tp = self._t(p)
        if self.mode == "target":
            if self.target is None:
                raise ValueError("target mode requires a target value")
            tt = math.log10(max(self.target, 1e-30)) if self.transform == "log10" else self.target
            return torch.abs(tp - tt) / self.std
        z = (tp - self.mean) / self.std
        return z if self.mode == "minimize" else -z

    def invert(self, y_norm: torch.Tensor) -> torch.Tensor:
        """y_norm -> physical property (for reporting). Valid for minimize/maximize.

        For `target` mode this is not a true inverse (abs loses sign); callers
        should report the property value directly instead.
        """
        tp = (self.mean - y_norm * self.std) if self.mode == "maximize" \
            else (y_norm * self.std + self.mean)
        if self.transform == "log10":
            return 10.0 ** tp
        return tp


def compute_property_norm(processed_h5: str, property_name: str,
                          transform: str | None = None,
                          chunk: int = 512) -> tuple[float, float]:
    """(mean, std) of transform(property) over the TRUE particles in a processed
    flow dataset. Gives each property a principled z-score (for norm_emit_4d this
    reproduces the flow's stored target_mean/std)."""
    fn, default_t = PROPERTY_REGISTRY[property_name]
    t = transform or default_t
    vals: list[np.ndarray] = []
    with h5py.File(processed_h5, "r") as h:
        parts = h["particles"]  # (N, P, 6) physical
        n = parts.shape[0]
        for i in range(0, n, chunk):
            block = torch.tensor(np.asarray(parts[i:i + chunk]), dtype=torch.float64)
            with torch.no_grad():
                vals.append(fn(block).cpu().numpy())
    arr = np.concatenate(vals)
    if t == "log10":
        arr = np.log10(np.clip(arr, 1e-30, None))
    return float(arr.mean()), float(max(arr.std(), 1e-12))


def build_reward_spec(property_name: str, mode: str = "minimize", *,
                      flow=None, processed_h5: str | None = None,
                      transform: str | None = None,
                      target: float | None = None) -> RewardSpec:
    """Build a RewardSpec. For norm_emit_4d, reuse the flow's stored
    target_mean/std (matches the lofi v1 baseline exactly, no data scan). For
    other properties, derive the z-score from `processed_h5`."""
    if property_name not in PROPERTY_REGISTRY:
        raise KeyError(f"unknown property {property_name!r}; "
                       f"choices: {sorted(PROPERTY_REGISTRY)}")
    fn, default_t = PROPERTY_REGISTRY[property_name]
    t = transform or default_t

    if property_name == "norm_emit_4d" and flow is not None and t == "log10":
        mean = float(flow.target_mean.item())
        std = float(flow.target_std.item())
    else:
        if processed_h5 is None:
            raise ValueError(
                f"property {property_name!r} needs `processed_h5` to compute its "
                "z-score (only norm_emit_4d can reuse the flow's stored stats)")
        mean, std = compute_property_norm(processed_h5, property_name, transform=t)

    return RewardSpec(name=property_name, property_fn=fn, transform=t,
                      mean=mean, std=std, mode=mode, target=target)


# ----- joint aspect+tilt control via the (s1,s2) shape-vector target ----------

def aspect_tilt_to_s(aspect: float, tilt_deg: float) -> tuple[float, float]:
    """Desired (eigen aspect ≥1, tilt angle deg) -> target shape vector (s1*,s2*).
    r* = (a²−1)/(a²+1); s1*=r*·cos(2θ*), s2*=r*·sin(2θ*)."""
    a2 = float(aspect) ** 2
    r = (a2 - 1.0) / (a2 + 1.0)
    th = math.radians(float(tilt_deg))
    return r * math.cos(2.0 * th), r * math.sin(2.0 * th)


def s_to_aspect_tilt(s1: torch.Tensor, s2: torch.Tensor):
    """Shape vector (s1,s2) -> (eigen aspect, tilt deg). Tensor-friendly (eval)."""
    s1 = torch.as_tensor(s1, dtype=torch.float64)
    s2 = torch.as_tensor(s2, dtype=torch.float64)
    r = torch.sqrt(s1 * s1 + s2 * s2).clamp(max=1.0 - 1e-9)
    aspect = torch.sqrt((1.0 + r) / (1.0 - r))
    tilt_deg = 0.5 * torch.atan2(s2, s1) * (180.0 / math.pi)
    return aspect, tilt_deg


@dataclass
class ShapeTargetSpec:
    """Joint aspect+tilt reward: y_norm = ‖(s1,s2) − (s1*,s2*)‖ / scale, so the
    env reward = −y_norm drives the beam to a target ellipse (shape AND
    orientation) at once. `mean`/`std` are duck-typed for the env's obs scaling
    (the obs/last-y_norm channel is the O(1) tracking distance, already ~unit)."""

    target_s1: float
    target_s2: float
    scale: float = 0.3        # ~dataset std of the shape vector -> O(1) reward
    name: str = "shape_target"
    mean: float = 0.0
    std: float = 1.0

    @classmethod
    def from_aspect_tilt(cls, aspect: float, tilt_deg: float,
                         scale: float = 0.3) -> "ShapeTargetSpec":
        s1, s2 = aspect_tilt_to_s(aspect, tilt_deg)
        return cls(target_s1=s1, target_s2=s2, scale=scale)

    def reward_ynorm(self, parts: torch.Tensor) -> torch.Tensor:
        """(B,) tracking distance / scale. Differentiable w.r.t. the particles."""
        s1 = _s1(parts)
        s2 = _s2(parts)
        d = torch.sqrt((s1 - self.target_s1) ** 2
                       + (s2 - self.target_s2) ** 2 + 1e-30)
        return d / self.scale

    def achieved(self, parts: torch.Tensor):
        """(s1, s2) achieved by the beam (for eval reporting)."""
        return _s1(parts), _s2(parts)
