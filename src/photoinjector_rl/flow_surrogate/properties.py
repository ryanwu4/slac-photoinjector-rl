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


def _particle_energy(parts: torch.Tensor) -> torch.Tensor:
    """Per-particle total energy [eV]: sqrt(px^2+py^2+pz^2 + mc^2^2)."""
    p2 = parts[:, :, 3] ** 2 + parts[:, :, 4] ** 2 + parts[:, :, 5] ** 2
    return torch.sqrt(p2 + MC2 ** 2)


def _mean_energy(parts: torch.Tensor) -> torch.Tensor:
    return _particle_energy(parts).mean(dim=1)


def _energy_spread(parts: torch.Tensor) -> torch.Tensor:
    e = _particle_energy(parts)
    return e.std(dim=1, unbiased=False) / e.mean(dim=1)


# name -> (fn, default transform). transform chooses the space the z-score /
# target live in: log10 for positive, right-skewed quantities; identity else.
PROPERTY_REGISTRY: dict[str, tuple[Callable[[torch.Tensor], torch.Tensor], str]] = {
    "norm_emit_4d": (_norm_emit_4d, "log10"),
    "norm_emit_x": (_norm_emit_x, "log10"),
    "norm_emit_y": (_norm_emit_y, "log10"),
    "sigma_x": (_sigma_x, "log10"),
    "sigma_y": (_sigma_y, "log10"),
    "sigma_z": (_sigma_z, "log10"),
    "mean_energy": (_mean_energy, "identity"),
    "energy_spread": (_energy_spread, "log10"),
}


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
