"""
Joint aspect+tilt control env: drive the beam's transverse shape vector
(s1, s2) = ((σx²−σy²)/(σx²+σy²), 2·cov_xy/(σx²+σy²)) to a target, i.e. command a
desired (eigen aspect ratio, tilt angle) at once.

`ShapeTargetEnv` subclasses `FlowBunchEnv` and overrides only the reward seam:
reward = −‖(s1,s2) − (s1*,s2*)‖ / scale. Everything else (flow sampling, the
differentiable reparam path, reset/step graph management, the pre-reset cache for
eval) is inherited. `diffrl/` and `emittance_target/` are untouched — the obs is
still 6-D `[knobs, last_y_norm]` (the shape target is fixed per run), and the
actor/critic auto-size from `num_obs`.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import torch

from . import properties
from .diff_env import FlowBunchEnv
from .model import ConditionalAffineFlow
from .properties import ShapeTargetSpec


class ShapeTargetEnv(FlowBunchEnv):
    """FlowBunchEnv variant whose reward targets the (s1,s2) shape vector."""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device = "cuda:0",
        render: bool = False,
        seed: int = 0,
        episode_length: int = 64,
        stochastic_init: bool = True,
        MM_caching_frequency: int = 1,
        no_grad: bool = False,
        *,
        flow_ckpt: str | None = None,
        flow: ConditionalAffineFlow | None = None,
        norm_json: str | None = None,
        processed_h5: str | None = None,
        n_particles: int = 512,
        action_scale: float = 0.05,
        distgen_drift_std: float = 0.0,
        target_aspect: float = 2.0,
        target_tilt_deg: float = 0.0,
        scale: float = 0.3,
    ):
        spec = ShapeTargetSpec.from_aspect_tilt(target_aspect, target_tilt_deg, scale=scale)
        super().__init__(
            num_envs, device=device, render=render, seed=seed,
            episode_length=episode_length, stochastic_init=stochastic_init,
            MM_caching_frequency=MM_caching_frequency, no_grad=no_grad,
            flow_ckpt=flow_ckpt, flow=flow, norm_json=norm_json,
            processed_h5=processed_h5, n_particles=n_particles,
            action_scale=action_scale, distgen_drift_std=distgen_drift_std,
            reward_spec=spec,
        )
        self.target_aspect = float(target_aspect)
        self.target_tilt_deg = float(target_tilt_deg)
        self._last_s1s2 = torch.zeros((self.num_envs, 2), device=self.device)

    # ----- overridden reward seam: distance to the (s1*,s2*) target ----------

    def _sample_property_ynorm(self, knobs: torch.Tensor,
                               distgen: torch.Tensor) -> torch.Tensor:
        x = torch.cat([knobs, distgen], dim=-1)
        parts = self._flow.sample_physical(x, self._n_particles)      # (B, n, 6)
        s1 = properties._s1(parts)
        s2 = properties._s2(parts)
        spec: ShapeTargetSpec = self._reward_spec
        if not getattr(self, "_in_reset", False):
            self._last_s1s2 = torch.stack([s1, s2], dim=-1).detach()  # (B, 2)
            # generic scalar for the existing info["property_value"] path
            self._last_property = properties._eigen_aspect(parts).detach()
        d = torch.sqrt((s1 - spec.target_s1) ** 2
                       + (s2 - spec.target_s2) ** 2 + 1e-30)
        return d / spec.scale                                          # (B,) y_norm

    def step(self, action: torch.Tensor):
        obs, reward, done, info = super().step(action)
        s = self._last_s1s2                                            # (B,2) pre-reset
        aspect, tilt = properties.s_to_aspect_tilt(s[:, 0], s[:, 1])
        info["shape_s1s2"] = s.detach().clone()
        info["aspect"] = aspect.to(self.device, torch.float32)
        info["tilt_deg"] = tilt.to(self.device, torch.float32)
        return obs, reward, done, info
