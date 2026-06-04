"""
Differentiable batched env whose reward is a property COMPUTED from the flow's
sampled output bunch — the substrate for first-order MBRL over arbitrary beam
properties.

It subclasses `emittance_target.diff_env.DiffPhotoinjectorEnv` and overrides only
the reward seam (`_forward_surrogate`): instead of a scalar-MLP call, it samples a
cloud from the conditional flow, computes a (differentiable) property, and z-scores
it to `y_norm` (env reward = -y_norm, inherited). All graph management — detached
`initialize_trajectory`, `torch.where` partial-reset grad-safety, frozen-but-
differentiable surrogate, `obs_before_reset` — is inherited unchanged, so the env
plugs into `emittance_target.diffrl.{SHAC,BPTT}` exactly like the v1 env.

Gradient path (per step, no torch.no_grad): action -> knobs (clamp) ->
flow.condition_net -> reparameterized sample (z ~ N(0,I) constant) -> de-standardize
-> property (e.g. norm_emit_4d via compute_emittance_torch) -> z-score -> reward.
Flow weights are frozen, so only the actor learns.
"""
from __future__ import annotations

import torch

from photoinjector_rl.emittance_target.diff_env import DiffPhotoinjectorEnv

from .model import ConditionalAffineFlow
from .properties import RewardSpec, build_reward_spec


class FlowBunchEnv(DiffPhotoinjectorEnv):
    """DiffPhotoinjectorEnv variant whose reward comes from a flow-sampled bunch."""

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
        property: str = "norm_emit_4d",
        reward_mode: str = "minimize",
        transform: str | None = None,
        target: float | None = None,
        n_particles: int = 512,
        action_scale: float = 0.05,
        distgen_drift_std: float = 0.0,
    ):
        # Load flow + build reward spec BEFORE super().__init__: the parent's
        # __init__ ends with reset(), which calls our _forward_surrogate, so
        # these attributes must already exist.
        if flow is None:
            if flow_ckpt is None:
                raise ValueError("pass `flow` (instance) or `flow_ckpt` (path)")
            flow = ConditionalAffineFlow.load_from_checkpoint(
                str(flow_ckpt), map_location=torch.device(device))
        flow = flow.to(device).eval()
        self._flow = flow
        self._n_particles = int(n_particles)

        if processed_h5 is None and norm_json is not None:
            processed_h5 = str(norm_json).replace("_norm.json", ".h5")
        self._reward_spec: RewardSpec = build_reward_spec(
            property, reward_mode, flow=flow, processed_h5=processed_h5,
            transform=transform, target=target,
        )

        # Parent freezes the flow (requires_grad_(False)) and skips MLP loading
        # because we pass a pre-built `surrogate`. target_mean/std come from the
        # reward spec so the inherited physical_emit / obs scaling stay consistent.
        super().__init__(
            num_envs, device=device, render=render, seed=seed,
            episode_length=episode_length, stochastic_init=stochastic_init,
            MM_caching_frequency=MM_caching_frequency, no_grad=no_grad,
            surrogate=flow, target_mean=self._reward_spec.mean,
            target_std=self._reward_spec.std, action_scale=action_scale,
            distgen_drift_std=distgen_drift_std,
        )

    @property
    def reward_spec(self) -> RewardSpec:
        return self._reward_spec

    # ----- overridden reward seam -------------------------------------------

    def _sample_property_ynorm(self, knobs: torch.Tensor,
                               distgen: torch.Tensor) -> torch.Tensor:
        x = torch.cat([knobs, distgen], dim=-1)                       # (B, 11)
        parts = self._flow.sample_physical(x, self._n_particles)      # (B, n, 6) physical
        p = self._reward_spec.value(parts)                            # (B,)
        return self._reward_spec.normalize(p)                         # (B,) y_norm

    def _forward_surrogate(self, knobs: torch.Tensor,
                           distgen: torch.Tensor) -> torch.Tensor:
        """Reward quantity = normalized computed bunch property. Differentiable
        w.r.t. `knobs` (hence the action) unless `self.no_grad`."""
        if self.no_grad:
            with torch.no_grad():
                return self._sample_property_ynorm(knobs, distgen)
        return self._sample_property_ynorm(knobs, distgen)

    # ----- reporting helpers (eval) -----------------------------------------

    @torch.no_grad()
    def physical_property(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Invert the z-score (+ transform) → physical property value."""
        return self._reward_spec.invert(y_norm)

    @torch.no_grad()
    def physical_emit(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Alias kept for the surrogate-side eval harness (which calls
        `physical_emit`). Returns the inverted property in its physical units:
        m^2 for the default norm_emit_4d spec, the property's own units
        otherwise. Use `physical_property` for the unit-agnostic name."""
        return self._reward_spec.invert(y_norm)
