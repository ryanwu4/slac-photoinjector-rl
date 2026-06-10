"""
Batched, torch-only, differentiable photoinjector env — the surrogate-agnostic
base shared by every surrogate. Implements the env API expected by
`diffrl.SHAC` and `diffrl.BPTT` (see `DiffRL/envs/dflex_env.py` for the upstream
contract).

State per env i: (knobs_i in [0,1]^5, distgen_i in [0,1]^6, step_count_i).
Action: 5-D direction in [-1, 1]; clipped, scaled by `action_scale`, added to
knobs, clipped back to [0, 1] — gradient flows through the clamp inside the box.
Reward: -y_norm, where y_norm = surrogate-derived scalar for (knobs, distgen).
Obs: concat(knobs, last_y_norm) of shape (num_envs, 6).
Episodes end at `episode_length` steps; on `done` the env auto-resets.

The surrogate is **injected** (any frozen `torch.nn.Module` whose forward maps
`concat(knobs, distgen) -> (B, 1)` or `(B,)`), making the base adaptable: a new
surrogate plugs in by either passing a pre-loaded `surrogate` (+ target_mean /
target_std), overriding `_load_surrogate` to load from a checkpoint, or
overriding the `_forward_surrogate` reward seam (as the conditional-flow
`FlowBunchEnv` does). The base itself has no knowledge of any concrete surrogate.

The surrogate is frozen (requires_grad_(False)) but called *without*
`torch.no_grad()` when `no_grad=False`, so input gradients propagate from the
reward back to the action that produced the state. This is the property that
makes SHAC/BPTT viable here.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

N_KNOB = 5
N_DISTGEN = 6
N_INPUT = N_KNOB + N_DISTGEN  # 11; matches the surrogate's input_dim
OBS_DIM = N_KNOB + 1          # 5 knobs + 1 last y_norm
ACTION_DIM = N_KNOB


class DiffPhotoinjectorEnv:
    """Differentiable batched env over an injected, frozen surrogate.

    Constructor signature mirrors DiffRL's `DFlexEnv` so SHAC/BPTT can
    instantiate it via the standard kwargs.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device = "cuda:0",
        render: bool = False,
        seed: int = 0,
        episode_length: int = 64,
        stochastic_init: bool = True,
        MM_caching_frequency: int = 1,  # unused; kept for API compatibility
        no_grad: bool = False,
        *,
        surrogate_ckpt: str | Path | None = None,
        norm_json: str | Path | None = None,
        surrogate: torch.nn.Module | None = None,
        target_mean: float | None = None,
        target_std: float | None = None,
        action_scale: float = 0.05,
        distgen_drift_std: float = 0.0,
    ):
        del MM_caching_frequency, render
        self.num_envs = int(num_envs)
        self.num_obs = OBS_DIM
        self.num_actions = ACTION_DIM
        self.episode_length = int(episode_length)
        self.device = torch.device(device)
        self.stochastic_init = bool(stochastic_init)
        self.no_grad = bool(no_grad)
        self.action_scale = float(action_scale)
        self.distgen_drift_std = float(distgen_drift_std)

        # Seeded generator for reproducibility (independent of global torch
        # seed; SHAC's seeding() also runs).
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(int(seed))

        # ---- resolve surrogate + normalization ------------------------------
        # The base is surrogate-agnostic: either a pre-loaded `surrogate`
        # (+ target_mean/std) is injected, or a subclass loads one from a
        # checkpoint via `_load_surrogate`.
        if surrogate is None:
            surrogate, target_mean, target_std = self._load_surrogate(
                surrogate_ckpt, norm_json)

        if target_mean is None or target_std is None:
            raise ValueError("target_mean / target_std must be provided")

        surrogate = surrogate.to(self.device).eval()
        for p in surrogate.parameters():
            p.requires_grad_(False)
        self._surrogate = surrogate
        self._target_mean = float(target_mean)
        self._target_std = float(target_std)

        # ---- state buffers --------------------------------------------------
        self._knobs = torch.zeros((self.num_envs, N_KNOB), dtype=torch.float32,
                                  device=self.device)
        self._distgen = torch.zeros((self.num_envs, N_DISTGEN),
                                    dtype=torch.float32, device=self.device)
        self._step_count = torch.zeros((self.num_envs,), dtype=torch.long,
                                       device=self.device)
        self._last_y_norm = torch.zeros((self.num_envs,), dtype=torch.float32,
                                        device=self.device)

        self.reset()

    # ----- surrogate-loading hook (overridable) ------------------------------

    def _load_surrogate(self, surrogate_ckpt: str | Path | None,
                        norm_json: str | Path | None
                        ) -> tuple[torch.nn.Module, float, float]:
        """Load a surrogate from a checkpoint. The base is surrogate-agnostic
        and has no concrete model to load, so it requires an injected
        `surrogate`; subclasses (e.g. the legacy MLP env) override this to load
        a specific model + normalization from disk."""
        raise NotImplementedError(
            "Base DiffPhotoinjectorEnv requires a pre-loaded `surrogate` "
            "(+ target_mean / target_std). To load from a checkpoint, subclass "
            "and override `_load_surrogate`.")

    # ----- internal helpers --------------------------------------------------

    def _forward_surrogate(self, knobs: torch.Tensor,
                           distgen: torch.Tensor) -> torch.Tensor:
        """Run the frozen surrogate. Keeps grad iff caller's tensors have it."""
        x = torch.cat([knobs, distgen], dim=-1)
        if self.no_grad:
            with torch.no_grad():
                y = self._surrogate(x)
        else:
            y = self._surrogate(x)
        return y.squeeze(-1)

    def _compute_obs(self) -> torch.Tensor:
        return torch.cat([self._knobs, self._last_y_norm.unsqueeze(-1)], dim=-1)

    def _sample_uniform(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, generator=self._rng, dtype=torch.float32,
                          device=self.device)

    # ----- DiffRL env API ----------------------------------------------------

    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Resample knobs/distgen for the given envs (or all). Returns obs.

        For a partial reset, gradient on non-reset envs is preserved by
        constructing the new state via `torch.where(mask, fresh, current)`.
        The naive `self._knobs = self._knobs.detach(); self._knobs[env_ids]
        = new` pattern would silently kill grad for survivors — currently
        masked because our episodes terminate synchronously, but a footgun
        as soon as that assumption changes (early termination, async resets).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = int(env_ids.numel())
        if n == 0:
            return self._compute_obs()

        if self.stochastic_init:
            new_knobs = self._sample_uniform((n, N_KNOB))
            new_dist = self._sample_uniform((n, N_DISTGEN))
        else:
            new_knobs = torch.full((n, N_KNOB), 0.5, device=self.device)
            new_dist = torch.full((n, N_DISTGEN), 0.5, device=self.device)

        if n == self.num_envs:
            # All envs reset → no surviving graph to preserve; cheapest path.
            self._knobs = new_knobs
            self._distgen = new_dist
        else:
            mask = torch.zeros(self.num_envs, dtype=torch.bool,
                               device=self.device)
            mask[env_ids] = True
            # Detached scaffolds with `new_*` placed at env_ids; arbitrary
            # values elsewhere because torch.where masks them out.
            knobs_full = self._knobs.detach().clone()
            knobs_full[env_ids] = new_knobs
            self._knobs = torch.where(mask.unsqueeze(-1), knobs_full,
                                      self._knobs)
            dist_full = self._distgen.detach().clone()
            dist_full[env_ids] = new_dist
            self._distgen = torch.where(mask.unsqueeze(-1), dist_full,
                                        self._distgen)

        self._step_count[env_ids] = 0

        with torch.no_grad():
            y = self._forward_surrogate(self._knobs.detach(),
                                        self._distgen.detach())
        # Survivors keep their pre-reset y_norm; reset envs get the fresh one.
        if n == self.num_envs:
            self._last_y_norm = y
        else:
            mask = torch.zeros(self.num_envs, dtype=torch.bool,
                               device=self.device)
            mask[env_ids] = True
            self._last_y_norm = torch.where(mask, y, self._last_y_norm)
        return self._compute_obs()

    def initialize_trajectory(self) -> torch.Tensor:
        """Cut gradient between epochs; return current obs.

        SHAC / BPTT call this at the start of `compute_actor_loss` so the
        graph from a previous outer iteration is not retained.
        """
        self._knobs = self._knobs.detach()
        self._distgen = self._distgen.detach()
        self._last_y_norm = self._last_y_norm.detach()
        return self._compute_obs()

    def clear_grad(self) -> None:
        """Hard reset of the gradient graph (called by SHAC/BPTT's initialize_env)."""
        self._knobs = self._knobs.detach()
        self._distgen = self._distgen.detach()
        self._last_y_norm = self._last_y_norm.detach()

    def step(self, action: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Apply Δaction, query surrogate, return (obs, reward, done, info).

        Gradient flows: action → knobs (via clamp inside the box) → surrogate
        input → y_norm → reward = -y_norm → obs[..., -1].
        """
        if action.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(
                f"action shape {tuple(action.shape)} != "
                f"({self.num_envs}, {ACTION_DIM})")

        # Knob accumulator. clamp is differentiable inside [0,1]; grad=0 on
        # the boundary, acceptable since the surrogate was trained on [0,1]^11.
        knobs_next = torch.clamp(
            self._knobs + self.action_scale * action, 0.0, 1.0)

        # Optional distgen drift (hidden context); detached so it adds no
        # gradient signal but does inject stochasticity into the dynamics.
        if self.distgen_drift_std > 0.0:
            with torch.no_grad():
                drift = torch.randn(
                    self._distgen.shape, generator=self._rng,
                    dtype=torch.float32, device=self.device,
                ) * self.distgen_drift_std
                self._distgen = torch.clamp(self._distgen + drift, 0.0, 1.0)

        y_norm = self._forward_surrogate(knobs_next, self._distgen)

        # Commit the new state.
        self._knobs = knobs_next
        self._last_y_norm = y_norm
        self._step_count = self._step_count + 1

        reward = -y_norm
        done = self._step_count >= self.episode_length

        # Pre-reset obs — needed by SHAC's terminal-value bootstrap. Detached
        # because SHAC consumes it only inside no-grad code.
        obs_pre = self._compute_obs().detach()

        # Reset any envs that finished. This re-samples their knobs/distgen
        # and clears step_count; new state is detached (leaf tensor).
        done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if done_env_ids.numel() > 0:
            self.reset(done_env_ids)

        obs = self._compute_obs()
        info: dict[str, Any] = {"obs_before_reset": obs_pre}
        return obs, reward, done, info

    def render(self) -> None:
        return None

    # ----- helpers exposed for evaluation / plotting ------------------------

    @torch.no_grad()
    def physical_emit(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Invert log10 + z-score → norm_emit_4d in m²."""
        return 10.0 ** (y_norm * self._target_std + self._target_mean)
