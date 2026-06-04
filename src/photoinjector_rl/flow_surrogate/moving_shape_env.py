"""
Moving-target (goal-conditioned) controller env for beam (aspect, tilt).

`MovingShapeEnv` tracks a time-varying shape-vector setpoint (s1*(t), s2*(t)) that
is sampled per episode (curriculum: static -> step -> smooth). The setpoint is in
the observation, so ONE policy learns to track any schedule:

    obs (9-D) = [knobs(5), s1_cur, s2_cur, s1*(t), s2*(t)]
    reward    = -‖(s1(t),s2(t)) - (s1*(t),s2*(t))‖ / scale   (dense, differentiable)

The per-step target is indexed by the env's `_step_count` (the within-episode
step) so it tracks the EPISODE, not SHAC's shorter horizon. `_step_count` is NOT
reset by `initialize_trajectory` (inherited), which only cuts the graph.

Subclasses `FlowBunchEnv` and reuses its flow sampling / reparam path / reset+step
graph logic. `diffrl/` and `emittance_target/` are untouched (actor/critic/obs_rms
auto-size from `num_obs`). Current-shape obs features are detached sensors; the
knobs in the obs stay differentiable (so diffrl's dynamics chain is intact) and
the reward (the optimized signal) is fully differentiable w.r.t. the action.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from photoinjector_rl.emittance_target.diff_env import ACTION_DIM, N_KNOB

from . import properties
from .diff_env import FlowBunchEnv
from .model import ConditionalAffineFlow
from .properties import ShapeTargetSpec
from .shape_targets import CurriculumState
from .vec_env import FlowSurrogateVecEnv

GOAL_OBS_DIM = N_KNOB + 4  # knobs(5) + s1_cur + s2_cur + s1* + s2* = 9


class MovingShapeEnv(FlowBunchEnv):
    """Goal-conditioned env tracking a time-varying (s1*,s2*) shape setpoint."""

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
        curriculum: CurriculumState | None = None,
        scale: float = 0.3,
        action_rate_penalty: float = 0.0,
        fixed_target_traj: np.ndarray | None = None,
    ):
        # ---- state needed by reset()/_compute_obs(), set BEFORE super().__init__
        dev = torch.device(device)
        self._T = int(episode_length)
        self._curriculum = curriculum or CurriculumState()  # holds config (incl. r_max)
        self._traj_rng = np.random.default_rng(int(seed) + 9973)
        self._action_rate_penalty = float(action_rate_penalty)
        self._fixed_target_traj = (None if fixed_target_traj is None
                                   else np.asarray(fixed_target_traj, dtype=np.float64))
        self._target_traj = torch.zeros((int(num_envs), self._T, 2), device=dev)
        self._s1_cur = torch.zeros(int(num_envs), device=dev)   # current shape (obs sensor)
        self._s2_cur = torch.zeros(int(num_envs), device=dev)
        self._s1s2_info = torch.zeros((int(num_envs), 2), device=dev)  # pre-reset (info)
        self._prev_action = torch.zeros((int(num_envs), ACTION_DIM), device=dev)
        # spec carries only scale/mean/std (duck-typed); reward computed here.
        spec = ShapeTargetSpec(target_s1=0.0, target_s2=0.0, scale=float(scale))

        super().__init__(
            num_envs, device=device, render=render, seed=seed,
            episode_length=episode_length, stochastic_init=stochastic_init,
            MM_caching_frequency=MM_caching_frequency, no_grad=no_grad,
            flow_ckpt=flow_ckpt, flow=flow, norm_json=norm_json,
            processed_h5=processed_h5, n_particles=n_particles,
            action_scale=action_scale, distgen_drift_std=distgen_drift_std,
            reward_spec=spec,
        )
        self.num_obs = GOAL_OBS_DIM  # override parent's 6 -> 9 (diffrl auto-sizes)

    # ----- target trajectory sampling ---------------------------------------

    def _sample_traj_for(self, ids) -> None:
        for i in ids:
            if self._fixed_target_traj is not None:
                traj = self._fixed_target_traj
                if traj.shape[0] < self._T:                 # pad with the last setpoint
                    pad = np.repeat(traj[-1:], self._T - traj.shape[0], axis=0)
                    traj = np.concatenate([traj, pad], axis=0)
                elif traj.shape[0] > self._T:
                    traj = traj[:self._T]
            else:
                traj = self._curriculum.sample_trajectory(self._traj_rng, self._T)
            self._target_traj[int(i)] = torch.tensor(traj, dtype=torch.float32,
                                                     device=self.device)

    def _target_at_step(self):
        idx = self._step_count.clamp(max=self._T - 1)
        ar = torch.arange(self.num_envs, device=self.device)
        t = self._target_traj[ar, idx]                     # (B, 2)
        return t[:, 0], t[:, 1]

    # ----- overridden seams --------------------------------------------------

    def _compute_obs(self) -> torch.Tensor:
        ts1, ts2 = self._target_at_step()
        return torch.cat([
            self._knobs,                                   # (B,5) differentiable
            self._s1_cur.unsqueeze(-1), self._s2_cur.unsqueeze(-1),  # detached sensor
            ts1.unsqueeze(-1), ts2.unsqueeze(-1),          # setpoint (constant)
        ], dim=-1)                                         # (B, 9)

    def _sample_property_ynorm(self, knobs: torch.Tensor,
                               distgen: torch.Tensor) -> torch.Tensor:
        x = torch.cat([knobs, distgen], dim=-1)
        parts = self._flow.sample_physical(x, self._n_particles)
        s1 = properties._s1(parts)
        s2 = properties._s2(parts)
        # detached current-shape sensors for the obs (knobs stay differentiable).
        # Updated EVERY forward (incl. the auto-reset's) so the next obs reflects
        # the post-reset state of done envs.
        self._s1_cur = s1.detach()
        self._s2_cur = s2.detach()
        if not getattr(self, "_in_reset", False):
            # pre-reset caches for info/eval (guarded like FlowBunchEnv._last_property
            # so the auto-reset forward doesn't overwrite the terminal achieved shape).
            self._last_property = properties._eigen_aspect(parts).detach()
            self._s1s2_info = torch.stack([s1, s2], dim=-1).detach()
        ts1, ts2 = self._target_at_step()                  # target at current step
        d = torch.sqrt((s1 - ts1) ** 2 + (s2 - ts2) ** 2 + 1e-30)
        return d / self._reward_spec.scale                 # (B,) y_norm (reward=-y_norm)

    def reset(self, env_ids=None):
        super().reset(env_ids)                             # resets knobs/distgen/_step_count
        ids = (range(self.num_envs) if env_ids is None
               else [int(i) for i in env_ids.detach().cpu().tolist()])
        self._sample_traj_for(ids)
        return self._compute_obs()                         # obs with the NEW setpoints

    def step(self, action: torch.Tensor):
        obs, reward, done, info = super().step(action)
        if self._action_rate_penalty > 0.0:
            reward = reward - self._action_rate_penalty * torch.linalg.norm(
                action - self._prev_action, dim=-1)
            self._prev_action = action.detach()
        # Report the PRE-reset achieved shape (guarded cache), so on a done step
        # info reflects the terminal beam, not the freshly-reset one.
        s = self._s1s2_info
        aspect, tilt = properties.s_to_aspect_tilt(s[:, 0], s[:, 1])
        info["shape_s1s2"] = s.clone()
        info["aspect"] = aspect.to(self.device, torch.float32)
        info["tilt_deg"] = tilt.to(self.device, torch.float32)
        return obs, reward, done, info


class MovingShapeVecEnv(FlowSurrogateVecEnv):
    """SB3 VecEnv over MovingShapeEnv (9-D goal-conditioned obs) for PPO."""

    def __init__(self, n_envs: int, *, flow_ckpt, norm_json, device, seed,
                 curriculum: CurriculumState | None = None, processed_h5=None,
                 n_particles: int = 512, episode_length: int = 64,
                 action_scale: float = 0.05, distgen_drift_std: float = 0.0,
                 scale: float = 0.3):
        self.device = torch.device(device)
        self._terminal_emit_bonus = 0.0
        self._env = MovingShapeEnv(
            num_envs=int(n_envs), device=self.device, seed=int(seed),
            episode_length=int(episode_length), stochastic_init=True, no_grad=True,
            flow_ckpt=flow_ckpt, norm_json=norm_json, processed_h5=processed_h5,
            n_particles=int(n_particles), action_scale=float(action_scale),
            distgen_drift_std=float(distgen_drift_std), curriculum=curriculum,
            scale=float(scale),
        )
        nobs = self._env.num_obs
        observation_space = spaces.Box(
            low=np.array([0.0] * N_KNOB + [-np.inf] * (nobs - N_KNOB), dtype=np.float32),
            high=np.array([1.0] * N_KNOB + [np.inf] * (nobs - N_KNOB), dtype=np.float32),
            dtype=np.float32,
        )
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        VecEnv.__init__(self, int(n_envs), observation_space, action_space)
        self._actions = None
