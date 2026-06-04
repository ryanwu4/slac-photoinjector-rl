"""
GPU-batched SB3 ``VecEnv`` over the conditional-flow surrogate, for PPO.

The flow analog of ``emittance_target.vec_env.SurrogateVecEnv``: it wraps
``FlowBunchEnv`` (with ``no_grad=True`` — PPO is a black-box learner, so we do
NOT keep the flow's autograd graph) and advances all N sub-envs as a single
batched flow sample + property computation per step. Zero IPC, no per-env Python
stepping. The observation/action spaces are identical to the v1 env so a policy
trained here is directly comparable to (and loadable like) the MLP-surrogate
PPO policies.

Per-step ``info`` mirrors the v1 env (``emit_m2`` = the inverted reward property,
``log_emit_norm``, ``knobs_phys``) so SB3's VecMonitor / metrics callbacks work
unchanged. Episodes end only by time limit → every ``done`` is a truncation; we
set ``TimeLimit.truncated`` + ``terminal_observation`` for correct PPO
bootstrapping.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from photoinjector_rl.emittance_target import SETTING_BOUNDS, SETTING_KEYS
from photoinjector_rl.emittance_target.diff_env import ACTION_DIM, N_KNOB, OBS_DIM

from .diff_env import FlowBunchEnv

_KNOB_LO = np.array([SETTING_BOUNDS[k][0] for k in SETTING_KEYS[:N_KNOB]],
                    dtype=np.float32)
_KNOB_HI = np.array([SETTING_BOUNDS[k][1] for k in SETTING_KEYS[:N_KNOB]],
                    dtype=np.float32)
_KNOB_RANGE = _KNOB_HI - _KNOB_LO


def _resolve_indices(indices: None | int | Iterable[int], n: int) -> list[int]:
    if indices is None:
        return list(range(n))
    if isinstance(indices, int):
        return [indices]
    return list(indices)


class FlowSurrogateVecEnv(VecEnv):
    """Batched-on-GPU SB3 VecEnv backed by one ``FlowBunchEnv`` (no_grad)."""

    def __init__(
        self,
        n_envs: int,
        *,
        flow_ckpt: str | Path,
        norm_json: str | Path,
        device: str | torch.device,
        seed: int,
        processed_h5: str | None = None,
        property: str = "norm_emit_4d",
        reward_mode: str = "minimize",
        target: float | None = None,
        n_particles: int = 512,
        episode_length: int = 64,
        action_scale: float = 0.05,
        distgen_drift_std: float = 0.0,
        terminal_emit_bonus: float = 0.0,
        stochastic_init: bool = True,
    ):
        self.device = torch.device(device)
        self._terminal_emit_bonus = float(terminal_emit_bonus)
        self._env = FlowBunchEnv(
            num_envs=int(n_envs),
            device=self.device,
            seed=int(seed),
            episode_length=int(episode_length),
            stochastic_init=bool(stochastic_init),
            no_grad=True,
            flow_ckpt=flow_ckpt,
            norm_json=norm_json,
            processed_h5=processed_h5,
            property=property,
            reward_mode=reward_mode,
            target=target,
            n_particles=int(n_particles),
            action_scale=float(action_scale),
            distgen_drift_std=float(distgen_drift_std),
        )

        observation_space = spaces.Box(
            low=np.array([0.0] * N_KNOB + [-np.inf], dtype=np.float32),
            high=np.array([1.0] * N_KNOB + [np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32,
        )
        super().__init__(int(n_envs), observation_space, action_space)
        self._actions: np.ndarray | None = None

    def _to_np(self, t: torch.Tensor) -> np.ndarray:
        return t.detach().to("cpu", torch.float32).numpy()

    # ----- VecEnv API -------------------------------------------------------

    def reset(self) -> np.ndarray:
        obs = self._env.reset()
        self.reset_infos = [{} for _ in range(self.num_envs)]
        return self._to_np(obs)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        act = np.clip(self._actions, -1.0, 1.0).astype(np.float32)
        act_t = torch.from_numpy(act).to(self.device)

        obs_t, rew_t, done_t, info = self._env.step(act_t)

        obs = self._to_np(obs_t)
        rewards = self._to_np(rew_t)
        dones = done_t.detach().to("cpu").numpy().astype(bool)

        pre = self._to_np(info["obs_before_reset"])
        knobs_norm = pre[:, :N_KNOB]
        knobs_phys = _KNOB_LO + knobs_norm * _KNOB_RANGE
        y_norm = pre[:, -1]
        emit_m2 = self._to_np(
            self._env.physical_emit(info["obs_before_reset"][:, -1])
        )

        if self._terminal_emit_bonus > 0.0:
            rewards = rewards + dones.astype(np.float32) * (
                self._terminal_emit_bonus * (-y_norm)
            )

        infos: list[dict[str, Any]] = []
        for i in range(self.num_envs):
            info_i: dict[str, Any] = {
                "emit_m2": float(emit_m2[i]),
                "log_emit_norm": float(y_norm[i]),
                "knobs_phys": knobs_phys[i].copy(),
            }
            if dones[i]:
                info_i["terminal_observation"] = pre[i].copy()
                info_i["TimeLimit.truncated"] = True
            infos.append(info_i)

        return obs, rewards, dones, infos

    def close(self) -> None:
        return None

    def seed(self, seed: int | None = None):
        if seed is not None:
            self._env._rng.manual_seed(int(seed))
        return [seed for _ in range(self.num_envs)]

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        idx = _resolve_indices(indices, self.num_envs)
        if attr_name == "render_mode":
            return [None for _ in idx]
        if hasattr(self, attr_name):
            return [getattr(self, attr_name) for _ in idx]
        return [getattr(self._env, attr_name, None) for _ in idx]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None,
                   **method_kwargs) -> list[Any]:
        raise NotImplementedError(
            "FlowSurrogateVecEnv is a single batched env and does not support "
            "per-env env_method() calls."
        )

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        idx = _resolve_indices(indices, self.num_envs)
        return [False for _ in idx]

    def render(self, mode: str | None = None):
        return None
