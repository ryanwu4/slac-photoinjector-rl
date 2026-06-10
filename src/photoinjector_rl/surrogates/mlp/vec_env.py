"""
GPU-batched SB3 ``VecEnv`` over the v1 emittance surrogate.

PPO's default ``SubprocVecEnv`` layout runs N separate CPU worker processes,
each doing a *batch-1* surrogate forward and pickling obs / action / reward
across a pipe on every step. For a sub-microsecond MLP that is ~99% IPC +
thread-pool overhead, which is why PPO collected at ~55 env-steps/s while the
diff-RL algos (one batched GPU env) ran at thousands.

`SurrogateVecEnv` removes that overhead: it wraps `DiffPhotoinjectorEnv`
(with ``no_grad=True``) and advances ALL N sub-envs as a single ``(N, 11)``
surrogate forward on `device` — zero IPC, zero per-env Python stepping, no
torch thread oversubscription. PPO rollout collection then runs at the same
throughput as SHAC / BPTT, isolating the remaining (intrinsic) sample-
efficiency gap from the implementation artifact.

Episode statistics (``ep_rew_mean`` etc.) are not produced by this class; wrap
it in ``stable_baselines3.common.vec_env.VecMonitor`` — see ``build_vec_env``
in ``train_ppo.py``.

The per-step ``info`` dicts mirror ``PhotoinjectorEnv``'s (``emit_m2``,
``log_emit_norm``, ``knobs_phys``) so the existing `EpisodeMetricsCallback`
keeps working unchanged. Episodes here end only by time limit, so every
``done`` is a truncation: we set ``TimeLimit.truncated`` and
``terminal_observation`` so PPO bootstraps the terminal value correctly.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from . import SETTING_BOUNDS, SETTING_KEYS
from .diff_env import ACTION_DIM, DiffPhotoinjectorEnv, N_KNOB, OBS_DIM

# Physical bounds for the 5 controllable knobs (for the knobs_phys diagnostic).
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


class SurrogateVecEnv(VecEnv):
    """Batched-on-GPU SB3 VecEnv backed by one `DiffPhotoinjectorEnv`.

    All sub-envs share a single surrogate forward per step, so wall-clock
    throughput matches the diff-RL trainers rather than the old 16-process
    CPU layout. Drop-in for PPO's training env (wrap in `VecMonitor`).
    """

    def __init__(
        self,
        n_envs: int,
        *,
        surrogate_ckpt: str | Path,
        norm_json: str | Path,
        device: str | torch.device,
        seed: int,
        episode_length: int = 64,
        action_scale: float = 0.05,
        distgen_drift_std: float = 0.0,
        terminal_emit_bonus: float = 0.0,
        stochastic_init: bool = True,
    ):
        self.device = torch.device(device)
        self._terminal_emit_bonus = float(terminal_emit_bonus)
        # no_grad=True: PPO is a black-box learner here, so we do NOT keep the
        # surrogate's autograd graph (that is what SHAC/BPTT exploit instead).
        self._env = DiffPhotoinjectorEnv(
            num_envs=int(n_envs),
            device=self.device,
            seed=int(seed),
            episode_length=int(episode_length),
            stochastic_init=bool(stochastic_init),
            no_grad=True,
            surrogate_ckpt=surrogate_ckpt,
            norm_json=norm_json,
            action_scale=float(action_scale),
            distgen_drift_std=float(distgen_drift_std),
        )

        # 5 knobs in [0,1] + 1 z-scored log-emit (unbounded); identical to
        # PhotoinjectorEnv so a policy trained here transfers to the gym env.
        observation_space = spaces.Box(
            low=np.array([0.0] * N_KNOB + [-np.inf], dtype=np.float32),
            high=np.array([1.0] * N_KNOB + [np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32,
        )
        # super().__init__ calls self.get_attr("render_mode"); self._env and
        # self.device are already set above, so that call is safe.
        super().__init__(int(n_envs), observation_space, action_space)
        self._actions: np.ndarray | None = None

    # ----- helpers ----------------------------------------------------------

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
        # SB3 already clips Box actions to bounds, but clip defensively so the
        # surrogate is only ever queried inside the [0,1]^11 box it was fit on.
        act = np.clip(self._actions, -1.0, 1.0).astype(np.float32)
        act_t = torch.from_numpy(act).to(self.device)

        obs_t, rew_t, done_t, info = self._env.step(act_t)

        obs = self._to_np(obs_t)
        rewards = self._to_np(rew_t)
        dones = done_t.detach().to("cpu").numpy().astype(bool)

        # obs_before_reset == post-step state for EVERY env *before* the env
        # auto-resets the done ones. Use it for per-step metrics and for the
        # terminal-bootstrap observation (matches PhotoinjectorEnv's info,
        # whose values reflect the state the step landed in).
        pre = self._to_np(info["obs_before_reset"])
        knobs_norm = pre[:, :N_KNOB]
        knobs_phys = _KNOB_LO + knobs_norm * _KNOB_RANGE
        y_norm = pre[:, -1]
        emit_m2 = self._to_np(
            self._env.physical_emit(info["obs_before_reset"][:, -1])
        )

        # Terminal-emit bonus mirrors PhotoinjectorEnv.step: on the truncating
        # step add bonus * (-y_norm_terminal). terminal_emit_bonus=0 is a no-op.
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
                # Episodes end only at episode_length -> time-limit truncation,
                # so PPO should bootstrap V(terminal) rather than treat it as a
                # true absorbing state.
                info_i["terminal_observation"] = pre[i].copy()
                info_i["TimeLimit.truncated"] = True
            infos.append(info_i)

        return obs, rewards, dones, infos

    def close(self) -> None:
        # Nothing to tear down: single in-process env, no subprocess/pipe.
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
            "SurrogateVecEnv is a single batched env and does not support "
            "per-env env_method() calls."
        )

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        idx = _resolve_indices(indices, self.num_envs)
        return [False for _ in idx]

    def render(self, mode: str | None = None):
        return None
