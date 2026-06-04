"""
Tests for SurrogateVecEnv — the GPU-batched SB3 VecEnv that replaces the
16-process SubprocVecEnv layout for PPO collection (closes the wall-clock
"Gap A" vs SHAC/BPTT).

Contract under test:
  - SB3 VecEnv API shapes / dtypes (np, not torch).
  - Per-step info parity with PhotoinjectorEnv (emit_m2 / log_emit_norm /
    knobs_phys) so EpisodeMetricsCallback keeps working.
  - reward == -y_norm at the landed state.
  - Episodes truncate at episode_length with TimeLimit.truncated +
    terminal_observation set (so PPO bootstraps V(terminal)).
  - Composes with VecMonitor to produce per-episode stats.

These use the real checkpoint via the shared fixtures; they skip if the
checkpoint / norm JSON are absent.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env import VecMonitor

from photoinjector_rl.emittance_target.diff_env import (
    ACTION_DIM,
    N_KNOB,
    OBS_DIM,
)
from photoinjector_rl.emittance_target.vec_env import SurrogateVecEnv


def _make(checkpoint_path, norm_json_path, *, n_envs=4, episode_length=8):
    return SurrogateVecEnv(
        n_envs,
        surrogate_ckpt=str(checkpoint_path),
        norm_json=str(norm_json_path),
        device="cpu",
        seed=0,
        episode_length=episode_length,
        action_scale=0.05,
    )


def test_reset_and_step_shapes(checkpoint_path, norm_json_path):
    env = _make(checkpoint_path, norm_json_path, n_envs=4)
    obs = env.reset()
    assert obs.shape == (4, OBS_DIM)
    assert obs.dtype == np.float32

    obs, rew, done, infos = env.step(np.zeros((4, ACTION_DIM), np.float32))
    assert obs.shape == (4, OBS_DIM) and obs.dtype == np.float32
    assert rew.shape == (4,)
    assert done.shape == (4,) and done.dtype == np.bool_
    assert len(infos) == 4
    for info in infos:
        assert {"emit_m2", "log_emit_norm", "knobs_phys"} <= info.keys()
        assert np.asarray(info["knobs_phys"]).shape == (N_KNOB,)
        assert info["emit_m2"] > 0.0  # physical emittance in m²
    env.close()


def test_reward_is_negative_ynorm(checkpoint_path, norm_json_path):
    """reward == -y_norm at the state the step landed in (== -log_emit_norm)."""
    env = _make(checkpoint_path, norm_json_path, n_envs=4)
    env.reset()
    env.step_async(np.full((4, ACTION_DIM), 0.3, np.float32))
    _obs, rew, _done, infos = env.step_wait()
    y = np.array([info["log_emit_norm"] for info in infos], dtype=np.float32)
    assert np.allclose(rew, -y, atol=1e-5)
    env.close()


def test_action_clipping_keeps_knobs_in_box(checkpoint_path, norm_json_path):
    """Out-of-range actions must not push the surrogate input outside [0,1]."""
    env = _make(checkpoint_path, norm_json_path, n_envs=4)
    env.reset()
    obs, _rew, _done, _infos = env.step(
        np.full((4, ACTION_DIM), 5.0, np.float32)
    )
    knobs = obs[:, :N_KNOB]
    assert (knobs >= 0.0).all() and (knobs <= 1.0).all()
    env.close()


def test_time_limit_truncation(checkpoint_path, norm_json_path):
    env = _make(checkpoint_path, norm_json_path, n_envs=4, episode_length=4)
    env.reset()
    a = np.zeros((4, ACTION_DIM), np.float32)
    for _ in range(3):
        _obs, _rew, done, infos = env.step(a)
        assert not done.any()
        assert all("TimeLimit.truncated" not in i for i in infos)
    # 4th step hits episode_length -> all envs truncate.
    _obs, _rew, done, infos = env.step(a)
    assert done.all()
    for info in infos:
        assert info["TimeLimit.truncated"] is True
        assert np.asarray(info["terminal_observation"]).shape == (OBS_DIM,)
    env.close()


def test_vecmonitor_episode_stats(checkpoint_path, norm_json_path):
    """Wrapped in VecMonitor, a finished episode reports an `episode` dict
    (the source of SB3's ep_rew_mean / ep_len_mean)."""
    env = VecMonitor(
        _make(checkpoint_path, norm_json_path, n_envs=2, episode_length=5)
    )
    env.reset()
    a = np.zeros((2, ACTION_DIM), np.float32)
    infos = []
    for _ in range(5):
        _obs, _rew, done, infos = env.step(a)
    assert done.all()
    for info in infos:
        assert "episode" in info
        assert {"r", "l"} <= info["episode"].keys()
        assert info["episode"]["l"] == 5
    env.close()
