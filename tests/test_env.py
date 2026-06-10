"""
Unit + regression tests for PhotoinjectorEnv.

Unit tests use mock surrogates (no checkpoint needed). Regression tests use
the real trained EmittanceMLP and auto-skip if the checkpoint is absent.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import numpy as np
import pytest
import torch

from photoinjector_rl.surrogates.mlp.env import (
    N_DISTGEN,
    N_KNOB,
    PhotoinjectorEnv,
)


# ============================================================================
# Unit tests (mock surrogate; no checkpoint required)
# ============================================================================


def _make_env(surrogate, **kwargs) -> PhotoinjectorEnv:
    defaults = dict(target_mean=-10.34, target_std=0.17, max_steps=64)
    defaults.update(kwargs)
    return PhotoinjectorEnv(surrogate=surrogate, **defaults)


def test_action_space_shape_and_bounds(constant_surrogate) -> None:
    env = _make_env(constant_surrogate)
    assert env.action_space.shape == (N_KNOB,)
    assert env.action_space.dtype == np.float32
    np.testing.assert_array_equal(env.action_space.low, -1.0)
    np.testing.assert_array_equal(env.action_space.high, 1.0)


def test_observation_space_shape(constant_surrogate) -> None:
    env = _make_env(constant_surrogate)
    assert env.observation_space.shape == (N_KNOB + 1,)
    assert env.observation_space.dtype == np.float32


def test_reset_obs_shape_and_dtype(constant_surrogate) -> None:
    env = _make_env(constant_surrogate)
    obs, info = env.reset(seed=0)
    assert obs.shape == (N_KNOB + 1,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_step_returns_5_tuple(constant_surrogate) -> None:
    env = _make_env(constant_surrogate)
    env.reset(seed=0)
    out = env.step(np.zeros(N_KNOB, dtype=np.float32))
    assert isinstance(out, tuple) and len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert obs.shape == (N_KNOB + 1,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_action_clipping_keeps_knobs_in_unit_box(constant_surrogate) -> None:
    env = _make_env(constant_surrogate)
    env.reset(seed=0)
    # Hammer with extreme +1 actions for a while; knobs must stay in [0,1].
    for _ in range(20):
        obs, *_ = env.step(np.full(N_KNOB, 100.0, dtype=np.float32))
        assert np.all(obs[:N_KNOB] >= 0.0) and np.all(obs[:N_KNOB] <= 1.0)
    # And again with extreme negative actions.
    for _ in range(20):
        obs, *_ = env.step(np.full(N_KNOB, -100.0, dtype=np.float32))
        assert np.all(obs[:N_KNOB] >= 0.0) and np.all(obs[:N_KNOB] <= 1.0)


def test_drift_zero_holds_distgen_constant(constant_surrogate) -> None:
    env = _make_env(constant_surrogate, distgen_drift_std=0.0)
    _, info0 = env.reset(seed=0)
    distgen0 = info0["distgen_norm"].copy()
    info = info0
    for _ in range(10):
        _, _, _, _, info = env.step(np.zeros(N_KNOB, dtype=np.float32))
    np.testing.assert_array_equal(info["distgen_norm"], distgen0)


def test_drift_nonzero_changes_distgen(constant_surrogate) -> None:
    env = _make_env(constant_surrogate, distgen_drift_std=0.05)
    _, info0 = env.reset(seed=0)
    distgen0 = info0["distgen_norm"].copy()
    info = info0
    for _ in range(10):
        _, _, _, _, info = env.step(np.zeros(N_KNOB, dtype=np.float32))
    assert not np.array_equal(info["distgen_norm"], distgen0)
    # Drift respects the [0,1] clip.
    assert np.all(info["distgen_norm"] >= 0.0)
    assert np.all(info["distgen_norm"] <= 1.0)


def test_seed_determinism(constant_surrogate) -> None:
    env1 = _make_env(constant_surrogate, distgen_drift_std=0.05)
    env2 = _make_env(constant_surrogate, distgen_drift_std=0.05)
    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)
    np.testing.assert_array_equal(obs1, obs2)
    a = np.array([0.3, -0.4, 0.1, -0.2, 0.0], dtype=np.float32)
    for _ in range(5):
        o1, *_ = env1.step(a)
        o2, *_ = env2.step(a)
        np.testing.assert_array_equal(o1, o2)


def test_episode_truncates_at_max_steps(constant_surrogate) -> None:
    env = _make_env(constant_surrogate, max_steps=3)
    env.reset(seed=0)
    a = np.zeros(N_KNOB, dtype=np.float32)
    truncs = []
    for _ in range(3):
        _, _, term, trunc, _ = env.step(a)
        truncs.append(trunc)
        assert not term
    # First two are not truncated, third one IS truncated.
    assert truncs == [False, False, True]


def test_info_dict_keys(constant_surrogate) -> None:
    env = _make_env(constant_surrogate)
    _, info = env.reset(seed=0)
    for key in ("emit_m2", "log_emit_norm", "distgen_norm",
                "knobs_phys", "step_count"):
        assert key in info, f"missing {key} in info dict"
    assert info["distgen_norm"].shape == (N_DISTGEN,)
    assert info["knobs_phys"].shape == (N_KNOB,)
    assert info["step_count"] == 0
    assert info["emit_m2"] > 0  # log10-inverse is positive


def test_distgen_not_in_observation(deterministic_surrogate) -> None:
    """Privacy invariant: changing distgen state must NOT change obs[:5].

    Use a deterministic surrogate so we can verify the predictor sees the
    distgen (obs[5] changes), while the agent-visible knob columns don't.
    """
    env_a = _make_env(deterministic_surrogate)
    env_b = _make_env(deterministic_surrogate)
    obs_a, _ = env_a.reset(seed=1)
    obs_b, _ = env_b.reset(seed=2)
    # Different seeds => different distgens => different y_norm.
    assert obs_a[5] != obs_b[5]
    # But the 5 knob columns are independent of distgen seed (drawn from
    # different RNG state, so they also differ -- the right check is that
    # forcing identical knobs but different distgens still differs in obs[5]
    # without leaking into obs[:5]).
    env_b._knobs_norm = env_a._knobs_norm.copy()
    env_b._last_y_norm = env_b._forward()
    obs_b2 = env_b._observation()
    np.testing.assert_array_equal(obs_b2[:N_KNOB], obs_a[:N_KNOB])
    assert obs_b2[5] != obs_a[5]   # surrogate output still distgen-dependent


def test_invalid_action_shape_raises(constant_surrogate) -> None:
    env = _make_env(constant_surrogate)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.zeros(N_KNOB + 1, dtype=np.float32))


# ---------- device handling --------------------------------------------------


def test_explicit_cpu_device_places_surrogate_on_cpu() -> None:
    """A torch Module surrogate gets moved to the requested device."""
    import torch.nn as nn

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(11, 1)

        def forward(self, x):
            return self.fc(x)

    tiny = TinyMLP()
    env = _make_env(tiny, device="cpu")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (N_KNOB + 1,)
    # Parameters must be on CPU.
    for p in tiny.parameters():
        assert p.device.type == "cpu"


def test_gpu_device_smoke(checkpoint_path, norm_json_path) -> None:
    """If CUDA is available, surrogate lives on cuda:0 and step() works."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")

    env = PhotoinjectorEnv.from_checkpoint(
        ckpt_path=checkpoint_path,
        norm_json=norm_json_path,
        device="cuda:0",
        max_steps=8,
    )
    # Internal model should now sit on cuda:0.
    surrogate = env._surrogate  # noqa: SLF001 -- private but stable
    assert isinstance(surrogate, torch.nn.Module)
    param_devices = {p.device.type for p in surrogate.parameters()}
    assert param_devices == {"cuda"}

    obs, info = env.reset(seed=0)
    assert np.isfinite(info["emit_m2"])
    for _ in range(4):
        obs, r, term, trunc, info = env.step(
            np.zeros(N_KNOB, dtype=np.float32)
        )
        assert np.isfinite(r)
        assert np.isfinite(info["emit_m2"])


def test_terminal_emit_bonus_only_affects_last_step(constant_surrogate) -> None:
    """With terminal_emit_bonus=K, only the last step's reward is amplified."""
    # constant_surrogate returns y_norm = 0, so base reward = -0.
    # Use a deterministic non-zero surrogate so reward != 0.
    def fixed_surrogate(x):
        return torch.full((x.shape[0], 1), -0.5, dtype=torch.float32)

    base_env = _make_env(fixed_surrogate, max_steps=4, terminal_emit_bonus=0.0)
    base_env.reset(seed=0)
    base_rewards = []
    for _ in range(4):
        _, r, _, trunc, _ = base_env.step(np.zeros(N_KNOB, dtype=np.float32))
        base_rewards.append(float(r))
        if trunc:
            break

    # Same setup with terminal_bonus = 10
    bonus_env = _make_env(fixed_surrogate, max_steps=4, terminal_emit_bonus=10.0)
    bonus_env.reset(seed=0)
    bonus_rewards = []
    for _ in range(4):
        _, r, _, trunc, _ = bonus_env.step(np.zeros(N_KNOB, dtype=np.float32))
        bonus_rewards.append(float(r))
        if trunc:
            break

    # Base reward = -y_norm = 0.5 every step.
    assert all(abs(r - 0.5) < 1e-5 for r in base_rewards[:-1])
    # First N-1 steps match exactly; last step is base + bonus*(-y_norm).
    np.testing.assert_allclose(bonus_rewards[:-1], base_rewards[:-1], rtol=1e-5)
    expected_last = base_rewards[-1] + 10.0 * 0.5
    assert abs(bonus_rewards[-1] - expected_last) < 1e-4


def test_invalid_cuda_index_in_resolve_device() -> None:
    """Asking for a device past torch.cuda.device_count() exits with a message."""
    import torch
    from photoinjector_rl.surrogates.mlp.train_ppo import resolve_device

    # cpu always resolves.
    assert resolve_device("cpu") == "cpu"
    # auto resolves to either cpu or cuda:0 depending on hardware.
    assert resolve_device("auto") in ("cpu", "cuda:0")

    if torch.cuda.is_available():
        # Out-of-range cuda index should raise SystemExit.
        bad = f"cuda:{torch.cuda.device_count() + 5}"
        with pytest.raises(SystemExit):
            resolve_device(bad)
    else:
        # Without CUDA, any cuda: request should raise SystemExit.
        with pytest.raises(SystemExit):
            resolve_device("cuda:0")


# ============================================================================
# Regression tests (real checkpoint; skip if missing)
# ============================================================================


def test_from_checkpoint_smoke(checkpoint_path, norm_json_path) -> None:
    env = PhotoinjectorEnv.from_checkpoint(
        ckpt_path=checkpoint_path,
        norm_json=norm_json_path,
        max_steps=16,
    )
    obs, info = env.reset(seed=0)
    assert obs.shape == (N_KNOB + 1,)
    assert np.isfinite(info["emit_m2"])
    rng = np.random.default_rng(0)
    for _ in range(10):
        action = rng.uniform(-1.0, 1.0, N_KNOB).astype(np.float32)
        obs, reward, term, trunc, info = env.step(action)
        assert np.isfinite(reward)
        assert np.isfinite(info["emit_m2"])
        assert not term


def test_reset_regression_seed42(checkpoint_path, norm_json_path) -> None:
    """With seed=42, the post-reset y_norm and emittance are deterministic.

    Golden values were captured against the shipped hi-fi surrogate
    `trained/emittance_target_hifi/checkpoints/best-*.ckpt`. If you retrain,
    update these constants.
    """
    env = PhotoinjectorEnv.from_checkpoint(
        ckpt_path=checkpoint_path,
        norm_json=norm_json_path,
    )
    obs, info = env.reset(seed=42)
    assert obs[5] == pytest.approx(REG_RESET_SEED42_YNORM, abs=1e-4)
    assert info["emit_m2"] == pytest.approx(REG_RESET_SEED42_EMIT_M2, rel=1e-3)


def test_step_sequence_regression(checkpoint_path, norm_json_path) -> None:
    """Fixed seed + 5 fixed actions -> reproducible (obs[5], reward) trajectory."""
    env = PhotoinjectorEnv.from_checkpoint(
        ckpt_path=checkpoint_path,
        norm_json=norm_json_path,
    )
    env.reset(seed=42)
    actions = [
        np.array([0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, -0.5, 0.0, 0.0], dtype=np.float32),
        np.array([-0.5, -0.5, 0.0, 0.5, 0.5], dtype=np.float32),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    ]
    trajectory = []
    for a in actions:
        obs, reward, *_ = env.step(a)
        trajectory.append((float(obs[5]), float(reward)))
    for (y_obs, r), (y_gold, r_gold) in zip(trajectory, REG_STEP_SEQ_TRAJ):
        assert y_obs == pytest.approx(y_gold, abs=1e-4)
        assert r == pytest.approx(r_gold, abs=1e-4)


# ---------- Regression golden values -----------------------------------------
#
# Checkpoint: trained/emittance_target_hifi/checkpoints/best-epoch=191-val_loss=0.0060.ckpt
# If retrained, recompute these against the new checkpoint.

REG_RESET_SEED42_YNORM = -0.9407756924629211
REG_RESET_SEED42_EMIT_M2 = 3.25181485738963e-11
REG_STEP_SEQ_TRAJ = [
    # (y_norm_after_step, reward)
    (-0.9836502075195312, 0.9836502075195312),
    (-0.9899106621742249, 0.9899106621742249),
    (-1.0061023235321045, 1.0061023235321045),
    (-0.9641300439834595, 0.9641300439834595),
    (-1.021549940109253, 1.021549940109253),
]
