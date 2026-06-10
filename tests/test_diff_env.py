"""
Tests for DiffPhotoinjectorEnv — the batched torch env used by SHAC/BPTT.

The key invariant under test is that gradient flows from the action through
the surrogate to the reward; without this, SHAC/BPTT degenerate to noise.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import torch

from photoinjector_rl.surrogates.mlp.diff_env import (
    ACTION_DIM,
    OBS_DIM,
    DiffPhotoinjectorEnv,
)


def _make_env(num_envs: int = 4, episode_length: int = 8, seed: int = 0,
              **kwargs) -> DiffPhotoinjectorEnv:
    """Build an env with a stand-in mean-of-input surrogate.

    The mock returns y = mean(x) — differentiable wrt x — so the gradient
    check is unambiguous. (The same env wired to a real EmittanceMLP is
    covered by `test_diff_env_real_surrogate`.)
    """
    class MockSurrogate(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=-1, keepdim=True)

    return DiffPhotoinjectorEnv(
        num_envs=num_envs, device="cpu", seed=seed,
        episode_length=episode_length, stochastic_init=True, no_grad=False,
        surrogate=MockSurrogate(), target_mean=0.0, target_std=1.0,
        **kwargs,
    )


def test_shapes():
    env = _make_env(num_envs=4)
    assert env.num_obs == OBS_DIM == 6
    assert env.num_actions == ACTION_DIM == 5
    obs = env.reset()
    assert obs.shape == (4, OBS_DIM)
    action = torch.zeros((4, ACTION_DIM))
    obs2, rew, done, info = env.step(action)
    assert obs2.shape == (4, OBS_DIM)
    assert rew.shape == (4,)
    assert done.shape == (4,)
    assert info["obs_before_reset"].shape == (4, OBS_DIM)


def test_gradient_flows_from_action_to_reward():
    """The core invariant for SHAC/BPTT: a.grad must be non-zero after
    backproping a sum of rewards over multiple steps.
    """
    torch.manual_seed(0)
    env = _make_env(num_envs=4, episode_length=16)
    env.reset()
    # Single shared action tensor used across all steps (the actor would
    # normally re-evaluate per step; this just exercises the env graph).
    action = torch.zeros((4, ACTION_DIM), requires_grad=True)
    total_reward = torch.tensor(0.0)
    obs = env.initialize_trajectory()
    for _ in range(8):
        obs, rew, done, _info = env.step(action)
        total_reward = total_reward + rew.sum()
    loss = -total_reward
    loss.backward()
    assert action.grad is not None, "no gradient reached the action"
    assert action.grad.abs().sum().item() > 0.0, "gradient is identically zero"


def test_clear_grad_detaches_state():
    env = _make_env(num_envs=2, episode_length=8)
    env.reset()
    action = torch.zeros((2, ACTION_DIM), requires_grad=True)
    env.step(action)
    # state should now carry grad_fn because step() flowed gradient into knobs
    # (episode_length=8 > 1 step, so no env has reset yet).
    assert env._knobs.requires_grad or env._knobs.grad_fn is not None
    env.clear_grad()
    assert env._knobs.grad_fn is None
    assert env._distgen.grad_fn is None
    assert env._last_y_norm.grad_fn is None


def test_partial_reset_preserves_survivor_gradient():
    """The key invariant the torch.where reset() fix enforces: when SOME
    envs hit done and reset mid-trajectory, the others must still carry
    gradient on their state. Forced via a monkey-patched _step_count.
    """
    env = _make_env(num_envs=4, episode_length=100)
    env.reset()
    action = torch.zeros((4, ACTION_DIM), requires_grad=True)
    # First step: builds grad_fn on all envs.
    obs, _r, _d, _ = env.step(action)
    assert env._knobs.grad_fn is not None
    # Force only env 0 to be at terminal — next step's increment hits
    # episode_length for env 0 only.
    env._step_count[0] = env.episode_length - 1
    env._step_count[1:] = 1
    _, rew, done, _ = env.step(action)
    # env 0 should have reset; envs 1-3 should not.
    assert bool(done[0].item())
    assert not bool(done[1:].any().item())
    # Survivor grad must be intact: backpropping the survivor rewards reaches
    # the action.
    survivor_loss = -rew[1:].sum()
    survivor_loss.backward(retain_graph=True)
    assert action.grad is not None
    assert action.grad[1:].abs().sum().item() > 0


def test_initialize_trajectory_returns_detached_obs():
    env = _make_env(num_envs=4, episode_length=8)
    env.reset()
    action = torch.zeros((4, ACTION_DIM), requires_grad=True)
    env.step(action)
    # Pre-conditions: state should have grad_fn after a step.
    assert env._knobs.grad_fn is not None
    obs = env.initialize_trajectory()
    assert obs.shape == (4, OBS_DIM)
    assert obs.grad_fn is None
    assert env._knobs.grad_fn is None
    # After the cut, a fresh step should re-build the graph.
    _, rew, _, _ = env.step(action)
    assert rew.grad_fn is not None


def test_per_step_actions_each_get_gradient():
    """SHAC's actual use case: a new action tensor per step. Each step's
    action must receive a non-zero gradient.
    """
    torch.manual_seed(0)
    env = _make_env(num_envs=4, episode_length=16)
    env.reset()
    actions = [torch.zeros((4, ACTION_DIM), requires_grad=True)
               for _ in range(6)]
    obs = env.initialize_trajectory()
    total = torch.tensor(0.0)
    for a in actions:
        obs, rew, _, _ = env.step(a)
        total = total + rew.sum()
    (-total).backward()
    for t, a in enumerate(actions):
        assert a.grad is not None, f"no grad at step {t}"
        assert a.grad.abs().sum().item() > 0, f"zero grad at step {t}"


def test_surrogate_parameters_are_frozen():
    env = _make_env(num_envs=2, episode_length=4)
    for p in env._surrogate.parameters():
        assert not p.requires_grad, "surrogate must be frozen"


def test_obs_before_reset_is_finite_and_correct_shape():
    env = _make_env(num_envs=4, episode_length=3)
    env.reset()
    a = torch.zeros((4, ACTION_DIM))
    for _ in range(3):
        obs, rew, done, info = env.step(a)
    pre = info["obs_before_reset"]
    assert pre.shape == (4, OBS_DIM)
    assert torch.isfinite(pre).all()
    # Last column is the pre-reset y_norm; should equal the rewards we just
    # observed (since reward = -y_norm).
    assert torch.allclose(pre[:, -1], -rew, atol=1e-6)


def test_cross_epoch_no_leak():
    """After initialize_trajectory between two backwards, the second backward
    must succeed and must not back-propagate into actions from the first epoch.
    """
    env = _make_env(num_envs=2, episode_length=8)
    env.reset()

    def one_epoch() -> torch.Tensor:
        action = torch.zeros((2, ACTION_DIM), requires_grad=True)
        obs = env.initialize_trajectory()
        loss = torch.tensor(0.0)
        for _ in range(3):
            obs, rew, _, _ = env.step(action)
            loss = loss - rew.sum()
        return action, loss

    a1, loss1 = one_epoch()
    loss1.backward()
    g1 = a1.grad.clone()
    a2, loss2 = one_epoch()
    loss2.backward()
    # First epoch's action grad should be unchanged after second backward.
    assert torch.allclose(g1, a1.grad)
    assert a2.grad is not None and a2.grad.abs().sum() > 0


def test_done_resets_step_count_and_state():
    env = _make_env(num_envs=2, episode_length=3)
    env.reset()
    a = torch.zeros((2, ACTION_DIM))
    for step_idx in range(3):
        obs, rew, done, info = env.step(a)
    # After the 3rd step, done should fire and step_count resets to 0
    assert bool(done.all().item())
    assert int(env._step_count.max().item()) == 0


def test_determinism_same_seed():
    a = torch.zeros((4, ACTION_DIM))
    env1 = _make_env(num_envs=4, seed=42, episode_length=16)
    env2 = _make_env(num_envs=4, seed=42, episode_length=16)
    for _ in range(5):
        o1, r1, _, _ = env1.step(a)
        o2, r2, _, _ = env2.step(a)
        assert torch.allclose(o1, o2)
        assert torch.allclose(r1, r2)


def test_knobs_stay_in_unit_box():
    env = _make_env(num_envs=8, episode_length=16)
    env.reset()
    # huge action — should clip
    a = torch.ones((8, ACTION_DIM)) * 100.0
    for _ in range(4):
        env.step(a)
    assert (env._knobs >= 0.0).all()
    assert (env._knobs <= 1.0).all()


def test_no_grad_mode_disables_gradient():
    env = _make_env(num_envs=2, episode_length=4)
    env.no_grad = True
    env.reset()
    action = torch.zeros((2, ACTION_DIM), requires_grad=True)
    _, rew, _, _ = env.step(action)
    # rew should be detached when no_grad=True
    assert rew.grad_fn is None


def test_distgen_drift_perturbs_state():
    env = _make_env(num_envs=8, episode_length=16, distgen_drift_std=0.1)
    env.reset()
    d0 = env._distgen.clone()
    env.step(torch.zeros((8, ACTION_DIM)))
    assert not torch.allclose(d0, env._distgen)


# ---- Regression test with the real trained surrogate --------------------


def test_real_surrogate_loads_and_runs(checkpoint_path, norm_json_path):
    """Smoke: loading the real EmittanceMLP via constructor paths works and
    rewards are finite. Skipped when the trained checkpoint is missing.
    """
    env = DiffPhotoinjectorEnv(
        num_envs=4, device="cpu", seed=0, episode_length=8,
        stochastic_init=True, no_grad=False,
        surrogate_ckpt=str(checkpoint_path),
        norm_json=str(norm_json_path),
    )
    obs = env.reset()
    assert torch.isfinite(obs).all()
    a = torch.zeros((4, ACTION_DIM))
    obs2, rew, done, info = env.step(a)
    assert torch.isfinite(rew).all()
