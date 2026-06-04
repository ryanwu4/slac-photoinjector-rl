"""
Tests for the flow-based MBRL env (FlowBunchEnv) + the differentiable property
registry. The contract / gradient tests are the acceptance spec: the env must
satisfy the diffrl interface and produce a reward differentiable w.r.t. the
action, and every property must backprop through the sampled cloud.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import torch

from photoinjector_rl.flow_surrogate import N_INPUT
from photoinjector_rl.flow_surrogate.model import ConditionalAffineFlow
from photoinjector_rl.flow_surrogate.diff_env import FlowBunchEnv
from photoinjector_rl.flow_surrogate.properties import PROPERTY_REGISTRY, RewardSpec

DEV = "cpu"


def _tiny_flow() -> ConditionalAffineFlow:
    torch.manual_seed(0)
    # Non-degenerate de-norm buffers so sampled clouds have a realistic scale
    # (positions ~mm, momenta ~1e4 eV/c, pz offset ~6 MeV/c).
    pm = [0.0, 0.0, 9.4e-1, 0.0, 0.0, 6.3e6]
    ps = [2.8e-3, 2.7e-3, 3.7e-4, 3.3e4, 3.2e4, 2.2e5]
    return ConditionalAffineFlow(
        condition_dim=N_INPUT, latent_dim=6, hidden_dim=16, n_layers=4,
        particle_mean=pm, particle_std=ps,
        target_mean=-10.0, target_std=0.2,
    ).eval()


def _env(**kw) -> FlowBunchEnv:
    defaults = dict(num_envs=8, device=DEV, seed=0, episode_length=4,
                    flow=_tiny_flow(), property="norm_emit_4d", n_particles=64)
    defaults.update(kw)
    return FlowBunchEnv(**defaults)


# ----- diffrl contract -------------------------------------------------------

def test_env_attrs() -> None:
    env = _env()
    assert env.num_obs == 6 and env.num_actions == 5 and env.num_envs == 8
    assert env.episode_length == 4


def test_reset_step_shapes() -> None:
    env = _env()
    obs = env.reset()
    assert obs.shape == (8, 6) and torch.isfinite(obs).all()
    action = torch.zeros(8, 5)
    obs, rew, done, info = env.step(torch.tanh(action))
    assert obs.shape == (8, 6)
    assert rew.shape == (8,) and torch.isfinite(rew).all()
    assert done.shape == (8,)
    assert "obs_before_reset" in info and info["obs_before_reset"].shape == (8, 6)


def test_step_reward_is_differentiable_wrt_action() -> None:
    """The whole point of MBRL here: d(reward)/d(action) exists and is nonzero."""
    env = _env()
    env.reset()
    action = torch.zeros(8, 5, requires_grad=True)
    _obs, rew, _done, _info = env.step(action)
    assert rew.requires_grad
    rew.sum().backward()
    assert action.grad is not None
    assert torch.isfinite(action.grad).all()
    assert action.grad.abs().sum() > 0


def test_initialize_trajectory_detaches() -> None:
    env = _env()
    env.reset()
    env.step(torch.zeros(8, 5, requires_grad=True))
    obs = env.initialize_trajectory()
    assert not obs.requires_grad


def test_clear_grad_runs() -> None:
    env = _env()
    env.step(torch.zeros(8, 5))
    env.clear_grad()  # should not raise


# ----- property registry -----------------------------------------------------

def _phys_cloud(requires_grad: bool = False) -> torch.Tensor:
    torch.manual_seed(1)
    scale = torch.tensor([2.8e-3, 2.7e-3, 3.7e-4, 3.3e4, 3.2e4, 2.2e5])
    offset = torch.tensor([0.0, 0.0, 9.4e-1, 0.0, 0.0, 6.3e6])
    parts = torch.randn(4, 256, 6, dtype=torch.float64) * scale + offset
    parts.requires_grad_(requires_grad)
    return parts


def test_all_properties_shape_finite_positive() -> None:
    parts = _phys_cloud()
    for name, (fn, _t) in PROPERTY_REGISTRY.items():
        p = fn(parts)
        assert p.shape == (4,), name
        assert torch.isfinite(p).all(), name
        assert (p > 0).all(), name  # all initial-registry properties are > 0


def test_all_properties_differentiable() -> None:
    for name, (fn, _t) in PROPERTY_REGISTRY.items():
        parts = _phys_cloud(requires_grad=True)
        fn(parts).sum().backward()
        assert parts.grad is not None, name
        assert torch.isfinite(parts.grad).all(), name
        assert parts.grad.abs().sum() > 0, name


def test_reward_spec_minimize_invert_roundtrip() -> None:
    spec = RewardSpec(name="x", property_fn=PROPERTY_REGISTRY["sigma_x"][0],
                      transform="log10", mean=-3.0, std=0.5, mode="minimize")
    p = torch.tensor([1e-3, 2e-3, 5e-4])
    y = spec.normalize(p)
    back = spec.invert(y)
    torch.testing.assert_close(back, p, rtol=1e-4, atol=1e-9)


# ----- equivalence with the flow's scalar forward() (4D emittance) ----------

def test_norm_emit_4d_matches_flow_forward() -> None:
    """FlowBunchEnv(norm_emit_4d) y_norm == flow.forward(x) for the same z.

    Both z-score log10(norm_emit_4d) with the flow's target_mean/std; with the
    same particle count and RNG state they must coincide to fp precision.
    """
    flow = _tiny_flow()
    n = int(flow.hparams.forward_n_particles)  # flow.forward uses this many
    env = _env(flow=flow, n_particles=n)
    knobs = torch.rand(8, 5)
    distgen = torch.rand(8, 6)
    x = torch.cat([knobs, distgen], dim=-1)

    torch.manual_seed(123)
    y_env = env._forward_surrogate(knobs, distgen)
    torch.manual_seed(123)
    y_flow = flow(x).squeeze(-1)
    torch.testing.assert_close(y_env, y_flow, rtol=1e-5, atol=1e-5)


# ----- end-to-end SHAC integration (tiny, reuses diffrl unchanged) ----------

def test_shac_smoke_runs(tmp_path) -> None:
    from functools import partial
    from photoinjector_rl.emittance_target.diffrl import SHAC

    flow = _tiny_flow()
    env_fn = partial(FlowBunchEnv, flow=flow, property="norm_emit_4d", n_particles=32)
    cfg = {
        "params": {
            "diff_env": {"name": "FlowBunchEnv", "episode_length": 4,
                         "stochastic_env": True, "action_scale": 0.05,
                         "distgen_drift_std": 0.0},
            "general": {"seed": 0, "device": DEV, "render": False,
                        "train": True, "logdir": str(tmp_path)},
            "config": {"name": "smoke", "num_actors": 4, "steps_num": 4,
                       "max_epochs": 2, "actor_learning_rate": 5e-4,
                       "critic_learning_rate": 1e-3, "lr_schedule": "linear",
                       "gamma": 0.99, "critic_method": "td-lambda", "lambda": 0.95,
                       "target_critic_alpha": 0.2, "obs_rms": True, "ret_rms": False,
                       "critic_iterations": 4, "num_batch": 2, "truncate_grads": True,
                       "grad_norm": 1.0, "rew_scale": 1.0, "save_interval": 0,
                       "betas": [0.7, 0.95],
                       "player": {"games_num": 4, "deterministic": False}},
            "network": {"actor_mlp": {"units": [16, 16], "activation": "elu"},
                        "actor_logstd_init": -1.0, "critic": "CriticMLP",
                        "critic_mlp": {"units": [16, 16], "activation": "elu"}},
        }
    }
    algo = SHAC(cfg, env_fn=env_fn)
    algo.train()
    # SHAC writes best/final policy .pt into logdir.
    pts = list(tmp_path.glob("*.pt"))
    assert pts, "SHAC did not write a policy checkpoint"
