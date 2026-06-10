"""
Tests for the flow-based MBRL env (FlowBunchEnv) + the differentiable property
registry. The contract / gradient tests are the acceptance spec: the env must
satisfy the diffrl interface and produce a reward differentiable w.r.t. the
action, and every property must backprop through the sampled cloud.
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import torch

from photoinjector_rl.surrogates.flow import N_INPUT
from photoinjector_rl.surrogates.flow.model import ConditionalAffineFlow
from photoinjector_rl.surrogates.flow.diff_env import FlowBunchEnv
from photoinjector_rl.surrogates.flow.properties import PROPERTY_REGISTRY, RewardSpec

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


def test_aspect_ratio_target_reward() -> None:
    """aspect_ratio uses a log10 transform; target-mode y_norm is 0 at the target
    and symmetric for r vs 1/r (a multiplicative ratio)."""
    fn, transform = PROPERTY_REGISTRY["aspect_ratio"]
    assert transform == "log10"
    spec = RewardSpec(name="aspect_ratio", property_fn=fn, transform="log10",
                      mean=0.0, std=0.36, mode="target", target=1.0)
    torch.testing.assert_close(spec.normalize(torch.tensor([1.0])),
                               torch.zeros(1), atol=1e-6, rtol=0)
    y_off = spec.normalize(torch.tensor([2.0, 0.5]))   # r and 1/r about round
    assert (y_off > 0).all()
    torch.testing.assert_close(y_off[0], y_off[1], atol=1e-5, rtol=1e-4)


def test_aspect_ratio_target_env_differentiable() -> None:
    """Full path: FlowBunchEnv(property='aspect_ratio', reward_mode='target') builds
    from the dataset z-score, steps, and the reward is differentiable wrt the action.
    Data-gated on the processed dataset (needed for the non-emittance z-score)."""
    import os
    import pytest
    proc = "data/processed/flow_surrogate.h5"
    if not os.path.exists(proc):
        pytest.skip("data/processed/flow_surrogate.h5 absent")
    env = FlowBunchEnv(num_envs=4, device=DEV, seed=0, episode_length=4, no_grad=False,
                       flow=_tiny_flow(), processed_h5=proc, property="aspect_ratio",
                       reward_mode="target", target=1.5, n_particles=64)
    env.reset()
    a = torch.zeros(4, 5, requires_grad=True)
    _o, r, _d, _i = env.step(a)
    assert r.shape == (4,) and torch.isfinite(r).all()
    r.sum().backward()
    assert a.grad is not None and a.grad.abs().sum() > 0


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
    from photoinjector_rl.diffrl import SHAC

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


# ----- joint aspect + tilt shape control -----------------------------------

def _tilted_cloud(aspect: float, tilt_deg: float, n: int = 20000) -> torch.Tensor:
    """Synthetic (1,n,6) cloud with known eigen aspect + tilt (z/p columns zero)."""
    import math
    torch.manual_seed(0)
    minor = 1.0e-3
    u = torch.randn(n) * (minor * aspect)
    v = torch.randn(n) * minor
    th = math.radians(tilt_deg)
    x = u * math.cos(th) - v * math.sin(th)
    y = u * math.sin(th) + v * math.cos(th)
    z = torch.zeros(n)
    return torch.stack([x, y, z, z, z, z], dim=1)[None].double()


def test_shape_helpers_recover_known_cloud() -> None:
    from photoinjector_rl.surrogates.flow import properties as P
    parts = _tilted_cloud(3.0, 30.0)
    assert abs(P._eigen_aspect(parts).item() - 3.0) < 0.15
    assert abs(P._tilt_angle_deg(parts).item() - 30.0) < 2.0


def test_aspect_tilt_s_roundtrip() -> None:
    from photoinjector_rl.surrogates.flow import properties as P
    for a, t in [(2.0, 30.0), (3.0, -45.0), (1.5, 80.0)]:
        s1, s2 = P.aspect_tilt_to_s(a, t)
        aa, tt = P.s_to_aspect_tilt(torch.tensor([s1]), torch.tensor([s2]))
        assert abs(aa.item() - a) < 1e-4 and abs(tt.item() - t) < 1e-3


def test_s1_s2_differentiable() -> None:
    from photoinjector_rl.surrogates.flow import properties as P
    parts = _tilted_cloud(2.0, 20.0).float().requires_grad_(True)
    (P._s1(parts).sum() + P._s2(parts).sum()).backward()
    assert parts.grad is not None and parts.grad.abs().sum() > 0


def test_shape_target_spec_zero_at_target() -> None:
    from photoinjector_rl.surrogates.flow import properties as P
    spec = P.ShapeTargetSpec.from_aspect_tilt(3.0, 30.0)
    y = spec.reward_ynorm(_tilted_cloud(3.0, 30.0))
    assert y.item() < 0.05                                   # ~0 at the target
    y_off = spec.reward_ynorm(_tilted_cloud(3.0, -30.0))     # wrong tilt
    assert y_off.item() > y.item() + 0.5


def test_shape_target_env_differentiable() -> None:
    from photoinjector_rl.surrogates.flow.shape_env import ShapeTargetEnv
    env = ShapeTargetEnv(num_envs=4, device=DEV, seed=0, episode_length=4,
                         no_grad=False, flow=_tiny_flow(), n_particles=64,
                         target_aspect=2.0, target_tilt_deg=30.0)
    obs = env.reset()
    assert obs.shape == (4, 6)                               # fixed target -> 6-D obs
    a = torch.zeros(4, 5, requires_grad=True)
    _o, r, _d, info = env.step(a)
    assert r.shape == (4,) and torch.isfinite(r).all()
    assert "aspect" in info and "tilt_deg" in info and "shape_s1s2" in info
    r.sum().backward()
    assert a.grad is not None and a.grad.abs().sum() > 0


# ----- moving-target (goal-conditioned) aspect+tilt tracking ----------------

def test_sample_shape_trajectory_shape_and_reachable() -> None:
    import numpy as np
    from photoinjector_rl.surrogates.flow.shape_targets import (
        CurriculumConfig, sample_shape_trajectory)
    cfg = CurriculumConfig(r_max=0.7)
    rng = np.random.default_rng(0)
    for d in (0.0, 0.5, 1.0):
        tr = sample_shape_trajectory(rng, 32, d, cfg)
        assert tr.shape == (32, 2)
        assert np.all(np.sqrt((tr ** 2).sum(1)) <= cfg.r_max + 1e-6)  # config r_max honored


def test_curriculum_config_from_dict_and_eval_spec() -> None:
    import numpy as np
    from photoinjector_rl.surrogates.flow.shape_targets import (
        CurriculumConfig, build_eval_trajectories)
    cfg = CurriculumConfig.from_dict({"r_max": 0.5, "tilt_turns_hard": 3.0,
                                      "unknown_key": 1})       # unknown ignored
    assert cfg.r_max == 0.5 and cfg.tilt_turns_hard == 3.0
    trajs = build_eval_trajectories(16, {"tilt_rotation": {"aspect": 2.5, "turns": 0.5}})
    assert set(trajs) == {"tilt_rotation"} and trajs["tilt_rotation"].shape == (16, 2)
    assert set(build_eval_trajectories(16)) == {"staircase", "tilt_rotation", "aspect_ramp"}


def test_curriculum_difficulty_increases_variation() -> None:
    import numpy as np
    from photoinjector_rl.surrogates.flow.shape_targets import (
        CurriculumState, sample_shape_trajectory)
    rng = np.random.default_rng(1)

    def mean_var(progress, n=200):
        cur = CurriculumState(progress=progress)
        vs = [sample_shape_trajectory(rng, 32, cur.difficulty_for_episode(rng)).var(0).sum()
              for _ in range(n)]
        return float(np.mean(vs))

    # early curriculum (mostly static) varies less than late (steps + smooth).
    assert mean_var(0.0) < mean_var(1.0)


def _moving_env(**kw):
    from photoinjector_rl.surrogates.flow.moving_shape_env import MovingShapeEnv
    defaults = dict(num_envs=4, device=DEV, seed=0, episode_length=6,
                    flow=_tiny_flow(), n_particles=64, scale=0.3)
    defaults.update(kw)
    return MovingShapeEnv(**defaults)


def test_moving_shape_obs_layout_and_dim() -> None:
    env = _moving_env(no_grad=True)
    assert env.num_obs == 9 and env.num_actions == 5
    obs = env.reset()
    assert obs.shape == (4, 9)
    assert torch.allclose(obs[:, :5], env._knobs)            # knobs first
    # target dims (7,8) match the env's step-0 setpoint
    ts1, ts2 = env._target_at_step()
    assert torch.allclose(obs[:, 7], ts1) and torch.allclose(obs[:, 8], ts2)


def test_moving_shape_target_advances_with_step_count() -> None:
    import numpy as np
    from photoinjector_rl.surrogates.flow.shape_targets import eval_aspect_ramp
    T = 6
    traj = eval_aspect_ramp(T, tilt_deg=10.0, a0=1.5, a1=3.0)   # strictly varying r
    env = _moving_env(num_envs=3, episode_length=T + 4, no_grad=True,
                      fixed_target_traj=traj)
    obs = env.reset()
    seen = []
    for _ in range(T):
        seen.append(obs[0, 7:9].cpu().numpy().copy())
        obs, _r, _d, _i = env.step(torch.zeros(3, 5))
    assert np.allclose(np.stack(seen), traj, atol=1e-5)        # setpoint tracks step


def test_moving_shape_reward_differentiable() -> None:
    env = _moving_env(no_grad=False)
    env.reset()
    a = torch.zeros(4, 5, requires_grad=True)
    _o, r, _d, info = env.step(a)
    assert r.shape == (4,) and torch.isfinite(r).all()
    assert "aspect" in info and "tilt_deg" in info and "shape_s1s2" in info
    r.sum().backward()
    assert a.grad is not None and a.grad.abs().sum() > 0


def test_moving_shape_info_cache_survives_reset() -> None:
    # The info diagnostic cache (_s1s2_info) is _in_reset-guarded so a done-step
    # auto-reset reports the TERMINAL shape; the obs sensor (_s1_cur) still updates.
    env = _moving_env(no_grad=True, episode_length=8)
    env.reset()
    for _ in range(3):
        env.step(torch.full((4, 5), 0.5))
    info_before = env._s1s2_info.clone()
    cur_before = env._s1_cur.clone()
    env.reset()                                               # _in_reset guards info cache
    assert torch.equal(env._s1s2_info, info_before)           # terminal cache preserved
    assert not torch.equal(env._s1_cur, cur_before)           # obs sensor updated


def test_moving_shape_initialize_trajectory_keeps_step_count() -> None:
    env = _moving_env(no_grad=True, episode_length=10)
    env.reset()
    for _ in range(3):
        env.step(torch.zeros(4, 5))
    before = env._step_count.clone()
    obs = env.initialize_trajectory()
    assert obs.shape == (4, 9)
    assert torch.equal(env._step_count, before)               # NOT reset by init_traj
