"""Shape smoke tests for vendored ActorStochasticMLP / CriticMLP."""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import torch

from photoinjector_rl.emittance_target.diffrl.models import (
    ActorDeterministicMLP,
    ActorStochasticMLP,
    CriticMLP,
)


_CFG = {
    "actor_mlp": {"units": [32, 32], "activation": "elu"},
    "actor_logstd_init": -1.0,
    "critic_mlp": {"units": [32, 32], "activation": "elu"},
}


def test_actor_stochastic_shape():
    actor = ActorStochasticMLP(obs_dim=6, action_dim=5, cfg_network=_CFG,
                               device="cpu")
    obs = torch.randn(8, 6)
    a = actor(obs)
    assert a.shape == (8, 5)
    a_det = actor(obs, deterministic=True)
    assert a_det.shape == (8, 5)
    assert actor.get_logstd().shape == (5,)


def test_actor_deterministic_shape():
    actor = ActorDeterministicMLP(obs_dim=6, action_dim=5, cfg_network=_CFG,
                                  device="cpu")
    obs = torch.randn(8, 6)
    assert actor(obs).shape == (8, 5)
    assert actor.get_logstd() is None


def test_critic_shape():
    critic = CriticMLP(obs_dim=6, cfg_network=_CFG, device="cpu")
    obs = torch.randn(8, 6)
    v = critic(obs)
    assert v.shape == (8, 1)


def test_actor_gradient_flow():
    actor = ActorStochasticMLP(obs_dim=6, action_dim=5, cfg_network=_CFG,
                               device="cpu")
    obs = torch.randn(8, 6, requires_grad=True)
    a = actor(obs, deterministic=True)
    a.sum().backward()
    assert obs.grad is not None
    # at least one parameter should have a grad
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in actor.parameters())
