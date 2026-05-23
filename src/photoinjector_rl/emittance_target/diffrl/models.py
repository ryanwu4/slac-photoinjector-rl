"""
Vendored ActorDeterministicMLP, ActorStochasticMLP, CriticMLP from DiffRL.
model_utils helpers are inlined.
"""
# Copyright (c) 2022 NVIDIA CORPORATION. Header preserved from upstream.
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def _init(module: nn.Linear, weight_init, bias_init, gain: float = 1.0):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def _get_activation_func(name: str) -> nn.Module:
    n = name.lower()
    if n == "tanh":
        return nn.Tanh()
    if n == "relu":
        return nn.ReLU()
    if n == "elu":
        return nn.ELU()
    if n == "gelu":
        return nn.GELU()
    if n == "identity":
        return nn.Identity()
    raise NotImplementedError(f"Activation {name} not defined")


class ActorDeterministicMLP(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg_network: dict,
                 device: str = "cuda:0"):
        super().__init__()
        self.device = device
        self.layer_dims = [obs_dim] + cfg_network["actor_mlp"]["units"] + [action_dim]

        def init_(m):
            return _init(m, nn.init.orthogonal_,
                         lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i],
                                           self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(_get_activation_func(
                    cfg_network["actor_mlp"]["activation"]))
                modules.append(nn.LayerNorm(self.layer_dims[i + 1]))
        self.actor = nn.Sequential(*modules).to(device)
        self.action_dim = action_dim
        self.obs_dim = obs_dim

    def get_logstd(self):
        return None

    def forward(self, observations, deterministic: bool = False):
        return self.actor(observations)


class ActorStochasticMLP(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg_network: dict,
                 device: str = "cuda:0"):
        super().__init__()
        self.device = device
        self.layer_dims = [obs_dim] + cfg_network["actor_mlp"]["units"] + [action_dim]

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i],
                                     self.layer_dims[i + 1]))
            if i < len(self.layer_dims) - 2:
                modules.append(_get_activation_func(
                    cfg_network["actor_mlp"]["activation"]))
                modules.append(nn.LayerNorm(self.layer_dims[i + 1]))
            else:
                modules.append(_get_activation_func("identity"))
        self.mu_net = nn.Sequential(*modules).to(device)

        logstd = cfg_network.get("actor_logstd_init", -1.0)
        self.logstd = nn.Parameter(
            torch.ones(action_dim, dtype=torch.float32, device=device) * logstd
        )
        self.action_dim = action_dim
        self.obs_dim = obs_dim

    def get_logstd(self):
        return self.logstd

    def forward(self, obs, deterministic: bool = False):
        mu = self.mu_net(obs)
        if deterministic:
            return mu
        std = self.logstd.exp()
        return Normal(mu, std).rsample()

    def forward_with_dist(self, obs, deterministic: bool = False):
        mu = self.mu_net(obs)
        std = self.logstd.exp()
        if deterministic:
            return mu, mu, std
        return Normal(mu, std).rsample(), mu, std

    def evaluate_actions_log_probs(self, obs, actions):
        mu = self.mu_net(obs)
        std = self.logstd.exp()
        return Normal(mu, std).log_prob(actions)


class CriticMLP(nn.Module):
    def __init__(self, obs_dim: int, cfg_network: dict, device: str = "cuda:0"):
        super().__init__()
        self.device = device
        self.layer_dims = [obs_dim] + cfg_network["critic_mlp"]["units"] + [1]

        def init_(m):
            return _init(m, nn.init.orthogonal_,
                         lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i],
                                           self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(_get_activation_func(
                    cfg_network["critic_mlp"]["activation"]))
                modules.append(nn.LayerNorm(self.layer_dims[i + 1]))
        self.critic = nn.Sequential(*modules).to(device)
        self.obs_dim = obs_dim

    def forward(self, observations):
        return self.critic(observations)
