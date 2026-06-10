from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


# --- provided helpers, same as from diffRL --------------------------------------------------------

def _init(module: nn.Linear, weight_init, bias_init, gain: float = 1.0):
    """Initialize a Linear layer in place and return it (orthogonal-init helper)."""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def _get_activation_func(name: str) -> nn.Module:
    """Map an activation name -> nn.Module. Supports tanh/relu/elu/gelu/identity."""
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


# --- networks to implement ---------------------------------------------------

#again, mostly unchanged from DiffRL but hand coded for understanding
class ActorStochasticMLP(nn.Module):

    def __init__(self, obs_dim, action_dim, cfg_network,
                 device = "cuda:0"):
        super().__init__()
        layer_dims = [obs_dim] + cfg_network["actor_mlp"]["units"] + [action_dim]

        modules = []
        for i in range(len(layer_dims) - 1):
            modules.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                #add activate and layernorm after linear layer
                modules.append(_get_activation_func(cfg_network["actor_mlp"]["activation"]))
                modules.append(nn.LayerNorm(layer_dims[i + 1]))
            else:
                #just linear output
                modules.append(_get_activation_func("identity"))

        self.mu_net = nn.Sequential(*modules).to(device)
        logstd = cfg_network.get("actor_logstd_init", -1.0)

        #log std not parameterized by NN, just a learned vector for each action dim
        self.logstd = nn.Parameter(torch.ones(action_dim, dtype=torch.float32, device=device) * logstd)


    def get_logstd(self):
        return self.logstd

    def forward(self, obs, deterministic = False):
        mu = self.mu_net(obs)
        if deterministic:
            return mu
        else:
            std = torch.exp(self.logstd)
            dist = Normal(mu, std) #add noise 
            return dist.rsample()   



class CriticMLP(nn.Module):
    def __init__(self, obs_dim, cfg_network, device= "cuda:0"):
        super().__init__()
        layer_dims = [obs_dim] + cfg_network["critic_mlp"]["units"] + [1]

        #specific initialization pattern used in paper:
        init = lambda m: _init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        modules = []
        for i in range(len(layer_dims) - 1):
            modules.append(init(nn.Linear(layer_dims[i], layer_dims[i + 1])))
            if i < len(layer_dims) - 2:
                #add activate and layernorm after linear layer
                modules.append(_get_activation_func(cfg_network["critic_mlp"]["activation"]))
                modules.append(nn.LayerNorm(layer_dims[i + 1]))
            #no output layer special handling in this case

        self.critic = nn.Sequential(*modules).to(device)

    def forward(self, observations):
        return self.critic(observations)
