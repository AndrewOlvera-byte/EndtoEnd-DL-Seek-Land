from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src.registry import register_model


class BCPolicy(nn.Module):
    """Simple MLP policy for continuous actions with tanh squashing.

    Maps observation -> action in [-1, 1].
    """

    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int] = (256, 128), act_dim: int = 9) -> None:
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))

    def policy(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)


class GaussianActorCritic(nn.Module):
    """Actor-Critic with Gaussian policy and state-value head.

    Policy outputs mean; log_std is a learned parameter vector.
    """

    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int], act_dim: int, init_log_std: float = -0.5) -> None:
        super().__init__()
        activation = nn.Tanh()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(activation)
            last = h
        self.body = nn.Sequential(*layers)
        self.mu_head = nn.Linear(last, act_dim)
        self.value_head = nn.Linear(last, 1)
        self.log_std = nn.Parameter(torch.ones(act_dim) * float(init_log_std))

    def policy(self, x):
        # Accept either a Tensor (obs) or a TensorDict with observation fields
        if hasattr(x, "get"):
            td = x
            # observation may be nested: td["observation"]["observation"] (Gym dict obs)
            obs = None
            try:
                obs = td.get(("observation", "observation"))
            except Exception:
                try:
                    obs = td.get("observation")
                except Exception:
                    obs = None
            if obs is None:
                raise RuntimeError("Policy expected 'observation' in TensorDict")
            z = self.body(obs)
            mu = self.mu_head(z)
            std = torch.exp(self.log_std).expand_as(mu)
            dist = torch.distributions.Normal(mu, std)
            action = torch.tanh(dist.rsample())
            td.set("action", action)
            return td
        else:
            obs = x
            z = self.body(obs)
            mu = self.mu_head(z)
            std = torch.exp(self.log_std).expand_as(mu)
            dist = torch.distributions.Normal(mu, std)
            action = torch.tanh(dist.rsample())
            return action

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.body(obs)
        return self.value_head(z).squeeze(-1)


@register_model("bc_policy")
def build_bc_policy(obs_dim: int, hidden_sizes: Sequence[int] = (256, 128), act_dim: int = 9) -> nn.Module:
    return BCPolicy(obs_dim=obs_dim, hidden_sizes=list(hidden_sizes), act_dim=act_dim)


@register_model("gaussian_actor_critic")
def build_gac(obs_dim: int, hidden_sizes: Sequence[int], act_dim: int, init_log_std: float = -0.5) -> nn.Module:
    return GaussianActorCritic(obs_dim=obs_dim, hidden_sizes=list(hidden_sizes), act_dim=act_dim, init_log_std=init_log_std)


