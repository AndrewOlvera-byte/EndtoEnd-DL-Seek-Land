from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src.registry import register_model


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], output_dim: int, activation: str = "relu") -> None:
        super().__init__()
        act: nn.Module
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            raise ValueError("Unsupported activation")
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(act)
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCriticMLP(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int], action_dim: int, activation: str = "tanh") -> None:
        super().__init__()
        act: nn.Module = nn.Tanh() if activation == "tanh" else nn.ReLU()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(act)
            last = h
        self.body = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, action_dim)
        self.value_head = nn.Linear(last, 1)

    def policy(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.body(obs)
        return self.policy_head(z)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.body(obs)
        return self.value_head(z).squeeze(-1)


@register_model("mlp")
def build_mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int, activation: str = "relu") -> nn.Module:
    return MLP(input_dim=input_dim, hidden_sizes=list(hidden_sizes), output_dim=output_dim, activation=activation)


@register_model("actor_critic_mlp")
def build_actor_critic_mlp(obs_dim: int, hidden_sizes: Sequence[int], action_dim: int, activation: str = "tanh") -> nn.Module:
    return ActorCriticMLP(obs_dim=obs_dim, hidden_sizes=list(hidden_sizes), action_dim=action_dim, activation=activation)


