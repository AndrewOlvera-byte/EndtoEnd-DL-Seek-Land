from __future__ import annotations

from typing import List, Optional

import torch

import gymnasium as gym

try:  # pragma: no cover - import guard
    import gymnasium_robotics
except Exception:  # pragma: no cover - optional at import
    gymnasium_robotics = None

from torchrl.envs import GymWrapper

from src.registry import register_env


@register_env("franka_kitchen")
def build_franka_kitchen_env(
    tasks_to_complete: Optional[List[str]] = None,
    device: str | torch.device = "cpu",
) -> GymWrapper:
    if gymnasium_robotics is None:
        raise RuntimeError("gymnasium-robotics is not installed. Please add it to requirements.")
    gym.register_envs(gymnasium_robotics)
    env = gym.make(
        "FrankaKitchen-v1",
        tasks_to_complete=tasks_to_complete or [
            "microwave",
            "kettle",
            "slide_cabinet",
            "top_burner",
        ],
    )
    return GymWrapper(env, device=device)


