from __future__ import annotations
import math

from typing import Any, Iterable, Tuple, List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler

from src.registry import register_optimizer, register_scheduler


def _named_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
    no_decay_keys: Tuple[str, ...] = ("bias", "norm", "pos_embed", "cls_token"),
) -> List[dict]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        in_no_decay = False
        for key in no_decay_keys:
            if key in name or ("norm" in key and ("norm" in no_decay_keys)):
                in_no_decay = True
                break
        if in_no_decay:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@register_optimizer("adamw")
def build_adamw(
    params: Iterable[torch.nn.Parameter],
    lr: float,
    betas: Tuple[float, float],
    weight_decay: float,
    fused: bool = False,
    model: torch.nn.Module | None = None,
    no_decay_keys: Tuple[str, ...] = ("bias", "norm", "pos_embed", "cls_token"),
) -> Optimizer:
    extra: dict = {}
    if fused and torch.cuda.is_available():
        try:
            extra["fused"] = True
        except TypeError:
            # Older torch without fused AdamW support
            pass
    param_groups: Iterable[torch.nn.Parameter] | List[dict]
    if model is not None:
        param_groups = _named_parameter_groups(model, weight_decay, no_decay_keys)
    else:
        param_groups = params
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, weight_decay=weight_decay, **extra)


class LinearWarmupThenCosine(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, max_epochs: int, min_lr: float, last_epoch: int = -1):
        self.warmup_epochs = int(max(0, warmup_epochs))
        self.max_epochs = int(max_epochs)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # During warmup: linearly increase from 0 to base lr
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = float(self.last_epoch + 1) / float(max(1, self.warmup_epochs))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        # After warmup: cosine decay to min_lr
        t = self.last_epoch - self.warmup_epochs
        T = max(1, self.max_epochs - self.warmup_epochs)
        return [self.min_lr + (base_lr - self.min_lr) * (1 + math.cos(math.pi * t / T)) / 2 for base_lr in self.base_lrs]


@register_scheduler("cosine")
def build_cosine(optimizer: Optimizer, t_max: int = None, eta_min: float = 1e-6, warmup_epochs: int = 0, max_epochs: int = None, min_lr: float = None) -> _LRScheduler:
    # Backward compatibility: if warmup_epochs=0 and no max_epochs/min_lr provided, fall back to plain cosine
    if (warmup_epochs or max_epochs or min_lr) is not None:
        if max_epochs is None:
            max_epochs = t_max if t_max is not None else 1
        if min_lr is None:
            min_lr = eta_min
        return LinearWarmupThenCosine(optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs, min_lr=min_lr)
    return CosineAnnealingLR(optimizer, T_max=(t_max if t_max is not None else 1), eta_min=eta_min)


