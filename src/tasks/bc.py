from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.tasks.base import LitTaskBase
from src.registry import register_task


@register_task("bc")
class LitBehavioralCloning(LitTaskBase):
    """Behavioral Cloning task supporting discrete and continuous actions.

    Expected batch format: (obs, action)
    - discrete: action is LongTensor with shape [B]
    - continuous: action is FloatTensor with shape [B, action_dim]

    The model is expected to produce either:
    - discrete: logits of shape [B, num_actions]
    - continuous: action prediction (mean) of shape [B, action_dim]
    """

    def __init__(
        self,
        model: nn.Module,
        action_space: str,
        criterion: Optional[nn.Module] = None,
        optim_name: Optional[str] = None,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        sched_name: Optional[str] = None,
        sched_kwargs: Optional[Dict[str, Any]] = None,
        ema_enable: bool = False,
        ema_decay: float = 0.9999,
        auto_scale_lr_batch: Optional[int] = 256,
    ) -> None:
        super().__init__(
            model=model,
            optim_name=optim_name,
            optim_kwargs=optim_kwargs,
            sched_name=sched_name,
            sched_kwargs=sched_kwargs,
            ema_enable=ema_enable,
            ema_decay=ema_decay,
            auto_scale_lr_batch=auto_scale_lr_batch,
        )
        space = str(action_space).lower()
        if space not in ("discrete", "continuous"):
            raise ValueError("action_space must be 'discrete' or 'continuous'")
        self.action_space: str = space
        self.criterion: nn.Module
        if criterion is not None:
            self.criterion = criterion
        else:
            # Reasonable defaults
            self.criterion = nn.CrossEntropyLoss() if self.action_space == "discrete" else nn.MSELoss()

    def _forward_policy(self, obs: torch.Tensor) -> torch.Tensor:
        # Support models that expose a 'policy' method, else use forward
        if hasattr(self.model, "policy") and callable(getattr(self.model, "policy")):
            return getattr(self.model, "policy")(obs)  # type: ignore[misc]
        return self.forward(obs)

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        obs, act = batch
        logits_or_action = self._forward_policy(obs)
        if self.action_space == "discrete":
            # logits: [B, A], act: [B]
            loss = self.criterion(logits_or_action, act)
            with torch.no_grad():
                pred = logits_or_action.argmax(dim=-1)
                acc = (pred == act).float().mean()
                self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        else:
            # predicted actions: [B, D], act: [B, D]
            loss = self.criterion(logits_or_action, act)
            with torch.no_grad():
                mae = (logits_or_action - act).abs().mean()
                self.log(f"{stage}_mae", mae, prog_bar=True, on_step=False, on_epoch=True)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        loss = self._step(batch, stage="train")
        self.maybe_update_ema()
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _batch_idx: int) -> None:  # type: ignore[override]
        # If EMA enabled, evaluate EMA weights
        if self.ema_enable and self.model_ema is not None:
            model_backup = self.model
            self.model = self.model_ema
            self._step(batch, stage="val")
            self.model = model_backup
        else:
            self._step(batch, stage="val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _batch_idx: int) -> None:  # type: ignore[override]
        self._step(batch, stage="test")


