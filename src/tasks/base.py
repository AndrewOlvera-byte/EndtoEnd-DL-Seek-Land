from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import lightning.pytorch as pl

from src.registry import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY


class LitTaskBase(pl.LightningModule):
    """Shared Lightning base with optimizer/scheduler wiring and optional EMA.

    Subclasses should implement training/validation/test logic and call
    maybe_update_ema() at the end of training_step if EMA is enabled.
    """

    def __init__(
        self,
        model: Optional[nn.Module],
        optim_name: Optional[str],
        optim_kwargs: Optional[Dict[str, Any]] = None,
        sched_name: Optional[str] = None,
        sched_kwargs: Optional[Dict[str, Any]] = None,
        ema_enable: bool = False,
        ema_decay: float = 0.9999,
        auto_scale_lr_batch: Optional[int] = 256,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])  # keep model out of hparams to avoid large dumps
        self.model: Optional[nn.Module] = model
        self.ema_enable: bool = bool(ema_enable)
        self.ema_decay: float = float(ema_decay)
        self.model_ema: Optional[nn.Module] = None
        self.auto_scale_lr_batch: Optional[int] = auto_scale_lr_batch
        if self.ema_enable and self.model is not None:
            self._init_ema()

    def _init_ema(self) -> None:
        import copy

        self.model_ema = copy.deepcopy(self.model).eval() if self.model is not None else None
        if self.model_ema is not None:
            for p in self.model_ema.parameters():
                p.requires_grad_(False)

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - thin wrapper
        if self.model is None:
            raise RuntimeError("No model attached to this task.")
        return self.model(*args, **kwargs)

    def maybe_update_ema(self) -> None:
        if not self.ema_enable or self.model_ema is None or self.model is None:
            return
        with torch.no_grad():
            msd = self.model.state_dict()
            for k, v in self.model_ema.state_dict().items():
                if k in msd:
                    v.copy_(v * self.ema_decay + msd[k] * (1.0 - self.ema_decay))

    def configure_optimizers(self):  # type: ignore[override]
        # Tasks that perform manual optimization may choose to return None
        if self.hparams.get("optim_name", None) is None:
            return None

        optim_kwargs = dict(self.hparams.get("optim_kwargs", {}) or {})

        # Optional LR auto scaling by batch size relative to a base reference
        if (
            "lr" in optim_kwargs
            and self.trainer is not None
            and hasattr(self.trainer, "datamodule")
            and getattr(self.trainer.datamodule, "batch_size", None) is not None
            and self.auto_scale_lr_batch is not None
        ):
            base_lr = float(optim_kwargs["lr"])
            batch_size = int(self.trainer.datamodule.batch_size)  # type: ignore[attr-defined]
            scale = batch_size / float(self.auto_scale_lr_batch)
            optim_kwargs["lr"] = base_lr * scale

        # Support AdamW param group split when model is present
        if (
            self.hparams.get("optim_name") == "adamw"
            and self.model is not None
        ):
            optim_kwargs = {
                **optim_kwargs,
                "model": self.model,
                "no_decay_keys": tuple(getattr(self.hparams, "no_decay_keys", ("bias", "norm"))),
            }

        optimizer = OPTIMIZER_REGISTRY.build(self.hparams["optim_name"], self.parameters(), **optim_kwargs)

        sched_name: Optional[str] = self.hparams.get("sched_name", None)
        if not sched_name:
            return optimizer

        sched_kwargs = dict(self.hparams.get("sched_kwargs", {}) or {})
        scheduler = SCHEDULER_REGISTRY.build(sched_name, optimizer, **sched_kwargs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }



