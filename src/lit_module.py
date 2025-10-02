from typing import Any, Dict, Optional
import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl
from src.registry import LOSS_REGISTRY, OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY

class LitClassifier(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optim_name: str,
        optim_kwargs: Dict[str, Any],
        sched_name: str,
        sched_kwargs: Dict[str, Any],
        ema_enable: bool = False,
        ema_decay: float = 0.9999,
        auto_scale_lr_batch: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "criterion"])
        self.model = model
        self.criterion = criterion
        # EMA of weights
        self.ema_enable = bool(ema_enable)
        self.ema_decay = float(ema_decay)
        self.model_ema: Optional[nn.Module] = None
        if self.ema_enable:
            import copy
            self.model_ema = copy.deepcopy(self.model).eval()
            for p in self.model_ema.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optim_kwargs = dict(self.hparams.optim_kwargs)
        # LR auto scaling by batch size relative to 256
        if (
            "lr" in optim_kwargs
            and self.trainer is not None
            and hasattr(self.trainer, "datamodule")
            and hasattr(self.trainer.datamodule, "batch_size")
        ):
            base_lr = float(optim_kwargs["lr"])
            batch_size = int(self.trainer.datamodule.batch_size)
            scale = batch_size / 256.0
            optim_kwargs["lr"] = base_lr * scale

        # No-decay param groups require passing model and keys
        if self.hparams.optim_name == "adamw":
            optim_kwargs = {
                **optim_kwargs,
                "model": self.model,
                "no_decay_keys": tuple(getattr(self.hparams, "no_decay_keys", ("bias", "norm", "pos_embed", "cls_token"))),
            }

        opt = OPTIMIZER_REGISTRY.build(self.hparams.optim_name, self.parameters(), **optim_kwargs)
        sched = SCHEDULER_REGISTRY.build(self.hparams.sched_name, opt, **self.hparams.sched_kwargs)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch"
            }
        }
    
    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        with torch.no_grad():
            if y.ndim == 2:
                # soft targets case: use argmax for targets
                pred = logits.argmax(dim=1)
                targ = y.argmax(dim=1)
                acc = (pred == targ).float().mean()
            else:
                pred = logits.argmax(dim=1)
                acc = (pred == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def training_step(self, batch, _):
        assert self.training, "Expected module to be in train() mode during training_step"
        loss = self._shared_step(batch, "train")
        # EMA update
        if self.ema_enable and self.model_ema is not None:
            with torch.no_grad():
                msd = self.model.state_dict()
                for k, v in self.model_ema.state_dict().items():
                    if k in msd:
                        v.copy_(v * self.ema_decay + msd[k] * (1.0 - self.ema_decay))
        return loss

    def validation_step(self, batch, _):
        assert not self.training, "Expected module to be in eval() mode during validation_step"
        if self.ema_enable and self.model_ema is not None:
            # Evaluate EMA weights if enabled
            model_backup = self.model
            self.model = self.model_ema
            self._shared_step(batch, "val")
            self.model = model_backup
        else:
            self._shared_step(batch, "val")

    def test_step(self, batch, _):
        assert not self.training, "Expected module to be in eval() mode during test_step"
        self._shared_step(batch, "test")