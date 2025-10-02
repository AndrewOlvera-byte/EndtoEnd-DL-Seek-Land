import math
import lightning.pytorch as pl


class MixupCutmixScheduler(pl.Callback):
    def __init__(
        self,
        start: float = 1.0,
        end: float = 0.0,
        start_epoch: int = 0,
        end_epoch: int | None = None,
        schedule_type: str = "cosine",
    ) -> None:
        super().__init__()
        self.start = float(start)
        self.end = float(end)
        self.start_epoch = int(start_epoch)
        self.end_epoch = end_epoch
        self.schedule_type = str(schedule_type).lower()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.end_epoch is None:
            self.end_epoch = int(trainer.max_epochs)
        self._apply(trainer, trainer.current_epoch)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._apply(trainer, trainer.current_epoch)

    def _interp(self, epoch: int) -> float:
        if epoch <= self.start_epoch:
            return self.start
        if epoch >= (self.end_epoch or epoch):
            return self.end
        t = (epoch - self.start_epoch) / max(1, (self.end_epoch - self.start_epoch))
        if self.schedule_type == "linear":
            return (1 - t) * self.start + t * self.end
        # cosine schedule by default
        return self.end + (self.start - self.end) * 0.5 * (1.0 + math.cos(math.pi * t))

    def _apply(self, trainer: pl.Trainer, epoch: int) -> None:
        dm = getattr(trainer, "datamodule", None)
        if dm is None or not hasattr(dm, "set_mixup_params"):
            return
        prob = self._interp(epoch)
        dm.set_mixup_params(prob=prob)


