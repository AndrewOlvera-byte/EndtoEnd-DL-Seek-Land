from __future__ import annotations

from typing import Any, Dict

from lightning.pytorch.callbacks import TQDMProgressBar


class OneBasedTQDMProgressBar(TQDMProgressBar):
    def on_train_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[override]
        super().on_train_epoch_start(trainer, pl_module)
        try:
            if self.main_progress_bar is not None:
                self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch + 1}")
        except Exception:
            pass

    def get_metrics(self, trainer, pl_module) -> Dict[str, Any]:  # type: ignore[override]
        metrics = super().get_metrics(trainer, pl_module)
        # Lightning reports 0-based epoch index; display 1-based
        try:
            if "epoch" in metrics:
                value = metrics["epoch"]
                if isinstance(value, (int, float)):
                    metrics["epoch"] = value + 1
                else:
                    # best-effort for tensors/other types
                    metrics["epoch"] = int(value) + 1
        except Exception:
            pass
        return metrics


