import os
import json
import shutil
from pathlib import Path
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from src.utils.seed import set_seed
from src.utils.speed import speed_setup
from src.utils.progress import OneBasedTQDMProgressBar
from src.lit_module import LitClassifier
from src.registry import (
    MODEL_REGISTRY,
    DATAMODULE_REGISTRY,
    LOSS_REGISTRY,
)
# Side-effect imports to populate registries
import src.models.vit  # noqa: F401
import src.data.cifar10_datamodule  # noqa: F401
import src.optimizers  # noqa: F401
import src.losses  # noqa: F401
from src.utils.mixup_scheduler import MixupCutmixScheduler

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # Persist resolved config in the Hydra run dir (cwd is already the run dir)
    try:
        Path("config_dump.yaml").write_text(OmegaConf.to_yaml(cfg))
    except Exception:
        pass
    set_seed(cfg.seed)
    speed_setup(cfg.channels_last, cfg.cudnn_benchmark)

    # Instantiate model: prefer Hydra instantiate when _target_ is provided; otherwise use registry by name
    if isinstance(cfg.model, DictConfig) and "_target_" in cfg.model:
        model = instantiate(cfg.model)
    else:
        model = MODEL_REGISTRY.build(
            cfg.model.name,
            image_size=cfg.model.image_size,
            patch_size=cfg.model.patch_size,
            embed_dim=cfg.model.embed_dim,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            num_classes=cfg.model.num_classes,
            dropout=cfg.model.dropout,
            attn_dropout=cfg.model.attn_dropout,
        )
    if cfg.channels_last:
        # Safe for 2D inputs; ViT consumes 4D tensors before patching
        model = model.to(memory_format=torch.channels_last)

    if isinstance(cfg.data, DictConfig) and "_target_" in cfg.data:
        datamodule = instantiate(
            cfg.data,
            batch_size=cfg.io.batch_size,
            num_workers=cfg.io.num_workers,
            prefetch_factor=cfg.io.prefetch_factor,
            persistent_workers=cfg.io.persistent_workers,
            pin_memory=cfg.io.pin_memory,
        )
    else:
        datamodule = DATAMODULE_REGISTRY.build(
            cfg.data.name,
            root=cfg.data.root,
            download=cfg.data.download,
            mean=cfg.data.mean,
            std=cfg.data.std,
            batch_size=cfg.io.batch_size,
            num_workers=cfg.io.num_workers,
            prefetch_factor=cfg.io.prefetch_factor,
            persistent_workers=cfg.io.persistent_workers,
            pin_memory=cfg.io.pin_memory,
        )

    # Loss: build with only supported kwargs to avoid missing-attribute errors
    loss_cfg = OmegaConf.to_container(cfg.loss, resolve=True) if isinstance(cfg.loss, DictConfig) else dict(cfg.loss)
    loss_name = loss_cfg.get("name")
    loss_reduction = loss_cfg.get("reduction", "mean")
    loss_kwargs = {"reduction": loss_reduction}
    if loss_name == "cross_entropy":
        loss_kwargs["label_smoothing"] = float(loss_cfg.get("label_smoothing", 0.0))
    criterion = LOSS_REGISTRY.build(loss_name, **loss_kwargs)

    lit = LitClassifier(
        model=model,
        criterion=criterion,
        optim_name=cfg.optim.name,
        optim_kwargs={
            "lr": cfg.optim.lr,
            "betas": tuple(cfg.optim.betas),
            "weight_decay": cfg.optim.weight_decay,
            "fused": bool(getattr(cfg.optim, "fused", False)),
        },
        sched_name=cfg.sched.name,
        sched_kwargs={
            "t_max": int(cfg.sched.t_max) if cfg.sched.t_max is not None else int(cfg.optim.max_epochs),
            "eta_min": cfg.sched.eta_min,
            "warmup_epochs": int(getattr(cfg.optim, "warmup_epochs", 0)),
            "max_epochs": int(getattr(cfg.optim, "max_epochs", cfg.trainer.max_epochs)),
            "min_lr": float(getattr(cfg.optim, "min_lr", cfg.sched.eta_min)),
        },
        ema_enable=bool(getattr(cfg, "ema", {}).get("enable", False)),
        ema_decay=float(getattr(cfg, "ema", {}).get("decay", 0.9999)),
    )

    # Resolve Hydra run directory
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    # TensorBoard logger -> run_dir/tb/version_0
    tb_logger = TensorBoardLogger(save_dir=str(run_dir), name="tb")

    # Checkpoints -> run_dir/ckpts
    ckpt_dir = run_dir / "ckpts"
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="epoch{epoch:02d}-valacc{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )

    # Precision with bf16 fallback if unsupported
    requested_precision = str(cfg.precision)
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    precision = requested_precision
    if requested_precision.startswith("bf16") and not bf16_supported:
        precision = "16-mixed"

    # Optional model compile (Blackwell/CUDA 12.8 supported)
    compile_cfg = getattr(cfg.trainer, "compile", None)
    if compile_cfg and getattr(compile_cfg, "enable", False):
        try:
            model = torch.compile(
                model,
                mode=getattr(compile_cfg, "mode", "max-autotune"),
                fullgraph=getattr(compile_cfg, "fullgraph", False),
                dynamic=getattr(compile_cfg, "dynamic", True),
            )
        except Exception as e:
            print(f"[warn] torch.compile disabled: {e}")

    # Handle fused AdamW + gradient clipping incompatibility with a hard guard
    gradient_clip_val = float(getattr(cfg.trainer, "gradient_clip_val", 0.0))
    fused_requested = bool(getattr(cfg.optim, "fused", False))
    if fused_requested and gradient_clip_val > 0.0:
        raise RuntimeError(
            "Invalid configuration: optim.fused=True is not compatible with trainer.gradient_clip_val>0. "
            "Set optim.fused=false to keep clipping, or set trainer.gradient_clip_val=0.0 to keep fused."
        )

    # Build callbacks list
    callbacks = [ckpt_cb, OneBasedTQDMProgressBar(refresh_rate=1)]
    if bool(getattr(cfg.trainer, "lr_monitor", True)):
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    es_cfg = getattr(cfg.trainer, "early_stopping", None)
    if es_cfg and bool(getattr(es_cfg, "enable", False)):
        callbacks.append(
            EarlyStopping(
                monitor=str(getattr(es_cfg, "monitor", "val_acc")),
                mode=str(getattr(es_cfg, "mode", "max")),
                patience=int(getattr(es_cfg, "patience", 20)),
                min_delta=float(getattr(es_cfg, "min_delta", 0.0)),
            )
        )

    # Optional Mixup/CutMix schedule callback
    coll_cfg = getattr(cfg.data, "collator", None)
    sch_cfg = getattr(coll_cfg, "schedule", None) if coll_cfg is not None else None
    if sch_cfg and bool(getattr(sch_cfg, "enable", False)):
        callbacks.append(
            MixupCutmixScheduler(
                start=float(getattr(sch_cfg, "start", 1.0)),
                end=float(getattr(sch_cfg, "end", 0.0)),
                start_epoch=int(getattr(sch_cfg, "start_epoch", 0)),
                end_epoch=int(getattr(sch_cfg, "end_epoch", cfg.trainer.max_epochs)),
                schedule_type=str(getattr(sch_cfg, "type", "cosine")),
            )
        )

    accelerator = str(getattr(cfg.trainer, "accelerator", "auto"))

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        num_sanity_val_steps=getattr(cfg.trainer, "num_sanity_val_steps", 0),
        accumulate_grad_batches=getattr(cfg.trainer, "accumulate_grad_batches", 1),
        logger=tb_logger,
        callbacks=callbacks,
    )

    # Track total pipeline runtime (training + evaluation)
    _pipeline_t0 = time.perf_counter()
    trainer.fit(lit, datamodule=datamodule)
    trainer.test(lit, datamodule=datamodule)
    _total_time_sec = float(time.perf_counter() - _pipeline_t0)

    # After training: export best bundle under run_dir/best
    best_dir = run_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else None
    if best_ckpt_path and best_ckpt_path.exists():
        shutil.copy2(best_ckpt_path, best_dir / best_ckpt_path.name)

    # Export plain .pt state_dict for non-Lightning loading
    torch.save(lit.model.state_dict(), best_dir / "model.pt")

    # Write summary.json with key metrics and paths
    metrics = {}
    for k, v in trainer.callback_metrics.items():
        try:
            metrics[k] = float(v)
        except Exception:
            pass
    # Enumerate saved checkpoints (excluding last.ckpt) to verify top-k saving
    try:
        saved_ckpt_files = sorted([
            p.name for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt"
        ])
    except Exception:
        saved_ckpt_files = []

    # Collect top-k checkpoints and scores from the callback (best_k_models)
    top_k_entries = []
    try:
        best_k_models = getattr(ckpt_cb, "best_k_models", {}) or {}
        mode = getattr(ckpt_cb, "mode", "max")
        sortable = [
            {"filename": Path(path).name, "score": float(score)}
            for path, score in best_k_models.items()
        ]
        reverse = True if str(mode).lower() == "max" else False
        top_k_entries = sorted(sortable, key=lambda d: d["score"], reverse=reverse)
    except Exception:
        top_k_entries = []
    summary = {
        "best_checkpoint": best_ckpt_path.name if (best_ckpt_path and best_ckpt_path.exists()) else None,
        "best_score": float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else None,
        "tb_dir": str(run_dir / "tb"),
        "ckpt_dir": str(ckpt_dir),
        "metrics": metrics,
        # Checkpoint verification and configuration
        "save_top_k": int(getattr(ckpt_cb, "save_top_k", -1)),
        "saved_ckpts_count": len(saved_ckpt_files),
        "saved_ckpts": saved_ckpt_files,
        "top_k_checkpoints": top_k_entries,
        # Total runtime
        "total_time_sec": _total_time_sec,
    }
    (best_dir / "summary.json").write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
