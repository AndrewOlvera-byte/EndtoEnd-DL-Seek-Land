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
from src.registry import (
    MODEL_REGISTRY,
    DATAMODULE_REGISTRY,
    LOSS_REGISTRY,
    TASK_REGISTRY,
)
# Side-effect imports to populate registries
# Keep only relevant registrations
import src.models.mlp  # noqa: F401
import src.models.policies  # noqa: F401
import src.data.trajectory_datamodule  # noqa: F401
import src.data.minari_datamodule  # noqa: F401
import src.tasks.bc  # noqa: F401
import src.tasks.rl  # noqa: F401
import src.optimizers  # noqa: F401
import src.losses  # noqa: F401
from src.utils.mixup_scheduler import MixupCutmixScheduler

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Only print config if not running from Ray Tune (less noise)
    if not os.environ.get("TUNE_ORIG_WORKING_DIR"):
        print(OmegaConf.to_yaml(cfg))
    # Persist resolved config in the Hydra run dir (cwd is already the run dir)
    try:
        Path("config_dump.yaml").write_text(OmegaConf.to_yaml(cfg))
    except Exception:
        pass
    set_seed(cfg.seed)
    speed_setup(cfg.channels_last, cfg.cudnn_benchmark)

    # Build datamodule first (if any) so we can infer dims for model
    datamodule = None
    if hasattr(cfg, "data") and cfg.get("data") is not None:
        if isinstance(cfg.data, DictConfig) and "_target_" in cfg.data:
            datamodule = instantiate(
                cfg.data,
                batch_size=cfg.io.batch_size,
                num_workers=cfg.io.num_workers,
                prefetch_factor=cfg.io.prefetch_factor,
                persistent_workers=cfg.io.persistent_workers,
                pin_memory=cfg.io.pin_memory,
            )
        elif hasattr(cfg.data, "name"):
            datamodule = DATAMODULE_REGISTRY.build(
                cfg.data.name,
                **{k: v for k, v in dict(cfg.data).items() if k != "name"},
                batch_size=cfg.io.batch_size,
                num_workers=cfg.io.num_workers,
                prefetch_factor=cfg.io.prefetch_factor,
                persistent_workers=cfg.io.persistent_workers,
                pin_memory=cfg.io.pin_memory,
            )
        # Pre-setup to expose dims if available
        try:
            datamodule.prepare_data()
        except Exception:
            pass
        try:
            datamodule.setup()
        except Exception:
            pass

    # Instantiate model: prefer Hydra instantiate when _target_ is provided; otherwise use registry by name
    # If datamodule exposed dims, override config fields if present
    model_cfg = dict(cfg.model) if not (isinstance(cfg.model, DictConfig) and "_target_" in cfg.model) else cfg.model
    if datamodule is not None and hasattr(datamodule, "obs_dim") and getattr(datamodule, "obs_dim") is not None:
        if isinstance(model_cfg, DictConfig):
            model_cfg["obs_dim"] = int(getattr(datamodule, "obs_dim"))
            if "action_dim" in model_cfg:
                model_cfg["action_dim"] = int(getattr(datamodule, "act_dim", model_cfg.get("action_dim", 0)))
            if "act_dim" in model_cfg:
                model_cfg["act_dim"] = int(getattr(datamodule, "act_dim", model_cfg.get("act_dim", 0)))
        else:
            model_cfg["obs_dim"] = int(getattr(datamodule, "obs_dim"))
            if "action_dim" in model_cfg:
                model_cfg["action_dim"] = int(getattr(datamodule, "act_dim", model_cfg.get("action_dim", 0)))
            if "act_dim" in model_cfg:
                model_cfg["act_dim"] = int(getattr(datamodule, "act_dim", model_cfg.get("act_dim", 0)))

    if isinstance(cfg.model, DictConfig) and "_target_" in cfg.model:
        model = instantiate(model_cfg)
    else:
        model = MODEL_REGISTRY.build(
            model_cfg["name"],
            **{k: v for k, v in dict(model_cfg).items() if k != "name"}
        )
    if cfg.channels_last:
        # Safe for 2D inputs; ViT consumes 4D tensors before patching
        model = model.to(memory_format=torch.channels_last)

    # Task selection: build BC or RL LightningModule
    task_type = str(getattr(cfg.task, "name", "")).lower()
    if task_type not in ("bc", "rl"):
        raise RuntimeError("cfg.task.name must be 'bc' or 'rl'")

    ema_enable = bool(getattr(cfg, "ema", {}).get("enable", False))
    ema_decay = float(getattr(cfg, "ema", {}).get("decay", 0.9999))

    optim_kwargs = {
        "lr": cfg.optim.lr,
        "betas": tuple(cfg.optim.betas) if hasattr(cfg.optim, "betas") else (0.9, 0.999),
        "weight_decay": cfg.optim.weight_decay if hasattr(cfg.optim, "weight_decay") else 0.0,
        "fused": bool(getattr(cfg.optim, "fused", False)),
    }
    sched_kwargs = {
        "t_max": int(getattr(cfg.sched, "t_max", 0)) if getattr(cfg.sched, "t_max", None) is not None else int(getattr(cfg.optim, "max_epochs", cfg.trainer.max_epochs)),
        "eta_min": getattr(cfg.sched, "eta_min", 1e-6),
        "warmup_epochs": int(getattr(cfg.optim, "warmup_epochs", 0)),
        "max_epochs": int(getattr(cfg.optim, "max_epochs", cfg.trainer.max_epochs)),
        "min_lr": float(getattr(cfg.optim, "min_lr", getattr(cfg.sched, "eta_min", 1e-6))),
    }

    if task_type == "bc":
        # Loss for BC if provided
        criterion = None
        if hasattr(cfg, "loss") and cfg.loss is not None:
            loss_cfg = OmegaConf.to_container(cfg.loss, resolve=True) if isinstance(cfg.loss, DictConfig) else dict(cfg.loss)
            loss_name = loss_cfg.get("name")
            loss_kwargs = {k: v for k, v in loss_cfg.items() if k != "name"}
            criterion = LOSS_REGISTRY.build(loss_name, **loss_kwargs)

        lit = TASK_REGISTRY.build(
            "bc",
            model=model,
            action_space=str(getattr(cfg.task, "action_space", "discrete")),
            criterion=criterion,
            optim_name=cfg.optim.name,
            optim_kwargs=optim_kwargs,
            sched_name=cfg.sched.name,
            sched_kwargs=sched_kwargs,
            ema_enable=ema_enable,
            ema_decay=ema_decay,
        )
    else:
        # RL: build env via Hydra if provided (build before task)
        if hasattr(cfg, "env") and isinstance(cfg.env, DictConfig) and "_target_" in cfg.env:
            env = instantiate(cfg.env)
        else:
            raise RuntimeError("RL task requires cfg.env with a valid _target_")

        # Optionally infer dims from env to rebuild model if needed
        try:
            obs_dim = None
            try:
                obs_spec = getattr(env, "observation_spec")
                obs_dim = int(obs_spec["observation"]["observation"].shape[-1])
            except Exception:
                try:
                    td0 = env.reset()
                    obs_dim = int(td0.get(("observation", "observation")).shape[-1])
                except Exception:
                    pass
            act_dim = None
            try:
                act_spec = getattr(env, "action_spec")
                act_dim = int(act_spec.shape[-1])
            except Exception:
                try:
                    gym_env = getattr(env, "_env", None)
                    act_dim = int(gym_env.action_space.shape[0]) if gym_env is not None else None
                except Exception:
                    pass
            if obs_dim is not None and act_dim is not None and hasattr(cfg, "model"):
                model = MODEL_REGISTRY.build(
                    cfg.model.name,
                    **{
                        **{k: v for k, v in dict(cfg.model).items() if k != "name"},
                        "obs_dim": obs_dim,
                        "act_dim": act_dim,
                        "action_dim": act_dim,
                    }
                )
                if cfg.channels_last:
                    model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass

        lit = TASK_REGISTRY.build(
            "rl",
            model=model,
            env=env,
            rollout_steps=int(getattr(cfg.task, "rollout_steps", 128)),
            frames_per_batch=int(getattr(cfg.task, "frames_per_batch", 1024)),
            optim_name=cfg.optim.name,
            optim_kwargs=optim_kwargs,
            sched_name=cfg.sched.name,
            sched_kwargs=sched_kwargs,
            ema_enable=ema_enable,
            ema_decay=ema_decay,
            ppo_clip_coef=float(getattr(cfg.task, "ppo_clip_coef", 0.2)),
            ppo_entropy_coef=float(getattr(cfg.task, "ppo_entropy_coef", 0.0)),
            ppo_value_coef=float(getattr(cfg.task, "ppo_value_coef", 0.5)),
            ppo_update_epochs=int(getattr(cfg.task, "ppo_update_epochs", 4)),
            gae_gamma=float(getattr(cfg.task, "gae_gamma", 0.99)),
            gae_lambda=float(getattr(cfg.task, "gae_lambda", 0.95)),
        )

    # Resolve Hydra run directory
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    # TensorBoard logger -> run_dir/tb/version_0
    tb_logger = TensorBoardLogger(save_dir=str(run_dir), name="tb")

    # Checkpoints -> run_dir/ckpts
    ckpt_dir = run_dir / "ckpts"
    monitor_metric = str(getattr(cfg.trainer, "monitor", "val_loss" if task_type == "bc" else "eval_return"))
    monitor_mode = str(getattr(cfg.trainer, "monitor_mode", "min" if task_type == "bc" else "max"))
    ckpt_filename = str(getattr(cfg.trainer, "ckpt_filename", "epoch{epoch:02d}-{monitor}"))
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=ckpt_filename,
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=int(getattr(cfg.trainer, "save_top_k", 3)),
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
                monitor=str(getattr(es_cfg, "monitor", monitor_metric)),
                mode=str(getattr(es_cfg, "mode", monitor_mode)),
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
    if task_type == "bc":
        assert datamodule is not None, "BC requires a datamodule"
        trainer.fit(lit, datamodule=datamodule)
        trainer.test(lit, datamodule=datamodule)
    else:
        trainer.fit(lit)
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
