from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Tuple

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from pathlib import Path
import subprocess


def _run_training_with_overrides(overrides: List[str]) -> Tuple[bool, Dict[str, Any]]:
    # Force Hydra to write into the trial working directory for easy metric pickup
    overrides = list(overrides) + [
        "hydra.run.dir=.",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
        "hydra.output_subdir=null",
    ]
    cmd = [
        "python",
        "-m",
        "src.train_lightning",
    ] + [f"+{ov}" if not ov.startswith("hydra.") else ov for ov in overrides]
    env = dict(os.environ)
    # Ensure src package is importable inside Ray trial subprocess
    # Get the workspace root (assuming this script is in src/)
    workspace_root = str(Path(__file__).parent.parent.absolute())
    env["PYTHONPATH"] = env.get("PYTHONPATH", "")
    if workspace_root not in env["PYTHONPATH"].split(os.pathsep):
        env["PYTHONPATH"] = (env["PYTHONPATH"] + (os.pathsep if env["PYTHONPATH"] else "")) + workspace_root
    # Suppress verbose output by redirecting to devnull
    import subprocess as sp
    proc = subprocess.run(cmd, check=False, env=env, stdout=sp.DEVNULL, stderr=sp.PIPE, text=True)
    metrics: Dict[str, Any] = {}
    try:
        summary = json.loads(Path("best/summary.json").read_text())
        metrics = summary.get("metrics", {})
        # also include best_score under a generic key if available
        if summary.get("best_score") is not None:
            metrics["best_score"] = float(summary["best_score"])  # type: ignore[index]
    except Exception as e:
        print(f"[warn] could not read metrics: {e}")
    ok = proc.returncode == 0
    return ok, metrics


def _trainable(config: Dict[str, Any], metric_key: str):
    # Convert flat dict to Hydra-style overrides
    overrides = []
    for k, v in config.items():
        if isinstance(v, bool):
            v = str(v).lower()
        elif isinstance(v, (list, tuple)):
            # Convert list/tuple to Hydra list syntax: [512,256]
            v = "[" + ",".join(str(x) for x in v) + "]"
        overrides.append(f"{k}={v}")
    ok, metrics = _run_training_with_overrides(overrides)
    # Prefer the designated metric key; fallback to best_score if present
    value = metrics.get(metric_key, metrics.get("best_score", None))
    if value is not None:
        tune.report({metric_key: float(value)})
    else:
        # still report something to unblock schedulers
        tune.report({metric_key: -1e9})


def _postprocess_topk(analysis, metric: str, mode: str, topk: int) -> List[Dict[str, Any]]:
    trials = [t for t in analysis.trials if t.last_result and metric in t.last_result]
    reverse = True if mode == "max" else False
    trials.sort(key=lambda t: t.last_result.get(metric, float("nan")), reverse=reverse)
    top = trials[:topk]
    return [t.config for t in top]


def _save_topk(which: str, topk_cfgs: List[Dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"topk_{which}.json"
    path.write_text(json.dumps(topk_cfgs, indent=2))
    return path


def _run_topk(which: str, cfgs: List[Dict[str, Any]]):
    for i, cfg in enumerate(cfgs, start=1):
        print(f"\n[topk] Running {which} #{i}/{len(cfgs)}")
        overrides = []
        for k, v in cfg.items():
            if isinstance(v, bool):
                v = str(v).lower()
            elif isinstance(v, (list, tuple)):
                # Convert list/tuple to Hydra list syntax: [512,256]
                v = "[" + ",".join(str(x) for x in v) + "]"
            overrides.append(f"{k}={v}")
        _run_training_with_overrides(overrides)


def search_bc(debug: bool = False, topk: int = 0, run_topk: bool = False):
    metric = "val_loss"
    mode = "min"
    max_iters = 3 if debug else 50

    space = {
        "model.hidden_sizes": tune.choice([[256, 128], [512, 256]]),
        "optim.lr": tune.loguniform(1e-4, 3e-3),
        "optim.weight_decay": tune.loguniform(1e-6, 1e-3),
        "optim.max_epochs": max_iters,
        "data.dataset_id": tune.choice(["kitchen-complete-v1", "kitchen-mixed-v1", "kitchen-partial-v1"]),
        "io.batch_size": tune.choice([256, 512, 1024]),
    }
    scheduler = ASHAScheduler(metric=metric, mode=mode, max_t=max_iters, grace_period=1, reduction_factor=2)
    searcher = HyperOptSearch(metric=metric, mode=mode)

    print(f"\n{'='*80}")
    print(f"Starting BC Hyperparameter Search")
    print(f"{'='*80}")
    print(f"Metric: {metric} ({mode})")
    print(f"Trials: {2 if debug else 32}")
    print(f"Max iterations: {max_iters}")
    print(f"{'='*80}\n")

    analysis = tune.run(
        tune.with_parameters(_trainable, metric_key=metric),
        name="bc_tune",
        scheduler=scheduler,
        search_alg=searcher,
        num_samples=2 if debug else 32,
        resources_per_trial={"cpu": 4, "gpu": 1},
        config={
            "task.name": "bc",
            "task.action_space": "continuous",
            "model.name": "bc_policy",
            "data.name": "minari_kitchen_bc",
            "loss.name": "mse",
            **space,
        },
        verbose=1,  # Reduce verbosity
    )

    if topk > 0:
        top_cfgs = _postprocess_topk(analysis, metric=metric, mode=mode, topk=topk)
        out = _save_topk("bc", top_cfgs, Path(os.environ.get("OUTPUT_DIR", "output")) / "ray")
        print(f"\n{'='*80}")
        print(f"[BC] Saved top-{topk} configs to {out}")
        print(f"{'='*80}")
        if run_topk:
            print(f"\n{'='*80}")
            print(f"Running top-{topk} BC configurations for full training")
            print(f"{'='*80}")
            _run_topk("bc", top_cfgs)


def search_rl(debug: bool = False, topk: int = 0, run_topk: bool = False):
    metric = "eval_return"
    mode = "max"
    max_iters = 3 if debug else 500
    space = {
        "model.hidden_sizes": tune.choice([[256, 256], [512, 256]]),
        "optim.lr": tune.loguniform(1e-5, 3e-4),
        "task.rollout_steps": tune.choice([256, 512, 1024]),
        "task.frames_per_batch": tune.choice([4096, 8192, 16384]),
        "task.ppo_clip_coef": tune.choice([0.1, 0.2, 0.3]),
        "task.ppo_update_epochs": tune.choice([5, 10]),
        "optim.max_epochs": max_iters,
    }
    scheduler = ASHAScheduler(metric=metric, mode=mode, max_t=max_iters, grace_period=1, reduction_factor=2)
    searcher = HyperOptSearch(metric=metric, mode=mode)

    print(f"\n{'='*80}")
    print(f"Starting RL Hyperparameter Search")
    print(f"{'='*80}")
    print(f"Metric: {metric} ({mode})")
    print(f"Trials: {2 if debug else 24}")
    print(f"Max iterations: {max_iters}")
    print(f"{'='*80}\n")

    analysis = tune.run(
        tune.with_parameters(_trainable, metric_key=metric),
        name="rl_tune",
        scheduler=scheduler,
        search_alg=searcher,
        num_samples=2 if debug else 24,
        resources_per_trial={"cpu": 8, "gpu": 1},
        config={
            "task.name": "rl",
            "model.name": "gaussian_actor_critic",
            "env._target_": "src.envs.franka_kitchen.build_franka_kitchen_env",
            **space,
        },
        verbose=1,  # Reduce verbosity
    )

    if topk > 0:
        top_cfgs = _postprocess_topk(analysis, metric=metric, mode=mode, topk=topk)
        out = _save_topk("rl", top_cfgs, Path(os.environ.get("OUTPUT_DIR", "output")) / "ray")
        print(f"\n{'='*80}")
        print(f"[RL] Saved top-{topk} configs to {out}")
        print(f"{'='*80}")
        if run_topk:
            print(f"\n{'='*80}")
            print(f"Running top-{topk} RL configurations for full training")
            print(f"{'='*80}")
            _run_topk("rl", top_cfgs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=["bc", "rl"])  # which search
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--run-topk", action="store_true")
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    if args.which == "bc":
        search_bc(debug=args.debug, topk=args.topk, run_topk=args.run_topk)
    else:
        search_rl(debug=args.debug, topk=args.topk, run_topk=args.run_topk)


