from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import lightning.pytorch as pl

from torchrl.envs import EnvBase
from torchrl.collectors import SyncDataCollector
from torchrl.objectives.value import GAE

from src.tasks.base import LitTaskBase
from src.registry import register_task


@register_task("rl")
class LitRL(LitTaskBase):
    """RL task using TorchRL SyncDataCollector and PPO objective.

    This module expects the `model` to provide two callables:
    - policy(obs) -> distribution parameters or actions
    - value(obs) -> state value estimation

    The `env_builder` callable should construct a TorchRL EnvBase (optionally
    vectorized) when called without arguments.
    """

    def __init__(
        self,
        model: nn.Module,
        env: EnvBase,
        rollout_steps: int,
        frames_per_batch: int,
        optim_name: Optional[str] = None,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        sched_name: Optional[str] = None,
        sched_kwargs: Optional[Dict[str, Any]] = None,
        ema_enable: bool = False,
        ema_decay: float = 0.9999,
        ppo_clip_coef: float = 0.2,
        ppo_entropy_coef: float = 0.0,
        ppo_value_coef: float = 0.5,
        ppo_update_epochs: int = 4,
        gae_gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        super().__init__(
            model=model,
            optim_name=optim_name,
            optim_kwargs=optim_kwargs,
            sched_name=sched_name,
            sched_kwargs=sched_kwargs,
            ema_enable=ema_enable,
            ema_decay=ema_decay,
            auto_scale_lr_batch=None,  # RL usually not scaled by dataloader batch size
        )

        self.env: EnvBase = env
        self.frames_per_batch = int(frames_per_batch)
        self.rollout_steps = int(rollout_steps)
        self.manual_backward_enabled = True
        self.automatic_optimization = False  # manual optimization in RL

        # Advantage estimator
        self.gae = GAE(gamma=gae_gamma, lmbda=gae_lambda, value_network=getattr(self.model, "value"))
        self.ppo_clip_coef = float(ppo_clip_coef)
        self.ppo_entropy_coef = float(ppo_entropy_coef)
        self.ppo_value_coef = float(ppo_value_coef)
        self.ppo_update_epochs = int(ppo_update_epochs)

    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        pass

    def on_fit_start(self) -> None:  # type: ignore[override]
        # Build collector on fit start to ensure device/trainer is ready
        device = self.device
        self.env.set_seed(self.trainer.global_rank if self.trainer is not None else 0)
        self.collector = SyncDataCollector(
            self.env,
            policy=getattr(self.model, "policy"),
            frames_per_batch=self.frames_per_batch,
            total_frames=-1,
            device=device,
        )

    def _atanh(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _log_prob_tanh_normal(self, action: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        # Inverse transform
        pre_tanh = self._atanh(action.clamp_(-0.999999, 0.999999))
        std = torch.exp(log_std)
        base = torch.distributions.Normal(mu, std)
        log_prob = base.log_prob(pre_tanh)
        # Tanh correction: sum over dims
        correction = 2.0 * (torch.log(torch.tensor(2.0, device=action.device)) - pre_tanh - torch.nn.functional.softplus(-2.0 * pre_tanh))
        return (log_prob - correction).sum(dim=-1)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        # Collector yields batches of trajectories; TorchRL loss consumes tensordicts
        opt = self.optimizers()
        assert opt is not None
        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_value_loss = torch.tensor(0.0, device=self.device)
        total_entropy = torch.tensor(0.0, device=self.device)
        total_return = torch.tensor(0.0, device=self.device)
        num_updates = 0

        for data in self.collector:
            with torch.no_grad():
                data = data.to(self.device)
                data = self.gae(value_network=getattr(self.model, "value"), tensordict=data)
                try:
                    total_return += data.get(("next","reward")).sum()
                except Exception:
                    total_return += torch.tensor(0.0, device=self.device)

            # Prepare tensors
            try:
                obs = data.get(("observation", "observation"))
            except Exception:
                obs = data.get("observation")
            actions = data.get("action")
            # Old log probs collected by policy; fallback to recompute if missing
            old_log_prob = None
            try:
                old_log_prob = data.get("log_prob")
            except Exception:
                pass

            # Normalize advantages
            adv = data.get("advantage")
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            v_target = data.get("value_target")

            for _ in range(self.ppo_update_epochs):
                # Compute current policy log prob and value
                mu = getattr(self.model, "policy")(obs)  # if policy returns actions when given Tensor, ensure it returns mu instead
                # Our policy returns actions; we need means and log_std. Access model internals directly.
                body = getattr(self.model, "body")(obs)
                mu = getattr(self.model, "mu_head")(body)
                log_std = getattr(self.model, "log_std")
                curr_log_prob = self._log_prob_tanh_normal(actions, mu, log_std)
                if old_log_prob is None:
                    old_log_prob = curr_log_prob.detach()
                ratio = torch.exp(curr_log_prob - old_log_prob)
                # Policy loss (clipped surrogate)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_coef, 1.0 + self.ppo_clip_coef) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values = getattr(self.model, "value")(obs)
                value_loss = torch.nn.functional.mse_loss(values, v_target)

                # Entropy bonus from base distribution
                entropy = (0.5 + 0.5 * torch.log(2 * torch.pi * torch.exp(2 * log_std))).sum(dim=-1).mean()

                loss = policy_loss + self.ppo_value_coef * value_loss - self.ppo_entropy_coef * entropy
                opt.zero_grad(set_to_none=True)
                self.manual_backward(loss)
                opt.step()

                total_policy_loss += policy_loss.detach()
                total_value_loss += value_loss.detach()
                total_entropy += entropy.detach()
                num_updates += 1

            # limit training_step to one collector batch per step to keep epoch granularity
            break

        # Logging
        if num_updates > 0:
            self.log("train_policy_loss", total_policy_loss / num_updates, prog_bar=True, on_epoch=True, on_step=False)
            self.log("train_value_loss", total_value_loss / num_updates, prog_bar=False, on_epoch=True, on_step=False)
            self.log("train_entropy", total_entropy / num_updates, prog_bar=False, on_epoch=True, on_step=False)
            self.log("train_return", total_return, prog_bar=True, on_epoch=True, on_step=False)

        # EMA update if enabled
        self.maybe_update_ema()
        return total_policy_loss if isinstance(total_policy_loss, torch.Tensor) else torch.tensor(0.0)

    def validation_step(self, batch: Any, batch_idx: int) -> None:  # type: ignore[override]
        # Evaluation episodes without learning; report episodic return
        returns = []
        with torch.no_grad():
            for _ in range(3):
                td = self.env.rollout(self.rollout_steps, policy=getattr(self.model, "policy"))
                try:
                    R = td.get(("next", "reward")).sum()
                except Exception:
                    R = torch.tensor(0.0, device=self.device)
                returns.append(R)
        if len(returns) > 0:
            stacked = torch.stack([r if isinstance(r, torch.Tensor) else torch.tensor(r, device=self.device) for r in returns])
            self.log("eval_return", stacked.mean(), prog_bar=True, on_epoch=True, on_step=False)


