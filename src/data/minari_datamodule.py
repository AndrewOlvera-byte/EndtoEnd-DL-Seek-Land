from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

try:
    import minari  # type: ignore
except Exception as _e:  # pragma: no cover - optional dependency at import time
    minari = None

from src.registry import register_datamodule


class _MinariTransitionDataset(Dataset):
    """Transition dataset for Minari datasets.

    Iterates over (obs, act) pairs extracted from all episodes.
    Optionally applies observation normalization using provided stats.
    """

    def __init__(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        obs_mean: Optional[torch.Tensor] = None,
        obs_std: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        assert observations.shape[0] == actions.shape[0]
        self.observations = observations
        self.actions = actions
        self.obs_mean = obs_mean
        self.obs_std = obs_std

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = self.observations[index]
        act = self.actions[index]
        if self.obs_mean is not None and self.obs_std is not None:
            obs = (obs - self.obs_mean) / (self.obs_std + 1e-6)
        return obs, act


def _stack_transitions_from_minari(ds) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts and stacks all transitions (observation/action) from a Minari dataset.

    Observations are taken from the dict key 'observation'.
    """
    obs_list: List[torch.Tensor] = []
    act_list: List[torch.Tensor] = []
    for episode in ds.iterate_episodes():
        # Each episode contains numpy arrays; convert to torch tensors
        obs = episode.observations["observation"]  # shape [T, obs_dim]
        acts = episode.actions  # shape [T, act_dim]
        obs_list.append(torch.as_tensor(obs, dtype=torch.float32))
        act_list.append(torch.as_tensor(acts, dtype=torch.float32))
    observations = torch.cat(obs_list, dim=0)
    actions = torch.cat(act_list, dim=0)
    return observations, actions


@register_datamodule("minari_kitchen_bc")
class MinariKitchenBCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_id: str = "kitchen-complete-v1",
        split_ratio: float = 0.95,
        batch_size: int = 512,
        num_workers: int = 8,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        normalize_obs: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.dataset_id = dataset_id
        self.split_ratio = float(split_ratio)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.persistent_workers = bool(persistent_workers)
        self.pin_memory = bool(pin_memory)
        self.normalize_obs = bool(normalize_obs)
        self.cache_dir = cache_dir

        self.ds_train: Optional[Dataset] = None
        self.ds_val: Optional[Dataset] = None

        # Exposed after setup
        self.obs_dim: Optional[int] = None
        self.act_dim: Optional[int] = None
        self.obs_mean: Optional[torch.Tensor] = None
        self.obs_std: Optional[torch.Tensor] = None

    def prepare_data(self) -> None:  # type: ignore[override]
        if minari is None:
            raise RuntimeError("minari is not installed. Please add it to requirements.")
        # Download once
        try:
            if self.cache_dir is not None:
                # minari respects MINARI_DATA_DIR env; set temporarily if requested
                import os

                old = os.environ.get("MINARI_DATA_DIR", None)
                os.environ["MINARI_DATA_DIR"] = str(self.cache_dir)
                try:
                    minari.download_dataset(self.dataset_id)
                finally:
                    if old is not None:
                        os.environ["MINARI_DATA_DIR"] = old
                    else:
                        os.environ.pop("MINARI_DATA_DIR", None)
            else:
                minari.download_dataset(self.dataset_id)
        except Exception:
            # If download fails, load may still work if already cached
            pass

    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        assert minari is not None
        ds = minari.load_dataset(self.dataset_id)

        observations, actions = _stack_transitions_from_minari(ds)
        assert observations.ndim == 2 and actions.ndim == 2
        self.obs_dim = int(observations.shape[1])
        self.act_dim = int(actions.shape[1])

        # Compute normalization stats from the whole set, then split
        if self.normalize_obs:
            self.obs_mean = observations.mean(dim=0)
            self.obs_std = observations.std(dim=0).clamp_min(1e-6)

        N = observations.shape[0]
        n_train = int(N * self.split_ratio)
        train_obs = observations[:n_train]
        val_obs = observations[n_train:]
        train_act = actions[:n_train]
        val_act = actions[n_train:]

        self.ds_train = _MinariTransitionDataset(train_obs, train_act, self.obs_mean, self.obs_std)
        self.ds_val = _MinariTransitionDataset(val_obs, val_act, self.obs_mean, self.obs_std)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self.ds_train is not None
        return self._loader(self.ds_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self.ds_val is not None
        return self._loader(self.ds_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self.ds_val is not None
        return self._loader(self.ds_val, shuffle=False)


