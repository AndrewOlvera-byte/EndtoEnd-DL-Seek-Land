from __future__ import annotations

from typing import Any, Optional, Tuple

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from torchvision import transforms

from src.registry import register_datamodule


class _TrajectoryDataset(Dataset):
    """Minimal trajectory dataset reading .pt files with (obs, act) tensors.

    Expected directory structure:
      root/
        train/
          0000.pt, 0001.pt, ...  where each file stores a dict with keys 'obs', 'act'
        val/
          ... same format ...
    """

    def __init__(self, directory: str | Path, transform: Optional[Any] = None) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.files = sorted([p for p in self.directory.glob("*.pt")])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.files[index]
        payload = torch.load(path)
        obs = payload["obs"]
        act = payload["act"]
        if self.transform is not None:
            obs = self.transform(obs)
        return obs, act


@register_datamodule("trajectory_bc")
class TrajectoryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        obs_is_image: bool = False,
        normalize_mean: Optional[Tuple[float, float, float]] = None,
        normalize_std: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.obs_is_image = bool(obs_is_image)
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.ds_train: Optional[Dataset] = None
        self.ds_val: Optional[Dataset] = None

    def _build_transform(self) -> Optional[Any]:
        if not self.obs_is_image:
            return None
        tx = []
        tx.append(transforms.ConvertImageDtype(torch.float32))
        if self.normalize_mean is not None and self.normalize_std is not None:
            tx.append(transforms.Normalize(self.normalize_mean, self.normalize_std))
        return transforms.Compose(tx)

    def setup(self, stage: Optional[str] = None) -> None:
        transform = self._build_transform()
        self.ds_train = _TrajectoryDataset(self.root / "train", transform=transform)
        self.ds_val = _TrajectoryDataset(self.root / "val", transform=transform)

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

    def train_dataloader(self) -> DataLoader:
        assert self.ds_train is not None
        return self._loader(self.ds_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.ds_val is not None
        return self._loader(self.ds_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self.ds_val is not None
        return self._loader(self.ds_val, shuffle=False)


