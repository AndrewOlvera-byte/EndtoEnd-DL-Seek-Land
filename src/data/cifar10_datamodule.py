import os
from typing import Optional, Callable, List, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import lightning.pytorch as pl
from src.registry import register_datamodule
import random
import math

@register_datamodule("cifar10")
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        download: bool,
        mean,
        std,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        prefetch_factor: int,
        persistent_workers: bool,
        img_size: int = 32,
        aug: Optional[dict] = None,
        collator: Optional[dict] = None,
    ):
        super().__init__()
        self.root = root
        self.download = download
        self.mean, self.std = mean, std
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.img_size = img_size
        self.aug_cfg = aug or {}
        self.collator_cfg = collator or {"enable": False}
        self.collate_fn: Optional[Callable] = None

        # Live mixup/cutmix params (can be scheduled via callback)
        self._mix_prob = float((self.collator_cfg or {}).get("prob", 0.0))
        self._mixup_alpha = float((self.collator_cfg or {}).get("mixup_alpha", 0.0))
        self._cutmix_alpha = float((self.collator_cfg or {}).get("cutmix_alpha", 0.0))

    def set_mixup_params(
        self,
        prob: Optional[float] = None,
        mixup_alpha: Optional[float] = None,
        cutmix_alpha: Optional[float] = None,
    ):
        if prob is not None:
            self._mix_prob = max(0.0, min(1.0, float(prob)))
        if mixup_alpha is not None:
            self._mixup_alpha = max(0.0, float(mixup_alpha))
        if cutmix_alpha is not None:
            self._cutmix_alpha = max(0.0, float(cutmix_alpha))

    def prepare_data(self):
        datasets.CIFAR10(self.root, train=True,  download=self.download)
        datasets.CIFAR10(self.root, train=False, download=self.download)

    def setup(self, stage: Optional[str] = None):
        normalize = transforms.Normalize(self.mean, self.std)
        aug_list: List[transforms.Compose] = []
        # RandomCrop and Flip
        random_crop = (self.aug_cfg or {}).get("random_crop", {"size": self.img_size, "padding": 4})
        if random_crop:
            aug_list.append(transforms.RandomCrop(int(random_crop.get("size", self.img_size)), padding=int(random_crop.get("padding", 4))))
        if (self.aug_cfg or {}).get("random_flip", True):
            aug_list.append(transforms.RandomHorizontalFlip())
        # RandAugment
        randaug = (self.aug_cfg or {}).get("randaugment", {"enable": False})
        if bool(randaug.get("enable", False)):
            n = int(randaug.get("n", 2)); m = int(randaug.get("m", 9))
            try:
                aug_list.append(transforms.RandAugment(n=n, m=m))
            except Exception:
                pass
        # Compose train transforms
        train_tf = transforms.Compose([
            *aug_list,
            transforms.ToTensor(),
            normalize,
        ])
        # RandomErasing applied on tensor space
        random_erasing = (self.aug_cfg or {}).get("random_erasing", {"enable": False})
        self._train_post_tf: Optional[Callable] = None
        if bool(random_erasing.get("enable", False)):
            p = float(random_erasing.get("p", 0.25))
            self._train_post_tf = transforms.RandomErasing(p=p)

        test_tf = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        self.ds_train = datasets.CIFAR10(self.root, train=True, transform=train_tf)
        self.ds_val = datasets.CIFAR10(self.root, train=False, transform=test_tf)

        # Prepare optional Mixup/CutMix collate
        if bool(self.collator_cfg.get("enable", False)):
            num_classes = int(self.collator_cfg.get("num_classes", 10))

            def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
                return torch.nn.functional.one_hot(labels.to(torch.long), num_classes=num_classes).float()

            def _rand_bbox(W: int, H: int, lam: float) -> Tuple[int, int, int, int]:
                # From CutMix paper implementation
                cut_rat = math.sqrt(1.0 - lam)
                cut_w = int(W * cut_rat)
                cut_h = int(H * cut_rat)
                cx = random.randint(0, W)
                cy = random.randint(0, H)
                x1 = max(0, cx - cut_w // 2)
                y1 = max(0, cy - cut_h // 2)
                x2 = min(W, cx + cut_w // 2)
                y2 = min(H, cy + cut_h // 2)
                return x1, y1, x2, y2

            def mixup_cutmix_collate(batch: List[Tuple[torch.Tensor, int]]):
                images, targets = zip(*batch)
                images = torch.stack(list(images), dim=0)
                targets = torch.tensor(targets, dtype=torch.long)
                # Apply optional post-tensor train transform (RandomErasing) after normalization
                if self._train_post_tf is not None:
                    # RandomErasing expects CHW per sample
                    images = torch.stack([self._train_post_tf(img) for img in images], dim=0)
                if random.random() > self._mix_prob:
                    return images, targets  # no mix
                # choose mode per batch
                mode = "mixup" if random.random() < 0.5 else "cutmix"
                perm = torch.randperm(images.size(0))
                lam_m = 1.0
                if mode == "mixup" and self._mixup_alpha > 0:
                    lam = torch.distributions.Beta(self._mixup_alpha, self._mixup_alpha).sample().item()
                    lam_m = float(lam)
                    images = lam * images + (1.0 - lam) * images[perm]
                elif mode == "cutmix" and self._cutmix_alpha > 0:
                    lam = torch.distributions.Beta(self._cutmix_alpha, self._cutmix_alpha).sample().item()
                    B, C, H, W = images.size()
                    x1, y1, x2, y2 = _rand_bbox(W, H, lam)
                    images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
                    # Adjust lam to match pixel ratio
                    lam_m = 1.0 - ((x2 - x1) * (y2 - y1)) / float(W * H)
                # Build soft targets
                y1 = _one_hot(targets, num_classes)
                y2 = _one_hot(targets[perm], num_classes)
                soft_targets = lam_m * y1 + (1.0 - lam_m) * y2
                return images, soft_targets

            self.collate_fn = mixup_cutmix_collate
        else:
            self.collate_fn = None

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn if shuffle else None,
        )

    def train_dataloader(self): return self._loader(self.ds_train, shuffle=True)
    def val_dataloader(self):   return self._loader(self.ds_val,   shuffle=False)
    def test_dataloader(self):  return self._loader(self.ds_val,   shuffle=False)