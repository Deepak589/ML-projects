"""Dataset, transforms, splits, dataloaders."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from src import config


class HemorrhageDataset(Dataset):
    """Loads PNG CT scans from class-named subfolders."""

    def __init__(self, root: Path, transform: Optional[Callable] = None):
        root = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for label, class_name in enumerate(config.CLASS_NAMES):
            class_dir = root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    self.samples.append((img_path, label))
        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = np.array(Image.open(path).convert("RGB"))
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.samples]


def get_train_transforms() -> Callable:
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=25, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.05, 0.15), hole_width_range=(0.05, 0.15), p=0.3),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transforms() -> Callable:
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])


def _make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    labels_arr = np.array(labels)
    class_counts = np.bincount(labels_arr, minlength=config.NUM_CLASSES)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels_arr]
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.SEED,
    data_root: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Stratified 70/15/15 split with class-balanced train sampler."""
    root = Path(data_root) if data_root is not None else config.DATA_ROOT
    base = HemorrhageDataset(root, transform=None)
    indices = np.arange(len(base))
    labels = np.array(base.labels)

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels,
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        stratify=labels,
        random_state=seed,
    )
    val_size = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, temp_labels,
        test_size=(1 - val_size),
        stratify=temp_labels,
        random_state=seed,
    )

    train_base = HemorrhageDataset(root, transform=get_train_transforms())
    eval_base = HemorrhageDataset(root, transform=get_eval_transforms())

    train_set = Subset(train_base, train_idx.tolist())
    val_set = Subset(eval_base, val_idx.tolist())
    test_set = Subset(eval_base, test_idx.tolist())

    train_labels = labels[train_idx].tolist()
    sampler = _make_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
