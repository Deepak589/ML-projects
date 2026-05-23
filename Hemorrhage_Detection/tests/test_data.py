import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src import config
from src.data import (
    HemorrhageDataset,
    get_train_transforms,
    get_eval_transforms,
    get_dataloaders,
)


def _dataset_readable() -> bool:
    """Return True if at least one real Dataset image can be opened."""
    for class_name in config.CLASS_NAMES:
        class_dir = config.DATA_ROOT / class_name
        if not class_dir.is_dir():
            return False
        for p in class_dir.iterdir():
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                try:
                    with open(str(p), "rb") as f:
                        f.read(16)
                    return True
                except OSError:
                    return False
    return False


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    """Return real DATA_ROOT if readable, else a synthetic one with 30 images per class."""
    if _dataset_readable():
        return config.DATA_ROOT
    # Build synthetic dataset
    root = tmp_path_factory.mktemp("synthetic_dataset")
    rng = np.random.default_rng(0)
    for class_name in config.CLASS_NAMES:
        class_dir = root / class_name
        class_dir.mkdir()
        for i in range(30):
            arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(class_dir / f"{i:03d}.png")
    return root


def test_dataset_lists_both_classes(data_root):
    ds = HemorrhageDataset(data_root, transform=get_eval_transforms())
    labels = {sample[1] for sample in ds.samples}
    assert labels == {0, 1}, f"Expected both classes 0 and 1, got {labels}"


def test_dataset_returns_tensor_with_correct_shape(data_root):
    ds = HemorrhageDataset(data_root, transform=get_eval_transforms())
    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert label in (0, 1)


def test_get_dataloaders_returns_three_loaders(data_root):
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=4, num_workers=0, data_root=data_root
    )
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0


def test_train_loader_batch_shape(data_root):
    train_loader, _, _ = get_dataloaders(
        batch_size=4, num_workers=0, data_root=data_root
    )
    images, labels = next(iter(train_loader))
    assert images.shape == (4, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert labels.shape == (4,)


def test_train_transforms_output_shape():
    """Train transforms must produce a valid CHW float tensor of the expected spatial size."""
    import numpy as np
    from src.data import get_train_transforms

    rng = np.random.default_rng(1)
    arr = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    transform = get_train_transforms()
    out = transform(image=arr)["image"]
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert out.dtype == torch.float32
