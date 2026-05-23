# Hemorrhage Detection Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace legacy Keras1/Tkinter hemorrhage detector with a modular PyTorch + timm EfficientNetB0 classifier exposing a clean inference API for a future Streamlit UI.

**Architecture:** PyTorch + `timm` EfficientNetB0 pretrained on ImageNet, binary head. Modular `src/` package: `config`, `data`, `model`, `train`, `evaluate`, `predict`. Albumentations for augmentation, `WeightedRandomSampler` for class balance, AMP for speed, ROC-AUC-based best checkpointing with early stopping.

**Tech Stack:** Python 3.10+, PyTorch ≥ 2.1, timm ≥ 1.0, Albumentations ≥ 1.4, scikit-learn, PIL, matplotlib, seaborn, pandas, tqdm.

**Spec:** `docs/superpowers/specs/2026-05-21-hemorrhage-detection-modernization-design.md`

---

## File Structure

**Created:**
- `src/__init__.py` — empty package marker
- `src/config.py` — constants
- `src/data.py` — Dataset, transforms, splits, dataloaders
- `src/model.py` — `build_model()`
- `src/train.py` — training entry (`python -m src.train`)
- `src/evaluate.py` — evaluation entry (`python -m src.evaluate`)
- `src/predict.py` — Streamlit inference API
- `tests/__init__.py`
- `tests/test_data.py`
- `tests/test_model.py`
- `tests/test_predict.py`
- `requirements.txt`
- `README.md`
- `legacy/README.md`

**Moved:**
- `Main.py` → `legacy/Main.py`
- `yolomodel/` → `legacy/yolomodel/`

**Untouched:**
- `Dataset/` (training data)
- `testImages/` (manual smoke-test images)
- `Project details.txt`
- `run.bat`

---

### Task 1: Archive legacy files and scaffold project

**Files:**
- Move: `Main.py` → `legacy/Main.py`
- Move: `yolomodel/` → `legacy/yolomodel/`
- Create: `legacy/README.md`
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `checkpoints/.gitkeep`
- Create: `outputs/.gitkeep`

- [ ] **Step 1: Create directories**

```bash
mkdir -p legacy src tests checkpoints outputs
```

- [ ] **Step 2: Move legacy files**

```bash
git mv Main.py legacy/Main.py
git mv yolomodel legacy/yolomodel
```

- [ ] **Step 3: Write `legacy/README.md`**

```markdown
# Legacy

Archived Keras 1.x / TensorFlow 1.x era code. Tkinter GUI + small CNN mislabeled as "YOLO". Replaced by the PyTorch implementation in `src/`. Kept for reference only; not maintained.

- `Main.py` — original Tkinter GUI application
- `yolomodel/` — pre-saved `.npy` dataset cache, Keras JSON model, H5 weights, pickled history
```

- [ ] **Step 4: Write `requirements.txt`**

```
torch>=2.1
torchvision>=0.16
timm>=1.0
albumentations>=1.4
scikit-learn>=1.3
numpy>=1.24
pillow>=10.0
matplotlib>=3.7
seaborn>=0.13
pandas>=2.0
tqdm>=4.66
pytest>=7.4
```

- [ ] **Step 5: Write `src/__init__.py`**

```python
```

(Empty file — package marker.)

- [ ] **Step 6: Write `tests/__init__.py`**

```python
```

(Empty file.)

- [ ] **Step 7: Add `.gitkeep` files**

```bash
touch checkpoints/.gitkeep outputs/.gitkeep
```

- [ ] **Step 8: Commit**

```bash
git add legacy/ src/ tests/ checkpoints/ outputs/ requirements.txt
git commit -m "chore: archive legacy GUI code and scaffold modern PyTorch project"
```

---

### Task 2: Configuration module

**Files:**
- Create: `src/config.py`

- [ ] **Step 1: Write `src/config.py`**

```python
"""Central configuration for hemorrhage detection."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_ROOT = PROJECT_ROOT / "Dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "best_model.pt"

# Classes (order = label index)
CLASS_NAMES = ["normal", "hemorrhage"]
NUM_CLASSES = len(CLASS_NAMES)

# Model
MODEL_NAME = "efficientnet_b0"
IMAGE_SIZE = 224

# Training
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 7

# Splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ImageNet normalization (EfficientNet expects this)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Reproducibility
SEED = 42
```

- [ ] **Step 2: Quick import sanity check**

Run: `python -c "from src import config; print(config.MODEL_NAME, config.IMAGE_SIZE)"`
Expected: `efficientnet_b0 224`

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat(config): central configuration module"
```

---

### Task 3: Data module — Dataset, transforms, splits

**Files:**
- Create: `src/data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write failing test `tests/test_data.py`**

```python
import pytest
import torch
from src import config
from src.data import (
    HemorrhageDataset,
    get_train_transforms,
    get_eval_transforms,
    get_dataloaders,
)


def test_dataset_lists_both_classes():
    ds = HemorrhageDataset(config.DATA_ROOT, transform=get_eval_transforms())
    labels = {sample[1] for sample in ds.samples}
    assert labels == {0, 1}, f"Expected both classes 0 and 1, got {labels}"


def test_dataset_returns_tensor_with_correct_shape():
    ds = HemorrhageDataset(config.DATA_ROOT, transform=get_eval_transforms())
    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert label in (0, 1)


def test_get_dataloaders_returns_three_loaders():
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4, num_workers=0)
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0


def test_train_loader_batch_shape():
    train_loader, _, _ = get_dataloaders(batch_size=4, num_workers=0)
    images, labels = next(iter(train_loader))
    assert images.shape == (4, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert labels.shape == (4,)
```

- [ ] **Step 2: Run test to confirm failure**

Run: `pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data'`

- [ ] **Step 3: Write `src/data.py`**

```python
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
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
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
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Stratified 70/15/15 split with class-balanced train sampler."""
    base = HemorrhageDataset(config.DATA_ROOT, transform=None)
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

    train_base = HemorrhageDataset(config.DATA_ROOT, transform=get_train_transforms())
    eval_base = HemorrhageDataset(config.DATA_ROOT, transform=get_eval_transforms())

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
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_data.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat(data): Dataset, augmentations, stratified split, weighted sampler"
```

---

### Task 4: Model module

**Files:**
- Create: `src/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write failing test `tests/test_model.py`**

```python
import torch
from src import config
from src.model import build_model


def test_build_model_returns_module():
    model = build_model(num_classes=2, pretrained=False)
    assert isinstance(model, torch.nn.Module)


def test_model_forward_shape():
    model = build_model(num_classes=2, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2)
```

- [ ] **Step 2: Run test to confirm failure**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.model'`

- [ ] **Step 3: Write `src/model.py`**

```python
"""Model factory."""
from __future__ import annotations

import timm
import torch.nn as nn

from src import config


def build_model(
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
    model_name: str | None = None,
) -> nn.Module:
    """Create a timm model with a custom classification head."""
    name = model_name or config.MODEL_NAME
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    return model
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_model.py -v`
Expected: both tests PASS. (First run downloads no weights since `pretrained=False`.)

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat(model): timm-backed model factory"
```

---

### Task 5: Training module

**Files:**
- Create: `src/train.py`

No unit tests — training is verified by smoke-running in Task 8.

- [ ] **Step 1: Write `src/train.py`**

```python
"""Training entry point: AMP, early stopping, ROC-AUC checkpoint."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src import config
from src.data import get_dataloaders
from src.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * images.size(0)
            n += images.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds = logits.argmax(dim=1).numpy()
    labels_np = labels.numpy()
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": float(accuracy_score(labels_np, preds)),
        "auc": float(roc_auc_score(labels_np, probs)) if len(set(labels_np)) > 1 else 0.0,
    }


def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, use_amp) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--model", type=str, default=config.MODEL_NAME)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    set_seed(config.SEED)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    model = build_model(num_classes=config.NUM_CLASSES, pretrained=True, model_name=args.model)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)

    log_path = config.OUTPUT_DIR / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_auc", "lr"])

    best_auc = -1.0
    best_val_loss = float("inf")
    patience = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device, use_amp)
        val = evaluate(model, val_loader, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val['loss']:.4f} | val_acc={val['accuracy']:.4f} | "
            f"val_auc={val['auc']:.4f} | lr={lr_now:.2e}"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val["loss"], val["accuracy"], val["auc"], lr_now])

        if val["auc"] > best_auc:
            best_auc = val["auc"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": args.model,
                    "image_size": config.IMAGE_SIZE,
                    "class_names": config.CLASS_NAMES,
                    "val_metrics": val,
                    "epoch": epoch,
                },
                config.DEFAULT_CHECKPOINT,
            )
            print(f"  -> saved best checkpoint (val_auc={best_auc:.4f})")

        if val["loss"] < best_val_loss - 1e-4:
            best_val_loss = val["loss"]
            patience = 0
        else:
            patience += 1
            if patience >= config.EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"Done. Best val AUC: {best_auc:.4f}. Checkpoint: {config.DEFAULT_CHECKPOINT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Static check — import only**

Run: `python -c "from src import train; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "feat(train): AMP training loop with early stop and best-AUC checkpoint"
```

---

### Task 6: Evaluation module

**Files:**
- Create: `src/evaluate.py`

- [ ] **Step 1: Write `src/evaluate.py`**

```python
"""Evaluation entry point: metrics, confusion matrix, ROC curve."""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config
from src.data import get_dataloaders
from src.model import build_model


def collect_predictions(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    return (
        np.concatenate(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


def plot_confusion(cm, out_path):
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc(labels, probs, out_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(config.DEFAULT_CHECKPOINT))
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    args = parser.parse_args()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        model_name=ckpt.get("model_name", config.MODEL_NAME),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    _, _, test_loader = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    probs, preds, labels = collect_predictions(model, test_loader, device)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_binary": float(precision_score(labels, preds, zero_division=0)),
        "recall_binary": float(recall_score(labels, preds, zero_division=0)),
        "f1_binary": float(f1_score(labels, preds, zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "n_test": int(len(labels)),
    }
    metrics_path = config.OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(labels, preds)
    plot_confusion(cm, config.OUTPUT_DIR / "confusion_matrix.png")
    plot_roc(labels, probs, config.OUTPUT_DIR / "roc_curve.png")

    print(json.dumps(metrics, indent=2))
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Static check**

Run: `python -c "from src import evaluate; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/evaluate.py
git commit -m "feat(evaluate): test-set metrics, confusion matrix, ROC plot"
```

---

### Task 7: Prediction (Streamlit-facing API)

**Files:**
- Create: `src/predict.py`
- Create: `tests/test_predict.py`

- [ ] **Step 1: Write failing test `tests/test_predict.py`**

```python
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src import config
from src.model import build_model
from src.predict import load_model, predict_image


@pytest.fixture
def tmp_checkpoint(tmp_path: Path) -> Path:
    model = build_model(num_classes=2, pretrained=False)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": config.MODEL_NAME,
            "image_size": config.IMAGE_SIZE,
            "class_names": config.CLASS_NAMES,
        },
        ckpt_path,
    )
    return ckpt_path


def test_load_model_returns_module_and_classnames(tmp_checkpoint):
    model, class_names = load_model(str(tmp_checkpoint), device="cpu")
    assert isinstance(model, torch.nn.Module)
    assert class_names == config.CLASS_NAMES


def test_predict_image_accepts_pil(tmp_checkpoint):
    model, _ = load_model(str(tmp_checkpoint), device="cpu")
    img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
    result = predict_image(model, img, device="cpu")
    assert result["label"] in config.CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probs"].keys()) == set(config.CLASS_NAMES)


def test_predict_image_accepts_path(tmp_checkpoint, tmp_path):
    model, _ = load_model(str(tmp_checkpoint), device="cpu")
    img_path = tmp_path / "sample.png"
    Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)).save(img_path)
    result = predict_image(model, str(img_path), device="cpu")
    assert result["label"] in config.CLASS_NAMES


def test_predict_image_accepts_ndarray(tmp_checkpoint):
    model, _ = load_model(str(tmp_checkpoint), device="cpu")
    arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    result = predict_image(model, arr, device="cpu")
    assert result["label"] in config.CLASS_NAMES
```

- [ ] **Step 2: Run test to confirm failure**

Run: `pytest tests/test_predict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.predict'`

- [ ] **Step 3: Write `src/predict.py`**

```python
"""Inference API for Streamlit (or any external caller)."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src import config
from src.data import get_eval_transforms
from src.model import build_model

ImageInput = Union[str, Path, Image.Image, np.ndarray]


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model(
    checkpoint_path: str,
    device: str = "auto",
) -> tuple[nn.Module, list[str]]:
    """Load a trained checkpoint and return (model, class_names)."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    torch_device = _resolve_device(device)
    ckpt = torch.load(ckpt_path, map_location=torch_device)

    model_name = ckpt.get("model_name", config.MODEL_NAME)
    class_names = ckpt.get("class_names", config.CLASS_NAMES)

    model = build_model(num_classes=len(class_names), pretrained=False, model_name=model_name)
    model.load_state_dict(ckpt["state_dict"])
    model.to(torch_device)
    model.eval()
    return model, class_names


def _to_ndarray(image: ImageInput) -> np.ndarray:
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
        return np.array(img)
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image
    raise TypeError(f"Unsupported image type: {type(image)}")


def predict_image(
    model: nn.Module,
    image: ImageInput,
    device: str = "auto",
    class_names: list[str] | None = None,
) -> dict:
    """Predict the class of a single image.

    Returns: {"label": str, "confidence": float, "probs": {class: float, ...}}
    """
    torch_device = _resolve_device(device)
    class_names = class_names or config.CLASS_NAMES

    arr = _to_ndarray(image)
    transform = get_eval_transforms()
    tensor = transform(image=arr)["image"].unsqueeze(0).to(torch_device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    return {
        "label": class_names[idx],
        "confidence": float(probs[idx]),
        "probs": {name: float(p) for name, p in zip(class_names, probs)},
    }
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_predict.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `pytest -v`
Expected: all tests across `test_data.py`, `test_model.py`, `test_predict.py` PASS.

- [ ] **Step 6: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat(predict): inference API for Streamlit consumption"
```

---

### Task 8: End-to-end smoke run

This verifies the whole pipeline works on real data. Uses a tiny epoch count to stay fast.

- [ ] **Step 1: Run 1-epoch training**

Run: `python -m src.train --epochs 1 --batch-size 16 --num-workers 0`
Expected: prints `Device: cuda ...` (or cpu), 1 epoch completes, `checkpoints/best_model.pt` written, `outputs/training_log.csv` has 1 data row.

- [ ] **Step 2: Verify checkpoint exists**

Run: `python -c "import torch; c=torch.load('checkpoints/best_model.pt', map_location='cpu'); print(list(c.keys()))"`
Expected: includes `state_dict`, `model_name`, `image_size`, `class_names`, `val_metrics`, `epoch`.

- [ ] **Step 3: Run evaluation**

Run: `python -m src.evaluate`
Expected: prints JSON metrics, writes `outputs/metrics.json`, `outputs/confusion_matrix.png`, `outputs/roc_curve.png`.

- [ ] **Step 4: Sanity-check predict API on a real test image**

Pick any file from `testImages/` (e.g. `testImages/1.png`):

```bash
python -c "from src.predict import load_model, predict_image; m, c = load_model('checkpoints/best_model.pt'); print(predict_image(m, 'testImages/1.png'))"
```

Expected: dict like `{"label": "normal"|"hemorrhage", "confidence": 0.5x-0.9x, "probs": {...}}`.

- [ ] **Step 5: Commit outputs (optional artifact snapshot)**

If you want a record of the smoke run:

```bash
git add outputs/training_log.csv outputs/metrics.json outputs/confusion_matrix.png outputs/roc_curve.png
git commit -m "chore: smoke-run artifacts from 1-epoch training"
```

If you'd rather keep `outputs/` out of git, add it to `.gitignore` and skip this commit.

---

### Task 9: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

````markdown
# Intracranial Hemorrhage Detection

Binary classification of CT scans (normal vs. hemorrhage) using a pretrained EfficientNetB0 (PyTorch + timm).

## Project Structure

```
src/        # package: config, data, model, train, evaluate, predict
tests/      # pytest unit tests
Dataset/    # CT scan PNGs in class subfolders
testImages/ # ad-hoc images for manual inference
checkpoints/ # saved models
outputs/    # metrics + plots
legacy/     # archived Keras1/Tkinter implementation
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train --epochs 30 --batch-size 32
```

Outputs:
- `checkpoints/best_model.pt` — best val-AUC checkpoint
- `outputs/training_log.csv` — per-epoch metrics

## Evaluate

```bash
python -m src.evaluate --checkpoint checkpoints/best_model.pt
```

Outputs: `outputs/metrics.json`, `outputs/confusion_matrix.png`, `outputs/roc_curve.png`.

## Inference (Python API)

```python
from src.predict import load_model, predict_image

model, class_names = load_model("checkpoints/best_model.pt")
result = predict_image(model, "testImages/1.png")
# {"label": "hemorrhage", "confidence": 0.92, "probs": {"normal": 0.08, "hemorrhage": 0.92}}
```

Use these functions directly from a Streamlit app — no GUI code lives inside the package.

## Tests

```bash
pytest -v
```

## Configuration

All hyperparameters live in `src/config.py`. CLI flags on `src.train` / `src.evaluate` override defaults.
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with setup, train, evaluate, inference instructions"
```

---

## Verification Checklist (end of plan)

- [ ] `pytest -v` — all tests pass
- [ ] `python -m src.train --epochs 1 --batch-size 16 --num-workers 0` — runs end-to-end
- [ ] `python -m src.evaluate` — produces metrics.json + plots
- [ ] `from src.predict import load_model, predict_image` — works in a fresh Python session
- [ ] No `tkinter`, `cv2.imshow`, or GUI imports anywhere under `src/`
- [ ] `legacy/` contains the old `Main.py` and `yolomodel/`; nothing in `src/` references them
