# Hemorrhage Detection — Modernization Design

**Date:** 2026-05-21
**Status:** Approved for implementation

## Goal

Replace legacy Keras1-era Tkinter app with a modern, efficient PyTorch model for binary intracranial hemorrhage detection on CT scans. Strip all GUI code. Produce a modular package that a future Streamlit UI can import directly.

## Non-Goals

- Streamlit UI (separate project, out of scope).
- Dataset expansion or relabeling.
- Multi-class (subtype) classification — binary only (`normal` vs `hemorrhage`).
- DICOM ingestion — PNG inputs only (matches existing dataset).

## Dataset

- Existing folders: `Dataset/hemorrhage/`, `Dataset/normal/`.
- Labels: `normal=0`, `hemorrhage=1`.
- Stratified split 70/15/15 train/val/test, fixed seed.

## Architecture

PyTorch + `timm`. EfficientNetB0 pretrained on ImageNet, binary classification head (2-class softmax via CrossEntropyLoss).

Model name is configurable via `src/config.py` to allow swap to ResNet50, EfficientNetB3, etc., without code changes elsewhere.

## Repository Layout

```
HemorrhageDetection/
├── Dataset/                  # existing, untouched
│   ├── hemorrhage/
│   └── normal/
├── src/
│   ├── __init__.py
│   ├── config.py             # paths, hyperparams, seed
│   ├── data.py               # Dataset, transforms, splits, sampler
│   ├── model.py              # build_model() via timm
│   ├── train.py              # train loop, AMP, early stop, checkpointing
│   ├── evaluate.py           # test metrics, confusion matrix, ROC
│   └── predict.py            # load_model(), predict_image() — Streamlit entry
├── checkpoints/              # best_model.pt
├── outputs/                  # metrics.json, confusion_matrix.png, training_log.csv, roc_curve.png
├── legacy/                   # archived Main.py + yolomodel/
├── testImages/               # existing, untouched
├── requirements.txt
└── README.md
```

## Components

### `src/config.py`
Single source of truth. Constants:
- `DATA_ROOT`, `CHECKPOINT_DIR`, `OUTPUT_DIR`
- `MODEL_NAME = "efficientnet_b0"`
- `IMAGE_SIZE = 224`
- `BATCH_SIZE = 32`
- `NUM_WORKERS = 4`
- `EPOCHS = 30`
- `LR = 3e-4`
- `WEIGHT_DECAY = 1e-4`
- `EARLY_STOP_PATIENCE = 7`
- `SEED = 42`
- `CLASS_NAMES = ["normal", "hemorrhage"]`

CLI args in train/evaluate override config values.

### `src/data.py`
- `HemorrhageDataset(Dataset)`: scans class folders, loads PNG via PIL, applies transforms, returns `(tensor, label)`.
- `get_splits(seed)` → stratified 70/15/15 indices.
- `get_train_transforms()` (Albumentations): Resize(224), HorizontalFlip, Rotate(±15°), RandomBrightnessContrast, Normalize(ImageNet mean/std), ToTensorV2.
- `get_eval_transforms()`: Resize, Normalize, ToTensorV2.
- `get_dataloaders()` → train/val/test loaders. Train uses `WeightedRandomSampler` derived from class counts.

### `src/model.py`
- `build_model(num_classes=2, pretrained=True) -> nn.Module`
- Delegates to `timm.create_model(config.MODEL_NAME, pretrained=pretrained, num_classes=num_classes)`.
- Returns logits.

### `src/train.py`
- Loss: `CrossEntropyLoss` (no class weights — handled by sampler to avoid double-weighting).
- Optimizer: `AdamW(lr=3e-4, weight_decay=1e-4)`.
- Scheduler: `CosineAnnealingLR(T_max=EPOCHS)`.
- AMP: `torch.cuda.amp.autocast` + `GradScaler`.
- Early stopping on val loss, patience 7.
- Best checkpoint on val ROC-AUC → `checkpoints/best_model.pt`.
  Checkpoint contents: `{state_dict, model_name, image_size, class_names, val_metrics, epoch}`.
- Per-epoch row appended to `outputs/training_log.csv`: `epoch, train_loss, val_loss, val_acc, val_auc, lr`.
- CLI: `python -m src.train [--epochs N --batch-size N --lr F --model NAME]`.

### `src/evaluate.py`
Load checkpoint, run test set, compute and save:
- Metrics → `outputs/metrics.json`: accuracy, precision, recall, F1 (binary + macro), ROC-AUC, PR-AUC.
- `outputs/confusion_matrix.png` (seaborn heatmap, labeled axes).
- `outputs/roc_curve.png`.
- CLI: `python -m src.evaluate --checkpoint checkpoints/best_model.pt`.

### `src/predict.py` — Streamlit entry point
Public API:
```python
def load_model(checkpoint_path: str, device: str = "auto") -> tuple[nn.Module, list[str]]
def predict_image(model, image, device) -> dict
    # image: PIL.Image | str path | np.ndarray
    # returns {"label": str, "confidence": float, "probs": {"normal": float, "hemorrhage": float}}
```
Behavior:
- `device="auto"` → `cuda` if available, else `cpu`.
- Uses same eval transforms as training for consistency.
- No GUI, no `cv2.imshow`, no file dialogs. Pure functions.

## Data Flow

1. `train.py`: `data.get_dataloaders()` → `model.build_model()` → train loop → save best checkpoint + log.
2. `evaluate.py`: load checkpoint → run test loader → write metrics + plots.
3. `predict.py` (called by future Streamlit): load checkpoint once → call `predict_image()` per inference.

## Error Handling

- Boundary validation only: missing dataset dir, missing checkpoint, unreadable image → raise with clear message.
- Internal code trusts inputs (per repo conventions).
- AMP/CUDA: detect device, fall back to CPU without crashing.

## Testing Strategy

- Smoke test: 1-epoch training run on full dataset must complete and produce a checkpoint.
- `predict_image()` on a known hemorrhage sample and a known normal sample → asserted labels match.
- Verified via manual runs, not a formal test suite (small project).

## Legacy Migration

- Move `Main.py`, `yolomodel/` (including `.npy`, `.h5`, `.json`, `.pckl`) into `legacy/`.
- Add `legacy/README.md` noting these are archived Keras1-era artifacts, no longer used.
- `Project details.txt` stays at repo root (low value, harmless).

## Dependencies (`requirements.txt`)

```
torch>=2.1
torchvision>=0.16
timm>=1.0
albumentations>=1.4
scikit-learn>=1.3
numpy
pillow
matplotlib
seaborn
pandas
tqdm
```

## Success Criteria

- Old code archived in `legacy/`, no GUI imports anywhere in `src/`.
- `python -m src.train` runs end-to-end on local CUDA GPU.
- `python -m src.evaluate` produces `metrics.json` with ROC-AUC ≥ 0.90 on test split (target; not a hard gate — depends on dataset quality).
- `from src.predict import load_model, predict_image` works standalone in a fresh Python session.
- `requirements.txt` install on a clean venv produces a working environment.
