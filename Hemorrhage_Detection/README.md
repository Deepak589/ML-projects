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
