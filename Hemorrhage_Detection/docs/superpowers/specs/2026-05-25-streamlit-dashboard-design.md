# Streamlit Dashboard — Hemorrhage Detection

**Date:** 2026-05-25  
**Status:** Approved  

---

## Overview

Single-page Streamlit app with 3 tabs: Predict, Model Metrics, Training History. Dark medical theme. Portfolio/research showcase use case.

---

## Architecture

Single file `app.py` at project root. No changes to `src/`. Consumes existing public API only.

```
app.py
├── Tab 1: Predict
│   ├── Upload CT scan (PNG/JPG/DICOM-as-PNG)
│   ├── Side-by-side: original image | Grad-CAM heatmap overlay
│   ├── Result badge: HEMORRHAGE (red #ff4757) / NORMAL (green #2ed573)
│   ├── Confidence percentage + Plotly horizontal bar
│   └── Probability gauge (Plotly indicator) for P(hemorrhage)
│
├── Tab 2: Model Metrics
│   ├── 6 metric cards: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
│   ├── Confusion matrix heatmap (Plotly, not static PNG)
│   └── ROC curve (Plotly, not static PNG)
│
└── Tab 3: Training History
    ├── Plotly multi-line: train_loss + val_loss (with epoch hover)
    ├── Plotly multi-line: val_auc + val_f2
    └── Plotly single-line: learning rate schedule
```

---

## Theme

| Token | Value |
|-------|-------|
| Background | `#0a0f1e` |
| Surface | `#111827` |
| Accent cyan | `#00d4ff` |
| Hemorrhage red | `#ff4757` |
| Normal green | `#2ed573` |
| Text primary | `#e2e8f0` |
| Text muted | `#64748b` |

Applied via `st.markdown` with injected CSS (no external theme files needed).

---

## Data Flow

```
Startup
  └── st.cache_resource: load_model(checkpoints/best_model.pt)

Tab 1 — Predict
  User uploads image
    → PIL.Image
    → src.predict.predict_image(model, image) → {label, confidence, probs}
    → pytorch-grad-cam (EfficientNetB0 target layer: model.conv_head)
    → GradCAM heatmap → cv2 colormap INFERNO → alpha-blend onto original
    → Plotly: confidence bar + probability gauge

Tab 2 — Metrics
  outputs/metrics.json → metric cards
  outputs/metrics.json → Plotly confusion matrix heatmap
  outputs/metrics.json → Plotly ROC curve (fpr/tpr from re-running evaluate or stored)

Tab 3 — Training History
  outputs/training_log.csv → pandas DataFrame
    → Plotly line chart: epoch vs train_loss + val_loss
    → Plotly line chart: epoch vs val_auc + val_f2
    → Plotly line chart: epoch vs lr
```

**Note on ROC curve data:** `metrics.json` stores scalar AUC only, not fpr/tpr arrays. Options:
1. Store fpr/tpr in a separate `outputs/roc_data.json` during evaluate
2. Re-run evaluate on test set at dashboard load (slow, requires Dataset/)
3. Show static `outputs/roc_curve.png` as fallback if roc_data.json absent

**Decision:** Generate `outputs/roc_data.json` (fpr, tpr arrays) in `src/evaluate.py` on next run. Dashboard loads it if present, else shows static PNG. No blocking dependency.

---

## Grad-CAM Implementation

- Library: `grad-cam` (pytorch-grad-cam, `pip install grad-cam`)
- Target layer: `model.features[-1]` (EfficientNetB0 last conv block via timm)
- Method: `GradCAMPlusPlus` (better for small lesions than vanilla GradCAM)
- Output: heatmap resized to 224×224, INFERNO colormap, alpha=0.5 blend
- Displayed: side-by-side columns — original left, heatmap right

---

## Error States

| Condition | Behavior |
|-----------|----------|
| No checkpoint found | Warning banner top of page; Predict tab shows disabled state |
| Non-image file uploaded | Inline st.error below uploader |
| `metrics.json` missing | Tab 2 shows "Run `python -m src.evaluate` first" |
| `training_log.csv` missing | Tab 3 shows "Run `python -m src.train` first" |
| `roc_data.json` missing | ROC section shows static `outputs/roc_curve.png` |
| Grad-CAM import fails | Heatmap section hidden; prediction result still shown |

---

## Files Changed

| File | Action |
|------|--------|
| `app.py` | **New** — Streamlit dashboard |
| `requirements.txt` | **Edit** — add `grad-cam>=1.4`, `streamlit>=1.32` |
| `src/evaluate.py` | **Edit** — write `outputs/roc_data.json` with fpr/tpr arrays |

No other `src/` files touched.

---

## Dependencies

```
streamlit>=1.32
grad-cam>=1.4        # pytorch-grad-cam
plotly>=5.18
```

Already in requirements.txt: torch, timm, pillow, pandas, scikit-learn, numpy.

---

## Success Criteria

- Upload any image from `testImages/` → result displayed in <3s on CPU
- Grad-CAM heatmap visible and plausibly highlights scan region
- All 3 tabs render without errors when all output files present
- Graceful degradation when output files missing
- `streamlit run app.py` starts with no import errors
