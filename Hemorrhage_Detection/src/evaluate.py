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
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config
from src.data import get_dataloaders
from src.model import build_model
from src.predict import tta_softmax


def collect_predictions(model, loader, device, tta: bool = False):
    """Return (probs_of_hemorrhage, labels). Preds derived later via threshold."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if tta:
                probs = tta_softmax(model, images)[:, 1].cpu().numpy()
            else:
                probs = torch.softmax(model(images), dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def best_threshold(probs, labels, beta: float = 2.0) -> float:
    """Pick threshold maximizing F-beta (beta>1 favors recall) on given split."""
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        score = fbeta_score(labels, (probs >= t).astype(int), beta=beta, zero_division=0)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


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
    parser.add_argument("--tta", action="store_true", help="Test-time augmentation")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="P(hemorrhage) decision threshold. Default: auto-tuned on val (F2-optimal).",
    )
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

    _, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # Threshold: use given value, else tune on val (never on test — avoids leakage).
    if args.threshold is not None:
        threshold = args.threshold
    else:
        val_probs, val_labels = collect_predictions(model, val_loader, device, tta=args.tta)
        threshold = best_threshold(val_probs, val_labels, beta=2.0)
    print(f"Decision threshold: {threshold:.3f} | TTA: {args.tta}")

    probs, labels = collect_predictions(model, test_loader, device, tta=args.tta)
    preds = (probs >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "tta": bool(args.tta),
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
