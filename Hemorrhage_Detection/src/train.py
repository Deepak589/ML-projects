"""Training entry point: AMP, early stopping, ROC-AUC checkpoint."""
from __future__ import annotations

import argparse
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, fbeta_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src import config
from src.data import get_dataloaders
from src.model import build_model


class FocalLoss(nn.Module):
    """Multiclass focal loss operating on logits + integer labels."""

    def __init__(self, gamma: float = config.FOCAL_GAMMA, alpha: float = config.FOCAL_ALPHA):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


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
    thresh_preds = (probs >= 0.5).astype(int)
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": float(accuracy_score(labels_np, preds)),
        "auc": float(roc_auc_score(labels_np, probs)) if len(set(labels_np)) > 1 else 0.0,
        "f2": float(fbeta_score(labels_np, thresh_preds, beta=2, zero_division=0)),
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

    loss_fn = FocalLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)

    log_path = config.OUTPUT_DIR / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_auc", "val_f2", "lr"])

    ckpt_metric = config.CHECKPOINT_METRIC  # "f2" or "auc"
    best_metric = -1.0
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
            f"val_auc={val['auc']:.4f} | val_f2={val['f2']:.4f} | lr={lr_now:.2e}"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, train_loss, val["loss"], val["accuracy"], val["auc"], val["f2"], lr_now]
            )

        if val[ckpt_metric] > best_metric:
            best_metric = val[ckpt_metric]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": args.model,
                    "image_size": config.IMAGE_SIZE,
                    "class_names": config.CLASS_NAMES,
                    "val_metrics": val,
                    "epoch": epoch,
                },
                config.V2_CHECKPOINT,
            )
            print(f"  -> saved best checkpoint (val_{ckpt_metric}={best_metric:.4f}) -> {config.V2_CHECKPOINT}")

        if val["loss"] < best_val_loss - 1e-4:
            best_val_loss = val["loss"]
            patience = 0
        else:
            patience += 1
            if patience >= config.EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"Done. Best val {ckpt_metric}: {best_metric:.4f}. Checkpoint: {config.V2_CHECKPOINT}")


if __name__ == "__main__":
    main()
