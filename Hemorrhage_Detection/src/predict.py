"""Inference API for Streamlit (or any external caller)."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

from src import config
from src.data import get_eval_transforms
from src.model import build_model

ImageInput = Union[str, Path, Image.Image, np.ndarray]


def tta_softmax(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Average softmax over test-time-augmentation views.

    Views match training augmentations (hflip, small rotation) so they stay
    in-distribution. x: normalized (B, C, H, W) tensor on the model's device.
    Returns mean softmax probs (B, num_classes).
    """
    views = [x, torch.flip(x, dims=[3])]  # original + horizontal flip
    for angle in (-10, 10):
        rot = TF.rotate(x, angle)
        views.append(rot)
        views.append(torch.flip(rot, dims=[3]))
    probs = torch.stack([torch.softmax(model(v), dim=1) for v in views])
    return probs.mean(dim=0)


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
    tta: bool = False,
    threshold: float = config.DECISION_THRESHOLD,
) -> dict:
    """Predict the class of a single image.

    tta: average over flip/rotation views for steadier probs.
    threshold: P(hemorrhage) >= threshold => "hemorrhage" (binary case only).
    Returns: {"label": str, "confidence": float, "probs": {class: float, ...}}
    """
    torch_device = _resolve_device(device)
    class_names = class_names or config.CLASS_NAMES

    arr = _to_ndarray(image)
    transform = get_eval_transforms()
    tensor = transform(image=arr)["image"].unsqueeze(0).to(torch_device)

    model.eval()
    with torch.no_grad():
        if tta:
            probs = tta_softmax(model, tensor)[0].cpu().numpy()
        else:
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    if len(class_names) == 2:
        idx = 1 if probs[1] >= threshold else 0
    else:
        idx = int(np.argmax(probs))
    return {
        "label": class_names[idx],
        "confidence": float(probs[idx]),
        "probs": {name: float(p) for name, p in zip(class_names, probs)},
    }
