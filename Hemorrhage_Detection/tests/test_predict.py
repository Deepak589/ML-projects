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
