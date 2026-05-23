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
