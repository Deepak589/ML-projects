import torch
import torch.nn as nn
from src import config
from src.model import build_model
from src.train import FocalLoss


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


def test_focal_loss_returns_scalar():
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    logits = torch.randn(8, 2)
    labels = torch.randint(0, 2, (8,))
    loss = loss_fn(logits, labels)
    assert loss.shape == (), "FocalLoss should return a scalar"


def test_focal_loss_gamma0_alpha1_matches_cross_entropy():
    """FocalLoss(gamma=0, alpha=1) must match nn.CrossEntropyLoss on same inputs."""
    focal = FocalLoss(gamma=0.0, alpha=1.0)
    ce = nn.CrossEntropyLoss()
    torch.manual_seed(0)
    logits = torch.randn(16, 2)
    labels = torch.randint(0, 2, (16,))
    assert torch.allclose(focal(logits, labels), ce(logits, labels), atol=1e-5)
