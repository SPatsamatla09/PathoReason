"""ResNet-18 baseline for MHIST fine-tuning."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from utils.config import NUM_CLASSES


def build_resnet18_finetune(
    *,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a ResNet-18 model for full fine-tuning on MHIST.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to initialize from ImageNet weights.

    Returns:
        ResNet-18 with a task-specific classification head.

    Raises:
        ValueError: If num_classes is invalid.
    """
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one.")

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    for parameter in model.parameters():
        parameter.requires_grad = True

    return model
