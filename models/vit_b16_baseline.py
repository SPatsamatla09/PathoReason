"""ViT-B/16 linear-probe baseline for MHIST classification."""

from __future__ import annotations

from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

from utils.config import NUM_CLASSES


def build_vit_b16_linear_probe(
    *,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a pretrained ViT-B/16 with a trainable linear classification head.

    The transformer backbone remains frozen, while only the final
    classification layer is optimized on MHIST.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to initialize from ImageNet weights.

    Returns:
        A ViT-B/16 configured for linear probing.

    Raises:
        ValueError: If num_classes is not greater than one.
    """
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one.")

    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model = vit_b_16(weights=weights)

    # Freeze the complete pretrained backbone.
    for parameter in model.parameters():
        parameter.requires_grad = False

    input_features = model.heads.head.in_features
    model.heads.head = nn.Linear(input_features, num_classes)

    # The newly created classification head is trainable by default.
    return model
