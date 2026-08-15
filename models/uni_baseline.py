"""Frozen UNI encoder for MHIST linear-probe experiments."""

from __future__ import annotations

import torch
from torch import nn


class UNIEncoder(nn.Module):
    """UNI feature extractor returning 1,024-dimensional embeddings."""

    embedding_dim = 1024

    def __init__(self) -> None:
        super().__init__()

        try:
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
        except ImportError as exc:
            raise ImportError(
                "UNI requires timm. Install it with: pip install timm"
            ) from exc

        self.model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )

        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        data_config = resolve_data_config(
            self.model.pretrained_cfg,
            model=self.model,
        )
        self.transform = create_transform(**data_config)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one frozen UNI embedding per image."""
        features = self.model(images)

        if features.ndim != 2:
            raise RuntimeError(
                f"Expected UNI output [B, D], got {tuple(features.shape)}"
            )

        return features
