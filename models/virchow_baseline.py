"""Frozen Virchow encoder for MHIST linear-probe experiments."""

from __future__ import annotations

import torch
from torch import nn


class VirchowEncoder(nn.Module):
    """Virchow feature extractor returning 2,560-dimensional embeddings."""

    embedding_dim = 2560

    def __init__(self) -> None:
        super().__init__()

        try:
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            from timm.layers import SwiGLUPacked
        except ImportError as exc:
            raise ImportError(
                "Virchow requires timm. Install it with: pip install timm"
            ) from exc

        self.model = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

        self.model.eval()

        data_config = resolve_data_config(
            self.model.pretrained_cfg,
            model=self.model,
        )
        self.transform = create_transform(**data_config)

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return concatenated class-token and mean-patch embeddings."""
        tokens = self.model(images)

        if tokens.ndim != 3:
            raise RuntimeError(
                f"Expected Virchow token output [B, N, D], got {tuple(tokens.shape)}"
            )

        class_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        mean_patch_token = patch_tokens.mean(dim=1)

        return torch.cat([class_token, mean_patch_token], dim=-1)
