"""Utilities for extracting and caching frozen foundation-model embeddings."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def extract_embeddings(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract frozen embeddings and labels from a dataset."""
    encoder.eval()
    encoder.to(device)

    embeddings = []
    labels = []

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["label"]

        features = encoder(images)

        if features.ndim != 2:
            raise ValueError(
                f"Encoder must return [batch, features], got {tuple(features.shape)}"
            )

        embeddings.append(features.cpu())
        labels.append(targets.cpu())

    return torch.cat(embeddings), torch.cat(labels)


def save_embedding_cache(
    path: Path,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    metadata: dict | None = None,
) -> None:
    """Save embeddings and labels for reuse across linear-probe runs."""
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
        },
        path,
    )


def load_embedding_cache(
    path: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load previously cached embeddings and labels."""
    if not path.is_file():
        raise FileNotFoundError(f"Embedding cache not found: {path}")

    cache = torch.load(path, map_location="cpu")

    if "embeddings" not in cache or "labels" not in cache:
        raise ValueError(
            f"Invalid embedding cache at {path}: "
            "expected 'embeddings' and 'labels'."
        )

    embeddings = cache["embeddings"]
    labels = cache["labels"]

    if embeddings.ndim != 2:
        raise ValueError(
            f"Cached embeddings must be [N, D], got {tuple(embeddings.shape)}"
        )

    if len(embeddings) != len(labels):
        raise ValueError(
            "Cached embeddings and labels have different sample counts."
        )

    return embeddings, labels
