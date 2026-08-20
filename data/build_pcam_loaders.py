"""PyTorch DataLoaders for streaming PatchCamelyon."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.pcam_data_loader import PCamDataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


@dataclass(frozen=True, slots=True)
class PCamLoaders:
    """Container for PCam train, validation, and test DataLoaders."""

    train: DataLoader
    valid: DataLoader
    test: DataLoader


def build_train_transform() -> transforms.Compose:
    """Return augmentation and normalization for PCam training."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_test_transform() -> transforms.Compose:
    """Return deterministic preprocessing for PCam evaluation."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_pcam_loaders(
    batch_size: int = 32,
    num_workers: int = 0,
) -> PCamLoaders:
    """Build streaming PCam train, validation, and test DataLoaders."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    train_dataset = PCamDataset(
        partition="train",
        transform=build_train_transform(),
        shuffle=True,
        seed=42,
        max_samples=20_000,
    )
    valid_dataset = PCamDataset(
        partition="valid",
        transform=build_test_transform(),
        max_samples=4_000,
    )
    test_dataset = PCamDataset(
        partition="test",
        transform=build_test_transform(),
        max_samples=4_000,
    )

    options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    return PCamLoaders(
        train=DataLoader(train_dataset, **options),
        valid=DataLoader(valid_dataset, **options),
        test=DataLoader(test_dataset, **options),
    )
