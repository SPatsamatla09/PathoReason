"""Reproducible PyTorch DataLoaders for the MHIST dataset."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.mhist_data_loader import MHISTDataset
from utils.config import BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS, SEED


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class MHISTLoaders:
    """Container for official MHIST train and test DataLoaders."""

    train: DataLoader
    test: DataLoader


def build_train_transform() -> transforms.Compose:
    """Return data augmentation and normalization for baseline training."""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_test_transform() -> transforms.Compose:
    """Return deterministic normalization for baseline evaluation."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_mhist_loaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = SEED,
    train_transform=None,
    test_transform=None,
) -> MHISTLoaders:
    """
    Build DataLoaders using the official MHIST train/test partitions.

    Args:
        batch_size: Number of images per batch.
        num_workers: Number of worker processes used for loading.
        seed: Seed controlling reproducible training-data shuffling.

    Returns:
        Train and test DataLoaders.

    Raises:
        ValueError: If batch_size or num_workers is invalid.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    if train_transform is None:
        train_transform = build_train_transform()
    if test_transform is None:
        test_transform = build_test_transform()

    train_dataset = MHISTDataset(
        partition="train",
        transform=train_transform,
    )
    test_dataset = MHISTDataset(
        partition="test",
        transform=test_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    common_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common_options,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_options,
    )

    return MHISTLoaders(train=train_loader, test=test_loader)
