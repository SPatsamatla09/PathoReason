"""Streaming PyTorch dataset implementation for PatchCamelyon (PCam)."""

from __future__ import annotations

from typing import Any, Callable, Iterator

from datasets import load_dataset
from PIL import Image
from torch.utils.data import IterableDataset


DATASET_ID = "1aurent/PatchCamelyon"
DATASET_REVISION = "e4bd149e7a868a9d811fdd9f9a9fb78c05c104ab"

LABEL_TO_NAME: dict[int, str] = {
    0: "normal",
    1: "tumor",
}

VALID_PARTITIONS: set[str] = {"train", "valid", "test"}


class PCamDataset(IterableDataset):
    """Stream PatchCamelyon images and labels from Hugging Face."""

    def __init__(
        self,
        partition: str,
        transform: Callable[[Image.Image], Any] | None = None,
        shuffle: bool = False,
        seed: int = 42,
        shuffle_buffer_size: int = 10_000,
        max_samples: int | None = None,
    ) -> None:
        if partition not in VALID_PARTITIONS:
            raise ValueError(
                f"Invalid partition {partition!r}. "
                f"Expected one of {sorted(VALID_PARTITIONS)}."
            )

        self.partition = partition
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_samples = max_samples

    def __iter__(self) -> Iterator[dict[str, Any]]:
        dataset = load_dataset(
            DATASET_ID,
            revision=DATASET_REVISION,
            split=self.partition,
            streaming=True,
        )

        if self.shuffle:
            dataset = dataset.shuffle(
                seed=self.seed,
                buffer_size=self.shuffle_buffer_size,
            )

        for index, sample in enumerate(dataset):
            if self.max_samples is not None and index >= self.max_samples:
                break

            image = sample["image"].convert("RGB")
            label = int(sample["label"])

            if self.transform is not None:
                image = self.transform(image)

            yield {
                "image": image,
                "label": label,
                "label_name": LABEL_TO_NAME[label],
                "image_name": f"pcam_{self.partition}_{index:06d}",
                "partition": self.partition,
            }
