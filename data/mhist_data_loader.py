"""PyTorch dataset implementation for the MHIST colorectal polyp dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from utils.config import ANNOTATIONS_FILE, IMAGES_DIR


LABEL_TO_INDEX: dict[str, int] = {
    "HP": 0,
    "SSA": 1,
}

VOTE_COLUMN = "Number of Annotators who Selected SSA (Out of 7)"

REQUIRED_COLUMNS: set[str] = {
    "Image Name",
    "Majority Vote Label",
    VOTE_COLUMN,
    "Partition",
}

VALID_PARTITIONS: set[str] = {"train", "test"}


class MHISTDataset(Dataset[dict[str, Any]]):
    """Load MHIST images, labels, partitions, and annotator metadata."""

    def __init__(
        self,
        partition: str,
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        """
        Initialize one official MHIST partition.

        Args:
            partition: Either ``train`` or ``test``.
            transform: Optional transformation applied to each PIL image.

        Raises:
            ValueError: If the partition or annotation values are invalid.
            FileNotFoundError: If required files or images are missing.
        """
        if partition not in VALID_PARTITIONS:
            raise ValueError(
                f"Invalid partition {partition!r}. "
                f"Expected one of {sorted(VALID_PARTITIONS)}."
            )

        self.partition = partition
        self.transform = transform
        self.images_dir = Path(IMAGES_DIR)
        self.annotations_file = Path(ANNOTATIONS_FILE)

        self._validate_paths()

        annotations = pd.read_csv(self.annotations_file)
        self._validate_annotations(annotations)

        self.annotations = (
            annotations.loc[annotations["Partition"] == partition]
            .copy()
            .reset_index(drop=True)
        )

        if self.annotations.empty:
            raise ValueError(
                f"No samples found in the {partition!r} partition."
            )

        self._validate_image_files()

    def __len__(self) -> int:
        """Return the number of samples in this partition."""
        return len(self.annotations)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one MHIST sample and its metadata."""
        if not 0 <= index < len(self):
            raise IndexError(
                f"Index {index} is outside the valid range "
                f"0 to {len(self) - 1}."
            )

        row = self.annotations.iloc[index]

        image_name = str(row["Image Name"])
        label_name = str(row["Majority Vote Label"])
        ssa_votes = int(row[VOTE_COLUMN])
        image_path = self.images_dir / image_name

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        majority_agreement = max(ssa_votes, 7 - ssa_votes)

        return {
            "image": image,
            "label": LABEL_TO_INDEX[label_name],
            "label_name": label_name,
            "image_name": image_name,
            "partition": self.partition,
            "ssa_votes": ssa_votes,
            "majority_agreement": majority_agreement,
        }

    def _validate_paths(self) -> None:
        """Confirm required MHIST paths exist."""
        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"MHIST images directory not found: {self.images_dir}"
            )

        if not self.annotations_file.is_file():
            raise FileNotFoundError(
                f"MHIST annotations file not found: "
                f"{self.annotations_file}"
            )

    @staticmethod
    def _validate_annotations(annotations: pd.DataFrame) -> None:
        """Validate the MHIST annotation schema and values."""
        missing_columns = REQUIRED_COLUMNS.difference(annotations.columns)

        if missing_columns:
            raise ValueError(
                "annotations.csv is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if annotations["Image Name"].isna().any():
            raise ValueError("annotations.csv contains missing image names.")

        if annotations["Image Name"].duplicated().any():
            duplicates = annotations.loc[
                annotations["Image Name"].duplicated(),
                "Image Name",
            ].tolist()

            raise ValueError(
                "Duplicate image names found in annotations.csv. "
                f"Examples: {duplicates[:5]}"
            )

        invalid_labels = set(
            annotations["Majority Vote Label"].dropna().unique()
        ).difference(LABEL_TO_INDEX)

        if invalid_labels:
            raise ValueError(
                f"Unexpected labels found: {sorted(invalid_labels)}"
            )

        invalid_partitions = set(
            annotations["Partition"].dropna().unique()
        ).difference(VALID_PARTITIONS)

        if invalid_partitions:
            raise ValueError(
                f"Unexpected partitions found: "
                f"{sorted(invalid_partitions)}"
            )

        if annotations[VOTE_COLUMN].isna().any():
            raise ValueError("Annotator vote counts contain missing values.")

        if not annotations[VOTE_COLUMN].between(0, 7).all():
            raise ValueError(
                "SSA annotator vote counts must be between 0 and 7."
            )

    def _validate_image_files(self) -> None:
        """Confirm every annotation points to an existing image."""
        missing_images = [
            image_name
            for image_name in self.annotations["Image Name"]
            if not (self.images_dir / str(image_name)).is_file()
        ]

        if missing_images:
            raise FileNotFoundError(
                f"{len(missing_images)} MHIST images are missing. "
                f"Examples: {missing_images[:5]}"
            )
