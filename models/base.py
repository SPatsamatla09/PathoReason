"""Shared prediction interface for PathoReason model tracks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from PIL import Image


VALID_LABELS = frozenset({"HP", "SSA"})


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Standardized output returned by every PathoReason predictor."""

    label: str
    confidence: float | None
    explanation: str | None = None
    evidence: tuple[Any, ...] = ()
    model_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in VALID_LABELS:
            raise ValueError(
                f"Invalid label {self.label!r}. "
                f"Expected one of {sorted(VALID_LABELS)}."
            )

        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0 and 1, or None."
            )

        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty.")


class BasePredictor(ABC):
    """Abstract interface implemented by Track A and Track B models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the predictor's stable identifier."""

    @abstractmethod
    def predict(
        self,
        image: Image.Image,
        prompt: str | Sequence[str],
    ) -> PredictionResult:
        """Return a standardized prediction for one image."""

    @staticmethod
    def validate_image(image: Image.Image) -> None:
        """Validate the shared image input contract."""
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Expected PIL.Image.Image, got {type(image).__name__}."
            )

        if image.mode != "RGB":
            raise ValueError(
                f"Expected RGB image, got image mode {image.mode!r}."
            )
