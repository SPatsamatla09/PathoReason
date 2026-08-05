"""Zero-shot PLIP predictor for MHIST classification."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from models.base import BasePredictor, PredictionResult


DEFAULT_CLASS_PROMPTS: dict[str, str] = {
    "HP": (
        "A histopathology image of a hyperplastic colorectal polyp "
        "with regular crypt architecture."
    ),
    "SSA": (
        "A histopathology image of a sessile serrated adenoma "
        "with serrated and distorted crypt architecture."
    ),
}


class PLIPPredictor(BasePredictor):
    """Run zero-shot MHIST classification using the PLIP model."""

    def __init__(
        self,
        model_id: str = "vinid/plip",
        device: str | None = None,
    ) -> None:
        self._model_name = model_id
        self.device = self._resolve_device(device)

        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    @property
    def model_name(self) -> str:
        """Return the Hugging Face model identifier."""
        return self._model_name

    def predict(
        self,
        image: Image.Image,
        prompt: str | Sequence[str] | None = None,
    ) -> PredictionResult:
        """Predict HP versus SSA using zero-shot text prompts."""
        self.validate_image(image)

        labels, prompts = self._prepare_prompts(prompt)

        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)
            probabilities = outputs.logits_per_image.softmax(dim=-1)[0]

        predicted_index = int(probabilities.argmax().item())
        predicted_label = labels[predicted_index]
        confidence = float(probabilities[predicted_index].item())

        class_probabilities = {
            label: float(probability.item())
            for label, probability in zip(labels, probabilities, strict=True)
        }

        return PredictionResult(
            label=predicted_label,
            confidence=confidence,
            explanation=None,
            evidence=(),
            model_name=self.model_name,
            metadata={
                "device": str(self.device),
                "class_probabilities": class_probabilities,
                "prompts": dict(zip(labels, prompts, strict=True)),
            },
        )

    @staticmethod
    def _prepare_prompts(
        prompt: str | Sequence[str] | None,
    ) -> tuple[list[str], list[str]]:
        """Normalize user-supplied prompts into HP and SSA class prompts."""
        labels = ["HP", "SSA"]

        if prompt is None:
            return labels, [DEFAULT_CLASS_PROMPTS[label] for label in labels]

        if isinstance(prompt, str):
            raise ValueError(
                "PLIP requires one prompt per class. "
                "Pass a sequence containing the HP and SSA prompts."
            )

        prompts = list(prompt)

        if len(prompts) != 2:
            raise ValueError(
                "PLIP requires exactly two prompts ordered as HP then SSA."
            )

        if not all(isinstance(item, str) and item.strip() for item in prompts):
            raise ValueError("All PLIP class prompts must be non-empty strings.")

        return labels, prompts

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        """Select an available inference device."""
        if device is not None:
            resolved = torch.device(device)
        elif torch.backends.mps.is_available():
            resolved = torch.device("mps")
        elif torch.cuda.is_available():
            resolved = torch.device("cuda")
        else:
            resolved = torch.device("cpu")

        return resolved
