"""GPT-4o predictor for structured MHIST pathology explanations."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Sequence
from io import BytesIO

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from models.base import BasePredictor, PredictionResult


DEFAULT_PROMPT = """
Classify this colorectal histopathology image as exactly one of:

- HP: Hyperplastic polyp
- SSA: Sessile serrated adenoma

Base the classification only on visible histological evidence.

Provide:
1. The predicted label.
2. A self-reported confidence from 0.0 to 1.0.
3. A concise diagnostic explanation.
4. A list of specific visible histological features supporting the prediction.

Do not mention features that are not visibly supported by the image.
""".strip()


RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["HP", "SSA"],
        },
        "verbalized_confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "explanation": {
            "type": "string",
            "minLength": 1,
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 1,
            },
        },
    },
    "required": [
        "label",
        "verbalized_confidence",
        "explanation",
        "evidence",
    ],
    "additionalProperties": False,
}


class GPT4Predictor(BasePredictor):
    """Track B predictor using an OpenAI vision-language model."""

    def __init__(
        self,
        model: str = "gpt-4o",
        image_detail: str = "high",
    ) -> None:
        """
        Initialize the API-backed pathology predictor.

        Args:
            model: OpenAI vision-capable model identifier.
            image_detail: Image processing detail: low, high, or auto.
        """
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY was not found. "
                "Add it to the project's .env file."
            )

        if image_detail not in {"low", "high", "auto"}:
            raise ValueError(
                "image_detail must be 'low', 'high', or 'auto'."
            )

        self.client = OpenAI(api_key=api_key)
        self._model_name = model
        self.image_detail = image_detail

    @property
    def model_name(self) -> str:
        """Return the API model identifier."""
        return self._model_name

    def predict(
        self,
        image: Image.Image,
        prompt: str | Sequence[str] | None,
    ) -> PredictionResult:
        """
        Generate an MHIST classification and structured explanation.

        The API model's verbalized confidence is retained in metadata for
        calibration auditing. It is not treated as a reliable probability
        until the confidence audit is completed.
        """
        self.validate_image(image)

        normalized_prompt = self._normalize_prompt(prompt)
        image_data_url = self._encode_image(image)

        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": normalized_prompt,
                        },
                        {
                            "type": "input_image",
                            "image_url": image_data_url,
                            "detail": self.image_detail,
                        },
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "mhist_pathology_prediction",
                    "description": (
                        "Structured HP-versus-SSA histopathology prediction."
                    ),
                    "strict": True,
                    "schema": RESPONSE_SCHEMA,
                }
            },
        )

        if not response.output_text:
            raise RuntimeError(
                "The OpenAI API returned no structured prediction text."
            )

        try:
            payload = json.loads(response.output_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "The OpenAI API returned invalid JSON."
            ) from exc

        label = str(payload["label"])
        explanation = str(payload["explanation"]).strip()
        evidence = tuple(
            str(item).strip()
            for item in payload["evidence"]
            if str(item).strip()
        )
        verbalized_confidence = float(
            payload["verbalized_confidence"]
        )

        return PredictionResult(
            label=label,
            confidence=None,
            explanation=explanation,
            evidence=evidence,
            model_name=self.model_name,
            metadata={
                "verbalized_confidence": verbalized_confidence,
                "confidence_status": "unaudited",
                "image_detail": self.image_detail,
                "response_id": response.id,
            },
        )


    @staticmethod
    def _normalize_prompt(
        prompt: str | Sequence[str] | None,
    ) -> str:
        """Normalize the shared prompt input into one API instruction."""
        if prompt is None:
            return DEFAULT_PROMPT

        if isinstance(prompt, str):
            normalized = prompt.strip()
            return normalized or DEFAULT_PROMPT

        prompt_parts = [
            str(item).strip()
            for item in prompt
            if str(item).strip()
        ]

        return "\n\n".join(prompt_parts) if prompt_parts else DEFAULT_PROMPT

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """Encode a PIL image as a PNG data URL."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{encoded}"
