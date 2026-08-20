"""GPT-4o predictor for structured MHIST pathology explanations."""

from __future__ import annotations

import base64
import json
import os
import re
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


FEATURE_VOCABULARY = [
    "nuclear atypia",
    "crypt/gland architecture",
    "serration",
    "mitotic figures",
    "necrosis",
    "stroma",
    "surface epithelium",
    "inflammatory infiltrate",
]

PCAM_FEATURE_VOCABULARY = [
    "tumor cell clusters",
    "nuclear pleomorphism",
    "high nuclear-cytoplasmic ratio",
    "mitotic figures",
    "glandular architecture",
    "necrosis",
    "lymphoid tissue",
    "stroma",
]

PROTOCOLS = {
    "mhist": {
        "labels": ["HP", "SSA"],
        "features": FEATURE_VOCABULARY,
    },
    "pcam": {
        "labels": ["normal", "tumor"],
        "features": PCAM_FEATURE_VOCABULARY,
    },
}

GRID_CELLS = [
    "A1", "A2", "A3", "A4",
    "B1", "B2", "B3", "B4",
    "C1", "C2", "C3", "C4",
    "D1", "D2", "D3", "D4",
]

def build_response_schema(protocol: str = "mhist") -> dict:
    """Build the strict structured-output schema for one dataset protocol."""
    if protocol not in PROTOCOLS:
        raise ValueError(
            f"Unknown protocol {protocol!r}. Expected one of {sorted(PROTOCOLS)}."
        )

    cfg = PROTOCOLS[protocol]
    return {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": list(cfg["labels"]),
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "feature": {
                            "type": "string",
                            "enum": list(cfg["features"]),
                        },
                        "grid_cells": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": GRID_CELLS,
                            },
                            "minItems": 1,
                        },
                        "description": {
                            "type": "string",
                            "minLength": 1,
                        },
                    },
                    "required": [
                        "feature",
                        "grid_cells",
                        "description",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "label",
            "confidence",
            "evidence",
        ],
        "additionalProperties": False,
    }


# Backward-compatible public constant: existing MHIST code sees the exact
# original protocol unless a predictor explicitly requests another protocol.
RESPONSE_SCHEMA: dict = build_response_schema("mhist")



def extract_json(raw: str) -> str:
    """Extract one JSON object from a possibly fenced/prefixed response."""
    s = raw.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if fence:
        s = fence.group(1).strip()

    start = s.find("{")
    if start < 0:
        raise ValueError("No opening JSON brace found.")

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]

    raise ValueError("Unbalanced JSON braces.")


def validate_payload(payload: object, protocol: str = "mhist") -> list[str]:
    """Return protocol violations without silently normalizing them."""
    if protocol not in PROTOCOLS:
        raise ValueError(
            f"Unknown protocol {protocol!r}. Expected one of {sorted(PROTOCOLS)}."
        )
    cfg = PROTOCOLS[protocol]
    violations: list[str] = []

    if not isinstance(payload, dict):
        return ["top-level response is not a JSON object"]

    if set(payload) != {"label", "confidence", "evidence"}:
        violations.append(
            "top-level keys must be exactly label, confidence, evidence"
        )

    label = payload.get("label")
    if label not in set(cfg["labels"]):
        violations.append(f"invalid label: {label!r}")

    confidence = payload.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        violations.append(f"invalid confidence: {confidence!r}")

    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        violations.append("evidence must be a list")
        return violations

    for i, item in enumerate(evidence):
        prefix = f"evidence[{i}]"

        if not isinstance(item, dict):
            violations.append(f"{prefix} must be an object")
            continue

        if set(item) != {"feature", "grid_cells", "description"}:
            violations.append(
                f"{prefix} must contain exactly "
                "feature, grid_cells, description"
            )

        feature = item.get("feature")
        if feature not in cfg["features"]:
            violations.append(
                f"{prefix}.feature outside frozen vocabulary: {feature!r}"
            )

        cells = item.get("grid_cells")
        if not isinstance(cells, list) or not cells:
            violations.append(
                f"{prefix}.grid_cells must be a non-empty list"
            )
        else:
            for cell in cells:
                if cell not in GRID_CELLS:
                    violations.append(
                        f"{prefix}.grid_cells contains invalid cell: {cell!r}"
                    )

        description = item.get("description")
        if not isinstance(description, str) or not description.strip():
            violations.append(
                f"{prefix}.description must be a non-empty string"
            )

    return violations


class GPT4Predictor(BasePredictor):
    """Track B predictor using an OpenAI vision-language model."""

    def __init__(
        self,
        model: str = "gpt-4o",
        image_detail: str = "high",
        protocol: str = "mhist",
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

        if protocol not in PROTOCOLS:
            raise ValueError(
                f"Unknown protocol {protocol!r}. Expected one of {sorted(PROTOCOLS)}."
            )

        self.protocol = protocol
        self.response_schema = build_response_schema(protocol)

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
        Generate a pathology classification and structured explanation.

        The API model's verbalized confidence is retained in metadata for
        calibration auditing. It is not treated as a reliable probability
        until the confidence audit is completed.
        """
        self.validate_image(image)

        normalized_prompt = self._normalize_prompt(prompt)
        image_data_url = self._encode_image(image)

        response = self.client.responses.create(
            model=self.model_name,
            temperature=1.0,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "pathoreason_prediction",
                    "schema": self.response_schema,
                    "strict": True,
                }
            },
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": image_data_url,
                            "detail": self.image_detail,
                        },
                        {
                            "type": "input_text",
                            "text": normalized_prompt,
                        },
                    ],
                }
            ],
        )

        if not response.output_text:
            raise RuntimeError(
                "The OpenAI API returned no structured prediction text."
            )

        raw_response = response.output_text.strip()

        try:
            extracted_json = extract_json(raw_response)
            payload = json.loads(extracted_json)
        except (ValueError, json.JSONDecodeError) as exc:
            err = RuntimeError(
                f"The OpenAI API returned unparseable JSON: {exc}"
            )
            err.raw_response = raw_response
            err.response_id = response.id
            raise err from exc

        violations = validate_payload(payload, protocol=self.protocol)

        if violations:
            err = RuntimeError(
                "The OpenAI API response violated the frozen protocol: "
                + "; ".join(violations)
            )
            err.raw_response = raw_response
            err.response_id = response.id
            err.protocol_violations = violations
            raise err

        top_level_key_order = list(payload.keys())

        expected_key_order = (
            ["evidence", "label", "confidence"]
            if normalized_prompt.lstrip().startswith(
                "You are shown"
            ) and '"evidence": [' in normalized_prompt
            and normalized_prompt.rfind('"evidence"')
                < normalized_prompt.rfind('"label"')
            else ["label", "confidence", "evidence"]
        )

        key_order_valid = (
            top_level_key_order == expected_key_order
        )

        label = str(payload["label"])
        verbalized_confidence = float(payload["confidence"])

        evidence = tuple(
            {
                "feature": str(item["feature"]),
                "grid_cells": [
                    str(cell)
                    for cell in item["grid_cells"]
                ],
                "description": str(item["description"]).strip(),
            }
            for item in payload["evidence"]
        )

        explanation = None

        usage = getattr(response, "usage", None)

        input_tokens = (
            getattr(usage, "input_tokens", None)
            if usage is not None
            else None
        )
        output_tokens = (
            getattr(usage, "output_tokens", None)
            if usage is not None
            else None
        )
        total_tokens = (
            getattr(usage, "total_tokens", None)
            if usage is not None
            else None
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
                "raw_response": raw_response,
                "top_level_key_order": top_level_key_order,
                "expected_key_order": expected_key_order,
                "key_order_valid": key_order_valid,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
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
