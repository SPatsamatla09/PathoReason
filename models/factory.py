"""Factory for constructing PathoReason model predictors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from models.base import BasePredictor
from models.gpt4_predictor import GPT4Predictor
from models.plip_predictor import PLIPPredictor


PredictorBuilder = Callable[..., BasePredictor]


_PREDICTOR_REGISTRY: dict[str, PredictorBuilder] = {
    "plip": PLIPPredictor,
    "gpt4": GPT4Predictor,
    "gpt-4o": GPT4Predictor,
    "openai": GPT4Predictor,
}


def build_predictor(
    predictor_name: str,
    **kwargs: Any,
) -> BasePredictor:
    """
    Construct a registered PathoReason predictor.

    Args:
        predictor_name: Registered predictor identifier.
        **kwargs: Keyword arguments passed to the predictor constructor.

    Returns:
        An initialized predictor implementing BasePredictor.

    Raises:
        ValueError: If the requested predictor is not registered.
    """
    normalized_name = predictor_name.strip().lower()

    try:
        builder = _PREDICTOR_REGISTRY[normalized_name]
    except KeyError as exc:
        available = ", ".join(sorted(_PREDICTOR_REGISTRY))
        raise ValueError(
            f"Unknown predictor {predictor_name!r}. "
            f"Available predictors: {available}."
        ) from exc

    predictor = builder(**kwargs)

    if not isinstance(predictor, BasePredictor):
        raise TypeError(
            f"Registered builder for {normalized_name!r} did not return "
            "a BasePredictor instance."
        )

    return predictor


def available_predictors() -> tuple[str, ...]:
    """Return all registered predictor identifiers."""
    return tuple(sorted(_PREDICTOR_REGISTRY))
