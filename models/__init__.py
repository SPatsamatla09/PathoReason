"""Model interfaces and predictor registry for PathoReason."""

from .base import BasePredictor, PredictionResult
from .factory import available_predictors, build_predictor
from .gpt4_predictor import GPT4Predictor
from .plip_predictor import PLIPPredictor

__all__ = [
    "BasePredictor",
    "PredictionResult",
    "PLIPPredictor",
    "GPT4Predictor",
    "build_predictor",
    "available_predictors",
]
