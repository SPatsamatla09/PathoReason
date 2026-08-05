"""Project-wide configuration utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "configs" / "config.yaml"


def load_config() -> dict[str, Any]:
    """Load and validate the project YAML configuration."""
    if not CONFIG_FILE.is_file():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")

    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)

    if not isinstance(loaded_config, dict):
        raise ValueError("config.yaml must contain a top-level mapping.")

    return loaded_config


config = load_config()

# Paths
DATA_ROOT = PROJECT_ROOT / config["paths"]["data_root"]
IMAGES_DIR = PROJECT_ROOT / config["paths"]["images"]
ANNOTATIONS_FILE = PROJECT_ROOT / config["paths"]["annotations"]
RESULTS_DIR = PROJECT_ROOT / config["paths"]["results"]
FIGURES_DIR = PROJECT_ROOT / config["paths"]["figures"]

# Dataset
IMAGE_SIZE = int(config["dataset"]["image_size"])
NUM_CLASSES = int(config["dataset"]["num_classes"])
CLASS_NAMES = tuple(config["dataset"]["class_names"])

# Training
SEED = int(config["seed"])
BATCH_SIZE = int(config["training"]["batch_size"])
NUM_WORKERS = int(config["training"]["num_workers"])
EPOCHS = int(config["training"]["epochs"])
LEARNING_RATE = float(config["training"]["learning_rate"])
WEIGHT_DECAY = float(config["training"]["weight_decay"])
