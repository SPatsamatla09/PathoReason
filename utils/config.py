"""Project-wide configuration utilities."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "configs" / "config.yaml"

with CONFIG_FILE.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

DATA_ROOT = PROJECT_ROOT / config["paths"]["data_root"]
IMAGES_DIR = PROJECT_ROOT / config["paths"]["images"]
ANNOTATIONS_FILE = PROJECT_ROOT / config["paths"]["annotations"]

IMAGE_SIZE = config["dataset"]["image_size"]
NUM_CLASSES = config["dataset"]["num_classes"]

SEED = config["seed"]
BATCH_SIZE = config["training"]["batch_size"]
NUM_WORKERS = config["training"]["num_workers"]
