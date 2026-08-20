"""Select and render the fixed PCam faithfulness subset.

Selects the first 10 normal and first 10 tumor samples encountered in the
canonical, unshuffled PatchCamelyon test stream. The Hugging Face dataset
revision is pinned by data.pcam_data_loader, making the selection reproducible.

The original 96x96 PCam image is resized to 224x224 before rendering the
4x4 localization grid used by the PathoReason Track B protocol.

Outputs are isolated from the frozen MHIST experiment:
    pcam/clean/
    pcam/gridded/
    pcam/selection_manifest.json
"""

from __future__ import annotations

import json
import os
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
PCAM_ROOT = os.path.join(ROOT, "pcam")
CLEAN_DIR = os.path.join(PCAM_ROOT, "clean")
GRIDDED_DIR = os.path.join(PCAM_ROOT, "gridded")
MANIFEST = os.path.join(PCAM_ROOT, "selection_manifest.json")

sys.path.insert(0, REPO)

from data.pcam_data_loader import (  # noqa: E402
    DATASET_ID,
    DATASET_REVISION,
    PCamDataset,
)
from grid_experiment.grid import draw_grid  # noqa: E402


N_PER_CLASS = 10
TARGET_SIZE = (224, 224)
PARTITION = "test"


def select():
    counts = {"normal": 0, "tumor": 0}
    selected = []

    dataset = PCamDataset(
        partition=PARTITION,
        shuffle=False,
    )

    for sample in dataset:
        label_name = sample["label_name"]

        if counts[label_name] >= N_PER_CLASS:
            continue

        counts[label_name] += 1
        selected.append(sample)

        if all(n == N_PER_CLASS for n in counts.values()):
            break

    if counts != {"normal": N_PER_CLASS, "tumor": N_PER_CLASS}:
        raise RuntimeError(f"Could not construct balanced subset: {counts}")

    return selected


def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(GRIDDED_DIR, exist_ok=True)

    selected = select()
    records = []

    for sample in selected:
        name = sample["image_name"] + ".png"

        clean = sample["image"].convert("RGB").resize(
            TARGET_SIZE,
            Image.Resampling.BILINEAR,
        )
        gridded = draw_grid(clean)

        clean_name = name
        grid_name = name.replace(".png", "_grid.png")

        clean.save(os.path.join(CLEAN_DIR, clean_name))
        gridded.save(os.path.join(GRIDDED_DIR, grid_name))

        records.append(
            {
                "image": name,
                "source_image_name": sample["image_name"],
                "partition": sample["partition"],
                "label": sample["label_name"],
                "label_id": sample["label"],
                "original_size": list(sample["image"].size),
                "rendered_size": list(TARGET_SIZE),
                "clean": f"pcam/clean/{clean_name}",
                "gridded": f"pcam/gridded/{grid_name}",
            }
        )

    payload = {
        "dataset": "PatchCamelyon",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "partition": PARTITION,
        "selection_rule": (
            "first 10 normal and first 10 tumor samples encountered in the "
            "canonical unshuffled test stream"
        ),
        "n_tiles": len(records),
        "class_counts": {
            "normal": sum(r["label"] == "normal" for r in records),
            "tumor": sum(r["label"] == "tumor" for r in records),
        },
        "source_size": [96, 96],
        "rendered_size": list(TARGET_SIZE),
        "grid": {
            "rows": 4,
            "cols": 4,
            "n_cells": 16,
        },
        "tiles": records,
    }

    with open(MANIFEST, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"selected {len(records)} PCam test tiles")
    print(f"class counts: {payload['class_counts']}")
    print(f"manifest: {MANIFEST}")
    print(f"clean: {CLEAN_DIR}")
    print(f"gridded: {GRIDDED_DIR}")


if __name__ == "__main__":
    main()
