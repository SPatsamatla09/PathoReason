"""Train and evaluate the ViT-B/16 linear-probe baseline on MHIST."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from data.build_loaders import build_mhist_loaders
from models.vit_b16_baseline import build_vit_b16_linear_probe
from training.engine import run_baseline_training, select_device
from utils.config import (
    EPOCHS,
    LEARNING_RATE,
    RESULTS_DIR,
    SEED,
    WEIGHT_DECAY,
)
from utils.reproducibility import set_global_seed


def main() -> None:
    """Run one reproducible ViT-B/16 linear-probe experiment."""
    set_global_seed(SEED)

    device = select_device()
    loaders = build_mhist_loaders()

    model = build_vit_b16_linear_probe()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError("ViT linear probe has no trainable parameters.")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    result = run_baseline_training(
        model=model,
        model_name="vit_b_16",
        training_mode="linear_probe",
        train_loader=loaders.train,
        test_loader=loaders.test,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=device,
    )

    output_path = Path(RESULTS_DIR) / "baselines" / "vit_b16.json"
    result.save_json(output_path)

    print(
        json.dumps(
            {
                "model": result.model_name,
                "device": result.device,
                "accuracy": result.test_accuracy,
                "roc_auc": result.test_roc_auc,
                "runtime_seconds": result.runtime_seconds,
                "accelerator_hours": result.accelerator_hours,
                "trainable_parameters": result.trainable_parameters,
                "total_parameters": result.total_parameters,
                "saved_to": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
