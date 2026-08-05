"""Train and evaluate the ResNet-18 fine-tuning baseline on MHIST."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from data.build_loaders import build_mhist_loaders
from models.resnet18_baseline import build_resnet18_finetune
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
    """Run one reproducible ResNet-18 baseline experiment."""
    set_global_seed(SEED)

    device = select_device()
    loaders = build_mhist_loaders()

    model = build_resnet18_finetune()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    result = run_baseline_training(
        model=model,
        model_name="resnet18",
        training_mode="fine_tune",
        train_loader=loaders.train,
        test_loader=loaders.test,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=device,
    )

    output_path = Path(RESULTS_DIR) / "baselines" / "resnet18.json"
    result.save_json(output_path)

    print(json.dumps(
        {
            "model": result.model_name,
            "device": result.device,
            "accuracy": result.test_accuracy,
            "roc_auc": result.test_roc_auc,
            "runtime_seconds": result.runtime_seconds,
            "accelerator_hours": result.accelerator_hours,
            "saved_to": str(output_path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
