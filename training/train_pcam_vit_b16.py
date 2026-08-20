"""Train and evaluate ViT-B/16 linear probe on PatchCamelyon."""

from pathlib import Path
import json
import torch

from data.build_pcam_loaders import build_pcam_loaders
from models.vit_b16_baseline import build_vit_b16_linear_probe
from training.engine import run_baseline_training, select_device
from utils.reproducibility import set_global_seed

SEED = 42
EPOCHS = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


def main():
    set_global_seed(SEED)
    device = select_device()
    loaders = build_pcam_loaders()

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

    checkpoint_path = Path("models") / f"A_pcam_seed{SEED}" / "vit_b16.pt"

    result = run_baseline_training(
        model=model,
        model_name="vit_b_16",
        training_mode="linear_probe",
        train_loader=loaders.train,
        test_loader=loaders.test,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    output_path = Path("results/pcam/baselines/vit_b16.json")
    result.save_json(output_path)

    print(json.dumps({
        "model": result.model_name,
        "dataset": "pcam",
        "device": result.device,
        "accuracy": result.test_accuracy,
        "roc_auc": result.test_roc_auc,
        "runtime_seconds": result.runtime_seconds,
        "accelerator_hours": result.accelerator_hours,
        "trainable_parameters": result.trainable_parameters,
        "total_parameters": result.total_parameters,
        "saved_to": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()
