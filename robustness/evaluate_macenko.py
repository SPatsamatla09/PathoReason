"""Evaluate PCam ResNet-18 robustness to Macenko stain normalization."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torchvision import transforms

from data.pcam_data_loader import PCamDataset
from models.resnet18_baseline import build_resnet18_finetune
from robustness.macenko import macenko_normalize
from training.engine import select_device


CHECKPOINT = Path("models/A_pcam_seed42/resnet18.pt")
OUTPUT = Path("results/pcam/robustness/macenko_resnet18.json")

SUBSET_SIZE = 500
IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PREPROCESS = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


@torch.inference_mode()
def predict(model, image, device):
    tensor = PREPROCESS(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    return int(probabilities.argmax().item()), float(probabilities[1].item())


def main() -> None:
    device = select_device()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model = build_resnet18_finetune()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = PCamDataset(
        partition="test",
        max_samples=SUBSET_SIZE,
    )

    targets = []

    original_predictions = []
    normalized_predictions = []

    original_probabilities = []
    normalized_probabilities = []

    prediction_flips = 0
    confidence_changes = []
    normalization_failures = 0

    for index, sample in enumerate(dataset):
        image = sample["image"]
        normalized = macenko_normalize(image)

        normalized_array = __import__("numpy").asarray(
            normalized, dtype=__import__("numpy").float32
        )
        normalization_failed = (
            normalized_array.mean() < 5
            or normalized_array.mean() > 250
            or normalized_array.std() < 2
        )

        if normalization_failed:
            normalization_failures += 1
            normalized = image

        original_pred, original_prob = predict(model, image, device)
        normalized_pred, normalized_prob = predict(model, normalized, device)

        target = int(sample["label"])

        targets.append(target)
        original_predictions.append(original_pred)
        normalized_predictions.append(normalized_pred)

        original_probabilities.append(original_prob)
        normalized_probabilities.append(normalized_prob)

        if original_pred != normalized_pred:
            prediction_flips += 1

        confidence_changes.append(
            abs(normalized_prob - original_prob)
        )

        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1}/{SUBSET_SIZE}")

    original_accuracy = accuracy_score(targets, original_predictions)
    normalized_accuracy = accuracy_score(targets, normalized_predictions)

    original_auc = roc_auc_score(targets, original_probabilities)
    normalized_auc = roc_auc_score(targets, normalized_probabilities)

    result = {
        "dataset": "pcam",
        "model": "resnet18",
        "checkpoint": str(CHECKPOINT),
        "subset_size": len(targets),
        "normalization": "macenko",
        "original": {
            "accuracy": float(original_accuracy),
            "roc_auc": float(original_auc),
        },
        "macenko": {
            "accuracy": float(normalized_accuracy),
            "roc_auc": float(normalized_auc),
        },
        "delta": {
            "accuracy": float(normalized_accuracy - original_accuracy),
            "roc_auc": float(normalized_auc - original_auc),
        },
        "normalization_failures": normalization_failures,
        "normalization_failure_rate": normalization_failures / len(targets),
        "failure_handling": "fallback_to_original_image",
        "prediction_flip_rate": prediction_flips / len(targets),
        "mean_absolute_probability_change": (
            sum(confidence_changes) / len(confidence_changes)
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2))

    print()
    print(json.dumps(result, indent=2))
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
