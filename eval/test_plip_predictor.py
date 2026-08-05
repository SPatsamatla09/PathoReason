"""Smoke test for the PLIP zero-shot predictor."""

from data.mhist_data_loader import MHISTDataset
from models.plip_predictor import PLIPPredictor


def main() -> None:
    dataset = MHISTDataset(partition="test")
    sample = dataset[0]

    predictor = PLIPPredictor()
    result = predictor.predict(sample["image"], prompt=None)

    print(f"Image: {sample['image_name']}")
    print(f"True label: {sample['label_name']}")
    print(f"Predicted label: {result.label}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Class probabilities: {result.metadata['class_probabilities']}")


if __name__ == "__main__":
    main()
