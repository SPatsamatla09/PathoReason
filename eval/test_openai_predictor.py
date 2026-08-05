"""Smoke test for the OpenAI pathology predictor."""

from data.mhist_data_loader import MHISTDataset
from models.gpt4_predictor import GPT4Predictor


def main() -> None:
    dataset = MHISTDataset(partition="test")
    sample = dataset[0]

    predictor = GPT4Predictor()

    result = predictor.predict(
        image=sample["image"],
        prompt=None,
    )

    print("=" * 60)
    print(f"Image: {sample['image_name']}")
    print(f"Ground Truth: {sample['label_name']}")
    print(f"Prediction: {result.label}")
    print(f"Confidence: {result.confidence}")
    print()
    print("Explanation")
    print("-" * 60)
    print(result.explanation)
    print()
    print("Evidence")
    print("-" * 60)
    for feature in result.evidence:
        print(f"• {feature}")
    print()
    print("Metadata")
    print("-" * 60)
    print(result.metadata)
    print("=" * 60)


if __name__ == "__main__":
    main()
