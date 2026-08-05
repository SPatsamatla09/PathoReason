"""Basic validation checks for the MHIST dataset implementation."""

from PIL import Image

from data.mhist_data_loader import MHISTDataset


def main() -> None:
    train_dataset = MHISTDataset(partition="train")
    test_dataset = MHISTDataset(partition="test")

    assert len(train_dataset) + len(test_dataset) == 3152

    sample = train_dataset[0]

    assert isinstance(sample["image"], Image.Image)
    assert sample["image"].mode == "RGB"
    assert sample["image"].size == (224, 224)
    assert sample["label"] in {0, 1}
    assert sample["label_name"] in {"HP", "SSA"}
    assert 0 <= sample["ssa_votes"] <= 7
    assert 4 <= sample["majority_agreement"] <= 7
    assert sample["partition"] == "train"

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print("MHIST data loader validation passed.")


if __name__ == "__main__":
    main()
