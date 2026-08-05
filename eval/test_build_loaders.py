"""Smoke test for the MHIST DataLoaders."""

from data.build_loaders import build_mhist_loaders


def main() -> None:
    loaders = build_mhist_loaders()

    train_batch = next(iter(loaders.train))
    test_batch = next(iter(loaders.test))

    print("Train image batch:", train_batch["image"].shape)
    print("Train labels:", train_batch["label"].shape)
    print("Test image batch:", test_batch["image"].shape)
    print("Test labels:", test_batch["label"].shape)


if __name__ == "__main__":
    main()
