"""Train and evaluate a UNI linear probe on MHIST."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from data.build_loaders import build_mhist_loaders
from models.uni_baseline import UNIEncoder
from training.engine import select_device, synchronize_device
from training.foundation_embeddings import (
    extract_embeddings,
    load_embedding_cache,
    save_embedding_cache,
)
from training.linear_probe import (
    evaluate_linear_probe,
    train_linear_probe,
)
from utils.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RESULTS_DIR,
    SEED,
    WEIGHT_DECAY,
)
from utils.reproducibility import set_global_seed


CACHE_DIR = Path("results") / "embedding_cache" / "uni"
TRAIN_CACHE = CACHE_DIR / "mhist_train.pt"
TEST_CACHE = CACHE_DIR / "mhist_test.pt"

CHECKPOINT_PATH = (
    Path("models") / f"A_mhist_seed{SEED}" / "uni.pt"
)


def main() -> None:
    """Run the frozen-UNI MHIST linear-probe experiment."""
    set_global_seed(SEED)
    device = select_device()

    synchronize_device(device)
    start_time = time.perf_counter()

    if TRAIN_CACHE.is_file() and TEST_CACHE.is_file():
        print("Loading cached UNI embeddings...")

        train_embeddings, train_labels = load_embedding_cache(TRAIN_CACHE)
        test_embeddings, test_labels = load_embedding_cache(TEST_CACHE)

    else:
        print("Loading UNI encoder...")
        encoder = UNIEncoder()

        print("Building MHIST loaders with UNI preprocessing...")
        loaders = build_mhist_loaders(
            batch_size=1,
            train_transform=encoder.transform,
            test_transform=encoder.transform,
        )

        print("Extracting MHIST train embeddings...")
        train_embeddings, train_labels = extract_embeddings(
            encoder,
            loaders.train,
            device,
        )
        save_embedding_cache(
            TRAIN_CACHE,
            train_embeddings,
            train_labels,
            metadata={
                "model": "uni",
                "dataset": "mhist",
                "split": "train",
                "seed": SEED,
                "embedding_dim": UNIEncoder.embedding_dim,
            },
        )

        print("Extracting MHIST test embeddings...")
        test_embeddings, test_labels = extract_embeddings(
            encoder,
            loaders.test,
            device,
        )
        save_embedding_cache(
            TEST_CACHE,
            test_embeddings,
            test_labels,
            metadata={
                "model": "uni",
                "dataset": "mhist",
                "split": "test",
                "seed": SEED,
                "embedding_dim": UNIEncoder.embedding_dim,
            },
        )

        del encoder

        if device.type == "mps":
            torch.mps.empty_cache()

    if train_embeddings.shape[1] != UNIEncoder.embedding_dim:
        raise RuntimeError(
            "Unexpected UNI embedding dimension: "
            f"{train_embeddings.shape[1]}"
        )

    print(
        "Embedding shapes:",
        tuple(train_embeddings.shape),
        tuple(test_embeddings.shape),
    )

    classifier = train_linear_probe(
        train_embeddings,
        train_labels,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        device=device,
    )

    metrics = evaluate_linear_probe(
        classifier,
        test_embeddings,
        test_labels,
        device=device,
    )

    synchronize_device(device)
    runtime_seconds = time.perf_counter() - start_time
    accelerator_hours = (
        runtime_seconds / 3600.0
        if device.type in {"cuda", "mps"}
        else 0.0
    )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": classifier.state_dict(),
            "model_name": "uni",
            "training_mode": "linear_probe",
            "embedding_dim": train_embeddings.shape[1],
            "seed": SEED,
        },
        CHECKPOINT_PATH,
    )

    result = {
        "model_name": "uni",
        "training_mode": "linear_probe",
        "device": str(device),
        "epochs": EPOCHS,
        "trainable_parameters": sum(
            p.numel() for p in classifier.parameters()
        ),
        "total_parameters": None,
        "linear_probe_parameters": sum(
            p.numel() for p in classifier.parameters()
        ),
        "runtime_seconds": runtime_seconds,
        "runtime_hours": runtime_seconds / 3600.0,
        "gpu_hours": (
            runtime_seconds / 3600.0
            if device.type == "cuda"
            else 0.0
        ),
        "accelerator_hours": accelerator_hours,
        "test_loss": metrics["loss"],
        "test_accuracy": metrics["accuracy"],
        "test_roc_auc": metrics["roc_auc"],
    }

    output_path = Path(RESULTS_DIR) / "baselines" / "uni.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print(
        json.dumps(
            {
                "model": "uni",
                "device": str(device),
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics["roc_auc"],
                "runtime_seconds": runtime_seconds,
                "accelerator_hours": accelerator_hours,
                "checkpoint": str(CHECKPOINT_PATH),
                "saved_to": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
