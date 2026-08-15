"""Shared training and evaluation engine for PathoReason baselines."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Classification metrics collected on one dataset partition."""

    loss: float
    accuracy: float
    roc_auc: float


@dataclass(frozen=True, slots=True)
class BaselineRunResult:
    """Reproducible summary of one baseline training run."""

    model_name: str
    training_mode: str
    device: str
    epochs: int
    trainable_parameters: int
    total_parameters: int
    runtime_seconds: float
    runtime_hours: float
    gpu_hours: float
    accelerator_hours: float
    test_loss: float
    test_accuracy: float
    test_roc_auc: float

    def save_json(self, output_path: Path) -> None:
        """Save the run summary as formatted JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(asdict(self), file, indent=2)


def select_device() -> torch.device:
    """Select the best available PyTorch compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    """Synchronize asynchronous accelerator operations for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return trainable and total parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return trainable, total


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    """Train a model for one epoch and return mean training loss."""
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Training DataLoader produced no samples.")

    return total_loss / total_samples


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EvaluationMetrics:
    """Evaluate a binary classifier using loss, accuracy, and ROC-AUC."""
    model.eval()

    total_loss = 0.0
    total_samples = 0
    targets: list[int] = []
    predictions: list[int] = []
    positive_probabilities: list[float] = []

    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)

        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(
                "Baseline models must return logits with shape [batch, 2]."
            )

        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)
        predicted_labels = probabilities.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        targets.extend(labels.cpu().tolist())
        predictions.extend(predicted_labels.cpu().tolist())
        positive_probabilities.extend(
            probabilities[:, 1].cpu().tolist()
        )

    if total_samples == 0:
        raise RuntimeError("Evaluation DataLoader produced no samples.")

    if len(set(targets)) < 2:
        raise ValueError(
            "ROC-AUC requires both HP and SSA samples in the test set."
        )

    return EvaluationMetrics(
        loss=total_loss / total_samples,
        accuracy=float(accuracy_score(targets, predictions)),
        roc_auc=float(roc_auc_score(targets, positive_probabilities)),
    )


def run_baseline_training(
    *,
    model: nn.Module,
    model_name: str,
    training_mode: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    epochs: int,
    device: torch.device | None = None,
    checkpoint_path: Path | None = None,
) -> BaselineRunResult:
    """Train and evaluate one baseline while recording compute time."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")

    resolved_device = device or select_device()
    model = model.to(resolved_device)
    criterion = nn.CrossEntropyLoss()

    trainable_parameters, total_parameters = count_parameters(model)

    synchronize_device(resolved_device)
    start_time = time.perf_counter()

    for epoch_index in range(epochs):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=resolved_device,
        )

        print(
            f"Epoch {epoch_index + 1:02d}/{epochs:02d} "
            f"| train_loss={train_loss:.4f}"
        )

    synchronize_device(resolved_device)
    runtime_seconds = time.perf_counter() - start_time

    test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=resolved_device,
    )

    runtime_hours = runtime_seconds / 3600.0

    gpu_hours = (
        runtime_hours
        if resolved_device.type == "cuda"
        else 0.0
    )
    accelerator_hours = (
        runtime_hours
        if resolved_device.type in {"cuda", "mps"}
        else 0.0
    )

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "training_mode": training_mode,
                "epochs": epochs,
            },
            checkpoint_path,
        )

    return BaselineRunResult(
        model_name=model_name,
        training_mode=training_mode,
        device=str(resolved_device),
        epochs=epochs,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        runtime_seconds=runtime_seconds,
        runtime_hours=runtime_hours,
        gpu_hours=gpu_hours,
        accelerator_hours=accelerator_hours,
        test_loss=test_metrics.loss,
        test_accuracy=test_metrics.accuracy,
        test_roc_auc=test_metrics.roc_auc,
    )
