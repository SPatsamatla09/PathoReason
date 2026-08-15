"""Linear probing on cached foundation-model embeddings."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def train_linear_probe(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    num_classes: int = 2,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    device: torch.device,
) -> nn.Linear:
    """Train a linear classifier on frozen cached embeddings."""

    if train_embeddings.ndim != 2:
        raise ValueError("train_embeddings must have shape [N, D].")

    if len(train_embeddings) != len(train_labels):
        raise ValueError("Embeddings and labels must contain the same number of samples.")

    dataset = TensorDataset(train_embeddings.float(), train_labels.long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    classifier = nn.Linear(train_embeddings.shape[1], num_classes).to(device)

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    classifier.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = classifier(embeddings)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        mean_loss = total_loss / len(dataset)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={mean_loss:.4f}"
        )

    return classifier


@torch.inference_mode()
def evaluate_linear_probe(
    classifier: nn.Linear,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a cached-embedding linear probe."""

    from sklearn.metrics import accuracy_score, roc_auc_score

    classifier.eval()

    embeddings = embeddings.float().to(device)
    labels = labels.long().to(device)

    logits = classifier(embeddings)
    probabilities = torch.softmax(logits, dim=1)

    predictions = logits.argmax(dim=1)

    targets = labels.cpu().numpy()
    predicted_labels = predictions.cpu().numpy()
    positive_probabilities = probabilities[:, 1].cpu().numpy()

    return {
        "accuracy": float(
            accuracy_score(targets, predicted_labels)
        ),
        "roc_auc": float(
            roc_auc_score(targets, positive_probabilities)
        ),
        "loss": float(
            nn.functional.cross_entropy(logits, labels).item()
        ),
    }
