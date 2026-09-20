"""Small PyTorch models and evaluation helpers for the centralized baseline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: Iterable[int] = (128, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dimensions = [input_dim, *hidden_dims]
        for left, right in zip(dimensions, dimensions[1:]):
            layers.extend([nn.Linear(left, right), nn.ReLU(), nn.Dropout(0.1)])
        layers.append(nn.Linear(dimensions[-1], num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_size_kb(model: nn.Module) -> float:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / 1024


def train_model(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int, learning_rate: float, weight_decay: float, seed: int) -> float:
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).long()), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
    return time.perf_counter() - started


def evaluate_model(model: nn.Module, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(x_test)).argmax(dim=1).numpy()
    matrix = confusion_matrix(y_test, predictions)
    false_positive = matrix.sum(axis=0) - np.diag(matrix)
    true_negative = matrix.sum() - matrix.sum(axis=0) - matrix.sum(axis=1) + np.diag(matrix)
    negative_cases = false_positive + true_negative
    per_class_fpr = np.divide(false_positive, negative_cases, out=np.zeros_like(false_positive, dtype=float), where=negative_cases != 0)
    return {"accuracy": accuracy_score(y_test, predictions), "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0), "precision": precision_score(y_test, predictions, average="macro", zero_division=0), "recall": recall_score(y_test, predictions, average="macro", zero_division=0), "fpr": float(per_class_fpr.mean()), "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0), "predictions": predictions.tolist()}


def save_metrics(metrics: dict, path: str | Path) -> None:
    serializable = {key: value for key, value in metrics.items() if key != "predictions"}
    Path(path).write_text(json.dumps(serializable, indent=2), encoding="utf-8")
