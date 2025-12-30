from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.fc_torch import train_torch, predict_torch, FCNetTorch
from training.metrics import accuracy


@dataclass(frozen=True)
class TorchTrainResult:
    model: FCNetTorch
    train_acc: float
    history: list[tuple[int, float]]


def train_torch_model(
    X: np.ndarray,
    y: np.ndarray,
    hidden_size: int,
    decay: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int = 1,
    log_every: int = 1000,
) -> TorchTrainResult:
    model, history = train_torch(
        X=X,
        y=y,
        hidden_size=hidden_size,
        decay=decay,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        log_every=log_every,
    )
    pred = predict_torch(X, model)
    acc = accuracy(pred, y)
    return TorchTrainResult(model=model, train_acc=acc, history=history)
