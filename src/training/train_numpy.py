from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.fc_numpy import train as train_core, predict
from training.metrics import accuracy


@dataclass(frozen=True)
class NumpyTrainResult:
    params: dict
    train_acc: float
    history: list[tuple[int, float]]


def train_numpy_model(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    hidden_size: int,
    decay: float,
    learning_rate: float,
    epochs: int,
    activation: str = "relu",
    seed: int = 1,
    log_every: int = 1000,
) -> NumpyTrainResult:
    params, history = train_core(
        X=X,
        y=y,
        n_classes=n_classes,
        hidden_size=hidden_size,
        decay=decay,
        learning_rate=learning_rate,
        epochs=epochs,
        activation=activation,
        seed=seed,
        log_every=log_every,
    )
    pred = predict(X, params, activation=activation)
    acc = accuracy(pred, y)
    return NumpyTrainResult(params=params, train_acc=acc, history=history)
