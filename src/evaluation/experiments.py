from __future__ import annotations

import os

from config.defaults import DEFAULT_SEED
from data.loaders import load_flower_dataset, load_spiral_dataset
from training.train_numpy import train_numpy_model
from training.train_torch import train_torch_model
from visualization.decision_boundary import plot_decision_boundary
from visualization.savefig import save_figure


def run_flower_numpy(hidden_size: int, decay: float, learning_rate: float, epochs: int, activation: str = "relu"):
    ds = load_flower_dataset(seed=DEFAULT_SEED)
    result = train_numpy_model(
        X=ds.X, y=ds.y, n_classes=ds.n_classes,
        hidden_size=hidden_size, decay=decay, learning_rate=learning_rate,
        epochs=epochs, activation=activation, seed=DEFAULT_SEED,
    )

    fig, _ = plot_decision_boundary(
        pred_func=lambda x: __predict_numpy(x, result.params, activation),
        X=ds.X, y=ds.y, title="Neural Network (NumPy) - Flower"
    )
    save_figure(fig, "outputs/figures/flower-boundary.jpg")
    return result


def run_spiral_numpy(hidden_size: int, decay: float, learning_rate: float, epochs: int, activation: str = "relu"):
    ds = load_spiral_dataset(seed=DEFAULT_SEED)
    result = train_numpy_model(
        X=ds.X, y=ds.y, n_classes=ds.n_classes,
        hidden_size=hidden_size, decay=decay, learning_rate=learning_rate,
        epochs=epochs, activation=activation, seed=DEFAULT_SEED,
    )

    fig, _ = plot_decision_boundary(
        pred_func=lambda x: __predict_numpy(x, result.params, activation),
        X=ds.X, y=ds.y, title="Neural Network (NumPy) - Spiral"
    )
    save_figure(fig, "outputs/figures/spiral-boundary.jpg")
    return result


def run_flower_torch(hidden_size: int, decay: float, learning_rate: float, epochs: int, batch_size: int):
    ds = load_flower_dataset(seed=DEFAULT_SEED)
    result = train_torch_model(
        X=ds.X, y=ds.y,
        hidden_size=hidden_size, decay=decay, learning_rate=learning_rate,
        epochs=epochs, batch_size=batch_size, seed=DEFAULT_SEED,
    )

    fig, _ = plot_decision_boundary(
        pred_func=lambda x: __predict_torch(x, result.model),
        X=ds.X, y=ds.y, title="Neural Network (PyTorch) - Flower"
    )
    save_figure(fig, "outputs/figures/flower-boundary.jpg")
    return result


def run_spiral_torch(hidden_size: int, decay: float, learning_rate: float, epochs: int, batch_size: int):
    ds = load_spiral_dataset(seed=DEFAULT_SEED)
    result = train_torch_model(
        X=ds.X, y=ds.y,
        hidden_size=hidden_size, decay=decay, learning_rate=learning_rate,
        epochs=epochs, batch_size=batch_size, seed=DEFAULT_SEED,
    )

    fig, _ = plot_decision_boundary(
        pred_func=lambda x: __predict_torch(x, result.model),
        X=ds.X, y=ds.y, title="Neural Network (PyTorch) - Spiral"
    )
    save_figure(fig, "outputs/figures/spiral-boundary.jpg")
    return result


def __predict_numpy(X, params, activation):
    from models.fc_numpy import predict
    return predict(X, params, activation=activation)


def __predict_torch(X, model):
    from models.fc_torch import predict_torch
    return predict_torch(X, model)
