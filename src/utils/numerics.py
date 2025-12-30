from __future__ import annotations

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(z.dtype)


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Stable sigmoid
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_grad(sigmoid_out: np.ndarray) -> np.ndarray:
    # derivative using output: s*(1-s)
    return sigmoid_out * (1.0 - sigmoid_out)


def stable_softmax(z: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Numerically stable softmax.
    """
    zmax = np.max(z, axis=axis, keepdims=True)
    expz = np.exp(z - zmax)
    denom = np.sum(expz, axis=axis, keepdims=True)
    return expz / (denom + 1e-12)
