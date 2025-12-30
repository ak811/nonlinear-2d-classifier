from __future__ import annotations

import numpy as np


def cross_entropy_from_probs(probs: np.ndarray, y_onehot: np.ndarray) -> float:
    """
    probs: (C, n)
    y_onehot: (C, n)
    returns average cross entropy
    """
    n = probs.shape[1]
    eps = 1e-12
    return float(-np.sum(y_onehot * np.log(probs + eps)) / n)


def l2_reg(W1: np.ndarray, W2: np.ndarray, decay: float) -> float:
    return float((decay / 2.0) * (np.sum(W1 * W1) + np.sum(W2 * W2)))
