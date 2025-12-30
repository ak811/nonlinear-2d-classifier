from __future__ import annotations

import numpy as np


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    y: (n,) integer labels
    returns: (C, n) one-hot matrix
    """
    n = y.shape[0]
    Y = np.zeros((num_classes, n), dtype=np.float64)
    Y[y.astype(int), np.arange(n)] = 1.0
    return Y
