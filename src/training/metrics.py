from __future__ import annotations

import numpy as np


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred == y))
