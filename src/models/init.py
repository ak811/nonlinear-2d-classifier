from __future__ import annotations

import numpy as np


def init_params(D: int, K: int, C: int, seed: int = 1) -> dict[str, np.ndarray]:
    """
    Initialize weights with i.i.d. N(0,1) and biases with zeros, matching the notebook intent.
    Shapes:
      W1: (K, D), b1: (K, 1)
      W2: (C, K), b2: (C, 1)
    """
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, 1.0, size=(K, D))
    b1 = np.zeros((K, 1), dtype=np.float64)
    W2 = rng.normal(0.0, 1.0, size=(C, K))
    b2 = np.zeros((C, 1), dtype=np.float64)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
