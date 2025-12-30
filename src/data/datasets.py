from __future__ import annotations

import numpy as np


def make_flower(n: int = 400, noise: float = 0.2, seed: int | None = 1):
    """
    Two-class "flower" dataset (2D). Returns X (2 x n), y (n,), n_classes=2.

    This is a common synthetic benchmark: polar petals + noise.
    """
    rng = np.random.default_rng(seed)

    n_per_class = n // 2
    X = np.zeros((2, n), dtype=np.float64)
    y = np.zeros(n, dtype=np.uint8)

    for j in range(2):
        ix = slice(j * n_per_class, (j + 1) * n_per_class)

        r = rng.uniform(0.0, 1.0, size=n_per_class)
        t = (
            j * np.pi
            + 4.0 * np.pi * r
            + rng.normal(0.0, noise, size=n_per_class)
        )

        X[0, ix] = r * np.sin(t)
        X[1, ix] = r * np.cos(t)
        y[ix] = j

    return X, y, 2


def make_spiral(n: int = 600, noise: float = 0.2, seed: int | None = 1):
    """
    Two-class "spiral" dataset (2D). Returns X (2 x n), y (n,), n_classes=2.
    """
    rng = np.random.default_rng(seed)

    n_per_class = n // 2
    X = np.zeros((2, n), dtype=np.float64)
    y = np.zeros(n, dtype=np.uint8)

    for j in range(2):
        ix = slice(j * n_per_class, (j + 1) * n_per_class)

        r = np.linspace(0.0, 1.0, n_per_class)
        t = (
            j * np.pi
            + 4.0 * np.pi * r
            + rng.normal(0.0, noise, size=n_per_class)
        )

        X[0, ix] = r * np.sin(t)
        X[1, ix] = r * np.cos(t)
        y[ix] = j

    return X, y, 2
