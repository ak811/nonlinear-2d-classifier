from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.defaults import DEFAULT_SEED, FLOWER_N, SPIRAL_N
from data.datasets import make_flower, make_spiral


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray  # (D, n)
    y: np.ndarray  # (n,)
    n_classes: int


def load_flower_dataset(n: int = FLOWER_N, seed: int = DEFAULT_SEED) -> Dataset:
    X, y, c = make_flower(n=n, seed=seed)
    return Dataset(X=X, y=y, n_classes=c)


def load_spiral_dataset(n: int = SPIRAL_N, seed: int = DEFAULT_SEED) -> Dataset:
    X, y, c = make_spiral(n=n, seed=seed)
    return Dataset(X=X, y=y, n_classes=c)
