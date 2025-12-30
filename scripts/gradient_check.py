import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from evaluation.gradient_check import gradient_check
from models.init import init_params


def main():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 100))
    y = rng.integers(0, 2, size=100, dtype=np.int64)

    params = init_params(D=8, K=5, C=2, seed=1)
    res = gradient_check(X=X, y=y, params=params, decay=1e-3, epsilon=1e-4, tol=1e-6)
    print(res)


if __name__ == "__main__":
    main()
