import numpy as np

from evaluation.gradient_check import gradient_check
from models.init import init_params


def test_gradient_check_sigmoid_passes():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 50))
    y = rng.integers(0, 2, size=50)

    params = init_params(D=6, K=4, C=2, seed=1)
    res = gradient_check(X=X, y=y, params=params, decay=1e-3, epsilon=1e-4, tol=1e-6)

    assert res["diff"] < 1e-6
