import numpy as np

from models.fc_numpy import forward, backward
from models.init import init_params


def test_shapes_relu():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 10))
    y = rng.integers(0, 3, size=10)

    params = init_params(D=4, K=6, C=3, seed=0)
    A3, cache = forward(X, params, activation="relu")
    grads = backward(y, params, cache, decay=1e-3)

    assert A3.shape == (3, 10)
    assert grads["dW1"].shape == (6, 4)
    assert grads["db1"].shape == (6, 1)
    assert grads["dW2"].shape == (3, 6)
    assert grads["db2"].shape == (3, 1)
