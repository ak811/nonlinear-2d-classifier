from __future__ import annotations

import copy
import numpy as np
from numpy.linalg import norm

from models.fc_numpy import forward, backward, compute_cost
from utils.numerics import sigmoid


def ravel_grads(grads: dict[str, np.ndarray]) -> np.ndarray:
    return np.hstack((grads["dW1"].ravel(), grads["db1"].ravel(), grads["dW2"].ravel(), grads["db2"].ravel()))


def compute_numerical_gradient(J, params: dict[str, np.ndarray], epsilon: float = 1e-4):
    """
    Numerically compute gradient of scalar function J(params).
    params dict contains W1,b1,W2,b2.
    Returns grads dict dW1,db1,dW2,db2 matching shapes.
    """
    num_grads = {}
    for name, value in params.items():
        g = np.zeros_like(value)
        it = np.nditer(value, flags=["multi_index"], op_flags=["readwrite"])

        while not it.finished:
            idx = it.multi_index

            params_plus = copy.deepcopy(params)
            params_minus = copy.deepcopy(params)

            params_plus[name][idx] = params_plus[name][idx] + epsilon
            params_minus[name][idx] = params_minus[name][idx] - epsilon

            J_plus = J(params_plus)
            J_minus = J(params_minus)

            g[idx] = (J_plus - J_minus) / (2.0 * epsilon)

            it.iternext()

        num_grads[name] = g

    return {"dW1": num_grads["W1"], "db1": num_grads["b1"], "dW2": num_grads["W2"], "db2": num_grads["b2"]}


def gradient_check(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, np.ndarray],
    decay: float = 1e-3,
    epsilon: float = 1e-4,
    tol: float = 1e-6,
) -> dict[str, float]:
    """
    Runs gradient check using sigmoid activation (differentiable everywhere).
    Returns a dict with 'diff' and norms.
    """
    # Analytic grads
    _, cache = forward(X, params, activation="sigmoid")
    analytic = backward(y, params, cache, decay=decay)

    # Numerical grads
    J = lambda p: compute_cost(X, y, p, decay=decay, activation="sigmoid")
    numeric = compute_numerical_gradient(J, params, epsilon=epsilon)

    ra = ravel_grads(analytic)
    rn = ravel_grads(numeric)

    diff = norm(rn - ra) / max(norm(rn + ra), 1e-12)

    return {
        "diff": float(diff),
        "norm_numeric": float(norm(rn)),
        "norm_analytic": float(norm(ra)),
        "passed": float(diff < tol),
    }
