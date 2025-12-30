from __future__ import annotations

import numpy as np

from models.init import init_params
from utils.numerics import relu, relu_grad, sigmoid, sigmoid_grad, stable_softmax
from utils.onehot import one_hot


def forward(X: np.ndarray, params: dict[str, np.ndarray], activation: str = "relu"):
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    Z2 = W1 @ X + b1  # (K, n)

    if activation.lower() == "relu":
        A2 = relu(Z2)
    elif activation.lower() == "sigmoid":
        A2 = sigmoid(Z2)
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    Z3 = W2 @ A2 + b2  # (C, n)
    A3 = stable_softmax(Z3, axis=0)  # (C, n)

    cache = {"X": X, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3, "activation": activation}
    return A3, cache


def compute_cost(X: np.ndarray, y: np.ndarray, params: dict[str, np.ndarray], decay: float,
                 activation: str = "relu") -> float:
    """
    Average cross entropy + L2 regularization:
      L = (1/n) sum_i CE(y_i, p_i) + (decay/2)(||W1||^2 + ||W2||^2)
    """
    n = X.shape[1]
    W1, W2 = params["W1"], params["W2"]

    probs, _ = forward(X, params, activation=activation)  # (C, n)
    Y = one_hot(y, num_classes=probs.shape[0])  # (C, n)

    eps = 1e-12
    ce = -np.sum(Y * np.log(probs + eps)) / n
    reg = (decay / 2.0) * (np.sum(W1 * W1) + np.sum(W2 * W2))
    return float(ce + reg)


def backward(y: np.ndarray, params: dict[str, np.ndarray], cache: dict, decay: float):
    """
    Reverse-mode gradients for the 2-layer NN.
    Uses averaged loss, so gradients include division by n.
    """
    W1, W2 = params["W1"], params["W2"]
    X = cache["X"]
    Z2, A2, A3 = cache["Z2"], cache["A2"], cache["A3"]
    activation = cache["activation"].lower()

    n = X.shape[1]
    Y = one_hot(y, num_classes=A3.shape[0])  # (C, n)

    # dZ3 = (A3 - Y) / n
    u2 = (A3 - Y) / n  # (C, n)
    dW2 = u2 @ A2.T + decay * W2  # (C, K)
    db2 = np.sum(u2, axis=1, keepdims=True)  # (C, 1)

    # Backprop through activation
    if activation == "relu":
        u1 = (W2.T @ u2) * relu_grad(Z2)  # (K, n)
    elif activation == "sigmoid":
        u1 = (W2.T @ u2) * sigmoid_grad(A2)  # (K, n), using A2 for sigmoid grad
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    dW1 = u1 @ X.T + decay * W1  # (K, D)
    db1 = np.sum(u1, axis=1, keepdims=True)  # (K, 1)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update_params(params: dict[str, np.ndarray], grads: dict[str, np.ndarray],
                  learning_rate: float) -> dict[str, np.ndarray]:
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    return params


def predict(X: np.ndarray, params: dict[str, np.ndarray], activation: str = "relu") -> np.ndarray:
    probs, _ = forward(X, params, activation=activation)
    return np.argmax(probs, axis=0)


def train(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    hidden_size: int,
    decay: float,
    learning_rate: float,
    epochs: int,
    activation: str = "relu",
    seed: int = 1,
    log_every: int = 1000,
):
    D = X.shape[0]
    params = init_params(D, hidden_size, n_classes, seed=seed)

    history = []
    for epoch in range(epochs):
        probs, cache = forward(X, params, activation=activation)
        grads = backward(y, params, cache, decay=decay)
        params = update_params(params, grads, learning_rate=learning_rate)

        if log_every and (epoch % log_every == 0):
            c = compute_cost(X, y, params, decay=decay, activation=activation)
            history.append((epoch, c))
            print(f"Epoch {epoch}: cost {c:.6f}")

    return params, history
