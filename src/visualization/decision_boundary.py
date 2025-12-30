from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(pred_func, X: np.ndarray, y: np.ndarray, title: str = ""):
    """
    X: (2, n) expected for visualization
    y: (n,)
    pred_func: function that accepts Xgrid in shape (2, m) and returns labels (m,)
    """
    if X.shape[0] != 2:
        raise ValueError("Decision boundary plotting expects 2D inputs: X shape should be (2, n).")

    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()].T  # (2, M)

    Z = pred_func(grid)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.6)

    ax.scatter(X[0, :], X[1, :], c=y, s=20, edgecolors="k")
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    return fig, ax
