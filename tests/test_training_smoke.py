import numpy as np

from models.fc_numpy import train, predict
from training.metrics import accuracy


def test_training_smoke_runs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2, 200))
    y = (X[0, :] + 0.25 * X[1, :] > 0).astype(int)
    n_classes = 2

    params, _ = train(
        X=X, y=y, n_classes=n_classes,
        hidden_size=8, decay=0.0, learning_rate=0.1,
        epochs=200, activation="relu", seed=0, log_every=0
    )
    pred = predict(X, params, activation="relu")
    acc = accuracy(pred, y)

    assert acc > 0.75
