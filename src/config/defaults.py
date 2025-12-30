from __future__ import annotations

# Reproducibility
DEFAULT_SEED = 1

# Dataset sizes (feel free to tune)
FLOWER_N = 400
SPIRAL_N = 600

# NumPy model hyperparams
NUMPY_FLOWER = {
    "hidden_size": 20,
    "decay": 0.0,
    "learning_rate": 0.05,
    "epochs": 20000,
    "activation": "relu",
}

NUMPY_SPIRAL = {
    "hidden_size": 100,
    "decay": 1e-3,
    "learning_rate": 1.0,
    "epochs": 10000,
    "activation": "relu",
}

# Torch model hyperparams (mirrors the notebook style)
TORCH_FLOWER = {
    "hidden_size": 20,
    "decay": 0.0,
    "learning_rate": 0.05,
    "epochs": 20000,
    "batch_size": 200,  # "full-ish" batch
}

TORCH_SPIRAL = {
    "hidden_size": 100,
    "decay": 1e-3,
    "learning_rate": 1.0,
    "epochs": 10000,
    "batch_size": 300,
}
