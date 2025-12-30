# ml-flower-spiral-nn

A small, self-contained project that trains a 2-layer fully connected neural network on classic synthetic 2D datasets (**flower** and **spiral**) using:

- **NumPy** (manual forward/backprop + gradient descent)
- **PyTorch** (equivalent architecture + SGD)

It also includes **gradient checking** (with sigmoid activation) to verify the NumPy backprop implementation.

## What this repo does

- Generates synthetic datasets (no external downloads)
- Trains a 2-layer MLP (Linear → ReLU → Linear)
- Plots decision boundaries and saves them to `outputs/figures/`
- Runs gradient checking to compare analytic vs numerical gradients
- Includes basic `pytest` tests

## Quickstart

### 1) Install

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2) Run training

NumPy (Flower):
```bash
python scripts/train_flower_numpy.py
```

NumPy (Spiral):
```bash
python scripts/train_spiral_numpy.py
```

PyTorch (Flower):
```bash
python scripts/train_flower_torch.py
```

PyTorch (Spiral):
```bash
python scripts/train_spiral_torch.py
```

Plots are saved to:

- `outputs/figures/flower-boundary.jpg`
- `outputs/figures/spiral-boundary.jpg`

## Results (saved plots)

### Flower decision boundary

![Flower decision boundary](outputs/figures/flower-boundary.jpg)

### Spiral decision boundary

![Spiral decision boundary](outputs/figures/spiral-boundary.jpg)

## Gradient checking (NumPy)

Gradient checking uses a **sigmoid** hidden activation so the network is differentiable everywhere (ReLU is not differentiable at 0, which makes numerical gradients disagree near 0).

Run:

```bash
python scripts/gradient_check.py
```

The printed output includes the relative difference:

\[
\frac{\|g_{num}-g_{ana}\|}{\|g_{num}+g_{ana}\|}
\]

A typical pass condition is `diff < 1e-6`.

## Tests

```bash
pytest
```

## Project structure

```text
ml-flower-spiral-nn/
├─ src/                    # library code (NumPy + Torch + data + viz)
├─ scripts/                # runnable entrypoints
├─ outputs/
│  ├─ figures/             # saved plots
│  └─ logs/                # optional logs
└─ tests/                  # pytest tests
```

## Notes

- The code assumes **2D inputs** for decision boundary plotting.
- Randomness is controlled via seeds in `src/config/defaults.py` for reproducibility.
