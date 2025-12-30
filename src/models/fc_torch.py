from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class FCNetTorch(nn.Module):
    """
    2-layer fully-connected network:
      Linear(D->K) + ReLU + Linear(K->C)
    """
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def init_like_numpy(model: FCNetTorch, seed: int = 1) -> None:
    """
    Initialize weights ~ N(0,1), biases=0 to mirror the numpy setup.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.fc1.weight.copy_(torch.normal(0.0, 1.0, size=model.fc1.weight.shape, generator=g))
        model.fc1.bias.zero_()
        model.fc2.weight.copy_(torch.normal(0.0, 1.0, size=model.fc2.weight.shape, generator=g))
        model.fc2.bias.zero_()


def train_torch(
    X: np.ndarray,  # (D, n)
    y: np.ndarray,  # (n,)
    hidden_size: int,
    decay: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int = 1,
    log_every: int = 1000,
):
    torch.manual_seed(seed)
    device = torch.device("cpu")

    D, n = X.shape
    num_classes = int(np.max(y)) + 1

    model = FCNetTorch(D, hidden_size, num_classes).to(device)
    init_like_numpy(model, seed=seed)

    X_t = torch.tensor(X.T, dtype=torch.float32, device=device)  # (n, D)
    y_t = torch.tensor(y, dtype=torch.long, device=device)       # (n,)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=decay)

    history = []
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        if log_every and ((epoch + 1) % log_every == 0):
            history.append((epoch + 1, float(loss.item())))
            print(f"Epoch [{epoch + 1}/{epochs}], loss {loss.item():.6f}")

    return model, history


@torch.no_grad()
def predict_torch(X: np.ndarray, model: FCNetTorch) -> np.ndarray:
    device = next(model.parameters()).device
    X_t = torch.tensor(X.T, dtype=torch.float32, device=device)  # (n, D)
    logits = model(X_t)
    pred = torch.argmax(logits, dim=1).cpu().numpy()
    return pred
