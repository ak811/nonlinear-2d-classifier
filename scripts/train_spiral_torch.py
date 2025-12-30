import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config.defaults import TORCH_SPIRAL
from evaluation.experiments import run_spiral_torch


def main():
    r = run_spiral_torch(
        hidden_size=TORCH_SPIRAL["hidden_size"],
        decay=TORCH_SPIRAL["decay"],
        learning_rate=TORCH_SPIRAL["learning_rate"],
        epochs=TORCH_SPIRAL["epochs"],
        batch_size=TORCH_SPIRAL["batch_size"],
    )
    print(f"Train accuracy: {r.train_acc * 100:.2f}%.")


if __name__ == "__main__":
    main()
