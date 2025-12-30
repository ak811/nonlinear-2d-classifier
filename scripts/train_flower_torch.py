import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config.defaults import TORCH_FLOWER
from evaluation.experiments import run_flower_torch


def main():
    r = run_flower_torch(
        hidden_size=TORCH_FLOWER["hidden_size"],
        decay=TORCH_FLOWER["decay"],
        learning_rate=TORCH_FLOWER["learning_rate"],
        epochs=TORCH_FLOWER["epochs"],
        batch_size=TORCH_FLOWER["batch_size"],
    )
    print(f"Train accuracy: {r.train_acc * 100:.2f}%.")


if __name__ == "__main__":
    main()
