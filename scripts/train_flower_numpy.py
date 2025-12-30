import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config.defaults import NUMPY_FLOWER
from evaluation.experiments import run_flower_numpy


def main():
    r = run_flower_numpy(
        hidden_size=NUMPY_FLOWER["hidden_size"],
        decay=NUMPY_FLOWER["decay"],
        learning_rate=NUMPY_FLOWER["learning_rate"],
        epochs=NUMPY_FLOWER["epochs"],
        activation=NUMPY_FLOWER["activation"],
    )
    print(f"Train accuracy: {r.train_acc * 100:.2f}%.")


if __name__ == "__main__":
    main()
