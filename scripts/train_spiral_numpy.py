import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config.defaults import NUMPY_SPIRAL
from evaluation.experiments import run_spiral_numpy


def main():
    r = run_spiral_numpy(
        hidden_size=NUMPY_SPIRAL["hidden_size"],
        decay=NUMPY_SPIRAL["decay"],
        learning_rate=NUMPY_SPIRAL["learning_rate"],
        epochs=NUMPY_SPIRAL["epochs"],
        activation=NUMPY_SPIRAL["activation"],
    )
    print(f"Train accuracy: {r.train_acc * 100:.2f}%.")


if __name__ == "__main__":
    main()
