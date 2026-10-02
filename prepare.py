"""Download Fashion-MNIST and save its original arrays for the pipeline."""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from tensorflow import keras


def project_root() -> Path:
    """Return the repository root before and after this script moves into src/."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent if script_dir.name == "src" else script_dir


def main() -> None:
    raw_dir = project_root() / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    arrays = {
        "x_train.npy": x_train,
        "y_train.npy": y_train,
        "x_test.npy": x_test,
        "y_test.npy": y_test,
    }

    for filename, array in arrays.items():
        output_path = raw_dir / filename
        np.save(output_path, array)
        print(
            f"Saved {output_path.relative_to(project_root())} "
            f"shape={array.shape} dtype={array.dtype}"
        )

    print("Fashion-MNIST preparation completed successfully.")


if __name__ == "__main__":
    main()
