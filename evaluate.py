"""Evaluate the trained Fashion-MNIST ANN and write reproducible metrics."""

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def project_root() -> Path:
    """Return the repository root before and after this script moves into src/."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent if script_dir.name == "src" else script_dir


def save_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: Path
) -> None:
    """Create and save a labeled confusion-matrix image."""
    matrix = confusion_matrix(y_true, y_pred, labels=range(10))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )
    figure, axis = plt.subplots(figsize=(11, 9))
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title("Fashion-MNIST Test Confusion Matrix")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    root = project_root()
    processed_dir = root / "data" / "processed"
    models_dir = root / "models"

    x_test = np.load(processed_dir / "x_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    model_path = models_dir / "model.h5"
    model = keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    confusion_matrix_path = models_dir / "confusion_matrix.png"
    save_confusion_matrix(y_test, predicted_labels, confusion_matrix_path)

    metrics = {
        "test_loss": round(float(test_loss), 6),
        "test_accuracy": round(float(test_accuracy), 6),
        "test_samples": int(y_test.shape[0]),
    }
    metrics_path = root / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")

    print(f"Evaluated {metrics['test_samples']} test samples")
    print(f"Test loss: {metrics['test_loss']:.6f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.4%}")
    print(f"Saved metrics to {metrics_path.relative_to(root)}")
    print(
        "Saved confusion matrix to "
        f"{confusion_matrix_path.relative_to(root)}"
    )
    print("Fashion-MNIST ANN evaluation completed successfully.")


if __name__ == "__main__":
    main()
