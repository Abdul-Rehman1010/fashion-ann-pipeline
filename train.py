"""Train a parameterized fully connected ANN on processed Fashion-MNIST."""

import csv
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import yaml
from tensorflow import keras


def project_root() -> Path:
    """Return the repository root before and after this script moves into src/."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent if script_dir.name == "src" else script_dir


def load_train_params(params_path: Path) -> dict[str, float | int]:
    """Load the ANN hyperparameters from params.yaml."""
    with params_path.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)["train"]

    return {
        "dense_units": int(params["dense_units"]),
        "dropout_rate": float(params["dropout_rate"]),
        "learning_rate": float(params["learning_rate"]),
        "epochs": int(params["epochs"]),
        "batch_size": int(params["batch_size"]),
    }


def load_processed_data(processed_dir: Path) -> tuple[np.ndarray, ...]:
    """Load the processed training and validation arrays."""
    return (
        np.load(processed_dir / "x_train.npy"),
        np.load(processed_dir / "y_train.npy"),
        np.load(processed_dir / "x_val.npy"),
        np.load(processed_dir / "y_val.npy"),
    )


def build_model(params: dict[str, float | int]) -> keras.Sequential:
    """Build the required Flatten-Dense-Dropout-Softmax ANN."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28)),
            keras.layers.Flatten(),
            keras.layers.Dense(int(params["dense_units"]), activation="relu"),
            keras.layers.Dropout(float(params["dropout_rate"])),
            keras.layers.Dense(10, activation="softmax"),
        ],
        name="fashion_mnist_ann",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=float(params["learning_rate"])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history(history: keras.callbacks.History, output_path: Path) -> None:
    """Write one CSV row per epoch from the Keras training history."""
    metric_names = list(history.history)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", *metric_names])
        writer.writeheader()
        for epoch_index in range(len(history.epoch)):
            row = {"epoch": epoch_index + 1}
            row.update(
                {
                    name: history.history[name][epoch_index]
                    for name in metric_names
                }
            )
            writer.writerow(row)


def main() -> None:
    root = project_root()
    processed_dir = root / "data" / "processed"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    params = load_train_params(root / "params.yaml")
    x_train, y_train, x_val, y_val = load_processed_data(processed_dir)

    print(f"Training parameters: {params}")
    print(
        f"Training samples: {x_train.shape[0]}, "
        f"validation samples: {x_val.shape[0]}"
    )

    keras.utils.set_random_seed(42)
    model = build_model(params)
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        verbose=2,
    )

    model_path = models_dir / "model.h5"
    history_path = models_dir / "history.csv"
    model.save(model_path)
    save_history(history, history_path)

    print(f"Saved trained model to {model_path.relative_to(root)}")
    print(f"Saved training history to {history_path.relative_to(root)}")
    print(
        "Final training accuracy: "
        f"{history.history['accuracy'][-1]:.4f}; "
        "final validation accuracy: "
        f"{history.history['val_accuracy'][-1]:.4f}"
    )
    print("Fashion-MNIST ANN training completed successfully.")


if __name__ == "__main__":
    main()
