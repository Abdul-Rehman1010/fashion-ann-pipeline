"""Preprocess the raw Fashion-MNIST arrays for model training."""

from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def project_root() -> Path:
    """Return the repository root before and after this script moves into src/."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent if script_dir.name == "src" else script_dir


def load_raw_arrays(raw_dir: Path) -> tuple[np.ndarray, ...]:
    """Load the four arrays created by the preparation stage."""
    return (
        np.load(raw_dir / "x_train.npy"),
        np.load(raw_dir / "y_train.npy"),
        np.load(raw_dir / "x_test.npy"),
        np.load(raw_dir / "y_test.npy"),
    )


def load_params(params_path: Path) -> tuple[float, int]:
    """Load and validate preprocessing settings from params.yaml."""
    with params_path.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    preprocess_params = params["preprocess"]
    validation_size = float(preprocess_params["validation_size"])
    seed = int(preprocess_params["seed"])

    if not 0.0 < validation_size < 1.0:
        raise ValueError("preprocess.validation_size must be between 0 and 1.")

    return validation_size, seed


def normalize(images: np.ndarray) -> np.ndarray:
    """Scale pixels to [0, 1] with clipped five-decimal precision."""
    scaled = np.divide(images, np.float32(255.0), dtype=np.float32)
    rounded = np.round(scaled, decimals=5)
    return np.clip(rounded, 0.0, 1.0)


def save_array(output_path: Path, array: np.ndarray, root: Path) -> None:
    """Save one processed array and print a concise verification line."""
    np.save(output_path, array)
    range_text = ""
    if np.issubdtype(array.dtype, np.floating):
        range_text = f" range=[{array.min():.1f}, {array.max():.1f}]"
    print(
        f"Saved {output_path.relative_to(root)} "
        f"shape={array.shape} dtype={array.dtype}{range_text}"
    )


def main() -> None:
    root = project_root()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_raw_arrays(raw_dir)
    validation_size, seed = load_params(root / "params.yaml")

    x_train = normalize(x_train)
    x_test = normalize(x_test)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=validation_size,
        random_state=seed,
        stratify=y_train,
    )

    arrays = {
        "x_train.npy": x_train,
        "y_train.npy": y_train,
        "x_val.npy": x_val,
        "y_val.npy": y_val,
        "x_test.npy": x_test,
        "y_test.npy": y_test,
    }

    for filename, array in arrays.items():
        save_array(processed_dir / filename, array, root)

    print(
        "Fashion-MNIST preprocessing completed successfully "
        f"with validation_size={validation_size} and seed={seed}."
    )


if __name__ == "__main__":
    main()
