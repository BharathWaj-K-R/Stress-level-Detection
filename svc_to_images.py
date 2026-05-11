from pathlib import Path

import joblib
import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "model" / "preprocessed_data.pkl"
OUTPUT_DIR = BASE_DIR / "images_from_svc"
IMG_SIZE = 128


def normalize_label(label):
    return str(label).strip().lower().replace(" stress", "")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Preprocessed data not found at {DATA_PATH}")

    X, y = joblib.load(DATA_PATH)
    X = np.asarray(X, dtype=np.float32)

    expected_features = IMG_SIZE * IMG_SIZE
    if X.ndim != 2 or X.shape[1] != expected_features:
        raise ValueError(
            f"Expected X to have shape (samples, {expected_features}), got {X.shape}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for index, (features, label) in enumerate(zip(X, y)):
        label_name = normalize_label(label)
        label_dir = OUTPUT_DIR / label_name
        label_dir.mkdir(parents=True, exist_ok=True)

        pixels = features.reshape((IMG_SIZE, IMG_SIZE))
        if pixels.max() <= 1.0:
            pixels = pixels * 255.0

        image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="L")
        image.save(label_dir / f"{index:04d}.png")

    print(f"Converted {len(X)} samples into image folders at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
