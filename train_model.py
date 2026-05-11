from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "model" / "preprocessed_data.pkl"
MODEL_PATH = BASE_DIR / "model" / "stress_model.pkl"
LABEL_MAP_PATH = BASE_DIR / "model" / "label_mapping.json"
IMG_SIZE = 128
RANDOM_STATE = 42

STRESS_ORDER = ["low", "medium", "high"]
LABEL_MAP = {
    "0": "Low Stress",
    "1": "Medium Stress",
    "2": "High Stress",
}


def normalize_label(label):
    return str(label).strip().lower().replace(" stress", "")


def encode_labels(labels):
    normalized = [normalize_label(label) for label in labels]
    unknown = sorted(set(normalized) - set(STRESS_ORDER))
    if unknown:
        raise ValueError(f"Unknown labels found in training data: {unknown}")
    return np.asarray([STRESS_ORDER.index(label) for label in normalized], dtype=np.int64)


def load_training_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

    X, y = joblib.load(DATA_PATH)
    X = np.asarray(X, dtype=np.float32)

    expected_features = IMG_SIZE * IMG_SIZE
    if X.ndim != 2 or X.shape[1] != expected_features:
        raise ValueError(
            f"Expected X to have shape (samples, {expected_features}), got {X.shape}"
        )

    if X.max() > 1.0:
        X = X / 255.0

    encoded_y = encode_labels(y)
    return X, encoded_y


def train_model():
    X, y = load_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "label_map": LABEL_MAP, "image_size": IMG_SIZE}, MODEL_PATH)

    with LABEL_MAP_PATH.open("w", encoding="utf-8") as file:
        json.dump(LABEL_MAP, file, indent=2)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Label map saved to: {LABEL_MAP_PATH}")
    print(f"Validation accuracy: {accuracy:.2%}")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=[LABEL_MAP[str(index)] for index in [0, 1, 2]],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    train_model()
