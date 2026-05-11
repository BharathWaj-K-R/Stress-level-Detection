from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "model" / "preprocessed_data.pkl"
USER_SAMPLES_PATH = BASE_DIR / "data" / "user_training_samples.pkl"
MODEL_PATH = BASE_DIR / "model" / "stress_model.pkl"
LABEL_MAP_PATH = BASE_DIR / "model" / "label_mapping.json"
METRICS_PATH = BASE_DIR / "model" / "training_metrics.json"
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

    if USER_SAMPLES_PATH.exists():
        user_X, user_y = joblib.load(USER_SAMPLES_PATH)
        user_X = np.asarray(user_X, dtype=np.float32)
        user_y = np.asarray(user_y, dtype=np.int64)

        if user_X.ndim != 2 or user_X.shape[1] != expected_features:
            raise ValueError(
                f"Expected user samples to have shape (samples, {expected_features}), got {user_X.shape}"
            )

        if user_X.max() > 1.0:
            user_X = user_X / 255.0

        X = np.vstack([X, user_X])
        encoded_y = np.concatenate([encoded_y, user_y])

    return X, encoded_y


def train_model():
    X, y = load_training_data()

    stratify = y if min(np.bincount(y, minlength=3)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_features="sqrt",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=[LABEL_MAP[str(index)] for index in [0, 1, 2]],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2]).tolist()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "label_map": LABEL_MAP,
            "image_size": IMG_SIZE,
            "version": "2.0",
            "model_name": "RandomForestClassifier",
        },
        MODEL_PATH,
    )

    LABEL_MAP_PATH.write_text(json.dumps(LABEL_MAP, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "classification_report": report,
                "confusion_matrix": matrix,
                "labels": LABEL_MAP,
                "total_samples": int(len(X)),
                "learned_samples": int(len(joblib.load(USER_SAMPLES_PATH)[0])) if USER_SAMPLES_PATH.exists() else 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Label map saved to: {LABEL_MAP_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
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
    print("Confusion matrix:")
    print(np.asarray(matrix))


if __name__ == "__main__":
    train_model()
