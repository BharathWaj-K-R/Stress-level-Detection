import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split

from preprocessing.feature_extraction import (
    FeatureExtractionConfig,
    batch_extract_features,
    get_feature_names,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "model" / "preprocessed_data.pkl"
UPGRADED_MODEL_DIR = BASE_DIR / "model" / "upgraded"
VERSIONED_MODEL_DIR = UPGRADED_MODEL_DIR / "versions"
ACTIVE_MODEL_DIR = UPGRADED_MODEL_DIR / "active"
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

    images = X.reshape((-1, IMG_SIZE, IMG_SIZE))
    encoded_y = encode_labels(y)
    return images, encoded_y


def build_feature_matrix(images, feature_mode, include_handwriting_features=True):
    config = FeatureExtractionConfig(
        mode=feature_mode,
        image_size=IMG_SIZE,
        include_handwriting_features=include_handwriting_features,
    )
    features = batch_extract_features(images, config=config)
    feature_names = get_feature_names(config)
    return features, feature_names, config


def get_next_version(feature_mode, root_dir=VERSIONED_MODEL_DIR):
    feature_dir = Path(root_dir) / feature_mode
    if not feature_dir.exists():
        return "v1.1.0"

    versions = []
    for child in feature_dir.iterdir():
        if child.is_dir() and child.name.startswith("v"):
            versions.append(child.name)

    if not versions:
        return "v1.1.0"

    latest = sorted(versions, key=lambda item: tuple(int(part) for part in item.lstrip("v").split(".")))[-1]
    major, minor, patch = [int(part) for part in latest.lstrip("v").split(".")]
    return f"v{major}.{minor}.{patch + 1}"


def get_output_paths(feature_mode, version, output_dir=VERSIONED_MODEL_DIR):
    output_dir = Path(output_dir) / feature_mode / version
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "directory": output_dir,
        "model": output_dir / "stress_model.pkl",
        "label_map": output_dir / "label_mapping.json",
        "metrics": output_dir / "training_metrics.json",
        "metadata": output_dir / "metadata.json",
    }


def compute_cross_validated_accuracy(X, y, random_state=RANDOM_STATE):
    estimator = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="accuracy")
    return float(np.mean(scores)), scores.tolist()


def promote_to_active(paths):
    ACTIVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ["stress_model.pkl", "label_mapping.json", "metadata.json"]:
        shutil.copy2(paths["directory"] / filename, ACTIVE_MODEL_DIR / filename)


def train_model(
    feature_mode="raw_pixels",
    output_dir=VERSIONED_MODEL_DIR,
    include_handwriting_features=True,
    version=None,
    promote=False,
):
    images, y = load_training_data()
    X, feature_names, config = build_feature_matrix(
        images,
        feature_mode=feature_mode,
        include_handwriting_features=include_handwriting_features,
    )
    cv_accuracy, cv_scores = compute_cross_validated_accuracy(X, y)

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
    report_text = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=[LABEL_MAP[str(index)] for index in [0, 1, 2]],
        zero_division=0,
    )

    version = version or get_next_version(feature_mode, root_dir=Path(output_dir))
    paths = get_output_paths(feature_mode, version, output_dir=Path(output_dir))
    joblib.dump(
        {
            "model": model,
            "label_map": LABEL_MAP,
            "image_size": IMG_SIZE,
            "feature_mode": feature_mode,
            "include_handwriting_features": include_handwriting_features,
            "feature_names": feature_names,
            "version": version,
        },
        paths["model"],
    )

    with paths["label_map"].open("w", encoding="utf-8") as file:
        json.dump(LABEL_MAP, file, indent=2)

    paths["metrics"].write_text(
        json.dumps(
            {
                "version": version,
                "feature_mode": feature_mode,
                "include_handwriting_features": include_handwriting_features,
                "training_samples": int(len(X)),
                "feature_count": int(X.shape[1]),
                "validation_accuracy": float(accuracy),
                "cross_validated_accuracy": float(cv_accuracy),
                "cross_validation_scores": cv_scores,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    paths["metadata"].write_text(
        json.dumps(
            {
                "version": version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "feature_extraction_method": feature_mode,
                "include_handwriting_features": include_handwriting_features,
                "training_set_size": int(len(X)),
                "cross_validated_accuracy": float(cv_accuracy),
                "semantic_version": version,
                "feature_count": int(X.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if promote:
        promote_to_active(paths)

    print(f"Feature mode: {feature_mode}")
    print(f"Model version: {version}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Model saved to: {paths['model']}")
    print(f"Label map saved to: {paths['label_map']}")
    print(f"Training metrics saved to: {paths['metrics']}")
    print(f"Metadata saved to: {paths['metadata']}")
    print(f"Cross-validated accuracy (5-fold): {cv_accuracy:.2%}")
    print(f"Validation accuracy: {accuracy:.2%}")
    print(report_text)

    return {
        "version": version,
        "feature_mode": feature_mode,
        "feature_count": int(X.shape[1]),
        "accuracy": float(accuracy),
        "cross_validated_accuracy": float(cv_accuracy),
        "model_path": str(paths["model"]),
        "metadata_path": str(paths["metadata"]),
        "metrics_path": str(paths["metrics"]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train handwriting stress models with selectable feature modes.")
    parser.add_argument(
        "--feature-mode",
        choices=["raw_pixels", "hog"],
        default="raw_pixels",
        help="Feature extraction mode to use for training.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(VERSIONED_MODEL_DIR),
        help="Directory where the candidate model artifacts should be written.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Optional semantic version to use instead of auto-incrementing.",
    )
    parser.add_argument(
        "--without-handwriting-features",
        action="store_true",
        help="Disable the three derived handwriting-specific features.",
    )
    parser.add_argument(
        "--promote-active",
        action="store_true",
        help="Copy the saved model, label map, and metadata into the upgraded active model directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        feature_mode=args.feature_mode,
        output_dir=Path(args.output_dir),
        include_handwriting_features=not args.without_handwriting_features,
        version=args.version,
        promote=args.promote_active,
    )
