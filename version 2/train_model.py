import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.feature_extraction import FeatureExtractionConfig, batch_extract_features, get_feature_names
from scripts.evaluate_model import evaluate_model


DATA_PATH = BASE_DIR / "model" / "preprocessed_data.pkl"
APPROVED_SAMPLES_PATH = BASE_DIR / "data" / "approved_samples.pkl"
ACTIVE_MODEL_DIR = BASE_DIR / "model" / "active"
VERSION_ROOT = BASE_DIR / "model" / "upgraded" / "versions"
REPORTS_DIR = BASE_DIR / "reports"
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


def load_base_images():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

    X, y = joblib.load(DATA_PATH)
    X = np.asarray(X, dtype=np.float32)
    if X.max() > 1.0:
        X = X / 255.0

    return X.reshape((-1, IMG_SIZE, IMG_SIZE)), encode_labels(y)


def load_approved_samples():
    if not APPROVED_SAMPLES_PATH.exists():
        return np.empty((0, IMG_SIZE, IMG_SIZE), dtype=np.float32), np.empty((0,), dtype=np.int64)

    images, labels = joblib.load(APPROVED_SAMPLES_PATH)
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def load_training_data():
    base_images, base_labels = load_base_images()
    approved_images, approved_labels = load_approved_samples()

    if len(approved_images):
        images = np.concatenate([base_images, approved_images], axis=0)
        labels = np.concatenate([base_labels, approved_labels], axis=0)
    else:
        images = base_images
        labels = base_labels

    return images, labels


def get_next_version(feature_mode):
    feature_dir = VERSION_ROOT / feature_mode
    if not feature_dir.exists():
        return "v2.1.0"

    versions = [path.name for path in feature_dir.iterdir() if path.is_dir() and path.name.startswith("v")]
    if not versions:
        return "v2.1.0"

    latest = sorted(versions, key=lambda item: tuple(int(part) for part in item.lstrip("v").split(".")))[-1]
    major, minor, patch = [int(part) for part in latest.lstrip("v").split(".")]
    return f"v{major}.{minor}.{patch + 1}"


def get_output_paths(feature_mode, version):
    version_dir = VERSION_ROOT / feature_mode / version
    version_dir.mkdir(parents=True, exist_ok=True)
    return {
        "directory": version_dir,
        "model": version_dir / "stress_model.pkl",
        "label_map": version_dir / "label_mapping.json",
        "metadata": version_dir / "metadata.json",
        "metrics": version_dir / "training_metrics.json",
    }


def build_feature_matrix(images, feature_mode, include_handwriting_features=True):
    config = FeatureExtractionConfig(
        mode=feature_mode,
        image_size=IMG_SIZE,
        include_handwriting_features=include_handwriting_features,
    )
    X = batch_extract_features(images, config=config)
    feature_names = get_feature_names(config)
    return X, feature_names


def compute_cv_accuracy(X, y):
    estimator = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_features="sqrt",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="accuracy")
    return float(np.mean(scores)), scores.tolist()


def promote_to_active(paths):
    ACTIVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ["stress_model.pkl", "label_mapping.json", "metadata.json"]:
        shutil.copy2(paths["directory"] / filename, ACTIVE_MODEL_DIR / filename)


def get_current_active_accuracy():
    metadata_path = ACTIVE_MODEL_DIR / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return float(metadata.get("cross_validated_accuracy", 0.0))


def train_model(feature_mode="hog", include_handwriting_features=True, version=None, promote=False):
    images, y = load_training_data()
    X, feature_names = build_feature_matrix(images, feature_mode, include_handwriting_features)
    cv_accuracy, cv_scores = compute_cv_accuracy(X, y)

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
    report_text = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=[LABEL_MAP[str(index)] for index in [0, 1, 2]],
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2]).tolist()

    version = version or get_next_version(feature_mode)
    paths = get_output_paths(feature_mode, version)
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
    paths["label_map"].write_text(json.dumps(LABEL_MAP, indent=2), encoding="utf-8")

    flat_images = images.reshape((len(images), -1))
    evaluation = evaluate_model(flat_images, y, paths["model"], feature_mode, cv_folds=5)
    evaluated_accuracy = float(evaluation["accuracy"])
    current_active_accuracy = get_current_active_accuracy()

    metadata = {
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_extraction_method": feature_mode,
        "include_handwriting_features": include_handwriting_features,
        "training_set_size": int(len(X)),
        "cross_validated_accuracy": float(cv_accuracy),
        "evaluated_accuracy": evaluated_accuracy,
        "semantic_version": version,
        "feature_count": int(X.shape[1]),
        "approved_feedback_samples": int(len(load_approved_samples()[0])),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths["metrics"].write_text(
        json.dumps(
            {
                "version": version,
                "feature_mode": feature_mode,
                "validation_accuracy": float(accuracy),
                "cross_validated_accuracy": float(cv_accuracy),
                "cross_validation_scores": cv_scores,
                "evaluated_accuracy": evaluated_accuracy,
                "confusion_matrix": matrix,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    promoted = False
    if promote:
        if current_active_accuracy is None or evaluated_accuracy >= current_active_accuracy:
            promote_to_active(paths)
            promoted = True

    print(f"Feature mode: {feature_mode}")
    print(f"Model version: {version}")
    print(f"Training samples: {len(X)}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Validation accuracy: {accuracy:.2%}")
    print(f"Cross-validated accuracy: {cv_accuracy:.2%}")
    print(f"Evaluated accuracy: {evaluated_accuracy:.2%}")
    if current_active_accuracy is not None:
        print(f"Current active accuracy: {current_active_accuracy:.2%}")
        print(f"Before/After difference: {evaluated_accuracy - current_active_accuracy:+.2%}")
    print(report_text)
    print("Confusion matrix:")
    print(np.asarray(matrix))
    print(f"Promoted to active: {promoted}")

    return {
        "version": version,
        "feature_mode": feature_mode,
        "validation_accuracy": float(accuracy),
        "cross_validated_accuracy": float(cv_accuracy),
        "evaluated_accuracy": evaluated_accuracy,
        "current_active_accuracy": current_active_accuracy,
        "promoted": promoted,
        "model_path": str(paths["model"]),
        "metadata_path": str(paths["metadata"]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Version 2 model from approved samples only.")
    parser.add_argument("--feature-mode", choices=["raw_pixels", "hog"], default="hog")
    parser.add_argument("--version", default=None)
    parser.add_argument("--promote-active", action="store_true")
    parser.add_argument("--without-handwriting-features", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        feature_mode=args.feature_mode,
        version=args.version,
        promote=args.promote_active,
        include_handwriting_features=not args.without_handwriting_features,
    )
