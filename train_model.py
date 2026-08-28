from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from preprocessing.feature_extraction import (
    FeatureExtractionConfig,
    batch_extract_features,
    get_feature_names,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_PATH = MODEL_DIR / "preprocessed_data.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

IMG_SIZE = 128
RANDOM_STATE = 42
STRESS_ORDER = ["low", "medium", "high"]
LABEL_MAP = {"0": "Low Stress", "1": "Medium Stress", "2": "High Stress"}


def _normalize_label(label: str) -> str:
    return str(label).strip().lower().replace(" stress", "")


def _encode_labels(labels) -> np.ndarray:
    normalized = [_normalize_label(label) for label in labels]
    unknown = sorted(set(normalized) - set(STRESS_ORDER))
    if unknown:
        raise ValueError(f"Unknown labels in training data: {unknown}")
    return np.asarray([STRESS_ORDER.index(label) for label in normalized], dtype=np.int64)


def _load_training_data() -> tuple[np.ndarray, np.ndarray]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

    images, labels = joblib.load(DATA_PATH)
    images = np.asarray(images, dtype=np.float32)
    if images.size == 0:
        raise ValueError("Training data is empty.")
    if images.max() > 1.0:
        images = images / 255.0

    images = images.reshape((-1, IMG_SIZE, IMG_SIZE))
    return images, _encode_labels(labels)


def train_model(
    feature_mode: str = "hog",
    include_handwriting_features: bool = True,
    version: str | None = None,
    promote: bool = False,
) -> dict:
    images, labels = _load_training_data()

    config = FeatureExtractionConfig(
        mode=feature_mode,
        image_size=IMG_SIZE,
        include_handwriting_features=include_handwriting_features,
    )
    features = batch_extract_features(images, config=config)
    feature_names = get_feature_names(config)

    classifier = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_features="sqrt",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(classifier, features, labels, cv=cv, scoring="accuracy")
    cv_accuracy = float(np.mean(cv_scores))

    stratify = labels if min(np.bincount(labels, minlength=3)) >= 2 else None
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    classifier.fit(train_x, train_y)
    predictions = classifier.predict(test_x)
    validation_accuracy = float(accuracy_score(test_y, predictions))

    if version is None:
        if METADATA_PATH.exists():
            try:
                current = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
                current_version = current.get("semantic_version", "v1.0.0")
                major, minor, patch = [int(part) for part in current_version.lstrip("v").split(".")]
                version = f"v{major}.{minor}.{patch + 1}"
            except (ValueError, TypeError, json.JSONDecodeError):
                version = "v2.0.0"
        else:
            version = "v2.0.0"

    current_accuracy = None
    if METADATA_PATH.exists():
        try:
            current_accuracy = float(
                json.loads(METADATA_PATH.read_text(encoding="utf-8")).get(
                    "cross_validated_accuracy", 0.0
                )
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            current_accuracy = None

    model_bundle = {
        "model": classifier,
        "label_map": LABEL_MAP,
        "image_size": IMG_SIZE,
        "feature_mode": feature_mode,
        "include_handwriting_features": include_handwriting_features,
        "feature_names": feature_names,
        "version": version,
    }

    promoted = False
    if promote and (current_accuracy is None or cv_accuracy >= current_accuracy):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_bundle, MODEL_DIR / "stress_model.pkl")
        (MODEL_DIR / "label_mapping.json").write_text(
            json.dumps(LABEL_MAP, indent=2),
            encoding="utf-8",
        )
        (MODEL_DIR / "metadata.json").write_text(
            json.dumps(
                {
                    "version": version,
                    "semantic_version": version,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "feature_extraction_method": feature_mode,
                    "include_handwriting_features": include_handwriting_features,
                    "training_set_size": int(len(features)),
                    "cross_validated_accuracy": cv_accuracy,
                    "evaluated_accuracy": validation_accuracy,
                    "feature_count": int(features.shape[1]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        promoted = True

    print(f"Feature mode        : {feature_mode}")
    print(f"Version             : {version}")
    print(f"Training samples    : {len(features)}")
    print(f"Feature count       : {features.shape[1]}")
    print(f"CV accuracy (5-fold): {cv_accuracy:.2%}")
    print(f"Validation accuracy : {validation_accuracy:.2%}")
    print(f"Promoted            : {promoted}")

    return {
        "version": version,
        "feature_mode": feature_mode,
        "validation_accuracy": validation_accuracy,
        "cross_validated_accuracy": cv_accuracy,
        "current_active_accuracy": current_accuracy,
        "promoted": promoted,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Retrain the stress-level classifier.")
    parser.add_argument("--feature-mode", choices=["hog", "raw_pixels"], default="hog")
    parser.add_argument("--version", default=None)
    parser.add_argument("--promote-active", action="store_true")
    parser.add_argument("--without-handwriting-features", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_model(
        feature_mode=args.feature_mode,
        version=args.version,
        promote=args.promote_active,
        include_handwriting_features=not args.without_handwriting_features,
    )
