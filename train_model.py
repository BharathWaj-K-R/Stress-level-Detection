from __future__ import annotations

import argparse
import json
import sys
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
APPROVED_SAMPLES_PATH = BASE_DIR / "data" / "approved_samples.pkl"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"

IMG_SIZE = 128
RANDOM_STATE = 42
STRESS_ORDER = ["low", "medium", "high"]
LABEL_MAP = {"0": "Low Stress", "1": "Medium Stress", "2": "High Stress"}


def _normalize_label(label: str) -> str:
    return str(label).strip().lower().replace(" stress", "")


def _encode_labels(labels) -> np.ndarray:
    normalized = [_normalize_label(lbl) for lbl in labels]
    unknown = sorted(set(normalized) - set(STRESS_ORDER))
    if unknown:
        raise ValueError(f"Unknown labels in training data: {unknown}")
    return np.asarray([STRESS_ORDER.index(lbl) for lbl in normalized], dtype=np.int64)


def _load_base_images() -> tuple[np.ndarray, np.ndarray]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")
    X, y = joblib.load(DATA_PATH)
    X = np.asarray(X, dtype=np.float32)
    if X.max() > 1.0:
        X = X / 255.0
    return X.reshape((-1, IMG_SIZE, IMG_SIZE)), _encode_labels(y)


def _load_approved() -> tuple[np.ndarray, np.ndarray]:
    if not APPROVED_SAMPLES_PATH.exists():
        return np.empty((0, IMG_SIZE, IMG_SIZE), dtype=np.float32), np.empty((0,), dtype=np.int64)
    imgs, labels = joblib.load(APPROVED_SAMPLES_PATH)
    return np.asarray(imgs, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _load_training_data() -> tuple[np.ndarray, np.ndarray]:
    base_imgs, base_labels = _load_base_images()
    approved_imgs, approved_labels = _load_approved()
    if len(approved_imgs):
        return (
            np.concatenate([base_imgs, approved_imgs], axis=0),
            np.concatenate([base_labels, approved_labels], axis=0),
        )
    return base_imgs, base_labels


def train_model(
    feature_mode: str = "hog",
    include_handwriting_features: bool = True,
    version: str | None = None,
    promote: bool = False,
) -> dict:
    images, y = _load_training_data()

    config = FeatureExtractionConfig(
        mode=feature_mode,
        image_size=IMG_SIZE,
        include_handwriting_features=include_handwriting_features,
    )
    X = batch_extract_features(images, config=config)
    feature_names = get_feature_names(config)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    clf_cv = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_features="sqrt",
    )
    cv_scores = cross_val_score(clf_cv, X, y, cv=cv, scoring="accuracy")
    cv_accuracy = float(np.mean(cv_scores))

    stratify = y if min(np.bincount(y, minlength=3)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
    )
    model = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_features="sqrt",
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    val_accuracy = float(accuracy_score(y_test, preds))

    if version is None:
        if METADATA_PATH.exists():
            try:
                meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
                v = meta.get("semantic_version", "v1.0.0")
                major, minor, patch = [int(p) for p in v.lstrip("v").split(".")]
                version = f"v{major}.{minor}.{patch + 1}"
            except Exception:
                version = "v2.0.0"
        else:
            version = "v2.0.0"

    current_acc = None
    if METADATA_PATH.exists():
        try:
            current_acc = float(
                json.loads(METADATA_PATH.read_text(encoding="utf-8")).get(
                    "cross_validated_accuracy", 0.0
                )
            )
        except Exception:
            pass

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "model": model,
        "label_map": LABEL_MAP,
        "image_size": IMG_SIZE,
        "feature_mode": feature_mode,
        "include_handwriting_features": include_handwriting_features,
        "feature_names": feature_names,
        "version": version,
    }

    promoted = False
    if promote:
        if current_acc is None or cv_accuracy >= current_acc:
            joblib.dump(model_bundle, MODEL_DIR / "stress_model.pkl")
            (MODEL_DIR / "label_mapping.json").write_text(
                json.dumps(LABEL_MAP, indent=2), encoding="utf-8"
            )
            (MODEL_DIR / "metadata.json").write_text(
                json.dumps(
                    {
                        "version": version,
                        "semantic_version": version,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "feature_extraction_method": feature_mode,
                        "include_handwriting_features": include_handwriting_features,
                        "training_set_size": int(len(X)),
                        "cross_validated_accuracy": cv_accuracy,
                        "evaluated_accuracy": val_accuracy,
                        "feature_count": int(X.shape[1]),
                        "approved_feedback_samples": int(len(_load_approved()[0])),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            promoted = True

    print(f"Feature mode      : {feature_mode}")
    print(f"Version           : {version}")
    print(f"Training samples  : {len(X)}")
    print(f"Feature count     : {X.shape[1]}")
    print(f"CV accuracy (5-fold): {cv_accuracy:.2%}")
    print(f"Val accuracy      : {val_accuracy:.2%}")
    print(f"Promoted          : {promoted}")

    return {
        "version": version,
        "feature_mode": feature_mode,
        "validation_accuracy": val_accuracy,
        "cross_validated_accuracy": cv_accuracy,
        "evaluated_accuracy": cv_accuracy,
        "current_active_accuracy": current_acc,
        "promoted": promoted,
    }


def _parse_args():
    p = argparse.ArgumentParser(description="Retrain the stress-level classifier.")
    p.add_argument("--feature-mode", choices=["hog", "raw_pixels"], default="hog")
    p.add_argument("--version", default=None)
    p.add_argument("--promote-active", action="store_true")
    p.add_argument("--without-handwriting-features", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_model(
        feature_mode=args.feature_mode,
        version=args.version,
        promote=args.promote_active,
        include_handwriting_features=not args.without_handwriting_features,
    )
