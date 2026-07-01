from __future__ import annotations

import json
import io
import warnings
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from preprocessing.feature_extraction import FeatureExtractionConfig, extract_feature_vector


DEFAULT_LABEL_MAP = {
    "0": "Low Stress",
    "1": "Medium Stress",
    "2": "High Stress",
    "low": "Low Stress",
    "medium": "Medium Stress",
    "high": "High Stress",
}
ALLOWED_EXTENSIONS = {".jpg", ".png"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024


def normalize_label_map(label_map):
    return {str(key): value for key, value in label_map.items()}


def load_label_map(label_map_path: Path | None):
    if label_map_path and label_map_path.exists():
        with label_map_path.open("r", encoding="utf-8") as file:
            return normalize_label_map(json.load(file))
    return normalize_label_map(DEFAULT_LABEL_MAP)


def load_model_bundle(model_path: Path, label_map_path: Path | None = None, metadata_path: Path | None = None):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        artifact = joblib.load(model_path)

    metadata = {}
    if metadata_path and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        label_map = normalize_label_map(artifact.get("label_map", load_label_map(label_map_path)))
        artifact_metadata = {
            "version": artifact.get("version", "legacy-v1"),
            "feature_extraction_method": artifact.get("feature_mode", "raw_pixels"),
            "include_handwriting_features": artifact.get("include_handwriting_features", False),
            "image_size": artifact.get("image_size", 128),
        }
        metadata = {**artifact_metadata, **metadata}
    else:
        model = artifact
        label_map = load_label_map(label_map_path)
        metadata = metadata or {
            "version": "legacy-v1",
            "feature_extraction_method": "raw_pixels",
            "include_handwriting_features": False,
            "image_size": 128,
        }

    return {
        "artifact": artifact,
        "model": model,
        "label_map": label_map,
        "metadata": metadata,
    }


def validate_uploaded_file(uploaded_file, allowed_extensions=ALLOWED_EXTENSIONS, max_size_bytes=MAX_FILE_SIZE_BYTES):
    filename = getattr(uploaded_file, "name", "")
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        return False, f"Unsupported file type. Allowed types: {allowed}"

    file_size = getattr(uploaded_file, "size", None)
    if file_size is None:
        file_size = len(uploaded_file.getvalue())

    if file_size > max_size_bytes:
        return False, f"File is too large. Maximum allowed size is {max_size_bytes // (1024 * 1024)} MB."

    return True, None


def open_uploaded_image(uploaded_file):
    try:
        payload = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        return ImageOps.exif_transpose(Image.open(io.BytesIO(payload))).convert("L")
    except (UnidentifiedImageError, OSError):
        return None


def pil_to_normalized_array(image: Image.Image, image_size: int = 128) -> np.ndarray:
    image = image.resize((image_size, image_size))
    pixels = np.asarray(image, dtype=np.float32)
    if pixels.max() > 1.0:
        pixels = pixels / 255.0
    return np.clip(pixels, 0.0, 1.0)


def get_feature_config(bundle, image_size=128):
    metadata = bundle.get("metadata", {})
    return FeatureExtractionConfig(
        mode=metadata.get("feature_extraction_method", "raw_pixels"),
        image_size=image_size,
        include_handwriting_features=metadata.get("include_handwriting_features", False),
    )


def extract_features_for_bundle(bundle, image: Image.Image, image_size=128) -> np.ndarray:
    normalized = pil_to_normalized_array(image, image_size=image_size)
    config = get_feature_config(bundle, image_size=image_size)
    features = extract_feature_vector(normalized, config=config)
    return features.reshape(1, -1).astype(np.float32)


def get_prediction_label(raw_label, label_map):
    label_key = str(raw_label)
    return label_map.get(label_key, label_key.replace("_", " ").title())


def predict_with_bundle(bundle, image: Image.Image, image_size=128):
    features = extract_features_for_bundle(bundle, image, image_size=image_size)
    model = bundle["model"]
    label_map = bundle["label_map"]
    raw_prediction = model.predict(features)[0]
    predicted_label = get_prediction_label(raw_prediction, label_map)

    probabilities = None
    probability_map = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = [str(value) for value in getattr(model, "classes_", range(len(probabilities)))]
        probability_map = {
            get_prediction_label(class_name, label_map): float(probability)
            for class_name, probability in zip(classes, probabilities)
        }

    return {
        "raw_prediction": raw_prediction,
        "predicted_label": predicted_label,
        "features": features,
        "probability_map": probability_map,
    }
