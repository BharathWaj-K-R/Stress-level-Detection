import sys
from pathlib import Path

import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from preprocessing.feature_extraction import FeatureExtractionConfig, extract_feature_vector


def load_sample_image():
    X, _ = joblib.load(BASE_DIR / "model" / "preprocessed_data.pkl")
    return np.asarray(X[0], dtype=np.float32).reshape(128, 128)


def test_raw_feature_shape_and_type():
    image = load_sample_image()
    features = extract_feature_vector(image, FeatureExtractionConfig(mode="raw_pixels"))

    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32
    assert features.shape == (16387,)


def test_hog_feature_shape_and_type():
    image = load_sample_image()
    features = extract_feature_vector(image, FeatureExtractionConfig(mode="hog"))

    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32
    assert features.shape == (8103,)
