import io

import joblib
import numpy as np
from PIL import Image

from inference.predictor import (
    extract_features_for_bundle,
    load_model_bundle,
    open_uploaded_image,
    pil_to_normalized_array,
    validate_uploaded_file,
)


class Upload:
    def __init__(self, name, payload):
        self.name = name
        self._payload = payload
        self.size = len(payload)

    def getvalue(self):
        return self._payload


class DummyModel:
    def predict(self, features):
        return np.array([0])


def make_png(width=320, height=180):
    image = Image.new("L", (width, height), color=255)
    pixels = image.load()
    for x in range(20, width - 20):
        y = int(height / 2 + 20 * np.sin(x / 14.0))
        if 0 <= y < height:
            pixels[x, y] = 0
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_validate_uploaded_file_accepts_jpeg_png_and_jpg():
    payload = make_png()
    for name in ["sample.jpg", "sample.jpeg", "sample.png"]:
        ok, message = validate_uploaded_file(Upload(name, payload))
        assert ok
        assert message is None


def test_validate_uploaded_file_rejects_empty_file():
    ok, message = validate_uploaded_file(Upload("sample.png", b""))
    assert not ok
    assert "empty" in message.lower()


def test_open_uploaded_image_rejects_invalid_bytes():
    assert open_uploaded_image(Upload("sample.png", b"not an image")) is None


def test_open_uploaded_image_converts_to_grayscale():
    image = Image.new("RGB", (160, 100), color=(20, 40, 80))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    opened = open_uploaded_image(Upload("sample.png", buffer.getvalue()))
    assert opened is not None
    assert opened.mode == "L"
    assert opened.size == (160, 100)


def test_pil_to_normalized_array_preserves_aspect_ratio_with_padding():
    image = Image.new("L", (320, 160), color=255)
    array = pil_to_normalized_array(image, image_size=128)
    assert array.shape == (128, 128)
    assert np.allclose(array[:32], 1.0)
    assert np.allclose(array[-32:], 1.0)


def test_model_bundle_can_be_loaded_from_joblib(tmp_path):
    model_path = tmp_path / "model.pkl"
    joblib.dump(
        {
            "model": DummyModel(),
            "label_map": {"0": "Low Stress"},
            "image_size": 128,
            "feature_mode": "raw_pixels",
            "include_handwriting_features": False,
            "version": "test-v1",
        },
        model_path,
    )
    bundle = load_model_bundle(model_path)
    assert bundle["metadata"]["version"] == "test-v1"
    assert bundle["metadata"]["feature_extraction_method"] == "raw_pixels"


def test_extract_features_matches_configured_feature_mode():
    image = Image.new("L", (128, 128), color=255)
    bundle = {
        "metadata": {
            "feature_extraction_method": "raw_pixels",
            "include_handwriting_features": False,
        }
    }
    features = extract_features_for_bundle(bundle, image)
    assert features.shape == (1, 128 * 128)
