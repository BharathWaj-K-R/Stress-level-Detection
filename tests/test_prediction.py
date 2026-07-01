import io
import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from inference.predictor import (
    load_model_bundle,
    open_uploaded_image,
    predict_with_bundle,
    validate_uploaded_file,
)


class FakeUpload:
    def __init__(self, name: str, payload: bytes):
        self.name = name
        self._payload = payload
        self.size = len(payload)
        self._buffer = io.BytesIO(payload)

    def seek(self, *args, **kwargs):
        return self._buffer.seek(*args, **kwargs)

    def read(self, *args, **kwargs):
        return self._buffer.read(*args, **kwargs)

    def getvalue(self):
        return self._payload


def create_mock_model():
    np.random.seed(42)
    X = np.random.rand(12, 16387).astype(np.float32)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    return model


def create_sample_upload():
    image = Image.new("L", (160, 160), color=255)
    for row in range(40, 120):
        for col in range(50, 110):
            image.putpixel((col, row), 0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return FakeUpload("sample.png", buffer.getvalue())


def test_validate_uploaded_file_accepts_png():
    upload = create_sample_upload()
    ok, message = validate_uploaded_file(upload)
    assert ok
    assert message is None


def test_prediction_function_with_mock_model():
    model = create_mock_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "stress_model.pkl"
        label_map_path = Path(tmpdir) / "label_mapping.json"
        metadata_path = Path(tmpdir) / "metadata.json"

        joblib.dump(
            {
                "model": model,
                "label_map": {"0": "Low Stress", "1": "Medium Stress", "2": "High Stress"},
                "image_size": 128,
                "feature_mode": "raw_pixels",
                "include_handwriting_features": True,
                "version": "vtest.1",
            },
            model_path,
        )
        label_map_path.write_text(json.dumps({"0": "Low Stress", "1": "Medium Stress", "2": "High Stress"}))
        metadata_path.write_text(json.dumps({"version": "vtest.1", "feature_extraction_method": "raw_pixels"}))

        bundle = load_model_bundle(model_path, label_map_path=label_map_path, metadata_path=metadata_path)
        upload = create_sample_upload()
        image = open_uploaded_image(upload)
        prediction = predict_with_bundle(bundle, image, image_size=128)

        assert prediction["predicted_label"] in {"Low Stress", "Medium Stress", "High Stress"}
        assert prediction["features"].shape[0] == 1
