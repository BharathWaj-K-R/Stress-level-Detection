from pathlib import Path

import numpy as np
from PIL import Image

from inference.predictor import load_model_bundle, predict_with_bundle


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "stress_model.pkl"
LABEL_MAP_PATH = ROOT / "model" / "label_mapping.json"
METADATA_PATH = ROOT / "model" / "metadata.json"


def test_packaged_model_loads_and_predicts():
    bundle = load_model_bundle(
        MODEL_PATH,
        label_map_path=LABEL_MAP_PATH,
        metadata_path=METADATA_PATH,
    )
    image = Image.fromarray(np.full((128, 128), 255, dtype=np.uint8), mode="L")
    result = predict_with_bundle(bundle, image, image_size=128)

    assert result["predicted_label"] in {
        "Low Stress",
        "Medium Stress",
        "High Stress",
    }
    assert result["features"].shape[0] == 1

    if result["probability_map"]:
        total = sum(result["probability_map"].values())
        np.testing.assert_allclose(total, 1.0, rtol=1e-6, atol=1e-6)
