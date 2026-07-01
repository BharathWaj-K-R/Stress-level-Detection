from pathlib import Path

import numpy as np
import streamlit as st

from inference.predictor import (
    load_model_bundle,
    open_uploaded_image,
    predict_with_bundle,
    validate_uploaded_file,
)
from preprocessing.quality_check import run_quality_checks


BASE_DIR = Path(__file__).resolve().parent
LEGACY_MODEL_PATH = BASE_DIR / "model" / "stress_model.pkl"
LEGACY_LABEL_MAP_PATH = BASE_DIR / "model" / "label_mapping.json"
UPGRADED_MODEL_PATH = BASE_DIR / "model" / "upgraded" / "active" / "stress_model.pkl"
UPGRADED_LABEL_MAP_PATH = BASE_DIR / "model" / "upgraded" / "active" / "label_mapping.json"
UPGRADED_METADATA_PATH = BASE_DIR / "model" / "upgraded" / "active" / "metadata.json"


def resolve_model_paths():
    if UPGRADED_MODEL_PATH.exists():
        return UPGRADED_MODEL_PATH, UPGRADED_LABEL_MAP_PATH, UPGRADED_METADATA_PATH
    return LEGACY_MODEL_PATH, LEGACY_LABEL_MAP_PATH, None


@st.cache_resource
def load_model():
    model_path, label_map_path, metadata_path = resolve_model_paths()
    return load_model_bundle(model_path, label_map_path=label_map_path, metadata_path=metadata_path)


st.set_page_config(page_title="Stress Level Detection", layout="centered")
st.title("Stress Level Detection from Handwriting")
st.markdown(
    "Upload a clear handwriting image in `.jpg` or `.png` format. "
    "The upgraded app checks quality before prediction and shows which model version is loaded."
)
st.warning(
    "Experimental / educational project only. This is not a clinical or diagnostic tool.",
    icon="⚠️",
)

try:
    bundle = load_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()

metadata = bundle.get("metadata", {})
model_version = metadata.get("semantic_version") or metadata.get("version", "legacy-v1")
feature_mode = metadata.get("feature_extraction_method", "raw_pixels")
st.caption(f"Loaded model version: `{model_version}` using `{feature_mode}` features")

uploaded_file = st.file_uploader("Upload handwriting image", type=["jpg", "png"])

if uploaded_file is not None:
    valid_file, validation_message = validate_uploaded_file(uploaded_file)
    if not valid_file:
        st.error(validation_message)
        st.stop()

    st.image(uploaded_file, caption="Uploaded Handwriting", use_container_width=True)

    if st.button("Predict Stress Level", type="primary"):
        image = open_uploaded_image(uploaded_file)
        if image is None:
            st.error("Please upload a valid JPG or PNG handwriting image.")
            st.stop()

        raw_pixels = np.asarray(image, dtype=np.float32)
        checks = run_quality_checks(raw_pixels)
        if not checks["passed"]:
            for warning in checks["warnings"]:
                st.warning(warning)
            st.stop()

        prediction = predict_with_bundle(bundle, image, image_size=128)
        st.success(f"Predicted Stress Level: {prediction['predicted_label']}")

        probability_map = prediction.get("probability_map")
        if probability_map:
            st.bar_chart(probability_map)
