from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "stress_model.pkl"
LABEL_MAP_PATH = BASE_DIR / "model" / "label_mapping.json"
IMG_SIZE = 128

DEFAULT_LABEL_MAP = {
    "0": "Low Stress",
    "1": "Medium Stress",
    "2": "High Stress",
    "low": "Low Stress",
    "medium": "Medium Stress",
    "high": "High Stress",
}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run `python train_model.py` first."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        artifact = joblib.load(MODEL_PATH)

    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        label_map = artifact.get("label_map", DEFAULT_LABEL_MAP)
    else:
        model = artifact
        label_map = load_label_map()

    return model, normalize_label_map(label_map)


def load_label_map():
    if LABEL_MAP_PATH.exists():
        with LABEL_MAP_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    return DEFAULT_LABEL_MAP


def normalize_label_map(label_map):
    return {str(key): value for key, value in label_map.items()}


def preprocess_image(uploaded_file):
    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("L")
    except (UnidentifiedImageError, OSError):
        st.error("Please upload a valid JPG, JPEG, or PNG handwriting image.")
        return None

    image = image.resize((IMG_SIZE, IMG_SIZE))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return pixels.reshape(1, -1)


def get_prediction_label(raw_label, label_map):
    label_key = str(raw_label)
    return label_map.get(label_key, label_key.replace("_", " ").title())


def get_confidence(model, features, raw_prediction):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = [str(value) for value in getattr(model, "classes_", range(len(probabilities)))]
        if str(raw_prediction) in classes:
            best_index = classes.index(str(raw_prediction))
        else:
            best_index = int(np.argmax(probabilities))
        return float(probabilities[best_index]) * 100, probabilities, classes

    return None, None, None


st.set_page_config(page_title="Stress Level Detection", layout="centered")
st.title("Stress Level Detection from Handwriting")
st.markdown(
    "Upload a clear handwriting image. The app will resize it to 128x128 pixels "
    "and classify the predicted stress level."
)

try:
    model, label_map = load_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()

uploaded_file = st.file_uploader("Upload handwriting image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Handwriting", use_container_width=True)

    if st.button("Predict Stress Level", type="primary"):
        features = preprocess_image(uploaded_file)

        if features is not None:
            raw_prediction = model.predict(features)[0]
            predicted_label = get_prediction_label(raw_prediction, label_map)
            confidence, probabilities, classes = get_confidence(model, features, raw_prediction)

            st.success(f"Predicted Stress Level: {predicted_label}")

            if confidence is not None:
                st.info(f"Confidence: {confidence:.2f}%")
                chart_data = {
                    get_prediction_label(class_name, label_map): float(probability)
                    for class_name, probability in zip(classes, probabilities)
                }
                st.bar_chart(chart_data)
            else:
                st.caption("This model does not provide probability scores.")
