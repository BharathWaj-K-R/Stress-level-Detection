from datetime import datetime
from pathlib import Path
import csv
import io
import json
import warnings

import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "stress_model.pkl"
LABEL_MAP_PATH = BASE_DIR / "model" / "label_mapping.json"
HISTORY_PATH = BASE_DIR / "data" / "prediction_history.csv"
USER_SAMPLES_PATH = BASE_DIR / "data" / "user_training_samples.pkl"
IMG_SIZE = 128

DEFAULT_LABEL_MAP = {
    "0": "Low Stress",
    "1": "Medium Stress",
    "2": "High Stress",
    "low": "Low Stress",
    "medium": "Medium Stress",
    "high": "High Stress",
}
STRESS_OPTIONS = {
    "Low Stress": 0,
    "Medium Stress": 1,
    "High Stress": 2,
}


st.set_page_config(
    page_title="Stress Level Detection V2",
    page_icon="SL",
    layout="wide",
)


def load_label_map():
    if LABEL_MAP_PATH.exists():
        with LABEL_MAP_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    return DEFAULT_LABEL_MAP


def normalize_label_map(label_map):
    return {str(key): value for key, value in label_map.items()}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run `python train_model.py` in this folder."
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


def open_image(image_file):
    try:
        image_file.seek(0)
        return ImageOps.exif_transpose(Image.open(image_file)).convert("L")
    except (UnidentifiedImageError, OSError):
        st.error("Please provide a valid JPG, JPEG, or PNG handwriting image.")
        return None


def analyze_image_quality(image):
    stat = ImageStat.Stat(image)
    brightness = stat.mean[0]
    contrast = stat.stddev[0]

    issues = []
    if brightness < 40:
        issues.append("The image looks too dark. Try capturing it with better lighting.")
    if brightness > 235:
        issues.append("The image looks too bright. Make sure the handwriting is visible.")
    if contrast < 18:
        issues.append("The handwriting may have low contrast. Use a darker pen or clearer background.")
    if min(image.size) < 128:
        issues.append("The image is small. A closer or higher-resolution capture may improve results.")

    return issues


def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return pixels.reshape(1, -1)


def label_for(raw_label, label_map):
    label_key = str(raw_label)
    return label_map.get(label_key, label_key.replace("_", " ").title())


def predict_stress(model, label_map, image):
    features = preprocess_image(image)
    raw_prediction = model.predict(features)[0]
    predicted_label = label_for(raw_prediction, label_map)

    confidence = None
    class_scores = {}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = [str(value) for value in getattr(model, "classes_", range(len(probabilities)))]
        for class_name, probability in zip(classes, probabilities):
            class_scores[label_for(class_name, label_map)] = float(probability)

        confidence = class_scores.get(predicted_label)
        if confidence is None:
            confidence = float(np.max(probabilities))

    return predicted_label, confidence, class_scores, features


def load_user_samples():
    if not USER_SAMPLES_PATH.exists():
        return np.empty((0, IMG_SIZE * IMG_SIZE), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X, y = joblib.load(USER_SAMPLES_PATH)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def save_learning_sample(features, label):
    USER_SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    X, y = load_user_samples()
    X = np.vstack([X, np.asarray(features, dtype=np.float32)])
    y = np.concatenate([y, np.asarray([label], dtype=np.int64)])
    joblib.dump((X, y), USER_SAMPLES_PATH)
    return len(y)


def retrain_model_from_feedback():
    from train_model import train_model

    train_model()
    load_model.clear()


def save_history(input_source, image_name, predicted_label, confidence):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if HISTORY_PATH.exists():
        with HISTORY_PATH.open("r", encoding="utf-8") as file:
            rows = [
                {
                    "timestamp": row.get("timestamp", ""),
                    "input_source": row.get("input_source", ""),
                    "image_name": row.get("image_name", ""),
                    "prediction": row.get("prediction", ""),
                }
                for row in csv.DictReader(file)
            ]

    rows.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_source": input_source,
            "image_name": image_name,
            "prediction": predicted_label,
        }
    )

    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "input_source", "image_name", "prediction"])
        for row in rows:
            writer.writerow([row["timestamp"], row["input_source"], row["image_name"], row["prediction"]])


def build_report(input_source, image_name, predicted_label, quality_issues):
    issue_text = "No major image-quality warnings." if not quality_issues else "\n".join(
        f"- {issue}" for issue in quality_issues
    )

    return f"""Stress Level Detection Report

Generated On: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Input Source: {input_source}
Image Name: {image_name}
Predicted Stress Level: {predicted_label}

Image Quality Notes:
{issue_text}

Disclaimer:
This result is generated for educational project purposes only. It is not a medical diagnosis and should not replace professional health advice.
"""


def show_header():
    st.markdown(
        """
        <style>
        .main-card {
            padding: 1.3rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #10251f 0%, #245346 100%);
            border: 1px solid rgba(16, 37, 31, 0.35);
            box-shadow: 0 18px 42px rgba(16, 37, 31, 0.18);
        }
        .main-card h1 {
            color: #fff8e6;
            margin-bottom: 0.35rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }
        .main-card p {
            color: #e8f0ec;
            font-size: 1.05rem;
            margin-bottom: 0;
        }
        .result-card {
            padding: 1rem;
            border-radius: 16px;
            background: #10251f;
            color: #f7f3e8;
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="main-card">
            <h1>Stress Level Detection from Handwriting - Version 2</h1>
            <p>
                Upload a handwriting image or capture one directly using your camera.
                The system preprocesses the image and predicts Low, Medium, or High stress.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_input_file():
    input_mode = st.radio(
        "Choose input method",
        ["Upload Image", "Use Camera"],
        horizontal=True,
    )

    if input_mode == "Upload Image":
        image_file = st.file_uploader(
            "Upload handwriting image",
            type=["jpg", "jpeg", "png"],
        )
        source = "Upload"
    else:
        image_file = st.camera_input("Capture handwriting image using camera")
        source = "Camera"

    return source, image_file


def render_history():
    st.subheader("Prediction History")
    if not HISTORY_PATH.exists():
        st.caption("No predictions saved yet.")
        return

    with HISTORY_PATH.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        st.caption("No predictions saved yet.")
        return

    st.dataframe(rows[-10:], use_container_width=True)
    st.download_button(
        "Download Full History CSV",
        data=HISTORY_PATH.read_text(encoding="utf-8"),
        file_name="prediction_history.csv",
        mime="text/csv",
    )


def render_learning_panel():
    latest = st.session_state.get("latest_prediction")
    if latest is None:
        return

    st.subheader("Teach the Model")
    st.caption(
        "If the prediction is wrong or you know the correct stress level, choose the correct label and save it. "
        "The model will retrain using this input."
    )
    actual_label = st.selectbox(
        "Correct stress level for this handwriting",
        list(STRESS_OPTIONS.keys()),
        index=list(STRESS_OPTIONS.keys()).index(latest["predicted_label"])
        if latest["predicted_label"] in STRESS_OPTIONS
        else 0,
    )

    if st.button("Save Input and Retrain Model", use_container_width=True):
        sample_count = save_learning_sample(latest["features"], STRESS_OPTIONS[actual_label])
        retrain_model_from_feedback()
        st.success(f"Model updated. Learned samples saved: {sample_count}")
        st.info("Run another prediction to use the updated model immediately.")


show_header()

try:
    model, label_map = load_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()

left_col, right_col = st.columns([1.1, 0.9])

with left_col:
    st.subheader("Input")
    source, image_file = get_input_file()

    if image_file is not None:
        display_image = Image.open(io.BytesIO(image_file.getvalue()))
        st.image(display_image, caption="Selected handwriting image", use_container_width=True)

with right_col:
    st.subheader("Prediction")

    if image_file is None:
        st.info("Choose an input method and provide a handwriting image to start.")
    elif st.button("Predict Stress Level", type="primary", use_container_width=True):
        image = open_image(image_file)
        if image is not None:
            quality_issues = analyze_image_quality(image)
            predicted_label, confidence, class_scores, features = predict_stress(model, label_map, image)
            confidence_text = "Not available" if confidence is None else f"{confidence * 100:.2f}%"

            st.markdown(
                f"""
                <div class="result-card">
                    <h2>{predicted_label}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if quality_issues:
                st.warning("Image quality suggestions:")
                for issue in quality_issues:
                    st.write(f"- {issue}")
            else:
                st.success("Image quality looks acceptable.")

            image_name = getattr(image_file, "name", "camera_capture.png")
            save_history(source, image_name, predicted_label, confidence)
            st.session_state["latest_prediction"] = {
                "features": features,
                "predicted_label": predicted_label,
                "image_name": image_name,
                "source": source,
            }

            report = build_report(source, image_name, predicted_label, quality_issues)
            st.download_button(
                "Download Prediction Report",
                data=report,
                file_name="stress_prediction_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

st.divider()
render_learning_panel()
st.divider()
render_history()
