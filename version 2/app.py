from datetime import datetime
from pathlib import Path
import csv
import io
import json
import os
import shutil
import sys
import uuid

import joblib
import numpy as np
import streamlit as st
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.predictor import (
    load_model_bundle,
    open_uploaded_image,
    pil_to_normalized_array,
    predict_with_bundle,
    validate_uploaded_file,
)
from preprocessing.quality_check import run_quality_checks
from train_model import train_model


LEGACY_MODEL_PATH = BASE_DIR / "model" / "stress_model.pkl"
LEGACY_LABEL_MAP_PATH = BASE_DIR / "model" / "label_mapping.json"
ACTIVE_MODEL_PATH = BASE_DIR / "model" / "active" / "stress_model.pkl"
ACTIVE_LABEL_MAP_PATH = BASE_DIR / "model" / "active" / "label_mapping.json"
ACTIVE_METADATA_PATH = BASE_DIR / "model" / "active" / "metadata.json"
HISTORY_PATH = BASE_DIR / "data" / "prediction_history.csv"
PENDING_REVIEW_DIR = BASE_DIR / "data" / "pending_review"
APPROVED_SAMPLES_PATH = BASE_DIR / "data" / "approved_samples.pkl"
APPROVED_ARCHIVE_DIR = BASE_DIR / "data" / "reviewed" / "approved"
REJECTED_ARCHIVE_DIR = BASE_DIR / "data" / "reviewed" / "rejected"
IMG_SIZE = 128

STRESS_OPTIONS = {
    "Low Stress": 0,
    "Medium Stress": 1,
    "High Stress": 2,
}


st.set_page_config(page_title="Stress Level Detection V2", page_icon="SL", layout="wide")


def resolve_model_paths():
    if ACTIVE_MODEL_PATH.exists():
        return ACTIVE_MODEL_PATH, ACTIVE_LABEL_MAP_PATH, ACTIVE_METADATA_PATH
    return LEGACY_MODEL_PATH, LEGACY_LABEL_MAP_PATH, None


@st.cache_resource
def load_model():
    model_path, label_map_path, metadata_path = resolve_model_paths()
    return load_model_bundle(model_path, label_map_path=label_map_path, metadata_path=metadata_path)


def save_history(input_source, image_name, predicted_label):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if HISTORY_PATH.exists():
        with HISTORY_PATH.open("r", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

    rows.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_source": input_source,
            "image_name": image_name,
            "prediction": predicted_label,
        }
    )

    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "input_source", "image_name", "prediction"])
        writer.writeheader()
        writer.writerows(rows)


def build_report(input_source, image_name, predicted_label, quality_warnings):
    issue_text = "No major image-quality warnings." if not quality_warnings else "\n".join(
        f"- {issue}" for issue in quality_warnings
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


def render_header(metadata):
    version = metadata.get("semantic_version") or metadata.get("version", "legacy-v2")
    feature_mode = metadata.get("feature_extraction_method", "raw_pixels")
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
        f"""
        <div class="main-card">
            <h1>Stress Level Detection from Handwriting</h1>
            <p>
                Version 2 prediction workspace with camera capture, pending review,
                and versioned retraining. Loaded model: <strong>{version}</strong> using
                <strong>{feature_mode}</strong> features.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "Experimental / educational project only. This is not a clinical or diagnostic tool.",
        icon="⚠️",
    )


def get_input_file():
    input_mode = st.radio("Choose input method", ["Upload Image", "Use Camera"], horizontal=True)
    if input_mode == "Upload Image":
        image_file = st.file_uploader("Upload handwriting image", type=["jpg", "png"])
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

    rows = list(csv.DictReader(HISTORY_PATH.open("r", encoding="utf-8")))
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


def queue_pending_review(image: Image.Image, source: str, image_name: str, predicted_label: str, actual_label: str):
    PENDING_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    record_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    record = {
        "id": record_id,
        "created_at": datetime.now().isoformat(),
        "source": source,
        "image_name": image_name,
        "predicted_label": predicted_label,
        "actual_label": actual_label,
        "image_array": pil_to_normalized_array(image, image_size=IMG_SIZE),
    }
    joblib.dump(record, PENDING_REVIEW_DIR / f"{record_id}.joblib")
    return record_id


def list_pending_samples():
    PENDING_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(PENDING_REVIEW_DIR.glob("*.joblib"))


def load_pending_record(path: Path):
    return joblib.load(path)


def load_approved_samples():
    if not APPROVED_SAMPLES_PATH.exists():
        return np.empty((0, IMG_SIZE, IMG_SIZE), dtype=np.float32), np.empty((0,), dtype=np.int64)
    images, labels = joblib.load(APPROVED_SAMPLES_PATH)
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def save_approved_samples(images, labels):
    APPROVED_SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((images, labels), APPROVED_SAMPLES_PATH)


def approve_pending_record(record_path: Path):
    record = load_pending_record(record_path)
    images, labels = load_approved_samples()
    image_array = np.asarray(record["image_array"], dtype=np.float32).reshape(1, IMG_SIZE, IMG_SIZE)
    label_value = np.asarray([STRESS_OPTIONS[record["actual_label"]]], dtype=np.int64)
    images = np.concatenate([images, image_array], axis=0)
    labels = np.concatenate([labels, label_value], axis=0)
    save_approved_samples(images, labels)

    APPROVED_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(str(record_path), str(APPROVED_ARCHIVE_DIR / record_path.name))


def reject_pending_record(record_path: Path):
    REJECTED_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(str(record_path), str(REJECTED_ARCHIVE_DIR / record_path.name))


def render_learning_panel():
    latest = st.session_state.get("latest_prediction")
    if latest is None:
        return

    st.subheader("Submit for Review")
    st.caption(
        "Choose the correct label and submit this sample for admin review. "
        "It will not be added to training until approved."
    )
    actual_label = st.selectbox(
        "Correct stress level for this handwriting",
        list(STRESS_OPTIONS.keys()),
        index=list(STRESS_OPTIONS.keys()).index(latest["predicted_label"])
        if latest["predicted_label"] in STRESS_OPTIONS
        else 0,
    )

    if st.button("Submit Sample for Review", use_container_width=True):
        record_id = queue_pending_review(
            latest["image"],
            latest["source"],
            latest["image_name"],
            latest["predicted_label"],
            actual_label,
        )
        st.success(f"Sample queued for admin review: {record_id}")


def render_admin_page():
    st.subheader("Admin Review")
    admin_password = os.getenv("STRESS_APP_ADMIN_PASSWORD")
    if not admin_password:
        st.info("Admin review is disabled because `STRESS_APP_ADMIN_PASSWORD` is not set.")
        return

    entered_password = st.text_input("Enter admin password", type="password")
    if entered_password != admin_password:
        st.warning("Enter the correct admin password to access pending reviews.")
        return

    pending_files = list_pending_samples()
    approved_count = len(load_approved_samples()[0])
    st.caption(f"Pending samples: {len(pending_files)} | Approved feedback samples: {approved_count}")

    if not pending_files:
        st.success("No pending samples to review.")
    else:
        for record_path in pending_files:
            record = load_pending_record(record_path)
            image = Image.fromarray((record["image_array"] * 255).astype(np.uint8))
            st.markdown(f"**{record['image_name']}**")
            st.caption(
                f"Queued: {record['created_at']} | Predicted: {record['predicted_label']} | "
                f"Reviewer label: {record['actual_label']}"
            )
            st.image(image, width=220)
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Approve {record['id']}", key=f"approve_{record['id']}"):
                    approve_pending_record(record_path)
                    st.success(f"Approved sample {record['id']}")
                    st.rerun()
            with col2:
                if st.button(f"Reject {record['id']}", key=f"reject_{record['id']}"):
                    reject_pending_record(record_path)
                    st.info(f"Rejected sample {record['id']}")
                    st.rerun()
            st.divider()

    if st.button("Retrain and Evaluate Approved Samples", type="primary", use_container_width=True):
        current_accuracy = None
        active_metadata_path = ACTIVE_METADATA_PATH
        if active_metadata_path.exists():
            active_metadata = json.loads(active_metadata_path.read_text(encoding="utf-8"))
            current_accuracy = active_metadata.get("cross_validated_accuracy")

        result = train_model(feature_mode="hog", promote=True)
        load_model.clear()
        st.success(f"Retraining completed for version {result['version']}")
        if current_accuracy is not None:
            st.write(f"Before accuracy: {current_accuracy:.2%}")
        st.write(f"After accuracy: {result['evaluated_accuracy']:.2%}")
        diff = result["evaluated_accuracy"] - (current_accuracy or 0.0)
        st.write(f"Difference: {diff:+.2%}")
        st.write(f"Promoted to active: {result['promoted']}")


try:
    bundle = load_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()

render_header(bundle.get("metadata", {}))

page = st.sidebar.radio("Page", ["Predict", "Admin Review"])

if page == "Predict":
    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.subheader("Input")
        source, image_file = get_input_file()

        if image_file is not None:
            valid_file, validation_message = validate_uploaded_file(image_file)
            if not valid_file:
                st.error(validation_message)
                st.stop()

            display_image = Image.open(io.BytesIO(image_file.getvalue()))
            st.image(display_image, caption="Selected handwriting image", use_container_width=True)

    with right_col:
        st.subheader("Prediction")

        if image_file is None:
            st.info("Choose an input method and provide a handwriting image to start.")
        elif st.button("Predict Stress Level", type="primary", use_container_width=True):
            image = open_uploaded_image(image_file)
            if image is None:
                st.error("Please provide a valid JPG or PNG handwriting image.")
                st.stop()

            raw_pixels = np.asarray(image, dtype=np.float32)
            checks = run_quality_checks(raw_pixels)
            if not checks["passed"]:
                for warning in checks["warnings"]:
                    st.warning(warning)
                st.stop()

            prediction = predict_with_bundle(bundle, image, image_size=IMG_SIZE)
            st.markdown(
                f"""
                <div class="result-card">
                    <h2>{prediction['predicted_label']}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

            image_name = getattr(image_file, "name", "camera_capture.jpg")
            save_history(source, image_name, prediction["predicted_label"])
            st.session_state["latest_prediction"] = {
                "image": image,
                "predicted_label": prediction["predicted_label"],
                "image_name": image_name,
                "source": source,
            }

            report = build_report(source, image_name, prediction["predicted_label"], checks["warnings"])
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
else:
    render_admin_page()
