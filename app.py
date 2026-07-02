from datetime import datetime
from pathlib import Path
import csv
import io
import json
import os
import shutil
import uuid

import joblib
import numpy as np
import streamlit as st
from PIL import Image

from inference.predictor import (
    load_model_bundle,
    open_uploaded_image,
    pil_to_normalized_array,
    predict_with_bundle,
    validate_uploaded_file,
)
from preprocessing.quality_check import run_quality_checks
from train_model import train_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "stress_model.pkl"
LABEL_MAP_PATH = MODEL_DIR / "label_mapping.json"
METADATA_PATH = MODEL_DIR / "metadata.json"

DATA_DIR = BASE_DIR / "data"
HISTORY_PATH = DATA_DIR / "prediction_history.csv"
PENDING_DIR = DATA_DIR / "pending_review"
APPROVED_SAMPLES_PATH = DATA_DIR / "approved_samples.pkl"
APPROVED_ARCHIVE_DIR = DATA_DIR / "reviewed" / "approved"
REJECTED_ARCHIVE_DIR = DATA_DIR / "reviewed" / "rejected"

IMG_SIZE = 128
STRESS_OPTIONS = {"Low Stress": 0, "Medium Stress": 1, "High Stress": 2}

st.set_page_config(
    page_title="Stress Level Detection",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: #0f1c18;
    }
    [data-testid="stSidebar"] * {
        color: #d4e8df !important;
    }
    .header-banner {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #0d2218 0%, #1a4332 60%, #0d2218 100%);
        border: 1px solid #1e4d36;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        margin-bottom: 1.5rem;
    }
    .header-banner h1 {
        color: #e8f5ee;
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.03em;
    }
    .header-banner p {
        color: #9ecfb5;
        font-size: 0.95rem;
        margin: 0;
    }
    .result-card {
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        background: #0d2218;
        border: 1px solid #1e4d36;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
        margin: 0.8rem 0;
        text-align: center;
    }
    .result-card .label {
        font-size: 0.85rem;
        color: #9ecfb5;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.3rem;
    }
    .result-card .value {
        font-size: 2rem;
        font-weight: 800;
        color: #5de8a0;
    }
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 0.8rem 0;
    }
    .metric-box {
        flex: 1;
        background: #0d2218;
        border: 1px solid #1e4d36;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .metric-box .m-label {
        font-size: 0.75rem;
        color: #9ecfb5;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-box .m-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #e8f5ee;
    }
    hr { border-color: #1e4d36 !important; }
    .stButton > button[kind="primary"] {
        background: #1a6641 !important;
        border: none !important;
        color: #e8f5ee !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #22854f !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return load_model_bundle(MODEL_PATH, label_map_path=LABEL_MAP_PATH, metadata_path=METADATA_PATH)


def save_history(source: str, image_name: str, predicted_label: str) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if HISTORY_PATH.exists():
        with HISTORY_PATH.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "image_name": image_name,
            "prediction": predicted_label,
        }
    )
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "source", "image_name", "prediction"])
        writer.writeheader()
        writer.writerows(rows)


def build_report(source: str, image_name: str, label: str, warnings: list) -> str:
    issues = "None detected." if not warnings else "\n".join(f"  • {w}" for w in warnings)
    return (
        f"Stress Level Detection Report\n"
        f"{'=' * 40}\n"
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Source    : {source}\n"
        f"File      : {image_name}\n"
        f"Result    : {label}\n\n"
        f"Image Quality Notes:\n{issues}\n\n"
        f"Disclaimer:\n"
        f"This result is for educational purposes only.\n"
        f"It is not a medical diagnosis or clinical assessment.\n"
    )


def queue_for_review(image: Image.Image, source: str, name: str, predicted: str, actual: str) -> str:
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    record_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    record = {
        "id": record_id,
        "created_at": datetime.now().isoformat(),
        "source": source,
        "image_name": name,
        "predicted_label": predicted,
        "actual_label": actual,
        "image_array": pil_to_normalized_array(image, image_size=IMG_SIZE),
    }
    joblib.dump(record, PENDING_DIR / f"{record_id}.joblib")
    return record_id


def list_pending() -> list:
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(PENDING_DIR.glob("*.joblib"))


def load_approved_samples():
    if not APPROVED_SAMPLES_PATH.exists():
        return np.empty((0, IMG_SIZE, IMG_SIZE), dtype=np.float32), np.empty((0,), dtype=np.int64)
    imgs, labels = joblib.load(APPROVED_SAMPLES_PATH)
    return np.asarray(imgs, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def save_approved_samples(imgs, labels) -> None:
    APPROVED_SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((imgs, labels), APPROVED_SAMPLES_PATH)


def approve_record(path: Path) -> None:
    record = joblib.load(path)
    imgs, labels = load_approved_samples()
    arr = np.asarray(record["image_array"], dtype=np.float32).reshape(1, IMG_SIZE, IMG_SIZE)
    lbl = np.asarray([STRESS_OPTIONS[record["actual_label"]]], dtype=np.int64)
    imgs = np.concatenate([imgs, arr], axis=0)
    labels = np.concatenate([labels, lbl], axis=0)
    save_approved_samples(imgs, labels)
    APPROVED_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(APPROVED_ARCHIVE_DIR / path.name))


def reject_record(path: Path) -> None:
    REJECTED_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(REJECTED_ARCHIVE_DIR / path.name))


with st.sidebar:
    st.markdown("## ✍️ Stress Detection")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Predict", "History", "Admin"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    try:
        bundle = load_model()
        meta = bundle.get("metadata", {})
        version = meta.get("semantic_version") or meta.get("version", "—")
        mode = meta.get("feature_extraction_method", "—")
        st.markdown(f"**Model version:** `{version}`")
        st.markdown(f"**Feature mode:** `{mode}`")
    except Exception:
        bundle = None

    st.markdown("---")
    st.caption("Educational project only — not a clinical tool.")


st.markdown(
    """
    <div class="header-banner">
        <h1>✍️ Stress Level Detection from Handwriting</h1>
        <p>Upload or capture a handwriting sample to analyse stress indicators using machine learning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "⚠️ Experimental / educational project only. Not a medical or clinical diagnostic tool.",
    icon=None,
)

if bundle is None:
    st.error("Failed to load model. Please check that model files exist.")
    st.stop()


if page == "Predict":
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("Input")
        input_mode = st.radio("Input method", ["Upload Image", "Camera Capture"], horizontal=True)
        if input_mode == "Upload Image":
            image_file = st.file_uploader(
                "Choose a handwriting image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
            )
            source = "Upload"
        else:
            image_file = st.camera_input("Capture handwriting")
            source = "Camera"

        if image_file is not None:
            ok, msg = validate_uploaded_file(image_file)
            if not ok:
                st.error(msg)
                st.stop()
            display = Image.open(io.BytesIO(image_file.getvalue()))
            st.image(display, caption="Selected image", use_container_width=True)

    with right:
        st.subheader("Result")

        if image_file is None:
            st.info("Provide a handwriting image on the left to begin.")
        else:
            if st.button("Analyse Stress Level", type="primary", use_container_width=True):
                image = open_uploaded_image(image_file)
                if image is None:
                    st.error("Could not read image. Please upload a valid JPG or PNG.")
                    st.stop()

                raw = np.asarray(image, dtype=np.float32)
                checks = run_quality_checks(raw)

                if not checks["passed"]:
                    for w in checks["warnings"]:
                        st.warning(w)
                    st.stop()

                with st.spinner("Analysing…"):
                    result = predict_with_bundle(bundle, image, image_size=IMG_SIZE)

                label = result["predicted_label"]
                prob_map = result.get("probability_map", {})

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="label">Predicted Stress Level</div>
                        <div class="value">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                m = checks["metrics"]
                ink = m.get("ink_density_ratio", 0)
                blur = m.get("laplacian_variance", 0)
                res = m.get("resolution", {})
                st.markdown(
                    f"""
                    <div class="metric-row">
                        <div class="metric-box">
                            <div class="m-label">Resolution</div>
                            <div class="m-value">{res.get('width','?')}×{res.get('height','?')}</div>
                        </div>
                        <div class="metric-box">
                            <div class="m-label">Ink Density</div>
                            <div class="m-value">{ink:.1%}</div>
                        </div>
                        <div class="metric-box">
                            <div class="m-label">Sharpness</div>
                            <div class="m-value">{blur:.4f}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if prob_map:
                    st.markdown("**Class probabilities**")
                    st.bar_chart(prob_map, use_container_width=True)

                image_name = getattr(image_file, "name", "camera_capture.jpg")
                save_history(source, image_name, label)
                st.session_state["latest"] = {
                    "image": image,
                    "label": label,
                    "name": image_name,
                    "source": source,
                    "warnings": checks["warnings"],
                }

                report = build_report(source, image_name, label, checks["warnings"])
                st.download_button(
                    "⬇ Download Report",
                    data=report,
                    file_name="stress_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

    latest = st.session_state.get("latest")
    if latest:
        st.divider()
        with st.expander("📝 Submit a correction for review", expanded=False):
            st.caption(
                "If the prediction seems wrong, select the correct label and submit. "
                "Your sample will be reviewed before it can be used for retraining."
            )
            keys = list(STRESS_OPTIONS.keys())
            default_idx = keys.index(latest["label"]) if latest["label"] in keys else 0
            actual = st.selectbox("Correct stress level", keys, index=default_idx)
            if st.button("Submit for Admin Review", use_container_width=True):
                rid = queue_for_review(
                    latest["image"], latest["source"], latest["name"],
                    latest["label"], actual,
                )
                st.success(f"Submitted for review (ID: `{rid}`)")


elif page == "History":
    st.subheader("Prediction History")

    if not HISTORY_PATH.exists():
        st.info("No predictions recorded yet.")
    else:
        with HISTORY_PATH.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            st.info("No predictions recorded yet.")
        else:
            st.caption(f"Showing last {min(len(rows), 20)} of {len(rows)} records.")
            st.dataframe(rows[-20:][::-1], use_container_width=True)
            st.download_button(
                "⬇ Download Full History CSV",
                data=HISTORY_PATH.read_text(encoding="utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv",
            )


elif page == "Admin":
    st.subheader("Admin Panel")

    admin_password = os.getenv("STRESS_APP_ADMIN_PASSWORD", "")
    if not admin_password:
        st.info(
            "Admin access is disabled. Set the `STRESS_APP_ADMIN_PASSWORD` "
            "environment variable to enable it.",
            icon="🔒",
        )
        st.stop()

    entered = st.text_input("Admin password", type="password", placeholder="Enter password…")
    if entered != admin_password:
        if entered:
            st.error("Incorrect password.")
        st.stop()

    st.success("Authenticated.", icon="✅")
    st.divider()

    pending = list_pending()
    approved_imgs, _ = load_approved_samples()
    col1, col2 = st.columns(2)
    col1.metric("Pending samples", len(pending))
    col2.metric("Approved samples", len(approved_imgs))

    st.divider()
    st.subheader("Pending Review")

    if not pending:
        st.success("No pending samples.")
    else:
        for path in pending:
            record = joblib.load(path)
            img = Image.fromarray((record["image_array"] * 255).astype(np.uint8))
            with st.container(border=True):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.image(img, width=180)
                with c2:
                    st.markdown(f"**{record['image_name']}**")
                    st.caption(
                        f"Submitted: {record['created_at'][:19]}  \n"
                        f"Predicted: **{record['predicted_label']}**  \n"
                        f"Reviewer label: **{record['actual_label']}**"
                    )
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("✅ Approve", key=f"ap_{record['id']}", use_container_width=True):
                            approve_record(path)
                            st.rerun()
                    with b2:
                        if st.button("❌ Reject", key=f"rj_{record['id']}", use_container_width=True):
                            reject_record(path)
                            st.rerun()

    st.divider()
    st.subheader("Retrain Model")
    st.caption(
        "Retrains using the base dataset plus all approved samples. "
        "The new model is promoted only if its accuracy matches or exceeds the current one."
    )

    feat_mode = st.selectbox("Feature mode", ["hog", "raw_pixels"], index=0)

    if st.button("🔄 Retrain & Promote", type="primary", use_container_width=True):
        current_acc = None
        if METADATA_PATH.exists():
            current_acc = json.loads(METADATA_PATH.read_text(encoding="utf-8")).get(
                "cross_validated_accuracy"
            )

        with st.spinner("Training… this may take a moment."):
            result = train_model(feature_mode=feat_mode, promote=True)

        load_model.clear()
        st.success(f"Retraining complete — version `{result['version']}`")

        before = f"{current_acc:.2%}" if current_acc is not None else "N/A"
        after = f"{result['evaluated_accuracy']:.2%}"
        diff = result["evaluated_accuracy"] - (current_acc or 0.0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Before", before)
        col2.metric("After", after)
        col3.metric("Change", f"{diff:+.2%}")
        st.info(f"Promoted to active: **{result['promoted']}**")
