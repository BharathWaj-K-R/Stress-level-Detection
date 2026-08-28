from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from inference.predictor import (
    extract_features_for_bundle,
    load_model_bundle,
    open_uploaded_image,
    predict_with_bundle,
    validate_uploaded_file,
)
from preprocessing.quality_check import run_quality_checks
from ui import apply_theme, card, footer, metric_card, page_header, result_card, section_heading, step_card


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "stress_model.pkl"
LABEL_MAP_PATH = MODEL_DIR / "label_mapping.json"
METADATA_PATH = MODEL_DIR / "metadata.json"
IMG_SIZE = 128
APP_VERSION = "2.1.0"
STRESS_LABELS = ("Low Stress", "Medium Stress", "High Stress")

st.set_page_config(
    page_title="Stress Classification Lab",
    page_icon="◌",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_model() -> dict:
    bundle = load_model_bundle(
        MODEL_PATH,
        label_map_path=LABEL_MAP_PATH,
        metadata_path=METADATA_PATH,
    )
    validate_model_bundle(bundle)
    return bundle


def validate_model_bundle(bundle: dict) -> None:
    model = bundle.get("model")
    if model is None or not hasattr(model, "predict"):
        raise RuntimeError("The packaged model does not expose a prediction interface.")

    metadata = bundle.get("metadata", {})
    configured_size = int(metadata.get("image_size", IMG_SIZE))
    if configured_size != IMG_SIZE:
        raise RuntimeError(
            f"Model image size {configured_size} does not match app image size {IMG_SIZE}."
        )

    if hasattr(model, "n_features_in_"):
        sample = Image.fromarray(np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8))
        produced = int(
            extract_features_for_bundle(bundle, sample, image_size=IMG_SIZE).shape[1]
        )
        expected = int(model.n_features_in_)
        if produced != expected:
            raise RuntimeError(
                f"Model feature mismatch: artifact expects {expected}, pipeline produces {produced}."
            )


def init_session() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("feedback", [])
    st.session_state.setdefault("latest_result", None)


def render_sidebar(bundle: dict) -> str:
    metadata = bundle.get("metadata", {})
    version = metadata.get("semantic_version") or metadata.get("version", "legacy")
    mode = metadata.get("feature_extraction_method", "unknown")

    with st.sidebar:
        st.markdown("## Stress Classification Lab")
        st.caption("Monochrome / light / session-first")
        st.divider()

        page = st.radio(
            "Workspace",
            ["Analyze", "Session history", "Methodology", "System"],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("No sign-in · No database · No external inference API")
        st.markdown(f"**Model**  \n`{version}`")
        st.markdown(f"**Features**  \n`{mode}`")
        st.divider()
        st.caption("Results live only in the current Streamlit session.")

    return page


def build_record(
    bundle: dict,
    result: dict,
    checks: dict,
    source: str,
    filename: str,
) -> dict:
    probabilities = result.get("probability_map") or {}
    confidence = max(probabilities.values()) if probabilities else 0.0
    metadata = bundle.get("metadata", {})
    metrics = checks.get("metrics", {})
    resolution = metrics.get("resolution", {})
    version = metadata.get("semantic_version") or metadata.get("version", "legacy")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "filename": filename,
        "prediction": result["predicted_label"],
        "confidence": float(confidence),
        "probability_map": {k: float(v) for k, v in probabilities.items()},
        "model_version": version,
        "feature_mode": metadata.get("feature_extraction_method", "unknown"),
        "ink_density_ratio": float(metrics.get("ink_density_ratio", 0.0)),
        "laplacian_variance": float(metrics.get("laplacian_variance", 0.0)),
        "resolution": f"{resolution.get('width', '?')} × {resolution.get('height', '?')}",
    }


def build_report(record: dict) -> str:
    probabilities = record.get("probability_map") or {}
    probability_text = "\n".join(
        f"  {label}: {value:.2%}" for label, value in probabilities.items()
    ) or "  Not available"

    return (
        "STRESS CLASSIFICATION LAB\n"
        "=========================\n"
        f"Generated (UTC) : {record['timestamp_utc']}\n"
        f"Source          : {record['source']}\n"
        f"File            : {record['filename']}\n"
        f"Prediction      : {record['prediction']}\n"
        f"Confidence      : {record['confidence']:.2%}\n"
        f"Model version   : {record['model_version']}\n"
        f"Feature mode    : {record['feature_mode']}\n\n"
        "CLASS DISTRIBUTION\n"
        f"{probability_text}\n\n"
        "INPUT QUALITY\n"
        f"  Resolution        : {record['resolution']}\n"
        f"  Ink density       : {record['ink_density_ratio']:.2%}\n"
        f"  Laplacian metric  : {record['laplacian_variance']:.6f}\n\n"
        "LIMITATION\n"
        "This is an experimental classification result. It is not a measurement, diagnosis, "
        "or clinical assessment of psychological stress. Do not use it for health, employment, "
        "academic, safety, or other high-impact decisions.\n"
    )


def history_csv(rows: list[dict]) -> str:
    output = io.StringIO()
    fields = [
        "timestamp_utc",
        "source",
        "filename",
        "prediction",
        "confidence",
        "ink_density_ratio",
        "laplacian_variance",
        "resolution",
        "model_version",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return output.getvalue()


def render_quality_cards(checks: dict) -> None:
    metrics = checks.get("metrics", {})
    resolution = metrics.get("resolution", {})
    columns = st.columns(3, gap="small")
    values = [
        ("Resolution", f"{resolution.get('width', '?')} × {resolution.get('height', '?')}"),
        ("Ink density", f"{metrics.get('ink_density_ratio', 0.0):.1%}"),
        ("Sharpness", f"{metrics.get('laplacian_variance', 0.0):.4f}"),
    ]
    for column, (label, value) in zip(columns, values):
        with column:
            metric_card(label, value)


def analyze_page(bundle: dict) -> None:
    section_heading(
        "Analyze",
        "Run one experimental classification from a handwriting image."
    )

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        with card():
            st.markdown("**Input sample**")
            st.caption("Use a clear, well-lit handwriting image with visible ink.")
            method = st.radio(
                "Input source",
                ["Upload image", "Camera"],
                horizontal=True,
                label_visibility="collapsed",
            )

            if method == "Upload image":
                uploaded = st.file_uploader(
                    "Choose handwriting image",
                    type=["jpg", "jpeg", "png"],
                    accept_multiple_files=False,
                    help="JPG, JPEG, or PNG. Maximum 5 MB.",
                )
            else:
                uploaded = st.camera_input("Capture handwriting")

            if uploaded is not None:
                valid, message = validate_uploaded_file(uploaded)
                if not valid:
                    st.error(message)
                    return

                preview = Image.open(io.BytesIO(uploaded.getvalue()))
                st.image(preview, caption="Input preview", use_container_width=True)

    with right:
        with card():
            st.markdown("**Quality gate**")
            st.caption("The sample must pass basic image checks before model inference.")

            if uploaded is None:
                st.markdown(
                    '<div class="ui-card-description">Your sample will be checked for resolution, sharpness, and visible ink here.</div>',
                    unsafe_allow_html=True,
                )
                return

            image = open_uploaded_image(uploaded)
            if image is None:
                st.error("The image could not be decoded. Upload a valid JPG, JPEG, or PNG file.")
                return

            checks = run_quality_checks(np.asarray(image, dtype=np.float32))
            render_quality_cards(checks)

            if not checks["passed"]:
                st.warning("Quality gate rejected this sample.")
                for warning in checks["warnings"]:
                    st.write(f"• {warning}")
                return

            st.success("Quality gate passed.")
            if st.button("Analyze sample", type="primary", use_container_width=True):
                with st.spinner("Extracting features and evaluating the model…"):
                    result = predict_with_bundle(bundle, image, image_size=IMG_SIZE)

                filename = getattr(uploaded, "name", "camera_capture.jpg")
                source = "Camera" if method == "Camera" else "Upload"
                record = build_record(bundle, result, checks, source, filename)
                st.session_state["latest_result"] = record
                st.session_state["history"].append(record)

    latest = st.session_state.get("latest_result")
    if not latest:
        return

    st.divider()
    section_heading("Latest result", "The result below describes the packaged model output, not a clinical assessment.")
    result_left, result_right = st.columns([0.95, 1.05], gap="large")

    with result_left:
        result_card(
            "Predicted class",
            latest["prediction"],
            f"Confidence estimate: {latest['confidence']:.1%} · Model {latest['model_version']}"
        )
        st.write("")
        st.download_button(
            "Download analysis report",
            data=build_report(latest),
            file_name="stress-classification-report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with result_right:
        with card():
            section_heading("Class distribution")
            if latest.get("probability_map"):
                st.bar_chart(latest["probability_map"], height=230)
            else:
                st.caption("Probability estimates are unavailable for this model artifact.")

    with card():
        section_heading("Correction note", "Corrections are stored only in this current session and are not sent to a database.")
        proposed = st.selectbox("Your proposed label", list(STRESS_LABELS))
        note = st.text_area(
            "Optional note",
            placeholder="Describe why this classification appears incorrect.",
            height=90,
        )
        if st.button("Add to session review queue"):
            st.session_state["feedback"].append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "filename": latest["filename"],
                    "prediction": latest["prediction"],
                    "proposed_label": proposed,
                    "note": note.strip(),
                }
            )
            st.success("Correction saved to this session.")


def history_page() -> None:
    rows = st.session_state.get("history", [])
    feedback = st.session_state.get("feedback", [])

    section_heading("Session history", "Nothing here is stored in a database. This is the current browser session only.")
    if not rows:
        with card():
            st.info("No analyses have been run in this session yet.")
        return

    average_confidence = sum(row["confidence"] for row in rows) / len(rows)
    counts = {label: sum(row["prediction"] == label for row in rows) for label in STRESS_LABELS}

    columns = st.columns(4, gap="small")
    summary = [
        ("Analyses", len(rows)),
        ("Avg confidence", f"{average_confidence:.1%}"),
        ("High stress", counts["High Stress"]),
        ("Review items", len(feedback)),
    ]
    for column, (label, value) in zip(columns, summary):
        with column:
            metric_card(label, value)

    st.write("")
    with card():
        section_heading("Prediction distribution")
        st.bar_chart(counts, height=230)

    with card():
        section_heading("Analysis log")
        table = [
            {
                "Time (UTC)": row["timestamp_utc"],
                "Prediction": row["prediction"],
                "Confidence": f"{row['confidence']:.1%}",
                "Source": row["source"],
                "File": row["filename"],
                "Model": row["model_version"],
            }
            for row in reversed(rows)
        ]
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button(
            "Download session CSV",
            data=history_csv(rows),
            file_name="stress-session-history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if feedback:
        with card():
            section_heading("Review queue")
            st.dataframe(feedback, use_container_width=True, hide_index=True)
            st.download_button(
                "Download review queue JSON",
                data=json.dumps(feedback, indent=2),
                file_name="stress-review-queue.json",
                mime="application/json",
                use_container_width=True,
            )


def methodology_page(bundle: dict) -> None:
    metadata = bundle.get("metadata", {})
    section_heading("Methodology", "A transparent view of the current inference pipeline.")

    with card():
        step_card(1, "Capture", "Receive handwriting through upload or browser camera capture.")
        step_card(2, "Quality gate", "Check image resolution, sharpness, and visible ink before inference.")
        step_card(3, "Preprocess", "Convert the input into the normalized representation expected by the packaged model.")
        step_card(4, "Classify", "Run the stored Random Forest classifier against the extracted feature vector.")
        step_card(5, "Export", "Download the result or session records without server-side persistence.")

    st.write("")
    with card():
        section_heading("Packaged model")
        rows = [
            {"Property": "Model version", "Value": metadata.get("semantic_version") or metadata.get("version", "unknown")},
            {"Property": "Feature mode", "Value": metadata.get("feature_extraction_method", "unknown")},
            {"Property": "Image size", "Value": metadata.get("image_size", IMG_SIZE)},
            {"Property": "Training set size", "Value": metadata.get("training_set_size", "not recorded")},
            {"Property": "Cross-validated accuracy", "Value": metadata.get("cross_validated_accuracy", "not recorded")},
            {"Property": "Evaluated accuracy", "Value": metadata.get("evaluated_accuracy", "not recorded")},
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.warning(
        "The current dataset is small. Model output is experimental and should not be treated as a validated measurement of psychological stress."
    )


def system_page(bundle: dict) -> None:
    model = bundle.get("model")
    metadata = bundle.get("metadata", {})
    section_heading("System", "Runtime information and the deployment contract used by the application.")

    left, right = st.columns(2, gap="large")
    with left:
        with card():
            section_heading("Deployment")
            st.code("streamlit run app.py", language="bash")
            st.caption(
                "The application is designed for Streamlit Community Cloud with app.py as the entrypoint. "
                "The packaged model is read from the repository."
            )
            st.markdown("**Architecture guarantees**")
            for item in [
                "No authentication or sign-in",
                "No application database",
                "No external inference service",
                "Session-only history",
                "In-memory exports",
            ]:
                st.write(f"• {item}")

    with right:
        with card():
            section_heading("Runtime checks")
            rows = [
                {"Check": "Model file", "Status": "READY" if MODEL_PATH.exists() else "MISSING"},
                {"Check": "Label mapping", "Status": "READY" if LABEL_MAP_PATH.exists() else "FALLBACK"},
                {"Check": "Prediction interface", "Status": "READY" if hasattr(model, "predict") else "FAILED"},
                {"Check": "Feature count", "Status": str(getattr(model, "n_features_in_", "unknown"))},
                {"Check": "Metadata", "Status": "PRESENT" if metadata else "FALLBACK"},
                {"Check": "Persistence", "Status": "SESSION ONLY"},
                {"Check": "Authentication", "Status": "NOT USED"},
                {"Check": "Database", "Status": "NOT USED"},
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    init_session()
    apply_theme()

    try:
        bundle = load_model()
    except Exception as exc:
        st.error("The packaged model could not be loaded safely.")
        st.code(str(exc), language="text")
        st.stop()

    metadata = bundle.get("metadata", {})
    version = metadata.get("semantic_version") or metadata.get("version", "legacy")
    mode = metadata.get("feature_extraction_method", "unknown")

    page_header(
        "HANDWRITING ANALYSIS / EXPERIMENTAL SYSTEM",
        "Stress Classification Lab",
        "A clean, session-first interface for experimental handwriting classification. No account, database, or external inference service is required.",
        status=f"Model {version} · {mode} · application {APP_VERSION}",
    )

    page = render_sidebar(bundle)

    if page == "Analyze":
        analyze_page(bundle)
    elif page == "Session history":
        history_page()
    elif page == "Methodology":
        methodology_page(bundle)
    else:
        system_page(bundle)

    footer(
        "Experimental software project. Outputs are not medical diagnoses or validated measures of psychological stress. "
        "Do not use them for health, employment, academic, safety, or other high-impact decisions."
    )


if __name__ == "__main__":
    main()
