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
from ui import (
    FAVICON_PATH,
    apply_theme,
    brand_block,
    card,
    empty_result_card,
    footer,
    hero_header,
    metric_card,
    result_card,
    section_heading,
    step_card,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "stress_model.pkl"
LABEL_MAP_PATH = MODEL_DIR / "label_mapping.json"
METADATA_PATH = MODEL_DIR / "metadata.json"
IMG_SIZE = 128
MAX_UPLOAD_MB = 5
APP_VERSION = "2.3.1"
STRESS_LABELS = ("Low Stress", "Medium Stress", "High Stress")

st.set_page_config(
    page_title="Stress Classification Lab",
    page_icon=str(FAVICON_PATH),
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
    model = bundle.get("model")

    with st.sidebar:
        brand_block()
        st.divider()

        page = st.radio(
            "Workspace",
            ["Analyze", "Session history", "Methodology", "System"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**System Status**")
        st.markdown(
            '<span class="ui-status"><span class="ui-dot"></span>Ready</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Model**  \n`{version}`")
        st.markdown(f"**Features**  \n`{mode}`")
        st.markdown(
            f"**Inference**  \n`{'READY' if hasattr(model, 'predict') else 'FAILED'}`"
        )

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
        "probability_map": {key: float(value) for key, value in probabilities.items()},
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
    values = [
        ("Resolution", f"{resolution.get('width', '?')} × {resolution.get('height', '?')}"),
        ("Ink density", f"{metrics.get('ink_density_ratio', 0.0):.1%}"),
        ("Sharpness", f"{metrics.get('laplacian_variance', 0.0):.4f}"),
    ]
    columns = st.columns(3, gap="small")
    for column, (label, value) in zip(columns, values):
        with column:
            metric_card(label, value)


def latest_result_panel() -> None:
    latest = st.session_state.get("latest_result")
    with card():
        section_heading("Latest Result")
        if not latest:
            empty_result_card(
                "Upload an image to see results",
                "The analysis will appear here with classification and a confidence estimate.",
            )
            return

        result_card(
            "Predicted class",
            latest["prediction"],
            f"Confidence estimate: <strong>{latest['confidence']:.1%}</strong> · Model {latest['model_version']}",
        )
        st.download_button(
            "Download analysis report",
            data=build_report(latest),
            file_name="stress-classification-report.txt",
            mime="text/plain",
            use_container_width=True,
        )

        if latest.get("probability_map"):
            st.markdown("**Class distribution**")
            st.bar_chart(latest["probability_map"], height=190)


def quick_stats() -> None:
    rows = st.session_state.get("history", [])
    counts = {label: sum(row["prediction"] == label for row in rows) for label in STRESS_LABELS}
    avg_conf = sum(row["confidence"] for row in rows) / len(rows) if rows else None

    with card():
        st.markdown(
            '<div class="ui-section-title">Quick Stats <span class="ui-muted">(This Session)</span></div>',
            unsafe_allow_html=True,
        )
        values = [
            ("Samples analyzed", str(len(rows))),
            ("Avg. confidence", f"{avg_conf:.1%}" if avg_conf is not None else "-"),
            ("High stress", str(counts["High Stress"]) if rows else "-"),
            ("Low stress", str(counts["Low Stress"]) if rows else "-"),
        ]
        columns = st.columns(4, gap="small")
        for column, (label, value) in zip(columns, values):
            with column:
                metric_card(label, value)


def analyze_page(bundle: dict) -> None:
    with card():
        section_heading(
            "Upload Handwriting Sample",
            "Provide a clear handwriting sample, review the preview, then run the analysis.",
        )

        input_col, preview_col = st.columns([1, 1], gap="large")

        with input_col:
            st.markdown("**1 · Input**")
            method = st.radio(
                "Input source",
                ["Upload image", "Camera"],
                horizontal=True,
                label_visibility="collapsed",
                key="analysis_input_method",
            )

            if method == "Upload image":
                uploaded = st.file_uploader(
                    "Choose handwriting image",
                    type=["jpg", "jpeg", "png"],
                    accept_multiple_files=False,
                    help=f"JPG, JPEG, or PNG. Maximum {MAX_UPLOAD_MB} MB.",
                    key="handwriting_upload",
                )
            else:
                uploaded = st.camera_input(
                    "Capture handwriting",
                    key="handwriting_camera",
                )

            st.caption("Use high-contrast handwriting with good lighting and enough visible writing.")

        upload_valid = False
        upload_message = ""
        image = None
        checks = None

        if uploaded is not None:
            upload_valid, upload_message = validate_uploaded_file(uploaded)
            if upload_valid:
                image = open_uploaded_image(uploaded)
                if image is None:
                    upload_valid = False
                    upload_message = "The image could not be decoded. Upload a valid image."
                else:
                    checks = run_quality_checks(np.asarray(image, dtype=np.float32))

        with preview_col:
            st.markdown("**2 · Preview**")
            if uploaded is None:
                empty_result_card(
                    "Preview area",
                    "Your handwriting sample will appear here after you select or capture an image.",
                )
            elif not upload_valid:
                st.error(upload_message)
            else:
                st.image(image, use_container_width=True)

        if uploaded is None:
            st.markdown("**Tips for best results**")
            tips = st.columns(4, gap="medium")
            for column, tip in zip(
                tips,
                [
                    "Clear, high-contrast writing",
                    "Good, even lighting",
                    "Avoid motion blur",
                    "Include enough writing",
                ],
            ):
                with column:
                    st.markdown(f"✓ {tip}")
            return

        if not upload_valid:
            return

        st.markdown("**3 · Quality Check**")
        render_quality_cards(checks)

        if not checks["passed"]:
            st.warning("The sample did not pass the image quality gate.")
            for warning in checks["warnings"]:
                st.write(f"• {warning}")
            return

        analyze_col, info_col = st.columns([1, 1], gap="large")
        with analyze_col:
            st.markdown("**4 · Analyze**")
            if st.button(
                "✦  Analyze Sample",
                type="primary",
                use_container_width=True,
                key="analyze_sample",
            ):
                with st.spinner("Extracting features and evaluating the model…"):
                    result = predict_with_bundle(bundle, image, image_size=IMG_SIZE)
                filename = getattr(uploaded, "name", "camera_capture.jpg")
                source = "Camera" if method == "Camera" else "Upload"
                record = build_record(bundle, result, checks, source, filename)
                st.session_state["latest_result"] = record
                st.session_state["history"].append(record)
                st.rerun()

        with info_col:
            st.markdown("**Workflow**")
            st.caption("Input → Preview → Quality Check → Analyze → Results")

    results_col, stats_col = st.columns([1, 1], gap="large")
    with results_col:
        latest_result_panel()
    with stats_col:
        quick_stats()

    with card():
        section_heading(
            "Session Notes",
            "Add an optional correction note to the current analysis session.",
        )
        latest = st.session_state.get("latest_result")
        if latest:
            proposed = st.selectbox("Your proposed label", list(STRESS_LABELS))
            note = st.text_area(
                "Optional note",
                placeholder="Describe why this classification appears incorrect.",
                height=80,
            )
            if st.button("Add correction to this session", key="add_session_correction"):
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
        else:
            st.caption("Run an analysis to enable correction notes.")


def history_page() -> None:
    rows = st.session_state.get("history", [])
    feedback = st.session_state.get("feedback", [])
    section_heading(
        "Session History",
        "Temporary analysis analytics for the current session.",
    )

    if not rows:
        with card():
            empty_result_card(
                "No analyses yet",
                "Run your first handwriting analysis and the session log will appear here.",
            )
        return

    average_confidence = sum(row["confidence"] for row in rows) / len(rows)
    counts = {label: sum(row["prediction"] == label for row in rows) for label in STRESS_LABELS}

    columns = st.columns(4, gap="small")
    for column, (label, value) in zip(
        columns,
        [
            ("Analyses", len(rows)),
            ("Avg. confidence", f"{average_confidence:.1%}"),
            ("High stress", counts["High Stress"]),
            ("Review items", len(feedback)),
        ],
    ):
        with column:
            metric_card(label, value)

    st.write("")
    with card():
        section_heading("Prediction Distribution")
        st.bar_chart(counts, height=230)

    with card():
        section_heading("Analysis Log")
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
            section_heading("Review Queue")
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
    section_heading(
        "Methodology",
        "Understand exactly how a handwriting sample moves through the system.",
    )

    with card():
        step_card(1, "Capture", "Receive handwriting through upload or browser camera capture.")
        step_card(2, "Quality gate", "Check resolution, sharpness, and visible ink before inference.")
        step_card(3, "Preprocess", "Normalize the image without distorting its aspect ratio.")
        step_card(4, "Extract features", "Build the HOG representation and engineered handwriting descriptors.")
        step_card(5, "Classify", "Run the packaged Random Forest classifier against the feature vector.")
        step_card(6, "Export", "Download the result or session records for local use.")

    st.write("")
    with card():
        section_heading("Packaged Model")
        rows = [
            {
                "Property": "Model version",
                "Value": metadata.get("semantic_version") or metadata.get("version", "unknown"),
            },
            {
                "Property": "Feature mode",
                "Value": metadata.get("feature_extraction_method", "unknown"),
            },
            {"Property": "Image size", "Value": metadata.get("image_size", IMG_SIZE)},
            {
                "Property": "Training set size",
                "Value": metadata.get("training_set_size", "not recorded"),
            },
            {
                "Property": "Cross-validated accuracy",
                "Value": metadata.get("cross_validated_accuracy", "not recorded"),
            },
            {
                "Property": "Evaluated accuracy",
                "Value": metadata.get("evaluated_accuracy", "not recorded"),
            },
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.warning(
        "The current dataset is small. Model output is experimental and should not be treated as a validated measurement of psychological stress."
    )


def system_page(bundle: dict) -> None:
    model = bundle.get("model")
    metadata = bundle.get("metadata", {})
    section_heading(
        "System",
        "Runtime information and the deployment contract used by the application.",
    )

    left, right = st.columns(2, gap="large")
    with left:
        with card():
            st.markdown("### Deployment")
            st.code("streamlit run app.py", language="bash")
            st.caption(
                "Designed for Streamlit Community Cloud with app.py as the entrypoint. "
                "The packaged model is read directly from the repository."
            )
            st.markdown("**Runtime characteristics**")
            for item in [
                "Local model inference",
                "Session-based analysis state",
                "In-memory exports",
                "Packaged model artifacts",
            ]:
                st.markdown(f"✓ {item}")

    with right:
        with card():
            st.markdown("### Runtime checks")
            rows = [
                {"Check": "Model file", "Status": "READY" if MODEL_PATH.exists() else "MISSING"},
                {
                    "Check": "Label mapping",
                    "Status": "READY" if LABEL_MAP_PATH.exists() else "FALLBACK",
                },
                {
                    "Check": "Prediction interface",
                    "Status": "READY" if hasattr(model, "predict") else "FAILED",
                },
                {"Check": "Feature count", "Status": str(getattr(model, "n_features_in_", "unknown"))},
                {"Check": "Metadata", "Status": "PRESENT" if metadata else "FALLBACK"},
                {"Check": "Persistence", "Status": "SESSION"},
                {"Check": "Favicon", "Status": "READY" if FAVICON_PATH.exists() else "MISSING"},
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

    hero_header(
        "Stress Classification Lab",
        "AI-powered handwriting analysis for experimental stress classification.",
        "HANDWRITING ANALYSIS / EXPERIMENTAL SYSTEM",
        [
            "Local Inference",
            f"Model {version}",
            f"Features {mode}",
            "Session Analysis",
        ],
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
        f"Stress Classification Lab · application {APP_VERSION} · model {version} · features {mode}. "
        "Experimental software; outputs are not medical diagnoses or validated measures of psychological stress."
    )


if __name__ == "__main__":
    main()
