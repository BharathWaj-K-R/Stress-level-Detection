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


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "stress_model.pkl"
LABEL_MAP_PATH = MODEL_DIR / "label_mapping.json"
METADATA_PATH = MODEL_DIR / "metadata.json"
IMG_SIZE = 128
APP_VERSION = "2.0.0"
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


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f7f7f5;
            --surface: #ffffff;
            --surface-soft: #f0f0ed;
            --border: #d7d7d2;
            --text: #121212;
            --muted: #686864;
            --black: #000000;
        }
        .stApp { background: var(--bg); color: var(--text); }
        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--border); }
        [data-testid="stSidebar"] * { color: var(--text) !important; }
        .shell { border-bottom: 1px solid var(--border); padding-bottom: 1.35rem; margin-bottom: 1.5rem; }
        .eyebrow { font-size: .68rem; letter-spacing: .16em; text-transform: uppercase; color: var(--muted); }
        .title { margin: .25rem 0 0; font-size: clamp(2.15rem, 4vw, 3.4rem); line-height: .98; letter-spacing: -.055em; font-weight: 850; color: var(--black); }
        .subtitle { max-width: 760px; margin-top: .75rem; color: var(--muted); line-height: 1.6; font-size: .95rem; }
        .status { margin-top: .85rem; color: var(--muted); font-size: .78rem; }
        .status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #111; margin-right: .45rem; }
        .surface { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1.15rem; }
        .hero-result { background: #111111; color: #fff; border-radius: 14px; padding: 1.45rem; }
        .result-label { color: #a8a8a3; font-size: .68rem; letter-spacing: .14em; text-transform: uppercase; }
        .result-value { margin-top: .3rem; font-size: clamp(2.1rem, 4vw, 3.25rem); font-weight: 850; letter-spacing: -.055em; line-height: 1; }
        .result-meta { color: #d2d2cd; font-size: .82rem; margin-top: .8rem; line-height: 1.55; }
        .metric-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: .9rem; min-height: 92px; }
        .metric-label { color: var(--muted); font-size: .67rem; text-transform: uppercase; letter-spacing: .1em; }
        .metric-value { color: var(--black); font-size: 1.26rem; font-weight: 780; margin-top: .25rem; }
        .mini { color: var(--muted); font-size: .8rem; line-height: 1.55; }
        .step { display: flex; gap: .8rem; padding: .8rem 0; border-bottom: 1px solid var(--border); }
        .step:last-child { border-bottom: 0; }
        .step-num { flex: 0 0 25px; width: 25px; height: 25px; border: 1px solid var(--border); border-radius: 50%; display:flex; align-items:center; justify-content:center; font-size:.7rem; font-weight:750; }
        .footer { border-top: 1px solid var(--border); margin-top: 2.5rem; padding-top: 1rem; color: #777771; font-size: .7rem; line-height:1.55; }
        .stButton > button, .stDownloadButton > button { background:#111 !important; color:#fff !important; border:1px solid #111 !important; border-radius:9px !important; font-weight:750 !important; min-height:2.55rem; }
        .stButton > button:hover, .stDownloadButton > button:hover { background:#383836 !important; border-color:#383836 !important; }
        div[data-testid="stFileUploader"] { background:#fff; border:1px dashed #bcbcb7; border-radius:12px; padding:.35rem; }
        .stTabs [data-baseweb="tab"] { color: var(--muted); }
        @media (max-width: 720px) { .title { font-size:2.25rem; } .surface, .hero-result { padding:1rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("feedback", [])
    st.session_state.setdefault("latest_result", None)


def render_header(bundle: dict) -> None:
    metadata = bundle.get("metadata", {})
    version = metadata.get("semantic_version") or metadata.get("version", "legacy")
    mode = metadata.get("feature_extraction_method", "unknown")
    st.markdown(
        f"""
        <div class="shell">
            <div class="eyebrow">HANDWRITING ANALYSIS / EXPERIMENTAL SYSTEM</div>
            <h1 class="title">Stress Classification Lab</h1>
            <div class="subtitle">
                A privacy-conscious browser application for experimental classification of handwriting
                samples. There are no user accounts, no sign-in flow, and no database.
            </div>
            <div class="status"><span class="status-dot"></span>Model {version} · {mode} · application {APP_VERSION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(bundle: dict) -> str:
    metadata = bundle.get("metadata", {})
    with st.sidebar:
        st.markdown("### Stress Classification Lab")
        st.caption("Monochrome light edition")
        st.divider()
        page = st.radio(
            "Workspace",
            ["Analyze", "Session history", "Methodology", "System"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Local-session architecture**")
        st.caption(
            "Analysis results exist only in Streamlit session state. Export them when you need a copy."
        )
        st.divider()
        st.markdown(
            f"**Model**  \n`{metadata.get('semantic_version') or metadata.get('version', 'legacy')}`"
        )
        st.markdown(
            f"**Feature mode**  \n`{metadata.get('feature_extraction_method', 'unknown')}`"
        )
    return page


def quality_cards(checks: dict) -> None:
    metrics = checks.get("metrics", {})
    resolution = metrics.get("resolution", {})
    items = [
        ("Resolution", f"{resolution.get('width', '?')} × {resolution.get('height', '?')}"),
        ("Ink density", f"{metrics.get('ink_density_ratio', 0.0):.1%}"),
        ("Sharpness", f"{metrics.get('laplacian_variance', 0.0):.4f}"),
    ]
    columns = st.columns(3)
    for column, (label, value) in zip(columns, items):
        column.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )


def build_record(bundle: dict, result: dict, checks: dict, source: str, filename: str) -> dict:
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
        f"  Ink density      : {record['ink_density_ratio']:.2%}\n"
        f"  Laplacian metric : {record['laplacian_variance']:.6f}\n\n"
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


def analyze_page(bundle: dict) -> None:
    st.markdown("## Analyze")
    st.caption("Upload or capture a clear handwriting sample. Accepted formats: JPG, JPEG, PNG. Maximum: 5 MB.")

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        with st.container(border=True):
            method = st.radio("Input source", ["Upload image", "Camera"], horizontal=True)
            if method == "Upload image":
                uploaded = st.file_uploader(
                    "Choose handwriting image",
                    type=["jpg", "jpeg", "png"],
                    accept_multiple_files=False,
                )
            else:
                uploaded = st.camera_input("Capture handwriting")

            if uploaded is not None:
                valid, message = validate_uploaded_file(uploaded)
                if not valid:
                    st.error(message)
                    return
                preview = Image.open(io.BytesIO(uploaded.getvalue()))
                st.image(preview, caption="Input sample", use_container_width=True)

    with right:
        with st.container(border=True):
            st.markdown("### Quality gate")
            if uploaded is None:
                st.markdown(
                    "<div class='mini'>Provide a sample to validate resolution, sharpness, and visible ink before model inference.</div>",
                    unsafe_allow_html=True,
                )
                return

            image = open_uploaded_image(uploaded)
            if image is None:
                st.error("The image could not be decoded. Upload a valid JPG, JPEG, or PNG file.")
                return

            checks = run_quality_checks(np.asarray(image, dtype=np.float32))
            quality_cards(checks)
            if not checks["passed"]:
                st.warning("The quality gate rejected this sample.")
                for item in checks["warnings"]:
                    st.write(f"• {item}")
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
        if latest:
            st.markdown("### Result")
            st.markdown(
                f"""
                <div class="hero-result">
                    <div class="result-label">Predicted class</div>
                    <div class="result-value">{latest['prediction']}</div>
                    <div class="result-meta">Confidence estimate: <strong>{latest['confidence']:.1%}</strong><br>
                    Experimental classification only. Not a clinical measurement.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("#### Class distribution")
            if latest.get("probability_map"):
                st.bar_chart(latest["probability_map"], height=220)
            st.download_button(
                "Download analysis report",
                data=build_report(latest),
                file_name="stress-classification-report.txt",
                mime="text/plain",
                use_container_width=True,
            )

            with st.expander("Add a correction note"):
                proposed = st.selectbox("Your proposed label", list(STRESS_LABELS))
                note = st.text_area("Optional note", placeholder="Describe why the classification appears incorrect.")
                if st.button("Save to session review queue"):
                    st.session_state["feedback"].append(
                        {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "filename": latest["filename"],
                            "prediction": latest["prediction"],
                            "proposed_label": proposed,
                            "note": note.strip(),
                        }
                    )
                    st.success("Correction saved for this session.")


def history_page() -> None:
    st.markdown("## Session history")
    rows = st.session_state.get("history", [])
    feedback = st.session_state.get("feedback", [])
    if not rows:
        st.info("No analyses have been run in this browser session.")
        return

    avg_conf = sum(row["confidence"] for row in rows) / len(rows)
    counts = {label: sum(row["prediction"] == label for row in rows) for label in STRESS_LABELS}
    c1, c2, c3, c4 = st.columns(4)
    for column, label, value in [
        (c1, "Analyses", len(rows)),
        (c2, "Avg confidence", f"{avg_conf:.1%}"),
        (c3, "High stress", counts["High Stress"]),
        (c4, "Review items", len(feedback)),
    ]:
        column.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Prediction distribution")
    st.bar_chart(counts, height=230)

    st.markdown("### Analysis log")
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
        st.markdown("### Review queue")
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
    st.markdown("## Methodology")
    st.markdown(
        """
        <div class="surface">
            <div class="step"><div class="step-num">1</div><div><strong>Capture</strong><br><span class="mini">A handwriting sample enters through upload or camera capture.</span></div></div>
            <div class="step"><div class="step-num">2</div><div><strong>Quality gate</strong><br><span class="mini">Resolution, sharpness and visible ink are checked before inference.</span></div></div>
            <div class="step"><div class="step-num">3</div><div><strong>Preprocess</strong><br><span class="mini">The image is normalized to the representation expected by the packaged model.</span></div></div>
            <div class="step"><div class="step-num">4</div><div><strong>Classify</strong><br><span class="mini">The trained Random Forest produces one of the stored stress-related labels.</span></div></div>
            <div class="step"><div class="step-num">5</div><div><strong>Export</strong><br><span class="mini">Results and session history can be downloaded locally without server-side storage.</span></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows = [
        {"Property": "Model version", "Value": metadata.get("semantic_version") or metadata.get("version", "unknown")},
        {"Property": "Feature mode", "Value": metadata.get("feature_extraction_method", "unknown")},
        {"Property": "Image size", "Value": metadata.get("image_size", IMG_SIZE)},
        {"Property": "Training set size", "Value": metadata.get("training_set_size", "not recorded")},
        {"Property": "Cross-validated accuracy", "Value": metadata.get("cross_validated_accuracy", "not recorded")},
        {"Property": "Evaluated accuracy", "Value": metadata.get("evaluated_accuracy", "not recorded")},
    ]
    st.markdown("### Packaged model metadata")
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.warning(
        "The repository's dataset is very small. This product therefore presents predictions as experimental "
        "classification outputs rather than validated measurements of psychological stress."
    )


def system_page(bundle: dict) -> None:
    model = bundle.get("model")
    metadata = bundle.get("metadata", {})
    st.markdown("## System")
    left, right = st.columns(2)
    with left:
        st.markdown("### Deployment contract")
        st.code(
            "streamlit run app.py --server.address 0.0.0.0 --server.port $PORT",
            language="bash",
        )
        st.markdown(
            "<div class='mini'>For Streamlit Community Cloud, select this repository, branch, and <code>app.py</code> entrypoint. The app reads its packaged model directly from the repository.</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("### Runtime checks")
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
    inject_css()
    try:
        bundle = load_model()
    except Exception as exc:
        st.error("The packaged model could not be loaded safely.")
        st.code(str(exc), language="text")
        st.stop()

    render_header(bundle)
    page = render_sidebar(bundle)

    if page == "Analyze":
        analyze_page(bundle)
    elif page == "Session history":
        history_page()
    elif page == "Methodology":
        methodology_page(bundle)
    else:
        system_page(bundle)

    st.markdown(
        "<div class='footer'>Experimental software project. Outputs are not medical diagnoses or validated measures of psychological stress. Do not use them for high-impact decisions.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
