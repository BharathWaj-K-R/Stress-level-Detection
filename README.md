# Stress Classification Lab

A monochrome, light-themed Streamlit application for **experimental handwriting-based stress classification**.

> **Important:** This is an educational/research prototype, not a medical device, diagnostic tool, or validated measure of psychological stress. Do not use predictions for health, employment, academic, safety, or other high-impact decisions.

## Product goals

The redesigned product keeps the original machine-learning pipeline while making the application simpler, safer to deploy, and easier to understand.

- Upload or capture handwriting from the browser
- Validate input quality before inference
- Run the packaged Random Forest model
- Show class distribution and confidence estimate
- Keep analysis history in the current Streamlit session only
- Add correction notes to a session-only review queue
- Export reports, history, and review notes locally
- Show model/methodology information in the app
- Use a monochrome light design system
- Require **no sign-in and no database**
- Avoid application-owned persistent runtime data

## Architecture

```text
Browser
  |
  v
Streamlit UI
  |
  +--> image validation
  |
  +--> image quality gate
  |
  +--> preprocessing / feature extraction
  |
  +--> packaged Random Forest model
  |
  +--> result + confidence estimate
  |
  +--> Streamlit session state
  |
  +--> local downloads (TXT / CSV / JSON)
```

Training remains a developer-side workflow in `train_model.py`. It is intentionally not triggered from the public application UI.

## Repository layout

```text
.
├── app.py
├── train_model.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── inference/
│   └── predictor.py
├── preprocessing/
│   ├── feature_extraction.py
│   └── quality_check.py
├── model/
│   ├── stress_model.pkl
│   ├── label_mapping.json
│   └── preprocessed_data.pkl
└── tests/
    ├── test_feature_extraction.py
    ├── test_quality_check.py
    └── test_predictor.py
```

Runtime files such as prediction history, correction queues, and approved samples are **not required by the redesigned app** and are ignored by Git when created locally by legacy tooling.

## Run locally

Use Python 3.11 for the same environment used by CI.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the application from the repository root:

```bash
streamlit run app.py
```

Run tests:

```bash
pytest -q
```

Compile-check all Python sources:

```bash
python -m compileall -q .
```

## Streamlit Community Cloud

The repository is organized for Streamlit Community Cloud with `app.py` as the root entrypoint, `requirements.txt` at the repository root, and `.streamlit/config.toml` for application configuration.

1. Select this GitHub repository.
2. Select the branch you want to deploy.
3. Set the entrypoint to `app.py`.
4. Keep the deployment Python version aligned with the version used for development and CI. The current project CI uses Python 3.11.
5. Keep `requirements.txt` as the app's single dependency file.

The application needs no database, sign-in provider, external inference API, secrets, or writable server-side runtime store for normal inference.

### Deployment safety

The redesigned application avoids application-owned persistent runtime files. The UI stores its analysis history in Streamlit session state and creates export files in memory for direct download.

For an already-running Community Cloud app, keep the current deployed branch unchanged until you have validated this redesign on a separate app or deployment target. After validation, switch the production app's configured branch deliberately rather than changing the live branch underneath it.

Streamlit Community Cloud uses the repository as the source for deployed apps and automatically rebuilds environments when dependency changes are detected. Its documented project layout uses a root dependency file and `.streamlit/config.toml` for configuration.

## Model and data limitations

The current repository contains a very small base dataset. The application therefore uses explicit experimental language and does not present model output as a clinical measurement.

The current preprocessing pipeline uses grayscale normalization, optional HOG features, and three engineered handwriting descriptors:

- ink density ratio
- stroke-width variance estimate
- slant-angle estimate

These are image features. They should not be interpreted as direct measurements of psychological state.

For credible research validation, future work should use a substantially larger, participant-aware dataset with a documented labeling protocol and participant-independent evaluation.

## Design system

The product uses a monochrome light visual language:

- Background: `#F7F7F5`
- Surface: `#FFFFFF`
- Secondary surface: `#F0F0ED`
- Border: `#D7D7D2`
- Primary text: `#121212`
- Muted text: `#686864`
- Primary action: `#111111`
- No gradients
- No colorful stress labels
- Minimal shadows and restrained rounded corners

The visual hierarchy comes from typography, spacing, borders, and contrast instead of saturated status colors.

## User experience

### Analyze

1. Select upload or camera input.
2. Submit a handwriting image.
3. Review the quality gate.
4. Run analysis.
5. Inspect the predicted class and class distribution.
6. Download a report.
7. Optionally record a correction note in the current session.

### Session history

The current session can show:

- number of analyses
- average confidence estimate
- prediction distribution
- analysis table
- session CSV export
- session correction queue
- correction queue JSON export

Refreshing or ending the session may clear this state. That is intentional because the product has no database or account system.

### Methodology

The app explains the analysis pipeline and displays packaged model metadata when available.

### System

The app exposes runtime health information such as whether the model file exists, whether label mapping is present, feature count, metadata availability, and the fact that authentication and database persistence are not used.

## Security and privacy posture

No credentials or sign-in flow are required. The application does not need an application-owned database and does not write ordinary predictions to a persistent backend store.

Uploaded images are processed in memory for the current interaction. Export files are generated for direct download rather than being stored by the application.

The upload layer enforces extension, size, decode, and pixel-count checks before inference.

## Developer training workflow

`train_model.py` remains available for offline experimentation and model rebuilding.

```bash
python train_model.py --feature-mode hog
```

Available feature modes:

- `hog`
- `raw_pixels`

The current training workflow uses cross-validation and a train/test split. Because the base dataset is very small, its metrics should not be presented as evidence of clinical validity. A future research-grade pipeline should add participant-aware splitting and stronger evaluation metrics.

## CI

GitHub Actions installs the pinned dependencies, compiles Python sources, and runs the test suite on Python 3.11.

## Status

**Release posture: experimental, Streamlit deployment ready with session-only persistence.**

The repository intentionally favors a small, deterministic runtime surface over adding infrastructure that the product does not need.
