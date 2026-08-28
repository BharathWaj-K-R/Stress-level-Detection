# Stress Classification Lab

A monochrome, light-themed Streamlit application for **experimental handwriting-based stress classification**.

> **Important:** This is an educational/research prototype, not a medical device, diagnostic tool, or validated measure of psychological stress. Do not use predictions for health, employment, academic, safety, or other high-impact decisions.

## Product

The app provides a focused workflow:

- Upload or capture a handwriting image.
- Validate file type, size, image dimensions, sharpness, and visible ink.
- Normalize the image while preserving its aspect ratio.
- Extract the feature representation expected by the packaged model.
- Run the local Random Forest classifier.
- Display the predicted class and classifier probability distribution.
- Keep analysis history and correction notes in the current Streamlit session.
- Export reports and session data directly from memory.

## Architecture

```text
Browser
  |
  v
Streamlit app (`app.py`)
  |
  +--> `ui.py` reusable monochrome UI system
  |
  +--> upload validation
  |
  +--> image quality gate
  |
  +--> preprocessing + feature extraction
  |
  +--> packaged Random Forest model
  |
  +--> result + probability estimate
  |
  +--> Streamlit session state
  |
  +--> in-memory TXT / CSV / JSON exports
```

Training is a developer-side workflow in `train_model.py`; it is not exposed through the public UI.

## Repository layout

```text
.
├── app.py
├── ui.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── assets/
│   ├── favicon.svg
│   └── hero-illustration.svg
├── inference/
│   ├── __init__.py
│   └── predictor.py
├── preprocessing/
│   ├── __init__.py
│   ├── feature_extraction.py
│   └── quality_check.py
├── model/
│   ├── stress_model.pkl
│   ├── label_mapping.json
│   └── preprocessed_data.pkl
└── tests/
    ├── test_feature_extraction.py
    ├── test_quality_check.py
    ├── test_predictor.py
    └── test_packaged_model.py
```

Every tracked file has a direct purpose in the product, model workflow, testing, or deployment configuration.

## UI system

`ui.py` implements a native Streamlit component layer inspired by shadcn/ui principles. The interface uses reusable brand, hero, card, metric, result, empty-state, methodology-step, and footer primitives.

## Model pipeline

The application loads the packaged model from `model/stress_model.pkl` with its label mapping and optional metadata. Before normal use, the app validates the model interface and feature dimensionality.

The input pipeline is:

1. extension and size validation
2. image decoding
3. EXIF orientation handling
4. grayscale conversion
5. aspect-ratio-preserving resize with padding
6. image quality checks
7. feature extraction
8. Random Forest prediction

The current feature system uses HOG plus optional engineered handwriting descriptors including ink density ratio, stroke-width variance estimate, and slant-angle estimate.

## Limitations

The current dataset is small. The application is therefore presented as an experimental classifier and not as a validated measurement of psychological stress.

The current training workflow uses cross-validation and a train/test split. A stronger research protocol should use a substantially larger participant-aware dataset, a documented labeling method, participant-independent evaluation, stronger metrics, and uncertainty analysis.

Classifier probabilities are model estimates and should not be interpreted as clinical probabilities.

## Local development

Use Python 3.11.

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

Install runtime dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Run tests separately:

```bash
python -m pip install pytest
python -m pytest -q
```

Compile-check Python sources:

```bash
python -m compileall -q .
```

## Streamlit Community Cloud

Deployment requires:

- Entry point: `app.py`
- Runtime dependencies: `requirements.txt`
- Streamlit configuration: `.streamlit/config.toml`
- Model artifacts: `model/`
- UI assets: `assets/`

Keep the deployed Python runtime aligned with Python 3.11.

## Data handling

Prediction history and correction notes exist only in Streamlit session state. Export files are generated in memory for direct download.

## Model training

Rebuild the packaged model locally with:

```bash
python train_model.py --feature-mode hog
```

Supported feature modes:

- `hog`
- `raw_pixels`

The training script writes active model artifacts only when `--promote-active` is explicitly supplied.

## Repository policy

Keep the repository limited to source code, tests, model artifacts, UI assets, deployment configuration, and documentation needed to run or rebuild the project.

Do not commit virtual environments, local secrets, editor settings, generated session files, or temporary datasets.

## Status

**Experimental Streamlit product.** The project intentionally favors a small, deterministic runtime surface and a simple deployment path over unnecessary infrastructure.
