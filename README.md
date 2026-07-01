# Stress Level Detection from Handwriting

This project predicts a handwriting sample's stress category as **Low**, **Medium**, or **High** using machine learning. The repository now contains:

- A legacy fallback app and model in the project root.
- An upgraded training/evaluation pipeline with raw-pixel and HOG feature modes.
- An experimental `version 2` app with camera capture, review-gated feedback, and versioned retraining.

## Data Source and Labels

The bundled training data lives in `model/preprocessed_data.pkl` and contains a small preprocessed educational dataset of handwriting samples.

- Image size: `128 x 128`
- Base dataset size: `30` samples
- Labels: `low`, `medium`, `high`

The original labels are encoded as:

- `0 -> Low Stress`
- `1 -> Medium Stress`
- `2 -> High Stress`

Because the dataset is very small, every model in this repository should be treated as **experimental** rather than clinically reliable.

## Feature Extraction Modes

The upgraded training pipeline supports two feature modes:

1. `raw_pixels`
   - Uses the normalized flattened grayscale image.
   - Adds three handwriting-specific derived features.

2. `hog`
   - Uses Histogram of Oriented Gradients from `skimage.feature.hog`.
   - Also adds the same three handwriting-specific derived features.

The shared handwriting-specific features are:

- `ink_density_ratio`
- `stroke_width_variance`
- `slant_angle_estimate`

## Model Versioning

Upgraded models are stored under:

```text
model/upgraded/versions/<feature_mode>/<semantic_version>/
```

Each version directory contains:

- `stress_model.pkl`
- `label_mapping.json`
- `training_metrics.json`
- `metadata.json`

The active upgraded model is copied to:

```text
model/upgraded/active/
```

The root Streamlit app prefers the upgraded active model when present and falls back to the original legacy model in `model/stress_model.pkl`.

`metadata.json` includes:

- timestamp
- feature extraction method
- training set size
- feature count
- cross-validated accuracy
- semantic version

## Evaluation Workflow

Train candidate models:

```powershell
python train_model.py --feature-mode raw_pixels --version v1.1.0
python train_model.py --feature-mode hog --version v1.1.0 --promote-active
```

Run the evaluation bundle:

```powershell
python scripts\run_evaluation.py
```

This writes outputs to `reports/`:

- confusion matrix images
- per-model classification reports
- feature importance charts
- an evaluation summary JSON

## Input Validation

Both apps now validate uploads before prediction:

- file type restricted to `.jpg` and `.png`
- max file size: `5 MB`
- minimum resolution check
- blur check using Laplacian variance
- handwriting-presence check using ink ratio

If a sample fails, prediction is blocked and the user sees a clear warning.

## Version 2 Admin Review Flow

The `version 2` app no longer retrains immediately from user feedback.

New user-labeled samples are first stored in:

```text
version 2/data/pending_review/
```

An admin reviewer can open the **Admin Review** page in the v2 app, authenticate with the environment variable:

```text
STRESS_APP_ADMIN_PASSWORD
```

Then the reviewer can:

- approve a pending sample
- reject a pending sample
- retrain from approved samples only

Approved samples are appended to:

```text
version 2/data/approved_samples.pkl
```

When retraining in v2:

- a new versioned model is created
- 5-fold evaluation is run
- before/after accuracy is printed
- the model is promoted to active only after evaluation

## Running the Apps

Root app:

```powershell
streamlit run app.py
```

Version 2 app:

```powershell
streamlit run "version 2\app.py"
```

## Disclaimer

This repository is for **educational and experimental use only**. It is **not** a clinical, medical, or diagnostic tool.
