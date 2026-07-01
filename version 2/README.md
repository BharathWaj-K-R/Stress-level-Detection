# Stress Level Detection from Handwriting - Version 2

This folder contains the experimental application for the project. It is isolated from the legacy root app so improvements can be developed without touching the original fallback model artifacts.

## What Version 2 Adds

- camera capture with `st.camera_input`
- input validation for file type and size
- quality checks before prediction
- version-aware model loading
- prediction history
- downloadable prediction reports
- **pending review** feedback flow
- admin-only approval/rejection of new samples
- versioned retraining from approved samples only

## Data and Labels

Version 2 uses:

- the bundled base dataset in `model/preprocessed_data.pkl`
- approved reviewer samples from `data/approved_samples.pkl`

Labels remain:

- `0 -> Low Stress`
- `1 -> Medium Stress`
- `2 -> High Stress`

## Feature Extraction Modes

Retraining supports:

- `raw_pixels`
- `hog`

Both modes can also append:

- ink density ratio
- stroke width variance
- slant angle estimate

## Model Versioning

Retrained models are saved under:

```text
version 2/model/upgraded/versions/<feature_mode>/<semantic_version>/
```

Each version includes:

- `stress_model.pkl`
- `label_mapping.json`
- `metadata.json`
- `training_metrics.json`

The currently promoted model is copied to:

```text
version 2/model/active/
```

## Admin Review Flow

User corrections do **not** retrain the model immediately anymore.

1. A user predicts on a handwriting sample.
2. The user submits the corrected label for review.
3. The sample is stored in:

```text
version 2/data/pending_review/
```

4. An admin opens the **Admin Review** page.
5. The admin authenticates using:

```text
STRESS_APP_ADMIN_PASSWORD
```

6. The admin approves or rejects each sample.
7. Approved samples are appended to:

```text
version 2/data/approved_samples.pkl
```

8. Retraining uses only the original dataset plus approved samples.
9. A new model version is evaluated before promotion to active.

## Install Requirements

```powershell
pip install -r requirements.txt
```

## Run the App

```powershell
streamlit run app.py
```

## Retrain From Approved Samples

```powershell
python train_model.py --feature-mode hog --promote-active
```

## Important Note

This application is for educational and experimental purposes only. It is not a clinical or diagnostic system.
