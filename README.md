# Stress Level Detection from Handwriting

A Streamlit web app that classifies handwriting samples into **Low**, **Medium**, or **High** stress levels using a Random Forest classifier trained on HOG features.

> **Disclaimer:** Educational / experimental project only. Not a clinical or diagnostic tool.

---

## Features

- Upload an image or capture via camera
- Image quality checks before prediction (resolution, blur, ink presence)
- Class probability breakdown
- Downloadable prediction report
- Prediction history log
- Admin panel for reviewing user-submitted corrections and retraining the model

---

## Project Structure

```
.
├── app.py                    # Main Streamlit app
├── train_model.py            # Retraining script
├── requirements.txt
├── inference/
│   └── predictor.py          # Model loading and prediction helpers
├── preprocessing/
│   ├── feature_extraction.py # HOG / raw-pixel feature extraction
│   └── quality_check.py      # Image quality validation
├── model/
│   ├── stress_model.pkl      # Trained model bundle
│   ├── label_mapping.json    # Class label map
│   ├── metadata.json         # Model version and accuracy info
│   └── preprocessed_data.pkl # Base training dataset (30 samples)
└── data/                     # Runtime data (auto-created)
    ├── prediction_history.csv
    ├── approved_samples.pkl
    └── pending_review/
```

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Admin Panel

Set the environment variable before running:

```bash
# Windows
set STRESS_APP_ADMIN_PASSWORD=yourpassword

# Linux / Mac
export STRESS_APP_ADMIN_PASSWORD=yourpassword
```

Then open the **Admin** page in the sidebar. From there you can:
- Review and approve / reject user-submitted corrections
- Retrain the model from the base dataset + approved samples
- The new model is only promoted if its accuracy is equal or better

---

## Retrain from CLI

```bash
python train_model.py --feature-mode hog --promote-active
```

Options:
- `--feature-mode` — `hog` (default) or `raw_pixels`
- `--promote-active` — overwrite the active model if accuracy improves
- `--without-handwriting-features` — disable the 3 derived handwriting features
- `--version` — set a specific version string (e.g. `v2.0.0`)

---

## Labels

| Code | Label |
|------|-------|
| 0 | Low Stress |
| 1 | Medium Stress |
| 2 | High Stress |

---

## Dataset

- 30 handwriting samples (10 per class), 128×128 px grayscale
- Stored in `model/preprocessed_data.pkl`
- Small dataset — treat all predictions as experimental
