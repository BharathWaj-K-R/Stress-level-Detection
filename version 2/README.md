# Stress Level Detection from Handwriting - Version 2

This folder contains an improved version of the project. It is separate from the original project files, so changes here do not interfere with the old code.

## New Features

- Upload handwriting image.
- Capture handwriting image directly using camera.
- Predict Low, Medium, or High stress.
- Check basic image quality.
- Save prediction history to CSV.
- Download a text prediction report.
- Learn from new labeled handwriting inputs.
- Keep model, data, reports, and history inside this folder.

## Folder Structure

```text
version 2/
    app.py
    train_model.py
    requirements.txt
    README.md
    model/
        preprocessed_data.pkl
        stress_model.pkl
        label_mapping.json
    data/
        prediction_history.csv
        user_training_samples.pkl
    reports/
```

## Install Requirements

```powershell
pip install -r requirements.txt
```

## Run the App

```powershell
streamlit run app.py
```

## Retrain the Model

```powershell
python train_model.py
```

## Teach the Model from App Input

1. Upload or capture a handwriting image.
2. Click **Predict Stress Level**.
3. In **Teach the Model**, choose the correct stress level.
4. Click **Save Input and Retrain Model**.

The new sample is saved in `data/user_training_samples.pkl`, and the model is retrained using the original dataset plus the newly labeled input.

## Important Note

This project is for educational use only. It is not a medical diagnostic tool. The model accuracy depends heavily on the size and quality of the handwriting dataset.
