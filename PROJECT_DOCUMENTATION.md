---
title: "Stress Level Detection from Handwriting"
subtitle: "Mini Project Documentation"
author: "Submitted by: Bharathwaj"
date: "2026"
---

\newpage

# Stress Level Detection from Handwriting

## Project Documentation

**Project Title:** Stress Level Detection from Handwriting

**Domain:** Machine Learning, Image Processing, Pattern Recognition

**Technology Used:** Python, Streamlit, Scikit-learn, NumPy, Pillow, Joblib

**Submitted by:** Bharathwaj

**Academic Year:** 2025-2026

\newpage

# Certificate

This is to certify that the project titled **"Stress Level Detection from Handwriting"** has been successfully completed as part of the mini project work. The project demonstrates the use of machine learning and image processing techniques to classify the stress level of a person from handwriting image samples.

The work includes preprocessing of handwriting images, model training, model saving, and a user-friendly web interface for prediction.

**Guide Signature:** ______________________

**Head of Department:** ______________________

**Date:** ______________________

\newpage

# Acknowledgement

I would like to express my sincere gratitude to my project guide, faculty members, and department for their guidance and encouragement throughout the development of this project. Their support helped in understanding the concepts of image preprocessing, machine learning model training, and application development.

I also thank my classmates, friends, and family for their support and motivation during the completion of this project. This project helped me gain practical knowledge in Python programming, machine learning, and deployment of a simple prediction system using Streamlit.

\newpage

# Abstract

Stress is a common psychological condition that can affect a person's health, productivity, and daily life. Traditional stress detection methods usually depend on questionnaires, medical instruments, or expert analysis. This project proposes a simple machine learning-based approach to predict stress level using handwriting images.

The system accepts a handwriting image as input, converts it into grayscale, resizes it into a fixed size of 128 x 128 pixels, normalizes the pixel values, and classifies the image into one of three stress levels: **Low Stress**, **Medium Stress**, or **High Stress**.

The project uses a trained machine learning model saved as a `.pkl` file. A Streamlit web application is used to provide an easy interface for uploading handwriting images and displaying the predicted stress level along with confidence values. The project demonstrates how handwriting patterns can be used as input features for automated stress-level classification.

\newpage

# Table of Contents

1. Introduction
2. Existing System
3. Proposed System
4. Objectives
5. Scope of the Project
6. System Requirements
7. Software and Tools Used
8. System Architecture
9. Dataset Description
10. Methodology
11. Module Description
12. Algorithm
13. Implementation
14. Testing
15. Results and Discussion
16. Advantages
17. Limitations
18. Future Enhancement
19. Conclusion
20. References

\newpage

# 1. Introduction

Stress detection is an important area in health monitoring and psychological assessment. People experience stress due to academic pressure, workload, personal issues, and other daily challenges. Detecting stress at an early stage can help individuals take preventive action and improve their mental well-being.

Handwriting is a behavioral biometric pattern. The shape, size, pressure, spacing, and style of handwriting may vary according to a person's emotional and psychological state. This project uses handwriting images to classify stress levels using machine learning.

The project is implemented using Python. The trained model is integrated with a Streamlit web application so that users can upload a handwriting image and receive a stress-level prediction.

\newpage

# 2. Existing System

In existing stress detection systems, stress is commonly identified using:

- Manual questionnaires and surveys.
- Medical devices such as ECG, EEG, and heart-rate sensors.
- Direct consultation with psychologists or healthcare professionals.
- Facial-expression or speech-based analysis.

These methods may require expert supervision, special hardware, or detailed user responses. They may also take more time and may not always be convenient for quick screening.

## Disadvantages of Existing System

- Requires manual effort or expert involvement.
- May need special medical devices.
- Can be time-consuming.
- Some users may not feel comfortable answering personal questions.
- Not always suitable for quick prediction in a simple software application.

\newpage

# 3. Proposed System

The proposed system predicts stress level from a handwriting image using machine learning. The user uploads a handwriting image through a Streamlit web interface. The image is preprocessed and converted into a numerical feature vector. The trained machine learning model then classifies the input into Low Stress, Medium Stress, or High Stress.

## Features of the Proposed System

- Accepts JPG, JPEG, and PNG handwriting images.
- Converts uploaded images into grayscale.
- Resizes images to 128 x 128 pixels.
- Normalizes pixel values for model prediction.
- Predicts stress level using a trained model.
- Displays confidence score when available.
- Provides a simple and user-friendly web interface.

\newpage

# 4. Objectives

The main objectives of this project are:

- To develop a system that predicts stress level from handwriting images.
- To preprocess handwriting images into a standard format suitable for machine learning.
- To train a machine learning model using preprocessed handwriting data.
- To classify stress into Low, Medium, and High categories.
- To build a Streamlit-based user interface for easy image upload and prediction.
- To provide prediction confidence for better user understanding.

\newpage

# 5. Scope of the Project

The scope of this project is limited to classifying stress level from handwriting images using a trained machine learning model. The system is intended for educational and prototype purposes.

The project can be used as a basic demonstration of how machine learning and image processing can be applied in behavioral analysis. It is not a medical diagnostic tool and should not be used as a replacement for professional mental-health evaluation.

\newpage

# 6. System Requirements

## Hardware Requirements

- Processor: Intel i3 or above
- RAM: Minimum 4 GB
- Storage: Minimum 500 MB free space
- Input Device: Keyboard and mouse
- Display: Standard monitor

## Software Requirements

- Operating System: Windows
- Programming Language: Python
- Python Libraries: Streamlit, NumPy, Pillow, Scikit-learn, Joblib
- Development Tool: Visual Studio Code, IDLE, or any Python editor
- Browser: Chrome, Edge, or any modern web browser

\newpage

# 7. Software and Tools Used

## Python

Python is used as the main programming language because it provides strong support for machine learning, image processing, and web application development.

## Streamlit

Streamlit is used to create the web interface. It allows users to upload handwriting images and view prediction results easily.

## Scikit-learn

Scikit-learn is used to train and save the machine learning model. The project uses a Random Forest classifier for stress classification.

## NumPy

NumPy is used for numerical operations and image-array processing.

## Pillow

Pillow is used to open, convert, resize, and preprocess uploaded images.

## Joblib

Joblib is used to save and load the trained machine learning model.

\newpage

# 8. System Architecture

The system follows a simple machine learning workflow.

```text
Handwriting Image
        |
        v
Image Upload using Streamlit
        |
        v
Image Preprocessing
Grayscale Conversion, Resize, Normalize
        |
        v
Feature Vector Creation
        |
        v
Trained Machine Learning Model
        |
        v
Stress Level Prediction
Low Stress / Medium Stress / High Stress
```

## Architecture Explanation

The user uploads a handwriting image through the web interface. The image is converted into grayscale and resized to 128 x 128 pixels. The pixel values are normalized and flattened into a feature vector. The trained model receives this vector and predicts the stress category.

\newpage

# 9. Dataset Description

The dataset used in this project is stored in preprocessed form. The file `model/preprocessed_data.pkl` contains the handwriting image features and their corresponding labels.

## Dataset Details

- Number of classes: 3
- Classes: Low Stress, Medium Stress, High Stress
- Image size: 128 x 128 pixels
- Feature size: 16,384 values per image
- Data format: Pickle file
- Number of available samples in the current project: 30

## Class Labels

The labels are mapped as follows:

| Numeric Label | Stress Level |
|---|---|
| 0 | Low Stress |
| 1 | Medium Stress |
| 2 | High Stress |

\newpage

# 10. Methodology

The project methodology consists of the following steps:

## Step 1: Data Collection

Handwriting samples are collected and grouped according to stress level. The current project uses already preprocessed handwriting data.

## Step 2: Preprocessing

Each handwriting image is converted into grayscale and resized to 128 x 128 pixels. The image is then normalized by converting pixel values into the range 0 to 1.

## Step 3: Feature Extraction

The resized image is converted into a one-dimensional vector of 16,384 pixel values. This vector is used as input for the machine learning model.

## Step 4: Model Training

A Random Forest classifier is trained using the preprocessed feature vectors and their stress labels.

## Step 5: Model Saving

The trained model is saved using Joblib as `model/stress_model.pkl`. The label mapping is saved as `model/label_mapping.json`.

## Step 6: Prediction

The Streamlit application loads the saved model, preprocesses the uploaded image, and predicts the stress level.

\newpage

# 11. Module Description

## 11.1 Application Module

File: `app.py`

This module provides the Streamlit web interface. It allows the user to upload an image and view the predicted stress level. It also displays the prediction confidence when probability scores are available.

## 11.2 Training Module

File: `train_model.py`

This module loads the preprocessed dataset, encodes the labels, trains the Random Forest model, evaluates the model, and saves the trained model.

## 11.3 Image Conversion Module

File: `svc_to_images.py`

This module converts the preprocessed feature vectors back into image files. This helps in visualizing the dataset and checking the image samples.

## 11.4 Model Folder

Folder: `model`

This folder stores the preprocessed data, trained model, and label mapping.

## 11.5 Test Folder

Folder: `test`

This folder contains sample handwriting images used for testing the prediction system.

\newpage

# 12. Algorithm

## Training Algorithm

1. Start the program.
2. Load preprocessed handwriting data from `model/preprocessed_data.pkl`.
3. Convert labels into numeric classes.
4. Split the data into training and testing sets.
5. Create a Random Forest classifier.
6. Train the model using the training data.
7. Predict labels for the testing data.
8. Calculate accuracy and classification report.
9. Save the model to `model/stress_model.pkl`.
10. Save the label mapping to `model/label_mapping.json`.
11. Stop the program.

## Prediction Algorithm

1. Start the Streamlit application.
2. Load the trained model.
3. Upload a handwriting image.
4. Convert the image into grayscale.
5. Resize the image to 128 x 128 pixels.
6. Normalize pixel values.
7. Convert the image into a feature vector.
8. Pass the vector to the trained model.
9. Display the predicted stress level and confidence.
10. Stop.

\newpage

# 13. Implementation

The project is implemented using Python. The important implementation details are explained below.

## Image Preprocessing

The uploaded handwriting image is opened using Pillow. It is converted to grayscale using the `convert("L")` method. The image is resized to 128 x 128 pixels. The pixel values are converted to floating-point values and divided by 255 to normalize them.

## Model Loading

The Streamlit application loads the model from:

```text
model/stress_model.pkl
```

The label mapping is loaded from:

```text
model/label_mapping.json
```

## Prediction

After preprocessing, the image is reshaped into a single row feature vector. The trained model predicts the stress level. If the model supports probability prediction, the application displays the confidence score.

\newpage

# 14. Testing

Testing is performed using sample images stored in the `test` folder. The system was tested with low, medium, and high stress handwriting samples.

## Test Cases

| Test Case | Input | Expected Output | Result |
|---|---|---|---|
| TC01 | Valid low-stress handwriting image | Low Stress | Passed |
| TC02 | Valid medium-stress handwriting image | Medium Stress | Passed |
| TC03 | Valid high-stress handwriting image | High Stress | Passed |
| TC04 | Invalid file type | Error message | Passed |
| TC05 | Missing model file | Error message | Passed |

## Sample Prediction Results

| Sample Image | Predicted Result |
|---|---|
| `test/low.png` | Low Stress |
| `test/high.png` | High Stress |
| `test/medium.png` | Medium Stress |
| `test/medium.jpg` | Medium Stress |

\newpage

# 15. Results and Discussion

The system successfully accepts handwriting images and predicts stress level using the trained model. The Streamlit interface makes the system easy to use. The model produces one of three outputs: Low Stress, Medium Stress, or High Stress.

The current dataset contains a small number of samples. Therefore, the project should be considered a prototype. With a larger and more diverse dataset, the accuracy and generalization of the model can be improved.

## Current Output

The final application displays:

- Uploaded handwriting image.
- Predicted stress level.
- Confidence percentage.
- Probability chart for all classes.

\newpage

# 16. Advantages

- Simple and easy-to-use interface.
- Does not require expensive hardware.
- Uses handwriting images as input.
- Provides quick prediction.
- Can be run locally on a normal computer.
- Uses common Python libraries.
- Model and label mapping are saved for reuse.

\newpage

# 17. Limitations

- The current dataset has limited samples.
- Prediction accuracy depends on the quality and variety of training data.
- The system is not a medical diagnostic tool.
- Handwriting style can vary greatly between individuals.
- Image quality, lighting, and background can affect prediction.
- More real-world testing is required for reliable use.

\newpage

# 18. Future Enhancement

The project can be improved in the following ways:

- Collect a larger handwriting dataset.
- Add more handwriting features such as slant, spacing, pressure, and stroke thickness.
- Use deep learning models such as CNNs when a larger dataset is available.
- Add user login and report generation.
- Store prediction history in a database.
- Improve UI design with better visual feedback.
- Add support for camera-based image capture.
- Validate the system with real stress-assessment data.

\newpage

# 19. Conclusion

The project **"Stress Level Detection from Handwriting"** demonstrates a machine learning-based approach for classifying stress levels from handwriting images. The system preprocesses the input image, converts it into a feature vector, and predicts the stress category using a trained Random Forest classifier.

The Streamlit application provides a simple interface for users to upload images and view predictions. Although the current implementation is a prototype with a limited dataset, it provides a strong foundation for future improvements. With more training data and advanced feature extraction, the system can become more accurate and useful for stress-level screening.

\newpage

# 20. References

1. Scikit-learn Documentation: https://scikit-learn.org/
2. Streamlit Documentation: https://docs.streamlit.io/
3. NumPy Documentation: https://numpy.org/doc/
4. Pillow Documentation: https://pillow.readthedocs.io/
5. Python Documentation: https://docs.python.org/

\newpage

# Appendix A: Project Files

| File or Folder | Description |
|---|---|
| `app.py` | Streamlit application for prediction |
| `train_model.py` | Model training script |
| `svc_to_images.py` | Converts feature vectors into images |
| `requirements.txt` | Required Python packages |
| `model/preprocessed_data.pkl` | Preprocessed handwriting dataset |
| `model/stress_model.pkl` | Saved trained model |
| `model/label_mapping.json` | Label mapping file |
| `test` | Sample images for testing |
| `images_from_svc` | Generated image dataset folders |

# Appendix B: How to Run the Project

## Install Requirements

```powershell
pip install -r requirements.txt
```

## Train the Model

```powershell
python train_model.py
```

## Run the Web Application

```powershell
streamlit run app.py
```

After running the command, the application opens in the browser. Upload a handwriting image and click **Predict Stress Level** to view the output.
