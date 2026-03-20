## Brain Tumor Detection using CNN
This project uses Deep Learning (Convolutional Neural Networks) to identify brain tumors from MRI/X-ray scans. It achieves 95%+ accuracy across four distinct categories.

## Overview
The system is designed to act as a diagnostic aid. Users can upload a brain scan through a web interface, and the model will:

Determine if a Tumor is present or if the brain is Healthy (No Tumor).

If a tumor is detected, it classifies it into one of three types: Glioma, Meningioma, or Pituitary.

### Dataset Information
The model was trained on a comprehensive dataset of 5,715 images, balanced across four classes to ensure high precision.

Training Set: [ https://drive.google.com/drive/folders/1u7O1TCNS6xYZn22tGlWdlz2mbxveellC?usp=sharing ]

Testing Set: [ https://drive.google.com/drive/folders/1jDDrdLlm5bENm6X9xBcsHlGtZ_wksfpj?usp=sharing ]

## Technology Stack
Frontend: HTML5, CSS3 (Custom Responsive UI)

Backend: Flask (Python)

AI/ML: TensorFlow, Keras, NumPy, OpenCV

Environment: Python 3.10.19

## Project Structure
model_training.ipynb: The notebook where the CNN model is built, trained, and evaluated.

brain_flask.py: The Flask server that handles image uploads and runs the prediction logic.

models/: Contains the saved .h5 or .keras model file.

templates/ & static/: The HTML and CSS files for the web interface.

## ⚙️ How to Run Locally
Clone the repository:
Bash
git clone https://github.com/Priyanshu17-z/brain_tumor_-detection_using_CNN.git
cd brain_tumor_-detection_using_CNN
Run using Python 3.10:
(Note: Ensure you use the 3.10.x version to avoid NumPy 2.x compatibility issues)

Bash
/usr/bin/python3.10 brain_flask.py
Open in Browser:
Go to http://127.0.0.1:5000

## Results
Accuracy: 95.2%
Classes:
Glioma: High detection rate
Meningioma: High detection rate
Pituitary: High detection rate
No Tumor: 0% False Positive rate in testing