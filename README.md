# Musical-Chord Detection System
**ML-powered Web App using Audio Signal Processing (Librosa) + Streamlit**

An end-to-end Machine Learning web application that detects musical chords from audio files using advanced audio signal processing and interactive visualization.

## 🌟 Overview
This project allows users to upload an audio file (WAV / MP3) and automatically:
* ✅ **Detect** the musical chord (Major/Minor).
* ✅ **Identify** the root note.
* ✅ **Display** prediction probabilities.
* ✅ **Visualize** waveform & spectrogram.
* ✅ **Highlight** notes on a virtual piano.
* ✅ **Provide** scientific & emotional interpretation.

## 📊 Dataset
The model was trained using the following dataset from Kaggle:
👉 [Musical Instrument Chord Classification](https://www.kaggle.com/datasets/deepcontractor/musical-instrument-chord-classification)

## ⚙️ How It Works

### 1. Feature Extraction
Audio is processed using `Librosa` to extract:
* **Chroma Features** & **MFCC** (Mel Frequency Cepstral Coefficients).
* **Spectral Contrast**.
* Statistical features: **Mean** & **Standard Deviation**.

### 2. Model Training
Multiple models were compared to find the best performance:
* Logistic Regression, KNN, Random Forest, **SVM**, and **XGBoost**.
* **Optimization:** `GridSearchCV` for hyperparameter tuning.
* **Data Quality:** `SMOTE` for class balancing & `StandardScaler` for normalization.

### 3. Deployment
* The best model is serialized as `chord_pipeline.pkl`.
* Built with a **pure Streamlit** interface (No HTML/CSS) for a clean, efficient UI.

## 📂 Project Structure
├── app.py                # Streamlit web app UI
├── main.py               # Data processing & Training pipeline
├── chord_pipeline.pkl    # Serialized best model
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation


👨‍💻 Author
Developed as part of a Machine Learning journey combining:
Audio Processing
Machine Learning
Interactive Web Apps
