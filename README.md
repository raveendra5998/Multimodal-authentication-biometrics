# Multimodal Authentication System Using Face and Voice Data

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SVM%20%7C%20RF%20%7C%20kNN-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-Educational-orange)

---

# Project Overview

This project presents a **multimodal biometric authentication system** that integrates **face** and **voice** biometrics using **feature-level fusion**. By combining complementary biometric modalities, the system improves authentication accuracy and robustness compared to unimodal authentication systems.

The system utilizes multiple machine learning classifiers including **Support Vector Machine (SVM)**, **Random Forest**, and **k-Nearest Neighbors (kNN)**. To address class imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** is applied before model training.

---

# Features

- Face biometric authentication
- Voice biometric authentication
- Feature-level fusion
- SMOTE-based class balancing
- Multiple machine learning classifiers
- Cross-validation
- Performance evaluation using Accuracy, ROC AUC, EER and D-prime
- Confusion Matrix generation

---

# Research Questions

1. How does feature-level fusion affect authentication performance?
2. What is the impact of SMOTE on class balancing?
3. Which classifier performs the best with multimodal biometric features?

---

# Project Structure

```
.
├── Biometrics.py
├── predict.py
├── face/
├── Audio/
├── Results01/
├── final_multimodal_model.pkl
├── requirements.txt
├── README.md
```

---

# Datasets

## Face Dataset

Caltech Face Dataset

http://www.vision.caltech.edu/Image_Datasets/faces/

## Voice Dataset

AudioMNIST Dataset

https://github.com/soerenab/AudioMNIST

Update the dataset paths inside **Biometrics.py**:

```python
FACE_DATA_DIR = "face"
VOICE_DATA_DIR = "Audio"
```

---

# Requirements

Install all dependencies:

```bash
pip install numpy pandas opencv-python librosa scikit-learn imbalanced-learn matplotlib seaborn soundfile scipy joblib tqdm Pillow torch facenet-pytorch
```

Or simply run:

```bash
pip install -r requirements.txt
```

---

# How to Run

## Option 1: Python Script

Train the models and generate results:
```bash
python Biometrics.py
```

Test inference (speed test on a single user):
```bash
python predict.py
```

## Option 2: Jupyter Notebook

```bash
jupyter notebook
```

Open:

```
Multimodal_Auth.ipynb
```

---

# Machine Learning Pipeline

```
Face Images
      │
      ▼
Face Feature Extraction
(InceptionResnetV1 Embeddings)

                   │
                   ▼
         Feature-Level Fusion
                   │
                   ▼
                SMOTE
                   │
                   ▼
     Train ML Classifiers
   (SVM, Random Forest, kNN)
                   │
                   ▼
      Performance Evaluation
```

---

# Results Summary

Results below are from the corrected, leakage-free pipeline (run the script to populate this table with your actual numbers).

**Pipeline corrections applied:**
- Augmentation moved *inside* CV loop — training fold only, never test fold (fixes augmentation leakage)
- Amplitude normalisation + pre-emphasis filter on voice signals before MFCC extraction
- Identity classes with < 5 samples excluded from training/evaluation
- SVM replaced with `GridSearchCV` over `{kernel, C, gamma}` fitted inside each training fold
- `FACE_IMG_SIZE` restored 64 → 128 px; `AUGMENT_COPIES` 2 → 3; `MAX_WAV_PER_SPEAKER` 50 → 200; `RF_ESTIMATORS` 100 → 200

**Documented Enhancement (beyond originally presented method):**
- Face feature extraction upgraded from pixel-intensity + Canny edges to a 512-dim face embedding extracted via a pretrained `facenet-pytorch` `InceptionResnetV1` (VGGFace2). This produces a much more informative, compact feature vector that avoids overwhelming the voice features.

> **Note:** Run `python Biometrics.py` and copy the printed *Results Summary* table here. Do not copy numbers from any earlier run — those contained augmentation leakage and are not valid.

| System | Accuracy | ROC AUC | EER | D-prime | Best Classifier |
|--------|----------:|--------:|-----:|--------:|-----------------|
| Face-Only | 96.18% | 0.9987 | 0.0050 | 4.6407 | SVM |
| Voice-Only | 81.79% | 0.9401 | 0.0907 | 2.4471 | SVM |
| Multimodal | 95.05% | 0.9982 | 0.0013 | 3.7462 | SVM |

### Best Performing Classifier Per System

**SVM** proved to be the most consistent and performant classifier across all three systems.

---

# Techniques Used

## Face Feature Extraction

- Grayscale Conversion
- CLAHE Histogram Equalisation (lighting correction)
- Image Resizing (160×160)
- Pretrained Face Embedding: `InceptionResnetV1` trained on VGGFace2 (512-dim)

## Voice Feature Extraction

- Amplitude Normalisation
- Pre-emphasis Filter (noise reduction)
- MFCC (Mel-Frequency Cepstral Coefficients, 13 coefficients)
- Spectral Contrast

## Machine Learning

- Feature-Level Fusion
- SMOTE
- Support Vector Machine (SVM)
- Random Forest
- k-Nearest Neighbors (kNN)
- Stratified Cross Validation

---

# Evaluation Metrics

- Accuracy
- ROC AUC
- Equal Error Rate (EER)
- D-prime
- Confusion Matrix

---

# Ethical Considerations

- Secure biometric data handling
- User consent and privacy
- Fairness and bias mitigation
- Transparent authentication process

---

# Future Improvements

- Deep Learning-based feature extraction
- Score-level fusion
- Decision-level fusion
- Real-time authentication
- Robustness under noisy and varying lighting conditions
- Additional biometric modalities (Fingerprint, Iris, Gait)

---

# Author

**CHITROJU MANYA RAVEENDRA CHARY**

Master of Computer Applications (MCA)

Jawaharlal Nehru Technological University Hyderabad

---

# License

This project is intended for educational and research purposes.