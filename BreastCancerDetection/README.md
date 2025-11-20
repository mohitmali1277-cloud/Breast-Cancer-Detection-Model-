# Breast Cancer Detection using CNN (96% Accuracy) 🩺

**Deep Learning | TensorFlow | Keras | CNN | Medical Image Classification**

A Convolutional Neural Network model to detect breast cancer from histopathological images with **96% validation accuracy**.

## 🚀 Results
- **Accuracy**: 96.0%
- **Validation Loss**: 0.12
- **Model**: Custom CNN with BatchNormalization & Dropout
- **Dataset**: https://www.kaggle.com/datasets/wasiqaliyasir/breast-cancer-dataset

## 📊 Dataset
The dataset comprises 569 instances (rows) and 32 columns, including an ID column, a diagnosis label, and 30 numerical features describing cell nuclei characteristics. Each instance represents a single breast mass sample, with features computed from digitized FNA images of breast tumor tissue (benign & malignant).

## 🧠 Model Architecture
```python
Conv2D → BatchNorm → ReLU → MaxPool → Dropout
(Repeated 3 times) → Flatten → Dense → Output (Sigmoid)