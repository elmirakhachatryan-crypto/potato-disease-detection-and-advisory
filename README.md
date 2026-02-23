# 🥔 Potato Disease Detection and Advisory System
### Կարտոֆիլի հիվանդությունների ճանաչում և խորհրդատվության տրամադրում
[![NBViewer](https://img.shields.io/badge/Open%20in-NBViewer-orange?logo=jupyter)](https://nbviewer.org/github/elmirakhachatryan-crypto/potato-disease-detection-and-advisory/blob/main/potato-disease-detection-and-advisory.ipynb)

A deep learning-based system that detects potato crop diseases from images and provides treatment recommendations for Armenian farmers.

---

## 📌 Project Overview

This project uses a Convolutional Neural Network (CNN) to classify potato leaf images into 3 categories and provide actionable advisory recommendations. The system is designed to help Armenian farmers quickly identify diseases and take appropriate action.

---

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | **98.03%** |
| Validation Accuracy | **98.84%** |
| Training Loss | 0.0648 |
| Validation Loss | 0.0285 |

---

## 🌿 Detected Classes

| Class | Images | Description |
|-------|--------|-------------|
| 🟢 Potato___Healthy | 152 | Healthy potato plant |
| 🟡 Potato___Early_Blight | 1000 | Early blight disease (*Alternaria solani*) |
| 🔴 Potato___Late_Blight | 1000 | Late blight disease (*Phytophthora infestans*) |

---

## 🛠️ Technologies Used

- **Python 3**
- **TensorFlow / Keras** — CNN model training
- **Streamlit** — Web interface
- **Google Colab** — Training environment
- **Google Drive** — Dataset and model storage
- **PIL / NumPy** — Image processing
- **Matplotlib / Sklearn** — Visualization and evaluation

---

## 📂 Project Structure

```
potato-disease-detection-and-advisory/
│
├── potato-disease-detection-and-advisory.ipynb   # Main notebook
└── README.md
```

---

## 🚀 How to Use

1. Open the notebook in **Google Colab**
2. Mount your Google Drive
3. Upload your dataset (PlantVillage - Potato classes)
4. Run all cells step by step
5. Launch the Streamlit app to test with your own images

---

## 📊 Dataset

- Source: [PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
- Total images: **2152**
- Classes: 3 (Healthy, Early Blight, Late Blight)

---

## 👨‍💻 Author

Developed as part of an AI project for potato crop disease recognition in Armenia.

---

## 📄 License

This project is licensed under the **MIT License**.
