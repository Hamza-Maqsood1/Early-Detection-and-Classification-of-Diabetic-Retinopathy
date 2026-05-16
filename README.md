# Early Detection and Classification of Diabetic Retinopathy

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-MobileNetV2-blue)
![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-95.95%25-brightgreen)
![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Domain](https://img.shields.io/badge/Domain-Medical%20AI-red)

## 📌 Overview

Diabetic Retinopathy (DR) is a leading cause of blindness worldwide, affecting millions of diabetic patients. Early detection is critical yet manual screening by ophthalmologists is time-consuming and inaccessible in many regions.

This project presents a **deep learning-based system** for automated early detection and severity grading of diabetic retinopathy using **MobileNetV2** with transfer learning from ImageNet.

---

## 🎯 Problem Statement

- DR affects **1 in 3 diabetic patients** globally
- Manual diagnosis requires specialist ophthalmologists scarce in developing regions
- Goal: Build an accurate, lightweight CNN model for automated 7-class DR severity grading

---

## 🔬 Methodology

### Model Architecture
- **Base Model:** MobileNetV2 (pretrained on ImageNet)
- **Custom Head:** GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.4) → Dense(128, ReLU) → Dropout(0.3) → Softmax
- **Training Strategy:** Two-phase transfer learning
  - Phase 1: Train custom head only (frozen base)
  - Phase 2: Fine-tune top layers with low learning rate

### Data Processing
- **Input Size:** 224 × 224 × 3
- **Normalization:** Pixel values scaled to [0, 1]
- **Data Augmentation:** Rotation, width/height shift, brightness variation, zoom, horizontal flip
- **Class Imbalance:** Handled via `compute_class_weight` (balanced)
- **Train/Val Split:** 80% training, 20% validation (stratified)

### Training Configuration
| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate (Phase 1) | 0.001 |
| Learning Rate (Phase 2) | 0.0001 |
| Batch Size | 32 |
| Loss Function | Categorical Crossentropy |
| Callbacks | ReduceLROnPlateau, EarlyStopping, ModelCheckpoint |

---

## 📊 Results

| Metric | Value |
|---|---|
| **Validation Accuracy** | **95.95%** |
| Classes | 7 |
| Model | MobileNetV2 + Transfer Learning |

### Severity Classes
| Class | Description |
|---|---|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe DR |
| 4 | Proliferative DR |
| 5 | Mild-Moderate DR |
| 6 | Moderate-Severe DR |

---

## 🛠️ Tech Stack

- **Framework:** TensorFlow / Keras
- **Model:** MobileNetV2
- **Language:** Python
- **Libraries:** NumPy, Matplotlib, Scikit-learn, OpenCV
- **Environment:** Google Colab (GPU)

---

## 📁 Repository Structure

```
├── Retinopathy.ipynb       # Main training notebook
├── README.md               # Project documentation
```

---

## 🚀 How to Run

1. Open `Retinopathy.ipynb` in Google Colab
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Upload your dataset to Google Drive
4. Update `zip_path` with your dataset path
5. Run all cells sequentially

---

## 🔮 Future Work

- Deploy as a web application for real-time screening
- Integrate GradCAM visualizations for explainability
- Test on larger datasets (EyePACS, Messidor-2)
- Extend to other retinal diseases (Glaucoma, AMD)

---

## 👤 Author

**Hamza Maqsood**
BS Artificial Intelligence University of Management and Technology, Lahore
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/hamza-maqsood1)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/Hamza-Maqsood1)
