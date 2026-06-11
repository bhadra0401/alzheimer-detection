#  Transfer Learning Framework for Early Alzheimer's Disease Prediction

<p align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Accuracy](https://img.shields.io/badge/Ensemble_Accuracy-97.7%25-brightgreen)

</p>

<p align="center">
  <a href="https://huggingface.co/spaces/bhadra0401/alzheimer-detection">
    <img src="https://img.shields.io/badge/_Live_Demo-Try_Now-success?style=for-the-badge">
  </a>
</p>

---

##  Overview

A healthcare-focused AI system that detects Alzheimer's Disease stages from Brain MRI scans using Transfer Learning and Ensemble Machine Learning.

### Classification Categories

* Mild Demented
* Moderate Demented
* Non Demented
* Very Mild Demented

---

##  Architecture

```text
MRI Scan
   ↓
Image Preprocessing
   ↓
MobileNetV2 / ResNet50
   ↓
Deep Feature Extraction
   ↓
SVM + Random Forest + Gradient Boosting
   ↓
Voting Ensemble
   ↓
Prediction
```

---

##  Dataset

**Dataset:** Augmented Alzheimer's MRI Dataset

| Class              | Images |
| ------------------ | -----: |
| Mild Demented      |  8,960 |
| Moderate Demented  |  6,464 |
| Non Demented       |  9,600 |
| Very Mild Demented |  8,960 |

**Total Images:** 33,984 MRI Scans

---

##  Models Used

### Deep Learning

* MobileNetV2 (Transfer Learning + Fine-Tuning)
* ResNet50 (Transfer Learning + Fine-Tuning)

### Machine Learning

* Support Vector Machine (SVM)
* Random Forest
* Gradient Boosting
* Voting Ensemble Classifier

---

##  Results

| Model                  |   Accuracy |
| ---------------------- | ---------: |
| MobileNetV2 Validation |     95.32% |
| SVM                    |     97.70% |
| Random Forest          |     97.39% |
| Gradient Boosting      |     97.46% |
| Ensemble Voting        | **97.70%** |

---

##  Application Screenshots

### Home Page

![Home Page](screenshots/home_page.png)

### Prediction Result

![Prediction Result](screenshots/prediction_result.png)

### Prediction Analysis

![Prediction Analysis](screenshots/prediction_analysis.png)

### Prediction Report

![Prediction Report](screenshots/prediction_report.png)

---

##  Run Locally

```bash
git clone https://github.com/bhadra0401/Alzheimer_Project.git

cd Alzheimer_Project

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

---

##  Tech Stack

* Python
* TensorFlow
* Keras
* Scikit-Learn
* NumPy
* Pandas
* OpenCV
* Streamlit

---

##  Future Improvements

* Grad-CAM Visualization
* Explainable AI (XAI)
* PDF Report Generation
* Cloud Deployment
* Multi-Patient Dashboard

---

##  Author

**Naga Veera Bhadra Kumar Akkala**

Data Science • Machine Learning • Deep Learning

⭐ If you found this project useful, consider giving it a star.
