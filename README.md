# Diabetes_Prediction_Logistic_Regression
AI-based logistic regression model using sparse EHR data for early Type 2 Diabetes detection in developing nations. Includes dataset, Python model, and full research paper.

# Diabetes Prediction using Logistic Regression

This project demonstrates how **AI and Logistic Regression** can predict **Type 2 Diabetes** using sparse Electronic Health Record (EHR) data — particularly focused on **developing nations** where digital health data is limited.

## 📊 Overview
The project uses the **DiaHealth Bangladesh dataset (2024)** with 5,437 patients. It handles missing data, scales biometric features, and classifies patients as diabetic or non-diabetic using logistic regression.

**Model Performance:**
- Accuracy: **77.94%**
- ROC AUC: **0.812**
- Sensitivity: **65.6%**
- Specificity: **78.7%**

## 🧰 Tech Stack
- Python 3.13
- Pandas
- NumPy
- scikit-learn

## 🧩 Features
- Handles missing data using median imputation
- Scales features for balanced performance
- Accepts user input for real-time diabetes prediction
- Works efficiently on small or incomplete datasets

## 🧪 How to Run
```bash
# 1️⃣ Install dependencies
pip install -r requirements.txt

# 2️⃣ Run the model
python predict_diabetes.py

