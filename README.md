# Support Vector Machine (SVM) Project

This project demonstrates the implementation of:

- SVC (Support Vector Classification)
- SVR (Support Vector Regression)

using Python, Machine Learning, and Streamlit.

---
## 🚀 Live Demo

🌐 Live App: https://svmsvr.streamlit.app/

# Project Overview

Support Vector Machine (SVM) is a supervised machine learning algorithm used for:

- Classification Problems
- Regression Problems

SVM works by finding the best boundary (hyperplane) that separates the data points with maximum margin.

This project contains:

1. SVC using Breast Cancer Dataset
2. SVR using CSV Dataset
3. Streamlit Web Application
4. Data Visualization
5. Model Evaluation

---

# Types of SVM Used

## 1. SVC (Support Vector Classification)

SVC is used for classification tasks.

### Dataset Used
Breast Cancer Dataset from sklearn

### Goal
Predict whether the tumor is:
- Malignant
- Benign

### Algorithms and Concepts Used
- SVC
- Train Test Split
- Feature Scaling
- Accuracy Score
- Confusion Matrix
- Classification Report

### Kernels Used
- Linear Kernel
- RBF Kernel
- Polynomial Kernel

### Workflow
1. Load Dataset
2. Split Features and Target
3. Train Test Split
4. Scale the Data
5. Train SVC Model
6. Make Predictions
7. Evaluate Accuracy

---

## 2. SVR (Support Vector Regression)

SVR is used for regression tasks.

### Dataset Used
TEST11.csv

### Goal
Predict continuous numerical values.

### Operations Performed
- Data Cleaning
- Handling Missing Values
- Outlier Detection using IQR
- Pair Plot
- Box Plot
- Feature Selection
- Model Training
- Prediction

### Evaluation Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R2 Score

### Workflow
1. Load CSV Dataset
2. Display Dataset Information
3. Handle Missing Values
4. Detect and Remove Outliers
5. Visualize Dataset
6. Split Features and Target
7. Train SVR Model
8. Make Predictions
9. Evaluate Model

---

# Files Included

| File Name | Description |
|-----------|-------------|
| app.py | Streamlit Application |
| SVC.ipynb | SVC Model Notebook |
| svr.ipynb | SVR Model Notebook |
| TEST11.csv | Dataset for SVR |
| requirements.txt | Required Libraries |
| README.md | Project Documentation |

---

# Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
