# app.py - SVM (SVC & SVR) Streamlit Application

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.preprocessing import StandardScaler

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(page_title="SVM - SVC & SVR", layout="wide")

# =====================================================
# TITLE
# =====================================================
st.title("Support Vector Machine (SVM)")
st.markdown("Machine Learning using SVC and SVR")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("SVM Navigation")

option = st.sidebar.radio(
    "Select Topic",
    [
        "What is SVM?",
        "What is SVC?",
        "What is SVR?",
        "SVC Implementation",
        "SVR Implementation"
    ]
)

# =====================================================
# WHAT IS SVM
# =====================================================
if option == "What is SVM?":

    st.header("What is Support Vector Machine (SVM)?")

    st.write(
        """
        Support Vector Machine (SVM) is a supervised machine learning algorithm.

        It is used for:
        - Classification problems
        - Regression problems

        SVM works by finding the best boundary (hyperplane) that separates data points.

        Main Goal of SVM:
        - Separate classes with maximum margin
        - Improve prediction accuracy

        Advantages:
        - Works well for small and medium datasets
        - High accuracy
        - Effective in high dimensional data

        Disadvantages:
        - Takes more time for large datasets
        - Difficult to understand visually
        """
    )

    st.subheader("Types of SVM")

    st.write(
        """
        1. SVC (Support Vector Classification)
           - Used for Classification problems
           - Example: Cancer detection, Spam detection

        2. SVR (Support Vector Regression)
           - Used for Regression problems
           - Example: Price prediction, Sales prediction
        """
    )

# =====================================================
# WHAT IS SVC
# =====================================================
elif option == "What is SVC?":

    st.header("What is SVC?")

    st.write(
        """
        SVC stands for Support Vector Classification.

        It is used for classification tasks.

        Example:
        - Predict whether cancer is Malignant or Benign
        - Email Spam Detection
        - Handwriting Recognition

        How SVC Works:
        - Finds the best line or boundary
        - Separates different classes
        - Maximizes the margin between classes

        Common Kernels:
        - Linear Kernel
        - Polynomial Kernel
        - RBF Kernel
        - Sigmoid Kernel
        """
    )

    st.subheader("Important Parameters")

    st.write(
        """
        1. C Parameter
           - Controls regularization
           - Large C → Less error but may overfit
           - Small C → More generalization

        2. Kernel
           - Defines decision boundary type

        3. Gamma
           - Controls influence of data points
        """
    )

# =====================================================
# WHAT IS SVR
# =====================================================
elif option == "What is SVR?":

    st.header("What is SVR?")

    st.write(
        """
        SVR stands for Support Vector Regression.

        It is used for regression problems.

        Example:
        - House Price Prediction
        - Sales Prediction
        - Stock Price Prediction

        SVR tries to fit the best line within a margin.

        Main Goal:
        - Predict continuous numerical values

        Advantages:
        - Handles non-linear data
        - Good prediction performance

        Disadvantages:
        - Slow for very large datasets
        """
    )

    st.subheader("Applications of SVR")

    st.write(
        """
        - Weather Forecasting
        - Market Prediction
        - Demand Forecasting
        - Financial Analysis
        """
    )

# =====================================================
# SVC IMPLEMENTATION
# =====================================================
elif option == "SVC Implementation":

    st.header("SVC Implementation using Breast Cancer Dataset")

    # -------------------------------------------------
    # LOAD DATASET
    # -------------------------------------------------
    data1 = load_breast_cancer()

    X = data1.data
    y = data1.target

    df = pd.DataFrame(X, columns=data1.feature_names)
    df['target'] = y

    st.subheader("Dataset")
    st.dataframe(df.head())

    # -------------------------------------------------
    # DATASET INFORMATION
    # -------------------------------------------------
    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # -------------------------------------------------
    # TRAIN TEST SPLIT
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # -------------------------------------------------
    # FEATURE SCALING
    # -------------------------------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------------------------------
    # MODEL SELECTION
    # -------------------------------------------------
    kernel_option = st.selectbox(
        "Select Kernel",
        ["linear", "rbf", "poly"]
    )

    c_value = st.slider("Select C Value", 0.1, 10.0, 1.0)

    # -------------------------------------------------
    # CREATE MODEL
    # -------------------------------------------------
    model = SVC(
        kernel=kernel_option,
        C=c_value,
        gamma='scale',
        probability=True
    )

    # -------------------------------------------------
    # TRAIN MODEL
    # -------------------------------------------------
    model.fit(X_train, y_train)

    # -------------------------------------------------
    # PREDICTIONS
    # -------------------------------------------------
    y_pred = model.predict(X_test)

    # -------------------------------------------------
    # EVALUATION
    # -------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Model Accuracy")
    st.success(f"Accuracy: {accuracy:.2f}")

    # -------------------------------------------------
    # CONFUSION MATRIX
    # -------------------------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # -------------------------------------------------
    # CLASSIFICATION REPORT
    # -------------------------------------------------
    st.subheader("Classification Report")

    report = classification_report(y_test, y_pred)
    st.text(report)

# =====================================================
# SVR IMPLEMENTATION
# =====================================================
elif option == "SVR Implementation":

    st.header("SVR Implementation")

    # -------------------------------------------------
    # FILE UPLOAD
    # -------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv']
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        # ---------------------------------------------
        # DISPLAY DATASET
        # ---------------------------------------------
        st.subheader("Dataset")
        st.dataframe(df.head())

        # ---------------------------------------------
        # DATASET INFO
        # ---------------------------------------------
        st.subheader("Dataset Information")
        st.write(df.info())

        st.subheader("Statistical Description")
        st.write(df.describe())

        # ---------------------------------------------
        # NULL VALUES
        # ---------------------------------------------
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        # ---------------------------------------------
        # BOXPLOT
        # ---------------------------------------------
        st.subheader("Box Plot")

        fig, ax = plt.subplots(figsize=(10, 5))
        df.boxplot(ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)

        # ---------------------------------------------
        # OUTLIER HANDLING USING IQR
        # ---------------------------------------------
        numeric_cols = df.select_dtypes(include=np.number).columns

        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1

        df_clean = df[
            ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) |
              (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        ]

        st.subheader("Dataset After Removing Outliers")
        st.write(df_clean.shape)

        # ---------------------------------------------
        # PAIRPLOT
        # ---------------------------------------------
        st.subheader("Pair Plot")

        fig = sns.pairplot(df_clean)
        st.pyplot(fig)

        # ---------------------------------------------
        # BAR PLOT
        # ---------------------------------------------
        st.subheader("Bar Plot")

        fig2, ax2 = plt.subplots()
        df_clean.iloc[:, 0].value_counts().head(10).plot(kind='bar', ax=ax2)
        st.pyplot(fig2)

        # ---------------------------------------------
        # SELECT FEATURES AND TARGET
        # ---------------------------------------------
        st.subheader("Select Features and Target")

        target_column = st.selectbox(
            "Select Target Column",
            df_clean.columns
        )

        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]

        # ---------------------------------------------
        # TRAIN TEST SPLIT
        # ---------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # ---------------------------------------------
        # FEATURE SCALING
        # ---------------------------------------------
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ---------------------------------------------
        # SVR MODEL
        # ---------------------------------------------
        model = SVR(kernel='rbf')

        # ---------------------------------------------
        # TRAIN MODEL
        # ---------------------------------------------
        model.fit(X_train, y_train)

        # ---------------------------------------------
        # PREDICTIONS
        # ---------------------------------------------
        y_pred = model.predict(X_test)

        # ---------------------------------------------
        # EVALUATION
        # ---------------------------------------------
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Model Evaluation")

        st.success(f"Mean Squared Error: {mse:.2f}")
        st.success(f"Mean Absolute Error: {mae:.2f}")
        st.success(f"R2 Score: {r2:.2f}")

        # ---------------------------------------------
        # ACTUAL VS PREDICTED
        # ---------------------------------------------
        st.subheader("Actual vs Predicted")

        result_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })

        st.dataframe(result_df.head(10))

        fig3, ax3 = plt.subplots()
        ax3.scatter(y_test, y_pred)
        ax3.set_xlabel("Actual Values")
        ax3.set_ylabel("Predicted Values")
        ax3.set_title("Actual vs Predicted")
        st.pyplot(fig3)

    else:
        st.warning("Please upload a CSV file.")
