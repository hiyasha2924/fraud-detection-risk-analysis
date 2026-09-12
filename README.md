# Fraud Detection & Risk Analysis Platform

An end-to-end machine learning project for detecting fraudulent credit-card transactions and assigning transaction-level risk scores.

## Project Overview

Credit-card fraud detection is a highly imbalanced classification problem because fraudulent transactions represent only a very small fraction of all transactions.

This project develops a machine learning pipeline using XGBoost and SMOTE to identify suspicious transactions and provide explainable risk scores.

## Features

- Exploratory Data Analysis
- Class-imbalance analysis
- Data preprocessing
- SMOTE-based oversampling
- XGBoost fraud classification
- ROC-AUC and PR-AUC evaluation
- Risk-score generation
- Low, Medium, High, and Critical risk categories
- SHAP-based model explainability
- CSV upload prediction
- Interactive Streamlit dashboard
- Population Stability Index for data-drift monitoring
- Automated testing using pytest

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib
- Seaborn
- Joblib
- Pytest

## Project Structure

```text
Fraud detection/
│
├── app.py
├── main.py
├── test_project.py
├── fraud_pipeline.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    ├── creditcard.csv
    └── test_transactions.csv