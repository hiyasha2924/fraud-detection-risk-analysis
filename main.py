import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Check library versions
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)


# Load the dataset
df = pd.read_csv("data/creditcard.csv")

# Display the first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Display the number of rows and columns
print("\nShape:", df.shape)

# Display all column names
print("\nColumns:")
print(df.columns)

# Display information about the dataset
print("\nDataset information:")
print(df.info())

# Display the number of legitimate and fraudulent transactions
print("\nClass distribution:")
print(df["Class"].value_counts())

# Display the percentage of legitimate and fraudulent transactions
print("\nClass percentage:")
print(df["Class"].value_counts(normalize=True) * 100)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum().sum())

# Display statistical summary of numerical columns
print("\nStatistical summary:")
print(df.describe())

# Display data types of all columns
print("\nData types:")
print(df.dtypes)

# Analyze transaction amounts
print("\nTransaction amount statistics:")
print(df["Amount"].describe())

# Compare transaction amounts for normal and fraudulent transactions
print("\nAverage transaction amount by class:")
print(df.groupby("Class")["Amount"].mean())

print("\nMedian transaction amount by class:")
print(df.groupby("Class")["Amount"].median())

import matplotlib.pyplot as plt

# Plot class distribution
class_counts = df["Class"].value_counts()

plt.figure(figsize=(6, 4))
plt.bar(["Normal", "Fraud"], class_counts.values)

plt.title("Normal vs Fraudulent Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.show()

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeature columns:")
print(X.columns.tolist())

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

from sklearn.preprocessing import StandardScaler

# Create a StandardScaler
scaler = StandardScaler()

# Columns that need scaling
columns_to_scale = ["Time", "Amount"]

# Fit scaler only on training data
scaler.fit(X_train[columns_to_scale])

# Transform training and testing data
X_train.loc[:, columns_to_scale] = scaler.transform(
    X_train[columns_to_scale]
)

X_test.loc[:, columns_to_scale] = scaler.transform(
    X_test[columns_to_scale]
)

print("\nScaled training data:")
print(X_train[columns_to_scale].head())

print("\nScaled testing data:")
print(X_test[columns_to_scale].head())

print("\nBefore SMOTE:")
print(y_train.value_counts())

from imblearn.over_sampling import SMOTE

# Create SMOTE object
smote = SMOTE(random_state=42)

# Apply SMOTE only to the training data
X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

print("\nOriginal training shape:", X_train.shape)
print("SMOTE training shape:", X_train_smote.shape)

from sklearn.linear_model import LogisticRegression

# Create Logistic Regression model
model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

# Train the model using the SMOTE-balanced training data
model.fit(X_train_smote, y_train_smote)

print("\nModel training completed!")

# Make predictions on the test set
y_pred = model.predict(X_test)

print("\nFirst 20 predictions:")
print(y_pred[:20])

# Get probability of fraud
y_prob = model.predict_proba(X_test)[:, 1]

print("\nFirst 20 fraud probabilities:")
print(y_prob[:20])

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)

# PR-AUC
pr_auc = average_precision_score(y_test, y_prob)

print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)

from xgboost import XGBClassifier

# Create XGBoost model
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

print("\nTraining XGBoost model...")

xgb_model.fit(
    X_train_smote,
    y_train_smote
)

print("XGBoost training completed!")

# Make predictions on the test set
xgb_pred = xgb_model.predict(X_test)

# Get fraud probabilities
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

print("\nFirst 20 XGBoost predictions:")
print(xgb_pred[:20])

print("\nFirst 20 XGBoost fraud probabilities:")
print(xgb_prob[:20])

# Evaluate XGBoost
print("\nXGBoost Confusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))

print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_pred))

xgb_roc_auc = roc_auc_score(y_test, xgb_prob)
xgb_pr_auc = average_precision_score(y_test, xgb_prob)

print("XGBoost ROC-AUC:", xgb_roc_auc)
print("XGBoost PR-AUC:", xgb_pr_auc)

import numpy as np
from sklearn.metrics import roc_curve

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, xgb_prob)

# Find the best recall while keeping FPR <= 1%
valid_indices = np.where(fpr <= 0.01)[0]

best_index = valid_indices[np.argmax(tpr[valid_indices])]

best_fpr = fpr[best_index]
best_recall = tpr[best_index]
best_threshold = thresholds[best_index]

print("\nRecall @ 1% FPR:")
print("Best threshold:", best_threshold)
print("FPR:", best_fpr)
print("Recall:", best_recall)

from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

# Create a pipeline where SMOTE is applied separately inside each CV fold
pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("xgb", XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ))
])

# Hyperparameter search space
param_grid = {
    "xgb__n_estimators": [100, 200, 300],
    "xgb__max_depth": [3, 5, 7],
    "xgb__learning_rate": [0.03, 0.05, 0.1],
    "xgb__subsample": [0.8, 1.0],
    "xgb__colsample_bytree": [0.8, 1.0]
}

# Stratified 3-fold cross-validation
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

# Hyperparameter tuning using PR-AUC
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=5,
    scoring="average_precision",
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("\nStarting hyperparameter tuning...")

random_search.fit(X_train, y_train)

print("\nHyperparameter tuning completed!")

print("\nBest parameters:")
print(random_search.best_params_)

print("\nBest cross-validation PR-AUC:")
print(random_search.best_score_)

# Get the best tuned model
best_model = random_search.best_estimator_

# Predict on untouched test data
tuned_pred = best_model.predict(X_test)
tuned_prob = best_model.predict_proba(X_test)[:, 1]

print("\nTuned XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, tuned_pred))

print("\nTuned XGBoost Classification Report:")
print(classification_report(y_test, tuned_pred))

tuned_roc_auc = roc_auc_score(y_test, tuned_prob)
tuned_pr_auc = average_precision_score(y_test, tuned_prob)

print("Tuned XGBoost ROC-AUC:", tuned_roc_auc)
print("Tuned XGBoost PR-AUC:", tuned_pr_auc)

# ==========================================
# Lesson 11: Recall @ 1% FPR
# ==========================================

from sklearn.metrics import roc_curve

# Calculate FPR, TPR and thresholds
fpr, tpr, thresholds = roc_curve(y_test, xgb_prob)

# Find thresholds where FPR is at most 1%
valid_indices = np.where(fpr <= 0.01)[0]

# Find the highest recall among those thresholds
best_index = valid_indices[np.argmax(tpr[valid_indices])]

best_threshold = thresholds[best_index]
best_fpr = fpr[best_index]
best_recall = tpr[best_index]

print("\n==========================================")
print("Recall @ 1% FPR")
print("==========================================")

print("Best threshold:", best_threshold)
print("FPR:", best_fpr)
print("Recall:", best_recall)

# ==========================================
# Lesson 12: XGBoost with scale_pos_weight
# ==========================================

# Calculate class imbalance ratio
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nClass imbalance ratio:")
print("Negative samples:", negative_count)
print("Positive samples:", positive_count)
print("scale_pos_weight:", scale_pos_weight)

# Create XGBoost model using class weighting
xgb_weighted = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=1.0,
    colsample_bytree=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

print("\nTraining weighted XGBoost model...")

xgb_weighted.fit(
    X_train,
    y_train
)

print("Weighted XGBoost training completed!")

# Make predictions
weighted_pred = xgb_weighted.predict(X_test)

# Get fraud probabilities
weighted_prob = xgb_weighted.predict_proba(X_test)[:, 1]

print("\nFirst 20 weighted XGBoost predictions:")
print(weighted_pred[:20])

print("\nFirst 20 weighted XGBoost fraud probabilities:")
print(weighted_prob[:20])

print("\nWeighted XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, weighted_pred))

print("\nWeighted XGBoost Classification Report:")
print(classification_report(y_test, weighted_pred))

weighted_roc_auc = roc_auc_score(y_test, weighted_prob)
weighted_pr_auc = average_precision_score(y_test, weighted_prob)

print("Weighted XGBoost ROC-AUC:", weighted_roc_auc)
print("Weighted XGBoost PR-AUC:", weighted_pr_auc)

# Calculate Recall @ 1% FPR
weighted_fpr, weighted_tpr, weighted_thresholds = roc_curve(
    y_test,
    weighted_prob
)

valid_indices = np.where(weighted_fpr <= 0.01)[0]

best_index = valid_indices[
    np.argmax(weighted_tpr[valid_indices])
]

weighted_best_threshold = weighted_thresholds[best_index]
weighted_best_fpr = weighted_fpr[best_index]
weighted_best_recall = weighted_tpr[best_index]

print("\nWeighted XGBoost Recall @ 1% FPR:")
print("Best threshold:", weighted_best_threshold)
print("FPR:", weighted_best_fpr)
print("Recall:", weighted_best_recall)

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# ==========================================
# Lesson 13: Model Performance Visualization
# ==========================================

# Logistic Regression ROC curve
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_prob)

# Baseline XGBoost ROC curve
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_prob)

# Weighted XGBoost ROC curve
weighted_fpr, weighted_tpr, _ = roc_curve(y_test, weighted_prob)

plt.figure(figsize=(8, 6))

plt.plot(
    lr_fpr,
    lr_tpr,
    label=f"Logistic Regression (AUC = {roc_auc_score(y_test, y_prob):.3f})"
)

plt.plot(
    xgb_fpr,
    xgb_tpr,
    label=f"XGBoost + SMOTE (AUC = {roc_auc_score(y_test, xgb_prob):.3f})"
)

plt.plot(
    weighted_fpr,
    weighted_tpr,
    label=f"XGBoost + Weight (AUC = {roc_auc_score(y_test, weighted_prob):.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()

plt.show()

# Calculate Precision-Recall curves

lr_precision, lr_recall, _ = precision_recall_curve(
    y_test,
    y_prob
)

xgb_precision, xgb_recall, _ = precision_recall_curve(
    y_test,
    xgb_prob
)

weighted_precision, weighted_recall, _ = precision_recall_curve(
    y_test,
    weighted_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    lr_recall,
    lr_precision,
    label=f"Logistic Regression (PR-AUC = {average_precision_score(y_test, y_prob):.3f})"
)

plt.plot(
    xgb_recall,
    xgb_precision,
    label=f"XGBoost + SMOTE (PR-AUC = {average_precision_score(y_test, xgb_prob):.3f})"
)

plt.plot(
    weighted_recall,
    weighted_precision,
    label=f"XGBoost + Weight (PR-AUC = {average_precision_score(y_test, weighted_prob):.3f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.grid()

plt.show()

import shap

# ==========================================
# Lesson 14: SHAP Explainability
# ==========================================

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(xgb_model)

print("SHAP explainer created!")

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

print("SHAP values calculated!")

# Calculate average feature importance
shap_importance = np.abs(shap_values).mean(axis=0)

feature_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean_ABS_SHAP": shap_importance
})

feature_importance = feature_importance.sort_values(
    by="Mean_ABS_SHAP",
    ascending=False
)

print("\nTop 15 features according to SHAP:")
print(feature_importance.head(15))

# SHAP summary plot
print("\nDisplaying SHAP summary plot...")

shap.summary_plot(
    shap_values,
    X_test,
    show=True
)

# ==========================================
# Lesson 15: Fraud Risk Scoring
# ==========================================

# Convert fraud probability into a 0-100 risk score
risk_scores = xgb_prob * 100

print("\nFirst 20 Fraud Risk Scores:")
print(risk_scores[:20])

def assign_risk_level(score):
    if score < 20:
        return "Low"
    elif score < 50:
        return "Medium"
    elif score < 80:
        return "High"
    else:
        return "Critical"

risk_levels = [
    assign_risk_level(score)
    for score in risk_scores
]

print("\nFirst 20 Risk Levels:")
print(risk_levels[:20])

risk_results = pd.DataFrame({
    "Actual_Class": y_test.values,
    "Fraud_Probability": xgb_prob,
    "Risk_Score": risk_scores,
    "Risk_Level": risk_levels
})

print("\nSample Risk Analysis:")
print(risk_results.head(20))

print("\nRisk Level Distribution:")
print(risk_results["Risk_Level"].value_counts())

print("\nFraud cases by Risk Level:")
print(
    risk_results[risk_results["Actual_Class"] == 1]["Risk_Level"]
    .value_counts()
)

print("\nTop 10 Highest-Risk Transactions:")

top_risk_transactions = risk_results.sort_values(
    by="Risk_Score",
    ascending=False
).head(10)

print(top_risk_transactions)

# ==========================================
# Explain the highest-risk transaction
# ==========================================

highest_risk_index = np.argmax(xgb_prob)

print("\nHighest-risk transaction index:")
print(highest_risk_index)

print("\nHighest-risk fraud probability:")
print(xgb_prob[highest_risk_index])

print("\nHighest-risk actual class:")
print(y_test.iloc[highest_risk_index])

print("\nTop contributing features:")
transaction_shap = pd.DataFrame({
    "Feature": X_test.columns,
    "SHAP_Value": shap_values[highest_risk_index]
})

transaction_shap["Absolute_SHAP"] = np.abs(
    transaction_shap["SHAP_Value"]
)

transaction_shap = transaction_shap.sort_values(
    by="Absolute_SHAP",
    ascending=False
)

print(transaction_shap.head(10))

import joblib

# Save the baseline XGBoost model
joblib.dump(xgb_model, "xgb_model.pkl")

print("\nXGBoost model saved successfully!")

# ==========================================
# Lesson 19: Final ML Pipeline
# ==========================================

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFinal feature shape:", X.shape)

X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training samples:", X_train_final.shape[0])
print("Testing samples:", X_test_final.shape[0])

numeric_scaled_features = ["Time", "Amount"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "scaler",
            StandardScaler(),
            numeric_scaled_features
        )
    ],
    remainder="passthrough"
)

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),

    ("smote", SMOTE(
        random_state=42
    )),

    ("xgb", XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    ))
])

print("\nTraining final pipeline...")

final_pipeline.fit(
    X_train_final,
    y_train_final
)

print("Final pipeline training completed!")

final_prob = final_pipeline.predict_proba(
    X_test_final
)[:, 1]

final_pred = final_pipeline.predict(
    X_test_final
)

print("\nFinal Pipeline Classification Report:")
print(
    classification_report(
        y_test_final,
        final_pred
    )
)

final_roc_auc = roc_auc_score(
    y_test_final,
    final_prob
)

final_pr_auc = average_precision_score(
    y_test_final,
    final_prob
)

print("Final Pipeline ROC-AUC:", final_roc_auc)
print("Final Pipeline PR-AUC:", final_pr_auc)

joblib.dump(
    final_pipeline,
    "fraud_pipeline.pkl"
)

print("\nFinal pipeline saved successfully!")

