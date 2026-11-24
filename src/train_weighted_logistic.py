import pandas as pd
import joblib
import os

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

# 1. Load datasets
train_data = pd.read_csv(
    "data/processed/scaled/train_scaled.csv"
)

validation_data = pd.read_csv(
    "data/processed/scaled/validation_scaled.csv"
)

# 2. Separate features and target
X_train = train_data.drop("Class", axis=1)
y_train = train_data["Class"]

X_validation = validation_data.drop("Class", axis=1)
y_validation = validation_data["Class"]

print("Training Shape:", X_train.shape)
print("Validation Shape:", X_validation.shape)


# Create Logistic Regression with class weighting
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_validation)

# Get fraud probabilities
probabilities = model.predict_proba(X_validation)[:, 1]

#Calculate evaluation metrics

accuracy = accuracy_score(y_validation, predictions)

precision = precision_score(
    y_validation,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_validation,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_validation,
    predictions,
    zero_division=0
)

ap = average_precision_score(
    y_validation,
    probabilities
)

roc_auc = roc_auc_score(
    y_validation,
    probabilities
)

#Display results
print("\nWEIGHTED LOGISTIC REGRESSION")
print("--------------------------------")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-Score:", round(f1, 4))
print("Average Precision:", round(ap, 4))
print("ROC-AUC:", round(roc_auc, 4))

# 8. Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_validation, predictions))

joblib.dump(
    model,
    "outputs/models/logistic_regression_balanced.pkl"
)