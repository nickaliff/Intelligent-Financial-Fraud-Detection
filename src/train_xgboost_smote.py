import pandas as pd
import os

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

# 1. Load scaled datasets
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

# 3. Display original class distribution
print("\nOriginal Training Distribution:")
print(y_train.value_counts())

# 4. Create SMOTE object
smote = SMOTE(
    sampling_strategy=0.5,
    random_state=42,
    k_neighbors=5
)

# 5. Apply SMOTE to training data
X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

# 6. Display new distribution
print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

print("\nNew Training Shape:")
print(X_train_smote.shape)

# 7. Create XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

# 8. Train using SMOTE data
model.fit(
    X_train_smote,
    y_train_smote
)

print("\nXGBoost SMOTE Training Completed!")

# 9. Predict validation data
predictions = model.predict(X_validation)

probabilities = model.predict_proba(
    X_validation
)[:, 1]

# 10. Calculate metrics
accuracy = accuracy_score(
    y_validation, predictions
)

precision = precision_score(
    y_validation, predictions,
    zero_division=0
)

recall = recall_score(
    y_validation, predictions,
    zero_division=0
)

f1 = f1_score(
    y_validation, predictions,
    zero_division=0
)

ap = average_precision_score(
    y_validation, probabilities
)

roc_auc = roc_auc_score(
    y_validation, probabilities
)

# 11. Display results
print("\nXGBOOST WITH SMOTE RESULTS")
print("--------------------------------")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-Score:", round(f1, 4))
print("Average Precision:", round(ap, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_validation, predictions))

# 12. Save model
os.makedirs("outputs/models", exist_ok=True)

model.save_model(
    "outputs/models/xgboost_smote.json"
)

print("\nSMOTE model saved successfully!")
