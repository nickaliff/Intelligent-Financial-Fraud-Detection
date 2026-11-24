import pandas as pd
import os

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

# 1. Load datasets
train_data = pd.read_csv(
    "data/processed/train.csv"
)

validation_data = pd.read_csv(
    "data/processed/validation.csv"
)

# 2. Separate features and target
X_train = train_data.drop(
    ["Time", "Class"], axis=1
)

y_train = train_data["Class"]

X_validation = validation_data.drop(
    ["Time", "Class"], axis=1
)

y_validation = validation_data["Class"]

print("Training Shape:", X_train.shape)
print("Validation Shape:", X_validation.shape)

# 3. Calculate class weight
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_weight = negative_count / positive_count

print("\nClass Weight Information")
print("----------------------------")
print("Legitimate:", negative_count)
print("Fraudulent:", positive_count)
print("Scale Pos Weight:", round(scale_weight, 2))

# 4. Create weighted XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_weight,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

# 5. Train model
model.fit(X_train, y_train)

print("\nWeighted XGBoost Training Completed!")

# 6. Generate predictions
predictions = model.predict(X_validation)

probabilities = model.predict_proba(
    X_validation
)[:, 1]

# 7. Calculate evaluation metrics
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

# 8. Display results
print("\nWEIGHTED XGBOOST RESULTS")
print("----------------------------")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-Score:", round(f1, 4))
print("Average Precision:", round(ap, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_validation, predictions))

# 9. Save model
os.makedirs("outputs/models", exist_ok=True)

model.save_model(
    "outputs/models/xgboost_weighted.json"
)

print("\nWeighted XGBoost model saved!")
