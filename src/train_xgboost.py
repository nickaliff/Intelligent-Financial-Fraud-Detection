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

# 3. Create XGBoost model
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

# 4. Train model
model.fit(X_train, y_train)

print("\nXGBoost Training Completed!")

# 5. Generate predictions
predictions = model.predict(X_validation)

# Get fraud probabilities
probabilities = model.predict_proba(
    X_validation
)[:, 1]

# 6. Calculate metrics
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

# 7. Display results
print("\nXGBOOST RESULTS")
print("--------------------------------")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-Score:", round(f1, 4))
print("Average Precision:", round(ap, 4))
print("ROC-AUC:", round(roc_auc, 4))

# 8. Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_validation, predictions))

# 9. Feature importance
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance.head(10))

# 10. Create output folders
os.makedirs("outputs/models", exist_ok=True)

# 11. Save model
model.save_model(
    "outputs/models/xgboost_baseline.json"
)

# 12. Save feature importance
importance.to_csv(
    "outputs/xgboost_feature_importance.csv",
    index=False
)

print("\nXGBoost model saved successfully!")
