import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier

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

# 3. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# 4. Train model
model.fit(X_train, y_train)

print("\nRandom Forest Training Completed!")

# 5. Make predictions
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
print("\nRANDOM FOREST RESULTS")
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

# 9. Calculate feature importance
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

# Sort by importance
importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance.head(10))

# 10. Save feature importance
os.makedirs("outputs", exist_ok=True)

importance.to_csv(
    "outputs/random_forest_feature_importance.csv",
    index=False
)

# 11. Save trained model
os.makedirs("outputs/models", exist_ok=True)

joblib.dump(
    model,
    "outputs/models/random_forest.pkl"
)

print("\nRandom Forest model saved!")
