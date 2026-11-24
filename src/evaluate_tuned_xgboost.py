import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)

# 1. Load validation data
data = pd.read_csv(
    "data/processed/validation.csv"
)

X_validation = data.drop(
    ["Time", "Class"], axis=1
)

y_validation = data["Class"]

# 2. Load original XGBoost model
original_model = XGBClassifier()

original_model.load_model(
    "outputs/models/xgboost_baseline.json"
)

# 3. Load tuned XGBoost model
tuned_model = XGBClassifier()

tuned_model.load_model(
    "outputs/models/xgboost_tuned.json"
)

# 4. Store models
models = {
    "Original XGBoost": original_model,
    "Tuned XGBoost": tuned_model
}

# 5. Evaluate both models
for name, model in models.items():

    print("\n", name)
    print("----------------------------")

    # Predictions
    predictions = model.predict(X_validation)

    # Fraud probabilities
    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # Evaluation metrics
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

    # Display results
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-Score:", round(f1, 4))
    print("Average Precision:", round(ap, 4))
    print("ROC-AUC:", round(roc_auc, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_validation, predictions
    ))
