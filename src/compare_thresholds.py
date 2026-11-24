import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

#Load validation data
data = pd.read_csv(
    "data/processed/scaled/validation_scaled.csv"
)

X_validation = data.drop("Class", axis=1)
y_validation = data["Class"]

#Load trained models
models = {
    "Logistic Regression": joblib.load(
        "outputs/models/logistic_regression.pkl"
    ),

    "Weighted Logistic Regression": joblib.load(
        "outputs/models/logistic_regression_balanced.pkl"
    )
}

#Define thresholds
thresholds = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9
]

#Compare models
for name, model in models.items():

    print("\n", name)
    print("--------------------------------")

    # Get fraud probabilities
    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    for threshold in thresholds:

        # Apply threshold
        predictions = (
            probabilities >= threshold
        ).astype(int)

        # Calculate metrics
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

        # Count true positives and false positives
        tp = (
            (predictions == 1) &
            (y_validation == 1)
        ).sum()

        fp = (
            (predictions == 1) &
            (y_validation == 0)
        ).sum()

        fn = (
            (predictions == 0) &
            (y_validation == 1)
        ).sum()

        # Display results
        print(
            f"Threshold: {threshold:.1f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f} | "
            f"TP: {tp} | FP: {fp} | FN: {fn}"
        )
