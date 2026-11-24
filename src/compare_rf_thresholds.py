import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# 1. Load validation data
data = pd.read_csv(
    "data/processed/validation.csv"
)

X_validation = data.drop(
    ["Time", "Class"], axis=1
)

y_validation = data["Class"]

# 2. Load Random Forest model
model = joblib.load(
    "outputs/models/random_forest.pkl"
)

# 3. Get fraud probabilities
probabilities = model.predict_proba(
    X_validation
)[:, 1]

# 4. Define thresholds
thresholds = [
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90
]

# 5. Test different thresholds
for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

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

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | "
        f"TP: {tp} | FP: {fp} | FN: {fn}"
    )
