import os
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBClassifier

# 1. Define feature columns
FEATURE_COLUMNS = [
    f"V{i}" for i in range(1, 29)
] + ["Amount"]

# 2. Set prediction threshold
THRESHOLD = 0.50

# 3. Model location
MODEL_PATH = "outputs/models/xgboost_tuned.json"

# 4. Check whether model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Tuned XGBoost model not found. "
        "Please generate the model first."
    )

# 5. Load trained model
model = XGBClassifier()
model.load_model(MODEL_PATH)

# 6. Create SHAP explainer
explainer = shap.TreeExplainer(model)

# 7. Create FastAPI application
app = FastAPI(
    title="Intelligent Financial Fraud Detection API",
    description="API for credit card fraud prediction",
    version="1.0"
)

print("Fraud Detection Model Loaded!")

# Request for one transaction
class TransactionRequest(BaseModel):
    features: dict[str, float]


# Request for multiple transactions
class BatchRequest(BaseModel):
    transactions: list[dict[str, float]]
{
    "features": {
        "V1": -1.35,
        "V2": 0.62,
        "V3": 1.24,
        "V4": -0.51,
        "V5": 0.33,
        "V6": -0.82,
        "V7": 0.15,
        "V8": -0.24,
        "V9": 0.47,
        "V10": -0.16,
        "V11": 0.38,
        "V12": -0.72,
        "V13": 0.21,
        "V14": -0.48,
        "V15": 0.55,
        "V16": -0.19,
        "V17": -0.63,
        "V18": 0.31,
        "V19": -0.27,
        "V20": 0.42,
        "V21": -0.35,
        "V22": 0.16,
        "V23": -0.12,
        "V24": 0.28,
        "V25": -0.44,
        "V26": 0.37,
        "V27": -0.22,
        "V28": 0.18,
        "Amount": 149.99
    }
}
# Validate transaction data
def prepare_data(transactions):

    # Check batch size
    if len(transactions) == 0:
        raise HTTPException(
            status_code=400,
            detail="No transactions provided."
        )

    if len(transactions) > 5000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 5000 transactions per request."
        )

    rows = []

    for features in transactions:

        # Check missing columns
        missing = [
            column for column in FEATURE_COLUMNS
            if column not in features
        ]

        # Check unexpected columns
        extra = [
            column for column in features
            if column not in FEATURE_COLUMNS
        ]

        if missing or extra:
            raise HTTPException(
                status_code=400,
                detail={
                    "missing_features": missing,
                    "unexpected_features": extra
                }
            )

        values = []

        for column in FEATURE_COLUMNS:

            value = features[column]

            if not np.isfinite(value):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value in {column}"
                )

            values.append(value)

        rows.append(values)

    return pd.DataFrame(
        rows,
        columns=FEATURE_COLUMNS
    )

# Generate predictions
def make_predictions(data):

    probabilities = model.predict_proba(
        data
    )[:, 1]

    results = []

    for probability in probabilities:

        predicted_class = int(
            probability >= THRESHOLD
        )

        if predicted_class == 1:
            label = "Fraudulent"
        else:
            label = "Legitimate"

        results.append({
            "fraud_probability": round(
                float(probability), 6
            ),
            "predicted_class": predicted_class,
            "predicted_label": label
        })

    return results

# Endpoint 1: Check API status
@app.get("/health")
def health_check():

    return {
        "status": "running",
        "model": "Tuned XGBoost",
        "threshold": THRESHOLD
    }


# Endpoint 2: Predict one transaction
@app.post("/predict")
def predict_transaction(request: TransactionRequest):

    data = prepare_data([
        request.features
    ])

    results = make_predictions(data)

    return results[0]


# Endpoint 3: Predict multiple transactions
@app.post("/predict/batch")
def predict_batch(request: BatchRequest):

    data = prepare_data(
        request.transactions
    )

    results = make_predictions(data)

    return {
        "total_transactions": len(results),
        "fraud_threshold": THRESHOLD,
        "predictions": results
    }


# Endpoint 4: Explain one transaction
@app.post("/explain")
def explain_transaction(request: TransactionRequest):

    data = prepare_data([
        request.features
    ])

    probability = model.predict_proba(
        data
    )[0][1]

    explanation = explainer(data)

    shap_values = explanation.values[0]

    base_value = float(
        np.array(explanation.base_values).reshape(-1)[0]
    )

    contributions = []

    for i, feature in enumerate(FEATURE_COLUMNS):

        contributions.append({
            "feature": feature,
            "feature_value": float(data.iloc[0][feature]),
            "shap_value": float(shap_values[i])
        })

    # Sort by absolute SHAP contribution
    contributions.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True
    )

    return {
        "fraud_probability": round(
            float(probability), 6
        ),
        "predicted_label": (
            "Fraudulent" if probability >= THRESHOLD
            else "Legitimate"
        ),
        "base_value": base_value,
        "explanation_scale": "Raw log-odds",
        "top_features": contributions[:10]
    }
