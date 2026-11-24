import pandas as pd
import matplotlib.pyplot as plt
import os

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    precision_recall_curve
)

# 1. Load testing dataset
test_data = pd.read_csv(
    "data/processed/test.csv"
)

# 2. Separate features and target
X_test = test_data.drop(
    ["Time", "Class"], axis=1
)

y_test = test_data["Class"]

print("Testing Shape:", X_test.shape)
print("Fraudulent Transactions:", y_test.sum())

# 3. Load tuned XGBoost model
model = XGBClassifier()

model.load_model(
    "outputs/models/xgboost_tuned.json"
)

print("\nTuned XGBoost Model Loaded!")

# 4. Get fraud probabilities
probabilities = model.predict_proba(
    X_test
)[:, 1]

# 5. Apply selected threshold
threshold = 0.50

predictions = (
    probabilities >= threshold
).astype(int)

print("\nPredictions Generated!")

# 6. Calculate evaluation metrics
accuracy = accuracy_score(
    y_test, predictions
)

precision = precision_score(
    y_test, predictions,
    zero_division=0
)

recall = recall_score(
    y_test, predictions,
    zero_division=0
)

f1 = f1_score(
    y_test, predictions,
    zero_division=0
)

ap = average_precision_score(
    y_test, probabilities
)

roc_auc = roc_auc_score(
    y_test, probabilities
)

# 7. Display results
print("\nFINAL TEST RESULTS")
print("--------------------------------")

print("Threshold:", threshold)
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-Score:", round(f1, 4))
print("Average Precision:", round(ap, 4))
print("ROC-AUC:", round(roc_auc, 4))

# 8. Display confusion matrix
cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")
print(cm)

# 9. Create output folders
os.makedirs("outputs/figures", exist_ok=True)

# 10. Plot confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"]
)

display.plot(cmap="Blues", values_format="d")

plt.title("Final XGBoost Test Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "outputs/figures/11_final_confusion_matrix.png",
    dpi=300
)

plt.show()
plt.close()

# 11. Calculate precision-recall values
precision_values, recall_values, thresholds = (
    precision_recall_curve(
        y_test,
        probabilities
    )
)

# 12. Plot precision-recall curve
plt.figure(figsize=(8, 5))

plt.plot(
    recall_values,
    precision_values
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Final XGBoost Precision-Recall Curve")

plt.tight_layout()

plt.savefig(
    "outputs/figures/12_final_precision_recall.png",
    dpi=300
)

plt.show()
plt.close()

print("\nEvaluation graphs saved!")

# 13. Create prediction report
prediction_report = pd.DataFrame({
    "Time": test_data["Time"],
    "Actual_Class": y_test,
    "Fraud_Probability": probabilities,
    "Predicted_Class": predictions
})

# 14. Save prediction report
prediction_report.to_csv(
    "outputs/final_test_predictions.csv",
    index=False
)

# 15. Save evaluation metrics
metrics_report = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "Average Precision",
        "ROC-AUC"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        ap,
        roc_auc
    ]
})

metrics_report.to_csv(
    "outputs/final_test_metrics.csv",
    index=False
)

print("\nFinal evaluation reports saved!")
