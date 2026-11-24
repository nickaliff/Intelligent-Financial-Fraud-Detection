import pandas as pd
import os

# Create reports folder
os.makedirs("reports", exist_ok=True)

# 1. Model comparison results
model_results = [
    {
        "Model": "Logistic Regression",
        "Variant": "Baseline",
        "Threshold": 0.10,
        "Precision": 0.7231,
        "Recall": 0.8545,
        "F1_Score": 0.7833,
        "Average_Precision": 0.7789,
        "ROC_AUC": 0.9819,
        "TP": 47,
        "FP": 18,
        "FN": 8
    },
    {
        "Model": "Logistic Regression",
        "Variant": "Class Weighted",
        "Threshold": 0.90,
        "Precision": 0.2193,
        "Recall": 0.9091,
        "F1_Score": 0.3534,
        "Average_Precision": 0.8367,
        "ROC_AUC": 0.9812,
        "TP": 50,
        "FP": 178,
        "FN": 5
    },
    {
        "Model": "Random Forest",
        "Variant": "Baseline",
        "Threshold": 0.30,
        "Precision": 0.9767,
        "Recall": 0.7636,
        "F1_Score": 0.8571,
        "Average_Precision": 0.8581,
        "ROC_AUC": 0.9853,
        "TP": 42,
        "FP": 1,
        "FN": 13
    },
    {
        "Model": "XGBoost",
        "Variant": "Baseline",
        "Threshold": 0.70,
        "Precision": 1.0000,
        "Recall": 0.7636,
        "F1_Score": 0.8660,
        "Average_Precision": 0.8553,
        "ROC_AUC": 0.9883,
        "TP": 42,
        "FP": 0,
        "FN": 13
    },
    {
        "Model": "XGBoost",
        "Variant": "Class Weighted",
        "Threshold": 0.95,
        "Precision": 0.9767,
        "Recall": 0.7636,
        "F1_Score": 0.8571,
        "Average_Precision": 0.8424,
        "ROC_AUC": 0.9922,
        "TP": 42,
        "FP": 1,
        "FN": 13
    },
    {
        "Model": "XGBoost",
        "Variant": "SMOTE",
        "Threshold": 0.90,
        "Precision": 0.9348,
        "Recall": 0.7818,
        "F1_Score": 0.8515,
        "Average_Precision": 0.8254,
        "ROC_AUC": 0.9847,
        "TP": 43,
        "FP": 3,
        "FN": 12
    },
    {
        "Model": "XGBoost",
        "Variant": "Tuned",
        "Threshold": 0.50,
        "Precision": 1.0000,
        "Recall": 0.7636,
        "F1_Score": 0.8660,
        "Average_Precision": 0.8570,
        "ROC_AUC": 0.9892,
        "TP": 42,
        "FP": 0,
        "FN": 13
    }
]

# 2. Convert to DataFrame
comparison = pd.DataFrame(model_results)

# 3. Sort by F1-score
comparison = comparison.sort_values(
    by="F1_Score",
    ascending=False
)

# 4. Save model comparison
comparison.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("Model comparison report saved!")


# 5. Final test evaluation
test_results = {
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "Average Precision",
        "ROC-AUC"
    ],
    "Score": [
        0.9996,
        0.9500,
        0.7308,
        0.8261,
        0.7543,
        0.9817
    ]
}

test_report = pd.DataFrame(test_results)

test_report.to_csv(
    "reports/final_test_metrics.csv",
    index=False
)

print("Final test metrics saved!")


# 6. Final confusion matrix
confusion_results = {
    "Actual_Class": [
        "Legitimate",
        "Fraudulent"
    ],
    "Predicted_Legitimate": [
        42503,
        14
    ],
    "Predicted_Fraud": [
        2,
        38
    ]
}

confusion_report = pd.DataFrame(confusion_results)

confusion_report.to_csv(
    "reports/final_confusion_matrix.csv",
    index=False
)

print("Final confusion matrix saved!")

print("\nAll reports generated successfully!")
