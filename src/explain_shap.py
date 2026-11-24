import pandas as pd
import matplotlib.pyplot as plt
import shap
import os

from xgboost import XGBClassifier

# 1. Load validation data
data = pd.read_csv(
    "data/processed/validation.csv"
)

# 2. Separate features and target
X_validation = data.drop(
    ["Time", "Class"], axis=1
)

y_validation = data["Class"]

# 3. Load tuned XGBoost model
model = XGBClassifier()

model.load_model(
    "outputs/models/xgboost_tuned.json"
)

print("SHAP Analysis Setup Completed!")
print("Validation Shape:", X_validation.shape)

# 4. Create SHAP explainer
explainer = shap.TreeExplainer(model)

# 5. Select 1,000 validation transactions
X_sample = X_validation.sample(
    n=1000,
    random_state=42
)

# 6. Calculate SHAP values
shap_values = explainer(X_sample)

print("\nSHAP Values Calculated!")
print("SHAP Shape:", shap_values.values.shape)

# 7. Generate SHAP feature importance
shap.plots.bar(
    shap_values,
    max_display=15,
    show=False
)

plt.title("SHAP Feature Importance")

plt.tight_layout()

plt.savefig(
    "outputs/figures/07_shap_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# 8. Generate SHAP summary plot
shap.plots.beeswarm(
    shap_values,
    max_display=15,
    show=False
)

plt.title("SHAP Summary Plot")

plt.tight_layout()

plt.savefig(
    "outputs/figures/08_shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# 9. Calculate mean absolute SHAP values
importance = pd.DataFrame({
    "Feature": X_sample.columns,
    "Mean_Absolute_SHAP": abs(
        shap_values.values
    ).mean(axis=0)
})

# 10. Sort by importance
importance = importance.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

print("\nTop 15 SHAP Features:")
print(importance.head(15))

# 11. Save results
os.makedirs("outputs", exist_ok=True)

importance.to_csv(
    "outputs/shap_feature_importance.csv",
    index=False
)

print("\nSHAP importance report saved!")

# 12. Get fraud probabilities
probabilities = model.predict_proba(
    X_validation
)[:, 1]

# 13. Predict using threshold 0.50
predictions = (
    probabilities >= 0.50
).astype(int)

# 14. Find correctly detected fraud transactions
fraud_indices = X_validation.index[
    (y_validation == 1) &
    (predictions == 1)
]

# 15. Select the first detected fraud transaction
sample_index = fraud_indices[0]

# Get the selected transaction
transaction = X_validation.loc[
    [sample_index]
]

# 16. Calculate SHAP values
local_shap = explainer(transaction)

# Display transaction information
print("\nIndividual Fraud Explanation")
print("----------------------------")

print("Transaction Index:", sample_index)

print("Actual Class:", y_validation.loc[sample_index])

print("Predicted Class:", predictions[sample_index])

print(
    "Fraud Probability:",
    round(probabilities[sample_index], 4)
)

# 17. Generate waterfall plot
shap.plots.waterfall(
    local_shap[0],
    max_display=15,
    show=False
)

plt.title("Individual Fraud Transaction Explanation")

plt.tight_layout()

plt.savefig(
    "outputs/figures/09_individual_fraud_explanation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nIndividual SHAP explanation saved!")

# 18. Separate fraudulent and legitimate transactions

fraud_data = X_validation[
    y_validation == 1
]

legitimate_data = X_validation[
    y_validation == 0
]

print("\nFraud Transactions:", len(fraud_data))
print("Legitimate Transactions:", len(legitimate_data))

# 19. Calculate SHAP values for all fraud transactions
fraud_shap = explainer(fraud_data)

# Calculate SHAP values for 1000 legitimate transactions
legitimate_sample = legitimate_data.sample(
    n=1000,
    random_state=42
)

legitimate_shap = explainer(legitimate_sample)

# 20. Calculate fraud-specific feature importance
fraud_importance = pd.DataFrame({
    "Feature": fraud_data.columns,
    "Mean_Absolute_SHAP": abs(
        fraud_shap.values
    ).mean(axis=0)
})

fraud_importance = fraud_importance.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

print("\nTop 10 Fraud-Specific SHAP Features:")
print(fraud_importance.head(10))

# 21. Save fraud-specific importance
fraud_importance.to_csv(
    "outputs/fraud_shap_importance.csv",
    index=False
)

# 22. Create fraud-specific SHAP summary
shap.plots.beeswarm(
    fraud_shap,
    max_display=15,
    show=False
)

plt.title("SHAP Summary for Fraudulent Transactions")

plt.tight_layout()

plt.savefig(
    "outputs/figures/10_fraud_shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nFraud-specific SHAP analysis completed!")
