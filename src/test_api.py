import pandas as pd
import requests

# 1. Load one validation transaction
data = pd.read_csv(
    "data/processed/validation.csv"
)

# 2. Select the first transaction
transaction = data.drop(
    ["Time", "Class"], axis=1
).iloc[0]

# 3. Convert transaction into a dictionary
features = transaction.to_dict()

# 4. Send transaction to API
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"features": features}
)

# 5. Display API response
print("API Status:", response.status_code)

print("\nPrediction Result:")
print(response.json())

# 6. Request SHAP explanation
explanation_response = requests.post(
    "http://127.0.0.1:8000/explain",
    json={"features": features}
)

print("\nSHAP Explanation:")
print(explanation_response.json())