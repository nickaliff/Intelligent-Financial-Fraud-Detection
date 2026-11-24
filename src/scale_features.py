import pandas as pd
import os
import joblib

from sklearn.preprocessing import StandardScaler

#1. Load the three datasets
train_data = pd.read_csv("data/processed/train.csv")
validation_data = pd.read_csv("data/processed/validation.csv")
test_data = pd.read_csv("data/processed/test.csv")

#2. Select input features
# Exclude Time and Class
feature_columns = [
    column for column in train_data.columns
    if column not in ["Time", "Class"]
]

#3. Separate input features and target
X_train = train_data[feature_columns]
y_train = train_data["Class"]

X_validation = validation_data[feature_columns]
y_validation = validation_data["Class"]

X_test = test_data[feature_columns]
y_test = test_data["Class"]

#4. Create scaler
scaler = StandardScaler()

#5. Fit using training data only
X_train_scaled = scaler.fit_transform(X_train)

#6. Transform validation and testing data
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

#7. Convert results back into DataFrames
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=feature_columns
)

X_validation_scaled = pd.DataFrame(
    X_validation_scaled,
    columns=feature_columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=feature_columns
)

#8. Add target column back
train_scaled = X_train_scaled.copy()
train_scaled["Class"] = y_train.reset_index(drop=True)

validation_scaled = X_validation_scaled.copy()
validation_scaled["Class"] = y_validation.reset_index(drop=True)

test_scaled = X_test_scaled.copy()
test_scaled["Class"] = y_test.reset_index(drop=True)

#9. Create output folder
os.makedirs("data/processed/scaled", exist_ok=True)

#10. Save scaled datasets
train_scaled.to_csv(
    "data/processed/scaled/train_scaled.csv",
    index=False
)

validation_scaled.to_csv(
    "data/processed/scaled/validation_scaled.csv",
    index=False
)

test_scaled.to_csv(
    "data/processed/scaled/test_scaled.csv",
    index=False
)

#11. Save scaler for future use
joblib.dump(
    scaler,
    "data/processed/scaled/scaler.pkl"
)

#12. Display results
print("Feature Scaling Completed!")
print("--------------------------------")

print("Number of Features:", len(feature_columns))

print("Training Shape:", train_scaled.shape)
print("Validation Shape:", validation_scaled.shape)
print("Testing Shape:", test_scaled.shape)

print("\nMissing Values:")
print(train_scaled.isnull().sum().sum())

print("\nTraining Feature Means (first 5):")
print(X_train_scaled.iloc[:, :5].mean().round(4))

print("\nScaled datasets saved successfully!")