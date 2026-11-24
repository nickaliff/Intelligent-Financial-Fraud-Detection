import pandas as pd
import os

from xgboost import XGBClassifier

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score

# Load training data
train_data = pd.read_csv(
    "data/processed/train.csv"
)

# Separate features and target
X_train = train_data.drop(
    ["Time", "Class"], axis=1
)

y_train = train_data["Class"]

print("Training Shape:", X_train.shape)
print("Fraudulent Transactions:", y_train.sum())

# Six configurations to compare

configurations = [
    {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1
    },
    {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1
    },
    {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1
    },
    {
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1
    },
    {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1
    },
    {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5
    }
]

# Create time-series cross-validation
time_split = TimeSeriesSplit(n_splits=3)

best_score = 0
best_parameters = None

# Test each configuration
for i, parameters in enumerate(configurations):

    print("\nConfiguration:", i + 1)
    print("----------------------------")

    fold_scores = []

    # Run three validation folds
    for fold, (train_index, val_index) in enumerate(
        time_split.split(X_train), start=1
    ):

        X_fold_train = X_train.iloc[train_index]
        y_fold_train = y_train.iloc[train_index]

        X_fold_val = X_train.iloc[val_index]
        y_fold_val = y_train.iloc[val_index]

        print(
            "Fold", fold,
            "| Training Fraud:", y_fold_train.sum(),
            "| Validation Fraud:", y_fold_val.sum()
        )

        # Create model
        model = XGBClassifier(
            n_estimators=parameters["n_estimators"],
            max_depth=parameters["max_depth"],
            learning_rate=parameters["learning_rate"],
            min_child_weight=parameters["min_child_weight"],
            subsample=parameters["subsample"],
            colsample_bytree=parameters["colsample_bytree"],
            reg_lambda=parameters["reg_lambda"],
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )

        # Train model
        model.fit(X_fold_train, y_fold_train)

        # Get fraud probabilities
        probabilities = model.predict_proba(
            X_fold_val
        )[:, 1]

        # Calculate Average Precision
        score = average_precision_score(
            y_fold_val,
            probabilities
        )

        fold_scores.append(score)

        print("Fold Average Precision:", round(score, 4))

    # Calculate average score
    mean_score = sum(fold_scores) / len(fold_scores)

    print(
        "\nAverage Precision:",
        round(mean_score, 4)
    )

    # Store best configuration
    if mean_score > best_score:

        best_score = mean_score
        best_parameters = parameters.copy()


# Display best configuration
print("\nBEST CONFIGURATION")
print("----------------------------")

print("Best Average Precision:", round(best_score, 4))
print("Best Parameters:")

print(best_parameters)

# Train final selected configuration on full training data

best_model = XGBClassifier(
    n_estimators=best_parameters["n_estimators"],
    max_depth=best_parameters["max_depth"],
    learning_rate=best_parameters["learning_rate"],
    min_child_weight=best_parameters["min_child_weight"],
    subsample=best_parameters["subsample"],
    colsample_bytree=best_parameters["colsample_bytree"],
    reg_lambda=best_parameters["reg_lambda"],
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

best_model.fit(X_train, y_train)

# Save selected model
os.makedirs("outputs/models", exist_ok=True)

best_model.save_model(
    "outputs/models/xgboost_tuned.json"
)

print("\nTuned XGBoost model saved successfully!")
