import pandas as pd
import sqlite3

# 1. Load cleaned data from SQLite
connection = sqlite3.connect(
    "data/processed/fraud_detection.db"
)

data = pd.read_sql_query(
    "SELECT * FROM transactions",
    connection
)

connection.close()

# 2. Sort transactions by time
data = data.sort_values("Time").reset_index(drop=True)

print("Total Dataset:", data.shape)

# 3. Calculate time boundaries
total_records = len(data)

train_position = int(total_records * 0.70)
validation_position = int(total_records * 0.85)

train_end_time = data.loc[train_position, "Time"]
validation_end_time = data.loc[validation_position, "Time"]

# 4. Create training dataset
train_data = data[
    data["Time"] <= train_end_time
]

# 5. Create validation dataset
validation_data = data[
    (data["Time"] > train_end_time) &
    (data["Time"] <= validation_end_time)
]

# 6. Create testing dataset
test_data = data[
    data["Time"] > validation_end_time
]

# 7. Display dataset information
datasets = {
    "Training": train_data,
    "Validation": validation_data,
    "Testing": test_data
}

for name, dataset in datasets.items():

    print("\n", name, "Dataset")
    print("----------------------------")

    print("Total Records:", len(dataset))

    print(
        "Legitimate:",
        (dataset["Class"] == 0).sum()
    )

    print(
        "Fraudulent:",
        (dataset["Class"] == 1).sum()
    )

    print(
        "Fraud Percentage:",
        round(dataset["Class"].mean() * 100, 3),
        "%"
    )

    print(
        "Time Range:",
        dataset["Time"].min(),
        "to",
        dataset["Time"].max()
    )

#8. Save the datasets
train_data.to_csv(
    "data/processed/train.csv",
    index=False
)

validation_data.to_csv(
    "data/processed/validation.csv",
    index=False
)

test_data.to_csv(
    "data/processed/test.csv",
    index=False
)

print("\nDatasets saved successfully!")