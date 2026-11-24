import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

#Create output folder for graphs
os.makedirs("outputs/figures", exist_ok=True)

connection = sqlite3.connect(
    "data/processed/fraud_detection.db"
)

#Load data from SQL
data = pd.read_sql_query(
    "SELECT * FROM transactions",
    connection
)

connection.close()

#Set graph style
sns.set_theme(style="whitegrid")

print("Dataset Shape:", data.shape)

print("----------------------------")

#Analysis 1: Transaction Class Distribution

class_counts = data["Class"].value_counts().sort_index()

print("\nTransaction Class Distribution:")
print(class_counts)

#Create bar chart
plt.figure(figsize=(7, 5))

sns.barplot(
    x=class_counts.index,
    y=class_counts.values
)

plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")
plt.xticks([0, 1], ["Legitimate", "Fraudulent"])

plt.tight_layout()

plt.savefig(
    "outputs/figures/01_class_distribution.png"
)

plt.show()
plt.close()

print("----------------------------")

#Analysis 2: Transaction Amount Distribution

print("\nTransaction Amount Summary:")
print(data.groupby("Class")["Amount"].describe())

plt.figure(figsize=(10, 5))

sns.histplot(
    data=data,
    x="Amount",
    hue="Class",
    bins=50,
    element="step",
    stat="density",
    common_norm=False
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Density")

plt.tight_layout()

plt.savefig(
    "outputs/figures/02_amount_distribution.png"
)

plt.show()
plt.close()

print("----------------------------")

#Log transformation for visualization

data["Log_Amount"] = np.log1p(data["Amount"])

plt.figure(figsize=(10, 5))

sns.histplot(
    data=data,
    x="Log_Amount",
    hue="Class",
    bins=50,
    element="step",
    stat="density",
    common_norm=False
)

plt.title("Log-Transformed Transaction Amount")
plt.xlabel("Log Transaction Amount")
plt.ylabel("Density")

plt.tight_layout()

plt.savefig(
    "outputs/figures/03_log_amount_distribution.png"
)

plt.show()
plt.close()

print("--------------------------")

#Analysis 3: Transaction Activity Over Time

#Convert seconds into elapsed hours
data["Hour"] = (data["Time"] / 3600).astype(int)

#Group transactions by hour
hourly_data = data.groupby("Hour").agg(
    Total_Transactions=("Class", "count"),
    Fraud_Transactions=("Class", "sum")
).reset_index()

#Calculate fraud percentage
hourly_data["Fraud_Percentage"] = (
    hourly_data["Fraud_Transactions"] /
    hourly_data["Total_Transactions"]
) * 100

print("\nHourly Transaction Summary:")
print(hourly_data.head())

# Display hours with highest fraud percentages

top_hours = hourly_data.sort_values(
    "Fraud_Percentage",
    ascending=False
)

print("\nHours with Highest Fraud Percentages:")
print(
    top_hours[
        [
            "Hour",
            "Total_Transactions",
            "Fraud_Transactions",
            "Fraud_Percentage"
        ]
    ].head(10)
)

#Plot fraud percentage
plt.figure(figsize=(12, 5))

sns.lineplot(
    data=hourly_data,
    x="Hour",
    y="Fraud_Percentage",
    marker="o"
)

plt.title("Fraud Percentage Across Time")
plt.xlabel("Elapsed Hour")
plt.ylabel("Fraud Percentage (%)")

plt.tight_layout()

plt.savefig(
    "outputs/figures/04_fraud_over_time.png"
)

plt.show()
plt.close()

print("--------------------------")

#Analysis 4: Feature Correlation

#Calculate correlations
correlations = data.corr(numeric_only=True)["Class"]

#Remove Class itself
correlations = correlations.drop("Class")

#Sort by absolute correlation
top_features = (
    correlations.abs()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Features Related to Fraud:")
print(correlations[top_features.index])

#Create heatmap
selected_columns = list(top_features.index) + ["Class"]

correlation_matrix = data[selected_columns].corr()

plt.figure(figsize=(11, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0
)

plt.title("Correlation Heatmap of Top 10 Features")

plt.tight_layout()

plt.savefig(
    "outputs/figures/05_correlation_heatmap.png"
)

plt.show()
plt.close()

print("--------------------------")

#Analysis 5: Feature Distribution Comparison

selected_features = list(top_features.index[:4])

print("\nSelected Features:")
print(selected_features)

#Create four charts
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for i, feature in enumerate(selected_features):

    sns.histplot(
        data=data,
        x=feature,
        hue="Class",
        bins=50,
        stat="density",
        common_norm=False,
        element="step",
        ax=axes[i // 2, i % 2]
    )

    axes[i // 2, i % 2].set_title(
        feature + " Distribution"
    )

plt.tight_layout()

plt.savefig(
    "outputs/figures/06_feature_distributions.png"
)

plt.show()
plt.close()