import pandas as pd
import sqlite3
import os

data = pd.read_csv("data/raw/creditcard.csv")
print("Original dataset:", data.shape)

data = data.drop_duplicates()
print("Cleaned dataset:", data.shape)

#Connect to SQLite database
connection = sqlite3.connect("data/processed/fraud_detection.db")

#Store data in a SQL table
data.to_sql(
    "transactions",
    connection,
    if_exists="replace",
    index=False
)

#Check total records stored
result = pd.read_sql_query(
    "SELECT COUNT(*) AS Total_Records FROM transactions",
    connection
)

print("\nDatabase Record Count:")
print(result)

#Close database connection
connection.close()

print("\nDatabase created successfully!")