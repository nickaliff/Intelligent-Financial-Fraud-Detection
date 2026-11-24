import sqlite3
import pandas as pd

connection = sqlite3.connect(
    "data/processed/fraud_detection.db"
)

#Query 1: Transaction count by class
print("1. Transaction Distribution")

query1 = """
SELECT Class, 
    COUNT(*) AS Total_Transactions
FROM transactions
GROUP BY Class;
"""

result1 = pd.read_sql_query(query1, connection)
print(result1)

#Query 2: Fraud percentage
print("\n2. Transaction Percentage")

query2 = """
SELECT Class,
    COUNT(*) AS Total_Transactions,
    ROUND(
        COUNT(*) * 100.0 /
        (SELECT COUNT(*) FROM transactions), 3
    ) AS Percentage
FROM transactions
GROUP BY Class;
"""

result2 = pd.read_sql_query(query2, connection)
print(result2)

#Query 3: Average transaction amount
print("\n3. Average Transaction Amount")

query3 = """
SELECT Class,
    COUNT(*) AS Total_Transactions,
    ROUND(AVG(Amount), 2) AS Average_Amount,
    ROUND(MAX(Amount), 2) AS Maximum_Amount
FROM transactions
GROUP BY Class;
"""

result3 = pd.read_sql_query(query3, connection)
print(result3)

#Query 4: Ten highest-value fraudulent transactions
print("\n4. Top 10 Fraudulent Transactions")

query4 = """
SELECT Time, Amount, Class
FROM transactions
WHERE Class = 1
ORDER BY Amount DESC
LIMIT 10;
"""

result4 = pd.read_sql_query(query4, connection)
print(result4)

connection.close()