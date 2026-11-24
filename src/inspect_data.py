import pandas as pd

data = pd.read_csv("data/raw/creditcard.csv")

print("Dataset Shape:")
print(data.shape)

print("\nFirst 5 Records:")
print(data.head())

print("\nDataset Information:")
data.info()

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Records:")
print(data.duplicated().sum())

print("\nTransaction Class Distribution:")
print(data["Class"].value_counts())

#Calculate fraud percentage
fraud_percentage = data["Class"].mean() * 100

print("\nFraud Percentage:")
print(round(fraud_percentage, 3), "%")

print("\nStatistical Summary:")
print(data.describe())