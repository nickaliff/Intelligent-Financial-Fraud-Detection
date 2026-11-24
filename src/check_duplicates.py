import pandas as pd

data = pd.read_csv("data/raw/creditcard.csv")

print("Duplicate records:", data.duplicated().sum())

#Display all records involved in duplicate groups
duplicate_rows = data[data.duplicated(keep=False)]

print("\nTotal rows involved in duplicate groups:")
print(len(duplicate_rows))

#Check the class distribution of duplicate records
print("\nDuplicate record class distribution:")
print(duplicate_rows["Class"].value_counts())

clean_data = data.drop_duplicates()

print("\nDataset shape after removing duplicates:")
print(clean_data.shape)

#Check class distribution after removal
print("\nClass distribution after removing duplicates:")
print(clean_data["Class"].value_counts())