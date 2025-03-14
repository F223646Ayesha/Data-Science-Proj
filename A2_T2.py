import pandas as pd
import numpy as np
from scipy.stats import zscore

# File paths
data_folder = r"D:\Data Science\A2"
merged_file = data_folder + r"\merged_data.csv"

# Load dataset
df = pd.read_csv(merged_file)

# Initial dataset overview
print("Initial Dataset Overview:")
print(df.info())
print("\nMissing Values Per Column:\n", df.isnull().sum())

# Handling missing data
missing_percentage = df.isnull().mean() * 100
threshold = 20  # Drop columns with more than 20% missing values
columns_to_drop = missing_percentage[missing_percentage > threshold].index
df.drop(columns=columns_to_drop, inplace=True)
print(f"Dropped columns with >{threshold}% missing values: {list(columns_to_drop)}")

# Fill missing values
num_cols = df.select_dtypes(include=["number"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns
df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()))
df[cat_cols] = df[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Convert period/time columns to datetime
if "period" in df.columns:
    df["period"] = pd.to_datetime(df["period"], errors="coerce")

# Ensure numerical columns are formatted correctly
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

# Convert categorical features
df[cat_cols] = df[cat_cols].astype("category")

# Remove duplicates
duplicates = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"Removed {duplicates} duplicate rows.")

# Detect and handle outliers using Z-score
z_scores = df[num_cols].apply(zscore)
outliers = (z_scores.abs() > 3).sum()
print("\nPotential Outliers Per Column:\n", outliers)

# Remove extreme outliers
df = df[(z_scores.abs() <= 3).all(axis=1)]
print(f"Removed {outliers.sum()} extreme outlier records.")

# Feature Engineering
if "period" in df.columns:
    df["hour"] = df["period"].dt.hour
    df["day"] = df["period"].dt.day
    df["month"] = df["period"].dt.month
    df["day_of_week"] = df["period"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    df["season"] = df["month"].apply(lambda x: "Winter" if x in [12, 1, 2] 
                                     else "Spring" if x in [3, 4, 5] 
                                     else "Summer" if x in [6, 7, 8] 
                                     else "Fall")

# Normalize numerical columns
df[num_cols] = df[num_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Save cleaned dataset
cleaned_file = data_folder + r"\cleaned_data.csv"
df.to_csv(cleaned_file, index=False)
print(f"\nCleaned dataset saved to: {cleaned_file}")

# Final summary
print("\nFinal Dataset Overview:")
print(df.info())
print("\nSample Data:\n", df.head())
