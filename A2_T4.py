import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import zscore

# File paths
data_folder = r"D:\Data Science\A2"
eda_plots_folder = os.path.join(data_folder, "eda_plots")
os.makedirs(eda_plots_folder, exist_ok=True)  # Ensure folder exists

cleaned_file = os.path.join(data_folder, "cleaned_data.csv")

# Load dataset
df = pd.read_csv(cleaned_file)

print("\nDataset Overview:")
print(df.info())

# Outlier Detection
num_cols = df.select_dtypes(include=["number"]).columns
outlier_summary = pd.DataFrame(index=num_cols, columns=["IQR Outliers", "Z-score Outliers"])

# Boxplot before handling outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[num_cols])
plt.title("Boxplot Before Outlier Removal")
plt.savefig(os.path.join(eda_plots_folder, "boxplot_before_outliers.png"))
plt.show()

def detect_outliers(df, method="iqr"):
    outliers = pd.DataFrame()
    for col in num_cols:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers[col] = ((df[col] < lower_bound) | (df[col] > upper_bound))
            outlier_summary.loc[col, "IQR Outliers"] = outliers[col].sum()
        elif method == "zscore":
            z_scores = zscore(df[col])
            outliers[col] = (np.abs(z_scores) > 3)
            outlier_summary.loc[col, "Z-score Outliers"] = outliers[col].sum()
    return outliers

# Apply both methods
detect_outliers(df, method="iqr")
detect_outliers(df, method="zscore")

# Save outlier summary
outlier_summary.to_csv(os.path.join(data_folder, "outlier_summary.csv"))
print("\nOutlier Summary (Saved to outlier_summary.csv):")
print(outlier_summary)

# Outlier Handling (Capping method)
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# Boxplot after handling outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[num_cols])
plt.title("Boxplot After Outlier Handling")
plt.savefig(os.path.join(eda_plots_folder, "boxplot_after_outliers.png"))
plt.show()

# Save cleaned dataset
cleaned_outliers_file = os.path.join(data_folder, "cleaned_outliers_data.csv")
df.to_csv(cleaned_outliers_file, index=False)
print(f"\nDataset with Outliers Handled Saved: {cleaned_outliers_file}")
