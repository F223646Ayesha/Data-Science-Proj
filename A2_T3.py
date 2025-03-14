import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew, kurtosis

# File paths
data_folder = r"D:\Data Science\A2"
eda_plots_folder = os.path.join(data_folder, "eda_plots")
os.makedirs(eda_plots_folder, exist_ok=True)

cleaned_file = os.path.join(data_folder, "cleaned_data.csv")

# Load dataset
df = pd.read_csv(cleaned_file)

# Dataset overview
print("\nDataset Overview:")
print(df.info())

# Statistical Summary
print("\nStatistical Summary for Numerical Features:")
num_cols = df.select_dtypes(include=["number"]).columns
stats_summary = df[num_cols].describe().T
stats_summary["skewness"] = df[num_cols].apply(skew)
stats_summary["kurtosis"] = df[num_cols].apply(kurtosis)
print(stats_summary)

# Save summary to CSV
stats_summary.to_csv(os.path.join(data_folder, "stats_summary.csv"))
print("Saved statistical summary to stats_summary.csv")

# Time Series Analysis
if "period" in df.columns and "value" in df.columns:
    df["period"] = pd.to_datetime(df["period"])
    df.set_index("period", inplace=True)

    plt.figure(figsize=(12, 6))
    plt.plot(df["value"], label="Electricity Demand", color="blue")
    plt.title("Electricity Demand Over Time")
    plt.xlabel("Time")
    plt.ylabel("Demand (MWh)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(eda_plots_folder, "time_series.png"))
    plt.show()

# Univariate Analysis
for col in num_cols:
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    sns.histplot(df[col], kde=True, bins=30, color="blue")
    plt.title(f"Histogram of {col}")

    plt.subplot(1, 3, 2)
    sns.boxplot(x=df[col], color="red")
    plt.title(f"Boxplot of {col}")

    plt.subplot(1, 3, 3)
    sns.kdeplot(df[col], color="green")
    plt.title(f"Density Plot of {col}")

    plt.tight_layout()
    plt.savefig(os.path.join(eda_plots_folder, f"hist_box_density_{col}.png"))
    plt.show()

# Correlation Analysis
plt.figure(figsize=(10, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(eda_plots_folder, "correlation_heatmap.png"))
plt.show()

# Time Series Decomposition
if "value" in df.columns:
    decomposition = seasonal_decompose(df["value"], model="additive", period=24)

    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(df["value"], label="Original", color="blue")
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend, label="Trend", color="green")
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal, label="Seasonality", color="red")
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid, label="Residuals", color="gray")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(eda_plots_folder, "time_series_decomposition.png"))
    plt.show()

    # Augmented Dickey-Fuller Test
    adf_test = adfuller(df["value"].dropna())
    print("\nAugmented Dickey-Fuller Test:")
    print(f"ADF Statistic: {adf_test[0]}")
    print(f"P-value: {adf_test[1]}")
    print("Critical Values:")
    for key, value in adf_test[4].items():
        print(f"\t{key}: {value}")

    if adf_test[1] < 0.05:
        print("✅ The time series is stationary.")
    else:
        print("❌ The time series is NOT stationary.")

print("\nEDA Completed Successfully! 🚀")
