import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# File paths
data_folder = r"D:\Data Science\A2"
eda_plots_folder = os.path.join(data_folder, "eda_plots")
os.makedirs(eda_plots_folder, exist_ok=True)  

processed_file = os.path.join(data_folder, "cleaned_outliers_data.csv")

# Load dataset
df = pd.read_csv(processed_file)

print("\nDataset Overview:")
print(df.info())

# Feature Engineering
if "period" in df.columns:
    df["period"] = pd.to_datetime(df["period"])
    df["hour"] = df["period"].dt.hour
    df["day"] = df["period"].dt.day
    df["month"] = df["period"].dt.month
    df["day_of_week"] = df["period"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Select features and target variable
features = ["hour", "day", "month", "day_of_week", "is_weekend", "temperature_2m"]
target = "value"

# Drop rows with missing target values
df = df.dropna(subset=[target])

# Split data into training and testing sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining Data: {X_train.shape[0]} rows, Testing Data: {X_test.shape[0]} rows")

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Save model performance metrics
metrics_df = pd.DataFrame({"MSE": [mse], "RMSE": [rmse], "R2 Score": [r2]})
metrics_df.to_csv(os.path.join(data_folder, "regression_metrics.csv"), index=False)

# Actual vs. Predicted Plot
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color="blue", alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
plt.xlabel("Actual Demand (MWh)")
plt.ylabel("Predicted Demand (MWh)")
plt.title("Actual vs. Predicted Electricity Demand")
plt.savefig(os.path.join(eda_plots_folder, "actual_vs_predicted.png"))
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=30, kde=True, color="purple")
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Residuals")
plt.title("Residual Distribution")
plt.savefig(os.path.join(eda_plots_folder, "residual_plot.png"))
plt.show()

print("\nRegression Modeling Completed Successfully! 🚀")
