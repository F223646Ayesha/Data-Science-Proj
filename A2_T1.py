import os
import glob
import pandas as pd
import json

# Define folder paths
weather_folder = r"D:\Data Science\A2\weather_data"
demand_folder = r"D:\Data Science\A2\electricity_data"
output_folder = r"D:\Data Science\A2"

# Scan directories for files
weather_files = glob.glob(os.path.join(weather_folder, "*.csv"))
demand_files = glob.glob(os.path.join(demand_folder, "*.json"))

print(f"Found {len(weather_files)} weather CSV files and {len(demand_files)} electricity demand JSON files.")

### Step 1: Load and Save Weather Data ###
weather_data = []
for file in weather_files:
    try:
        df = pd.read_csv(file, encoding="utf-8")
        print(f"Loaded {file}: {df.shape[0]} rows, {df.shape[1]} columns")
        print("Weather Data Columns:", df.columns)  # Debugging
        weather_data.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Merge all weather CSV files and save them
if weather_data:
    weather_df = pd.concat(weather_data, ignore_index=True)
    weather_output = os.path.join(output_folder, "weather_data.csv")
    weather_df.to_csv(weather_output, index=False)
    print(f"Weather data saved to: {weather_output}")
    print(f"Merged Weather Data: {weather_df.shape[0]} rows, {weather_df.shape[1]} columns")
else:
    weather_df = pd.DataFrame()
    print("No weather data found.")

### Step 2: Load and Save Electricity Demand Data ###
demand_data = []
for file in demand_files:
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            df = pd.DataFrame(data["response"]["data"])  # Extract relevant section
            print(f"Loaded {file}: {df.shape[0]} rows, {df.shape[1]} columns")
            print("Demand Data Columns:", df.columns)  # Debugging
            demand_data.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Merge all demand JSON files and save them
if demand_data:
    demand_df = pd.concat(demand_data, ignore_index=True)
    demand_output = os.path.join(output_folder, "electricity_data.csv")
    demand_df.to_csv(demand_output, index=False)
    print(f"Electricity demand data saved to: {demand_output}")
    print(f"Merged Demand Data: {demand_df.shape[0]} rows, {demand_df.shape[1]} columns")
else:
    demand_df = pd.DataFrame()
    print("No electricity demand data found.")

### Step 3: Prepare for Merging ###
print("\nWeather Columns:", weather_df.columns)
print("Demand Columns:", demand_df.columns)

# Rename 'date' in weather_df to 'period' for consistency
if "date" in weather_df.columns:
    weather_df.rename(columns={"date": "period"}, inplace=True)

# Convert to datetime format
weather_df["period"] = pd.to_datetime(weather_df["period"], errors="coerce")
demand_df["period"] = pd.to_datetime(demand_df["period"], errors="coerce")

# Fix timezone issue: Remove UTC from demand_df["period"]
if demand_df["period"].dt.tz is not None:
    demand_df["period"] = demand_df["period"].dt.tz_convert(None)  # Convert to naive datetime

if weather_df["period"].dt.tz is not None:
    weather_df["period"] = weather_df["period"].dt.tz_convert(None)  # Convert to naive datetime

### Step 4: Merge DataFrames ###
if "period" in weather_df.columns and "period" in demand_df.columns:
    merged_df = pd.merge(weather_df, demand_df, how="outer", on="period", suffixes=("_weather", "_demand"))
    print(f"Final Merged Data: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")

    # Save merged dataset
    merged_output = os.path.join(output_folder, "merged_data.csv")
    merged_df.to_csv(merged_output, index=False)
    print(f"Final merged dataset saved to: {merged_output}")
else:
    print("Error: 'period' column is missing in one or both datasets. Unable to merge.")
