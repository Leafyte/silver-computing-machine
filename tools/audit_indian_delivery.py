"""Comprehensive Data Quality Audit for Indian Food Delivery Dataset."""
import os
import pandas as pd
import numpy as np

csv_path = os.path.join("data", "train.csv")
df = pd.read_csv(csv_path)

print("=== 1. SHAPE & COLUMNS ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(df.columns.tolist())

print("\n=== 2. DTYPES & SAMPLE VALUES ===")
for col in df.columns:
    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "ALL_NA"
    print(f"  {col:30s} | {str(df[col].dtype):10s} | Sample: {repr(sample)}")

print("\n=== 3. WHITESPACE & MISSING VALUE MARKERS ===")
for col in df.columns:
    s = df[col].astype(str).str.strip()
    na_count = (s.isin(["", "NaN", "nan", "null", "None", "?", "NA", "N/A"]) | df[col].isna()).sum()
    pct = na_count / len(df) * 100
    if na_count > 0:
        print(f"  {col:30s} : {na_count:6d} missing ({pct:5.2f}%)")

print("\n=== 4. DUPLICATES ===")
print(f"Exact duplicate rows: {df.duplicated().sum()}")
cleaned_ids = df["ID"].astype(str).str.strip()
print(f"Unique IDs: {cleaned_ids.nunique()} out of {len(df)}")

print("\n=== 5. GPS COORDINATES AUDIT ===")
coord_cols = [
    "Restaurant_latitude", "Restaurant_longitude",
    "Delivery_location_latitude", "Delivery_location_longitude"
]
for col in coord_cols:
    val = df[col]
    print(f"  {col:30s} : min={val.min():.4f}, max={val.max():.4f}, zeros={(val == 0).sum()}, negatives={(val < 0).sum()}")

# Check coordinate pairs
# India roughly: lat 6 to 38 N, lon 68 to 98 E
zero_lat = (df["Restaurant_latitude"] == 0) | (df["Delivery_location_latitude"] == 0)
zero_lon = (df["Restaurant_longitude"] == 0) | (df["Delivery_location_longitude"] == 0)
print(f"Rows with zero lat/lon in Restaurant: {((df['Restaurant_latitude'] == 0) | (df['Restaurant_longitude'] == 0)).sum()}")
print(f"Rows with zero lat/lon in Delivery: {((df['Delivery_location_latitude'] == 0) | (df['Delivery_location_longitude'] == 0)).sum()}")

# Negative coordinates (Indian coordinates are all strictly positive North and East)
neg_coords = (df["Restaurant_latitude"] < 0) | (df["Restaurant_longitude"] < 0) | (df["Delivery_location_latitude"] < 0) | (df["Delivery_location_longitude"] < 0)
print(f"Rows with negative coordinates: {neg_coords.sum()}")

print("\n=== 6. TARGET DISTRIBUTION (Time_taken(min)) ===")
print("Raw target value counts sample:")
print(df["Time_taken(min)"].value_counts().head(10))
# Strip '(min) '
clean_target = df["Time_taken(min)"].astype(str).str.extract(r"(\d+)")[0].astype(float)
print("\nParsed target summary statistics:")
print(clean_target.describe())

print("\n=== 7. DATES & TIMES ===")
print(f"Order_Date unique sample: {df['Order_Date'].astype(str).str.strip().unique()[:5]}")
print(f"Time_Orderd unique sample: {df['Time_Orderd'].astype(str).str.strip().unique()[:5]}")
print(f"Time_Order_picked unique sample: {df['Time_Order_picked'].astype(str).str.strip().unique()[:5]}")

print("\n=== 8. CATEGORICAL VALUES (CLEANED) ===")
cat_cols = ["Weatherconditions", "Road_traffic_density", "Type_of_order", "Type_of_vehicle", "multiple_deliveries", "Festival", "City"]
for col in cat_cols:
    stripped = df[col].astype(str).str.strip()
    print(f"\n{col} distribution:")
    print(stripped.value_counts(dropna=False).head(10))
