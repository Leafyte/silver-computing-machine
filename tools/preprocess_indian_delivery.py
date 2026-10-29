"""
Indian Food Delivery Dataset (Kaggle - Gaurav Malik) Preprocessing Pipeline.

Reproducible, leak-free data cleaning, engineering, and feature view preparation
for Newton GBDT and Decision-Focused VQC models in DebugLeaf (QML-01).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import GroupShuffleSplit


def compute_haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Compute Haversine great-circle distance between two coordinate arrays in kilometers."""
    r = 6371.0  # Earth radius in kilometers
    phi1, lambda1, phi2, lambda2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return r * c


def compute_file_sha256(filepath: str | Path) -> str:
    """Calculate lowercase hexadecimal SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_pipeline(
    raw_csv_path: str = "data/train.csv",
    output_dir: str = "data",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Execute the end-to-end data audit, cleaning, feature engineering, and splitting pipeline."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 1: LOADING RAW DATASET")
    print("=" * 80)
    df_raw = pd.read_csv(raw_csv_path)
    total_raw_rows = len(df_raw)
    print(f"Loaded {raw_csv_path}: {total_raw_rows:,} rows, {df_raw.shape[1]} columns")

    # Clean whitespace across all string columns
    df = df_raw.copy()
    na_markers = {"", "nan", "NaN", "null", "None", "?", "conditions NaN", "NA", "N/A"}
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(list(na_markers), np.nan)

    print("\n" + "=" * 80)
    print("STEP 2: TARGET EXTRACTION & VALIDATION")
    print("=" * 80)
    # Target: Time_taken(min) -> extracted numeric minutes
    raw_target_str = df["Time_taken(min)"].astype(str)
    target_series = raw_target_str.str.extract(r"(\d+)")[0].astype(float)
    df["delivery_duration_min"] = target_series

    print(f"Target count: {df['delivery_duration_min'].notna().sum():,} / {total_raw_rows:,}")
    print(f"Target stats (min): min={df['delivery_duration_min'].min():.1f}, "
          f"mean={df['delivery_duration_min'].mean():.2f}, "
          f"median={df['delivery_duration_min'].median():.1f}, "
          f"max={df['delivery_duration_min'].max():.1f}, "
          f"std={df['delivery_duration_min'].std():.2f}")

    print("\n" + "=" * 80)
    print("STEP 3: GEOGRAPHIC AUDIT & SUSPICIOUS COORDINATE IDENTIFICATION")
    print("=" * 80)
    # India bounding box roughly: Lat 6.0N to 38.0N, Lon 68.0E to 98.0E
    zero_rest = (df["Restaurant_latitude"] == 0.0) | (df["Restaurant_longitude"] == 0.0)
    neg_rest = (df["Restaurant_latitude"] < 0.0) | (df["Restaurant_longitude"] < 0.0)
    out_of_india = (
        (df["Delivery_location_latitude"] < 6.0)
        | (df["Delivery_location_latitude"] > 38.0)
        | (df["Delivery_location_longitude"] < 68.0)
        | (df["Delivery_location_longitude"] > 98.0)
    )

    df["is_zero_coords"] = zero_rest
    df["is_negative_coords"] = neg_rest
    df["is_suspicious_coords"] = zero_rest | neg_rest | out_of_india
    df["is_model_eligible"] = ~df["is_suspicious_coords"]

    suspicious_count = int(df["is_suspicious_coords"].sum())
    eligible_count = int(df["is_model_eligible"].sum())
    print(f"Zero coordinate rows: {zero_rest.sum():,}")
    print(f"Negative restaurant coordinate rows: {neg_rest.sum():,}")
    print(f"Out-of-India delivery coordinate rows: {out_of_india.sum():,}")
    print(f"Total suspicious coordinate rows: {suspicious_count:,} ({suspicious_count / total_raw_rows * 100:.2f}%)")
    print(f"Model-eligible rows with validated coordinates: {eligible_count:,} ({eligible_count / total_raw_rows * 100:.2f}%)")

    # Compute Haversine distance
    df["haversine_distance_km"] = compute_haversine_km(
        df["Restaurant_latitude"].values,
        df["Restaurant_longitude"].values,
        df["Delivery_location_latitude"].values,
        df["Delivery_location_longitude"].values,
    )

    clean_dist = df.loc[df["is_model_eligible"], "haversine_distance_km"]
    print(f"Eligible Haversine distance stats (km): min={clean_dist.min():.2f}, "
          f"mean={clean_dist.mean():.2f}, median={clean_dist.median():.2f}, "
          f"max={clean_dist.max():.2f}, std={clean_dist.std():.2f}")

    print("\n" + "=" * 80)
    print("STEP 4: LOGISTICS FEATURE ENGINEERING & MIDNIGHT CROSSING HANDLING")
    print("=" * 80)
    # City code extracted from Delivery_person_ID prefix
    df["city_code"] = df["Delivery_person_ID"].str.extract(r"^([A-Z]+?)RES")[0]
    print(f"Extracted {df['city_code'].nunique()} city codes from Delivery_person_ID")

    # Dates & Day of Week
    order_dates = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y")
    df["order_date_parsed"] = order_dates
    df["day_of_week"] = order_dates.dt.dayofweek  # 0=Monday, 6=Sunday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Pickup time -> Departure hour
    pick_parts = df["Time_Order_picked"].str.split(":", expand=True)
    pick_h = pick_parts[0].astype(int)
    pick_m = pick_parts[1].astype(int)
    df["departure_hour"] = pick_h + pick_m / 60.0

    # Rush hour indicators in Indian urban logistics:
    # Lunch rush: 12:00 to 14:59 (12.0 - 15.0)
    # Dinner rush: 18:00 to 21:59 (18.0 - 22.0)
    df["is_lunch_rush"] = ((df["departure_hour"] >= 12.0) & (df["departure_hour"] < 15.0)).astype(int)
    df["is_dinner_rush"] = ((df["departure_hour"] >= 18.0) & (df["departure_hour"] < 22.0)).astype(int)
    df["is_rush_hour"] = (df["is_lunch_rush"] | df["is_dinner_rush"]).astype(int)

    # Order-to-pickup prep time (minutes) with midnight crossing handling
    ord_parts = df["Time_Orderd"].str.split(":", expand=True)
    ord_h = pd.to_numeric(ord_parts[0], errors="coerce")
    ord_m = pd.to_numeric(ord_parts[1], errors="coerce")

    # Calculate difference
    min_ord = ord_h * 60.0 + ord_m
    min_pick = pick_h * 60.0 + pick_m
    time_diff = min_pick - min_ord
    # Handle midnight crossings (e.g. ordered 23:55, picked 00:10 -> diff = -1425 -> add 1440 = 15)
    time_diff_corrected = np.where(time_diff < 0.0, time_diff + 1440.0, time_diff)

    df["order_to_pickup_min"] = time_diff_corrected
    df["order_to_pickup_missing_flag"] = df["order_to_pickup_min"].isna().astype(int)

    midnight_crossings = int(((time_diff < 0.0) & time_diff.notna()).sum())
    print(f"Midnight order-to-pickup crossings successfully resolved: {midnight_crossings:,}")

    # Weather conditions cleaning
    df["weather_clean"] = df["Weatherconditions"].str.replace(r"^conditions\s+", "", regex=True)
    df["is_adverse_weather"] = df["weather_clean"].isin(["Fog", "Stormy", "Sandstorms", "Windy"]).astype(int)

    # Road traffic density ordinal mapping
    traffic_map = {"Low": 0.0, "Medium": 1.0, "High": 2.0, "Jam": 3.0}
    df["traffic_density_clean"] = df["Road_traffic_density"]
    df["traffic_density_ordinal"] = df["Road_traffic_density"].map(traffic_map)

    # Vehicle condition, type, multiple deliveries, festival, city urbanization
    df["vehicle_condition"] = pd.to_numeric(df["Vehicle_condition"], errors="coerce")
    df["multiple_deliveries_clean"] = pd.to_numeric(df["multiple_deliveries"], errors="coerce")
    df["is_festival"] = (df["Festival"] == "Yes").astype(int)
    df["festival_clean"] = df["Festival"]
    df["city_type_clean"] = df["City"]
    df["type_of_vehicle_clean"] = df["Type_of_vehicle"]
    df["type_of_order_clean"] = df["Type_of_order"]

    df["delivery_person_age_clean"] = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")
    df["delivery_person_ratings_clean"] = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce")

    print("\n" + "=" * 80)
    print("STEP 5: REPRODUCIBLE LEAK-FREE COURIER-GROUP PARTITIONING")
    print("=" * 80)
    # Split strictly on model-eligible records by Delivery_person_ID
    df_eligible = df[df["is_model_eligible"]].copy()

    gss_outer = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=random_seed)
    train_idx_raw, temp_idx_raw = next(gss_outer.split(df_eligible, groups=df_eligible["Delivery_person_ID"]))

    temp_sub = df_eligible.iloc[temp_idx_raw]
    gss_inner = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=random_seed)
    val_sub_idx, test_sub_idx = next(gss_inner.split(temp_sub, groups=temp_sub["Delivery_person_ID"]))

    val_idx_raw = temp_idx_raw[val_sub_idx]
    test_idx_raw = temp_idx_raw[test_sub_idx]

    eligible_indices = df_eligible.index
    train_indices = eligible_indices[train_idx_raw]
    val_indices = eligible_indices[val_idx_raw]
    test_indices = eligible_indices[test_idx_raw]

    # Assign split partition labels across full dataset
    df["split_partition"] = "excluded"
    df.loc[train_indices, "split_partition"] = "train"
    df.loc[val_indices, "split_partition"] = "val"
    df.loc[test_indices, "split_partition"] = "test"

    print(f"Partition breakdown:")
    print(f"  Train:     {len(train_indices):6,d} ({len(train_indices) / eligible_count * 100:.1f}% of eligible)")
    print(f"  Val:       {len(val_indices):6,d} ({len(val_indices) / eligible_count * 100:.1f}% of eligible)")
    print(f"  Test:      {len(test_indices):6,d} ({len(test_indices) / eligible_count * 100:.1f}% of eligible)")
    print(f"  Excluded:  {suspicious_count:6,d} (anomalous coordinates)")

    train_couriers = set(df.loc[train_indices, "Delivery_person_ID"])
    val_couriers = set(df.loc[val_indices, "Delivery_person_ID"])
    test_couriers = set(df.loc[test_indices, "Delivery_person_ID"])

    print(f"Courier isolation:")
    print(f"  Train couriers: {len(train_couriers)}, Val: {len(val_couriers)}, Test: {len(test_couriers)}")
    print(f"  Train & Val courier overlap:  {len(train_couriers & val_couriers)}")
    print(f"  Train & Test courier overlap: {len(train_couriers & test_couriers)}")
    print(f"  Val & Test courier overlap:   {len(val_couriers & test_couriers)}")

    print(f"City representation (all 22 Indian cities):")
    print(f"  Train cities: {df.loc[train_indices, 'city_code'].nunique()} / 22")
    print(f"  Val cities:   {df.loc[val_indices, 'city_code'].nunique()} / 22")
    print(f"  Test cities:  {df.loc[test_indices, 'city_code'].nunique()} / 22")

    print("\n" + "=" * 80)
    print("STEP 6: TRAINING-FITTED IMPUTATION & SCALING ARTIFACTS")
    print("=" * 80)
    # Compute imputation parameters strictly from the train split
    train_df = df.loc[train_indices]

    impute_artifacts = {
        "delivery_person_age_median": float(train_df["delivery_person_age_clean"].median()),
        "delivery_person_ratings_median": float(train_df["delivery_person_ratings_clean"].median()),
        "order_to_pickup_min_median": float(train_df["order_to_pickup_min"].median()),
        "multiple_deliveries_median": float(train_df["multiple_deliveries_clean"].median()),
        "weather_clean_mode": str(train_df["weather_clean"].dropna().mode().iloc[0]),
        "traffic_density_ordinal_mode": float(train_df["traffic_density_ordinal"].dropna().mode().iloc[0]),
        "traffic_density_clean_mode": str(train_df["traffic_density_clean"].dropna().mode().iloc[0]),
        "city_type_clean_mode": str(train_df["city_type_clean"].dropna().mode().iloc[0]),
        "festival_clean_mode": str(train_df["festival_clean"].dropna().mode().iloc[0]),
    }

    # Scaling artifacts for VQC (MinMax on distance, target) fitted strictly on Train
    dist_min = float(train_df["haversine_distance_km"].min())
    dist_max = float(train_df["haversine_distance_km"].max())
    target_min = float(train_df["delivery_duration_min"].min())
    target_max = float(train_df["delivery_duration_min"].max())

    scaling_artifacts = {
        "haversine_distance_min": dist_min,
        "haversine_distance_max": dist_max,
        "delivery_duration_min": target_min,
        "delivery_duration_max": target_max,
        "vqc_distance_scale_range": [0.0, 1.0],
        "vqc_congestion_scale_range": [0.0, 1.0],
    }

    preprocessing_artifacts = {
        "pipeline_version": "1.0.0",
        "random_seed": random_seed,
        "imputation": impute_artifacts,
        "scaling": scaling_artifacts,
        "traffic_ordinal_mapping": traffic_map,
    }

    # Apply training-fitted imputations
    df["delivery_person_age_imp"] = df["delivery_person_age_clean"].fillna(impute_artifacts["delivery_person_age_median"])
    df["delivery_person_ratings_imp"] = df["delivery_person_ratings_clean"].fillna(impute_artifacts["delivery_person_ratings_median"])
    df["order_to_pickup_min_imp"] = df["order_to_pickup_min"].fillna(impute_artifacts["order_to_pickup_min_median"])
    df["multiple_deliveries_imp"] = df["multiple_deliveries_clean"].fillna(impute_artifacts["multiple_deliveries_median"])
    df["traffic_density_ordinal_imp"] = df["traffic_density_ordinal"].fillna(impute_artifacts["traffic_density_ordinal_mode"])
    df["weather_clean_imp"] = df["weather_clean"].fillna(impute_artifacts["weather_clean_mode"])
    df["city_type_clean_imp"] = df["city_type_clean"].fillna(impute_artifacts["city_type_clean_mode"])

    print("Imputation parameters fitted strictly on Train:")
    for k, v in impute_artifacts.items():
        print(f"  {k:35s} = {v}")

    print("\n" + "=" * 80)
    print("STEP 7: CONSTRUCTING THE TWO FEATURE VIEWS")
    print("=" * 80)
    # A. VQC View: 2 to 4 numerically stable, strictly [0, 1] bounded features
    # vqc_x0: Normalized distance in [0, 1]
    dist_span = dist_max - dist_min if dist_max > dist_min else 1.0
    df["vqc_x0_distance"] = np.clip((df["haversine_distance_km"] - dist_min) / dist_span, 0.0, 1.0)

    # vqc_x1: Urban Congestion Pressure composite in [0, 1]
    # Combines traffic density (0..3), rush hour (0 or 1), and adverse weather (0 or 1)
    raw_congestion = (
        (df["traffic_density_ordinal_imp"] / 3.0) * 0.60
        + df["is_rush_hour"] * 0.25
        + df["is_adverse_weather"] * 0.15
    )
    df["vqc_x1_congestion"] = np.clip(raw_congestion, 0.0, 1.0)

    # vqc_x2: Multiple deliveries workload in [0, 1]
    df["vqc_x2_multidelivery"] = np.clip(df["multiple_deliveries_imp"] / 3.0, 0.0, 1.0)

    # vqc_x3: Kitchen order-to-pickup prep lag in [0, 1]
    df["vqc_x3_preptime"] = np.clip((df["order_to_pickup_min_imp"] - 5.0) / 10.0, 0.0, 1.0)

    # Target normalized in [0, 1] for regression VQC
    target_span = target_max - target_min if target_max > target_min else 1.0
    df["target_normalized"] = np.clip((df["delivery_duration_min"] - target_min) / target_span, 0.0, 1.0)

    # Synthetic operational SLA breach scenario (standard 30-min food delivery SLA)
    sla_threshold_min = 30.0
    df["sla_threshold_min"] = sla_threshold_min
    df["is_sla_breach_synthetic"] = (df["delivery_duration_min"] > sla_threshold_min).astype(int)

    # B. GBDT Tabular View: Clean, encoded tabular feature set
    gbdt_cat_cols = [
        "weather_clean_imp", "traffic_density_clean", "type_of_order_clean",
        "type_of_vehicle_clean", "city_type_clean_imp", "city_code"
    ]
    # Generate clean dummy / one-hot encodings for categorical columns fitted strictly on training categories
    train_eligible = df[df["split_partition"] == "train"]
    one_hot_columns: Dict[str, List[str]] = {}
    gbdt_encoded_dfs = []

    for c in gbdt_cat_cols:
        unique_cats = sorted(train_eligible[c].dropna().unique())
        one_hot_columns[c] = unique_cats
        cat_dummies = pd.get_dummies(df[c], prefix=c, drop_first=False, dtype=int)
        # Keep only columns present in training
        expected_cols = [f"{c}_{cat}" for cat in unique_cats]
        for ec in expected_cols:
            if ec not in cat_dummies.columns:
                cat_dummies[ec] = 0
        gbdt_encoded_dfs.append(cat_dummies[expected_cols])

    gbdt_features_df = pd.concat([
        df[[
            "ID", "Delivery_person_ID", "split_partition", "delivery_duration_min",
            "haversine_distance_km", "departure_hour", "day_of_week", "is_weekend",
            "is_rush_hour", "is_lunch_rush", "is_dinner_rush",
            "delivery_person_age_imp", "delivery_person_ratings_imp",
            "order_to_pickup_min_imp", "order_to_pickup_missing_flag",
            "vehicle_condition", "multiple_deliveries_imp", "is_festival",
            "traffic_density_ordinal_imp", "is_adverse_weather"
        ]],
        *gbdt_encoded_dfs
    ], axis=1)

    print(f"GBDT feature set constructed: {gbdt_features_df.shape[1]} columns")

    print("\n" + "=" * 80)
    print("STEP 8: PERSISTING DELIVERABLES")
    print("=" * 80)

    # 1. Clean Parquet (All 45,593 rows)
    clean_parquet_path = out_path / "indian_delivery_clean.parquet"
    table_clean = pa.Table.from_pandas(df)
    pq.write_table(table_clean, clean_parquet_path, compression="snappy")
    print(f"[OK] Saved clean dataset: {clean_parquet_path} ({clean_parquet_path.stat().st_size:,} bytes)")

    # 2. GBDT Parquet (Eligible 41,522 rows)
    gbdt_eligible = gbdt_features_df[df["is_model_eligible"]].reset_index(drop=True)
    gbdt_parquet_path = out_path / "indian_delivery_gbdt.parquet"
    table_gbdt = pa.Table.from_pandas(gbdt_eligible)
    pq.write_table(table_gbdt, gbdt_parquet_path, compression="snappy")
    print(f"[OK] Saved GBDT view:     {gbdt_parquet_path} ({gbdt_parquet_path.stat().st_size:,} bytes, {len(gbdt_eligible):,} rows)")

    # 3. VQC Parquet (Eligible 41,522 rows)
    vqc_cols = [
        "ID", "Delivery_person_ID", "city_code", "split_partition",
        "vqc_x0_distance", "vqc_x1_congestion", "vqc_x2_multidelivery", "vqc_x3_preptime",
        "delivery_duration_min", "target_normalized", "sla_threshold_min", "is_sla_breach_synthetic"
    ]
    vqc_eligible = df.loc[df["is_model_eligible"], vqc_cols].reset_index(drop=True)
    vqc_parquet_path = out_path / "indian_delivery_vqc.parquet"
    table_vqc = pa.Table.from_pandas(vqc_eligible)
    pq.write_table(table_vqc, vqc_parquet_path, compression="snappy")
    print(f"[OK] Saved VQC view:      {vqc_parquet_path} ({vqc_parquet_path.stat().st_size:,} bytes, {len(vqc_eligible):,} rows)")

    # 4. Preprocessing Artifacts JSON
    artifacts_json_path = out_path / "preprocessing_artifacts.json"
    with open(artifacts_json_path, "w", encoding="utf-8") as f:
        json.dump(preprocessing_artifacts, f, indent=2)
    print(f"[OK] Saved artifacts:     {artifacts_json_path}")

    # 5. Split Manifest JSON
    split_manifest = {
        "dataset_name": "Indian Food Delivery Dataset (Kaggle - Gaurav Malik)",
        "source_csv": str(raw_csv_path),
        "source_sha256": compute_file_sha256(raw_csv_path),
        "total_raw_rows": total_raw_rows,
        "excluded_anomalous_rows": suspicious_count,
        "eligible_model_rows": eligible_count,
        "exclusion_reasons": {
            "zero_coordinates_null_island": int(zero_rest.sum()),
            "negative_restaurant_coordinates": int(neg_rest.sum()),
            "out_of_india_bounding_box": int(out_of_india.sum()),
        },
        "split_method": "GroupShuffleSplit on Delivery_person_ID",
        "random_seed": random_seed,
        "split_counts": {
            "train": int((df["split_partition"] == "train").sum()),
            "val": int((df["split_partition"] == "val").sum()),
            "test": int((df["split_partition"] == "test").sum()),
            "excluded": suspicious_count,
        },
        "courier_counts": {
            "total_unique": int(df["Delivery_person_ID"].nunique()),
            "train": len(train_couriers),
            "val": len(val_couriers),
            "test": len(test_couriers),
            "overlap_train_val": len(train_couriers & val_couriers),
            "overlap_train_test": len(train_couriers & test_couriers),
            "overlap_val_test": len(val_couriers & test_couriers),
        },
        "city_representation": {
            "total_cities": 22,
            "train_cities": int(df.loc[train_indices, "city_code"].nunique()),
            "val_cities": int(df.loc[val_indices, "city_code"].nunique()),
            "test_cities": int(df.loc[test_indices, "city_code"].nunique()),
        },
        "target_duration_stats": {
            "overall_mean": float(df["delivery_duration_min"].mean()),
            "overall_std": float(df["delivery_duration_min"].std()),
            "train_mean": float(df.loc[train_indices, "delivery_duration_min"].mean()),
            "val_mean": float(df.loc[val_indices, "delivery_duration_min"].mean()),
            "test_mean": float(df.loc[test_indices, "delivery_duration_min"].mean()),
            "synthetic_sla_threshold_min": sla_threshold_min,
            "overall_synthetic_breach_rate": float(df.loc[df["is_model_eligible"], "is_sla_breach_synthetic"].mean()),
            "train_synthetic_breach_rate": float(df.loc[train_indices, "is_sla_breach_synthetic"].mean()),
            "val_synthetic_breach_rate": float(df.loc[val_indices, "is_sla_breach_synthetic"].mean()),
            "test_synthetic_breach_rate": float(df.loc[test_indices, "is_sla_breach_synthetic"].mean()),
        },
        "file_digests": {
            "clean_parquet_sha256": compute_file_sha256(clean_parquet_path),
            "gbdt_parquet_sha256": compute_file_sha256(gbdt_parquet_path),
            "vqc_parquet_sha256": compute_file_sha256(vqc_parquet_path),
        },
    }
    split_manifest_path = out_path / "split_manifest.json"
    with open(split_manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)
    print(f"[OK] Saved split manifest: {split_manifest_path}")

    # 6. Feature Schema JSON
    gbdt_feature_names = [c for c in gbdt_eligible.columns if c not in ["ID", "Delivery_person_ID", "split_partition", "delivery_duration_min"]]
    feature_schema = {
        "dataset": "Indian Food Delivery Dataset (Cleaned & Preprocessed)",
        "target": {
            "name": "delivery_duration_min",
            "type": "float64",
            "unit": "minutes",
            "range": [10.0, 54.0],
            "description": "Total delivery duration in minutes from restaurant departure/pickup to customer drop-off.",
        },
        "synthetic_scenario_target": {
            "name": "is_sla_breach_synthetic",
            "type": "int64",
            "threshold": 30.0,
            "description": "Synthetic binary classification label indicating delivery duration exceeding 30 minutes.",
        },
        "gbdt_view": {
            "total_features": len(gbdt_feature_names),
            "features": gbdt_feature_names,
            "numeric_features": [
                "haversine_distance_km", "departure_hour", "day_of_week",
                "delivery_person_age_imp", "delivery_person_ratings_imp",
                "order_to_pickup_min_imp", "vehicle_condition", "multiple_deliveries_imp",
                "traffic_density_ordinal_imp"
            ],
            "binary_indicators": [
                "is_weekend", "is_rush_hour", "is_lunch_rush", "is_dinner_rush",
                "order_to_pickup_missing_flag", "is_festival", "is_adverse_weather"
            ],
            "one_hot_encoded_categories": one_hot_columns,
        },
        "vqc_view": {
            "features": {
                "vqc_x0_distance": {
                    "type": "float64",
                    "range": [0.0, 1.0],
                    "encoding": "Ry(pi * x0) angle rotation on qubit 0",
                    "source": "Haversine distance (km) MinMax-scaled using training bounds",
                },
                "vqc_x1_congestion": {
                    "type": "float64",
                    "range": [0.0, 1.0],
                    "encoding": "Ry(pi * x1) angle rotation on qubit 1",
                    "source": "Composite urban congestion index (traffic density ordinal, rush hour, adverse weather)",
                },
                "vqc_x2_multidelivery": {
                    "type": "float64",
                    "range": [0.0, 1.0],
                    "encoding": "Ry(pi * x2) angle rotation on qubit 2 (4-qubit extension)",
                    "source": "Multiple deliveries workload scaled to [0, 1]",
                },
                "vqc_x3_preptime": {
                    "type": "float64",
                    "range": [0.0, 1.0],
                    "encoding": "Ry(pi * x3) angle rotation on qubit 3 (4-qubit extension)",
                    "source": "Kitchen order-to-pickup prep time scaled to [0, 1]",
                },
            },
            "qubit_mapping_recommendations": {
                "2_qubit_primary": ["vqc_x0_distance", "vqc_x1_congestion"],
                "4_qubit_extended": ["vqc_x0_distance", "vqc_x1_congestion", "vqc_x2_multidelivery", "vqc_x3_preptime"],
            },
        },
    }
    feature_schema_path = out_path / "feature_schema.json"
    with open(feature_schema_path, "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=2)
    print(f"[OK] Saved feature schema: {feature_schema_path}")

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    return {
        "total_raw_rows": total_raw_rows,
        "excluded_rows": suspicious_count,
        "eligible_rows": eligible_count,
        "split_counts": split_manifest["split_counts"],
        "target_stats": split_manifest["target_duration_stats"],
        "gbdt_feature_count": len(gbdt_feature_names),
        "vqc_feature_count": len(feature_schema["vqc_view"]["features"]),
    }


if __name__ == "__main__":
    result = run_pipeline()
