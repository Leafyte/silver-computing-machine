"""Rebuild train OOF residuals with fold-local preprocessing, reusing the v3 model."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastparquet import ParquetFile
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "luna_hgbr_recovery_20261029T131809Z_v5"
RUNTIME = ROOT / ".runtime" / "run2" / RUN_ID
RELEASE = ROOT / "artifacts" / "run2_india" / RUN_ID
PREVIOUS_RUN = ROOT / "artifacts" / "run2_india" / "luna_hgbr_recovery_20261029T131809Z_v3"
TARGET = "delivery_duration_min"
N_SPLITS = 5
MODEL_CONFIG = {
    "class": "sklearn.ensemble.HistGradientBoostingRegressor",
    "max_iter": 100,
    "learning_rate": 0.08,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "early_stopping": False,
    "random_state": 42,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def metric_summary(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae_minutes": float(mean_absolute_error(y, pred)),
        "rmse_minutes": float(np.sqrt(mean_squared_error(y, pred))),
        "median_absolute_error_minutes": float(median_absolute_error(y, pred)),
    }


def strict_development(path: Path, columns: list[str]) -> pd.DataFrame:
    parquet = ParquetFile(path)
    filters = [[("split_partition", "in", ["train", "val"])]]
    parts = parquet.to_pandas(columns=["split_partition"], filters=filters, row_filter=True)
    if set(parts["split_partition"].astype(str).unique()) != {"train", "val"}:
        raise ValueError("target-bearing GBDT/VQC source must load train and val only")
    frame = parquet.to_pandas(columns=columns, filters=filters, row_filter=True)
    if set(frame["split_partition"].astype(str).unique()) != {"train", "val"}:
        raise ValueError("target-bearing read contained a sealed test partition")
    return frame


def fold_transform(
    gbdt_rows: pd.DataFrame,
    raw_train: pd.DataFrame,
    fit_idx: np.ndarray,
    feature_names: list[str],
    category_schema: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit every fold-specific GBDT imputer/mode on fold-fit couriers only."""
    row_ids = gbdt_rows["ID"].to_numpy()
    raw = raw_train.set_index("ID").loc[row_ids]
    fit_raw = raw.iloc[fit_idx]
    X = gbdt_rows.loc[:, feature_names].copy()
    audit: dict[str, Any] = {"imputers": {}, "fit_categories": {}}

    numeric_sources = {
        "delivery_person_age_imp": "delivery_person_age_clean",
        "delivery_person_ratings_imp": "delivery_person_ratings_clean",
        "order_to_pickup_min_imp": "order_to_pickup_min",
        "multiple_deliveries_imp": "multiple_deliveries_clean",
    }
    for output, source in numeric_sources.items():
        fit_values = pd.to_numeric(fit_raw[source], errors="coerce")
        fill = float(fit_values.median())
        values = pd.to_numeric(raw[source], errors="coerce").fillna(fill)
        X[output] = values.to_numpy(dtype=np.float64)
        audit["imputers"][output] = {"method": "median", "source": source, "value": fill}

    traffic_fit = pd.to_numeric(fit_raw["traffic_density_ordinal"], errors="coerce")
    traffic_mode = float(traffic_fit.mode().iloc[0])
    traffic_values = pd.to_numeric(raw["traffic_density_ordinal"], errors="coerce").fillna(traffic_mode)
    X["traffic_density_ordinal_imp"] = traffic_values.to_numpy(dtype=np.float64)
    audit["imputers"]["traffic_density_ordinal_imp"] = {
        "method": "mode", "source": "traffic_density_ordinal", "value": traffic_mode,
    }

    for prefix, source in (("weather_clean_imp", "weather_clean"), ("city_type_clean_imp", "city_type_clean")):
        fit_values = fit_raw[source].dropna().astype(str)
        mode = str(fit_values.mode().iloc[0])
        values = raw[source].astype("string").fillna(mode).astype(str)
        seen_categories = set(fit_values.tolist()) | {mode}
        for category in category_schema[prefix]:
            column = f"{prefix}_{category}"
            if column in X.columns:
                X[column] = ((values.to_numpy() == category) & (category in seen_categories)).astype(np.int8)
        audit["imputers"][prefix] = {"method": "mode", "source": source, "value": mode}

    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        raise ValueError("Fold-specific feature transform produced nonfinite values")
    one_hot_sources = {
        "traffic_density_clean": "traffic_density_clean",
        "type_of_order_clean": "type_of_order_clean",
        "type_of_vehicle_clean": "type_of_vehicle_clean",
        "city_code": "city_code",
    }
    for prefix, source in one_hot_sources.items():
        fit_categories = set(fit_raw[source].dropna().astype(str).unique().tolist())
        audit["fit_categories"][prefix] = sorted(fit_categories)
        values = raw[source].astype("string").fillna("")
        for category in category_schema[prefix]:
            column = f"{prefix}_{category}"
            if column in X.columns:
                X[column] = ((values.to_numpy(dtype=object) == category) & (category in fit_categories)).astype(np.int8)
    for prefix, source in (("weather_clean_imp", "weather_clean"), ("city_type_clean_imp", "city_type_clean")):
        audit["fit_categories"][prefix] = sorted(fit_raw[source].dropna().astype(str).unique().tolist())
    audit["fit_groups"] = sorted(fit_raw["Delivery_person_ID"].astype(str).unique().tolist())
    return X.astype(np.float32), audit


def main() -> int:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    RELEASE.mkdir(parents=True, exist_ok=True)
    for path in (RELEASE / "recovery_protocol.json", RUNTIME / "classical_development_predictions.parquet"):
        if path.exists():
            raise RuntimeError(f"v4 outputs already exist; immutable run refuses overwrite: {path}")

    parent_manifest_path = PREVIOUS_RUN / "recovered_gbdt_manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = ROOT / parent_manifest["model"]["checkpoint_path"]
    checkpoint = joblib.load(checkpoint_path)
    if checkpoint["target_name"] != TARGET or checkpoint["target_unit"] != "minutes":
        raise ValueError("Reused recovered model has unexpected target contract")
    schema = json.loads((ROOT / "data" / "feature_schema.json").read_text(encoding="utf-8-sig"))
    split = json.loads((ROOT / "data" / "split_manifest.json").read_text(encoding="utf-8-sig"))
    feature_names = list(checkpoint["feature_names"])
    category_schema = schema["gbdt_view"]["one_hot_encoded_categories"]
    protocol = {
        "run_id": RUN_ID,
        "protocol_written_at_utc": now(),
        "change_from_v3": "Correct OOF residual preprocessing to fit GBDT imputers and categorical modes within each GroupKFold training fold; use one VQC feature scaler fit on all original-train covariates and shared across OOF/validation; reuse the v3 final train-only checkpoint, with no GBDT search or refit.",
        "model_configuration": MODEL_CONFIG,
        "target": {"name": TARGET, "unit": "minutes"},
        "split_manifest_sha256": sha256(ROOT / "data" / "split_manifest.json"),
        "source_feature_schema_sha256": sha256(ROOT / "data" / "feature_schema.json"),
        "model_checkpoint_path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
        "model_checkpoint_sha256": sha256(checkpoint_path),
        "crossfit": {"method": "GroupKFold", "n_splits": N_SPLITS, "group": "Delivery_person_ID", "partition": "original train"},
        "test_labels": "not loaded, read, fit, or used in this v4 stage",
    }
    protocol_path = RELEASE / "recovery_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    gbdt_columns = ["ID", "Delivery_person_ID", "split_partition", TARGET, *feature_names]
    vqc_columns = ["ID", "Delivery_person_ID", "split_partition", "vqc_x0_distance", "vqc_x1_congestion"]
    gbdt = strict_development(ROOT / "data" / "indian_delivery_gbdt.parquet", gbdt_columns)
    vqc = strict_development(ROOT / "data" / "indian_delivery_vqc.parquet", vqc_columns)
    rows = gbdt.merge(vqc, on=["ID", "Delivery_person_ID", "split_partition"], validate="one_to_one", sort=False)
    train_mask = rows["split_partition"].eq("train").to_numpy()
    val_mask = rows["split_partition"].eq("val").to_numpy()
    tr = rows.loc[train_mask].reset_index(drop=True)
    va = rows.loc[val_mask].reset_index(drop=True)
    if len(tr) != 29592 or len(va) != 5989 or rows["split_partition"].eq("test").any():
        raise ValueError("Target-bearing development tables must contain only original train and val counts")

    # This source load has no target or target-derived columns. It includes raw
    # features from all partitions, then immediately keeps only original train
    # rows before any fold statistics are learned.
    raw_columns = [
        "ID", "Delivery_person_ID", "split_partition", "haversine_distance_km",
        "delivery_person_age_clean", "delivery_person_ratings_clean", "order_to_pickup_min",
        "multiple_deliveries_clean", "traffic_density_ordinal", "traffic_density_clean", "weather_clean",
        "city_type_clean", "type_of_order_clean", "type_of_vehicle_clean", "city_code",
        "is_rush_hour", "is_adverse_weather",
    ]
    raw_all = pd.read_parquet(ROOT / "data" / "indian_delivery_clean.parquet", engine="fastparquet", columns=raw_columns)
    raw_train = raw_all.loc[raw_all["split_partition"].eq("train")].reset_index(drop=True)
    del raw_all
    if len(raw_train) != len(tr) or raw_train["ID"].duplicated().any():
        raise ValueError("Target-free raw feature rows do not uniquely match original train partition")
    raw_train = tr[["ID", "Delivery_person_ID", "split_partition"]].merge(
        raw_train, on=["ID", "Delivery_person_ID", "split_partition"], how="left", validate="one_to_one", sort=False,
        suffixes=("", "_raw"),
    )
    raw_train = raw_train[[
        "ID", "Delivery_person_ID", "split_partition", "haversine_distance_km",
        "delivery_person_age_clean", "delivery_person_ratings_clean", "order_to_pickup_min",
        "multiple_deliveries_clean", "traffic_density_ordinal", "traffic_density_clean",
        "weather_clean", "city_type_clean", "type_of_order_clean", "type_of_vehicle_clean",
        "city_code", "is_rush_hour", "is_adverse_weather",
    ]]
    if raw_train["haversine_distance_km"].isna().any():
        raise ValueError("Raw feature join lost train rows")

    # Use a single QML feature mapping fit on all original-train covariates only.
    # This scaler is shared across OOF rows, validation, and eventual inference.
    raw_by_id = raw_train.set_index("ID").loc[tr["ID"].to_numpy()]
    distance_min = float(pd.to_numeric(raw_by_id["haversine_distance_km"]).min())
    distance_max = float(pd.to_numeric(raw_by_id["haversine_distance_km"]).max())
    traffic_series = pd.to_numeric(raw_by_id["traffic_density_ordinal"], errors="coerce")
    global_traffic_mode = float(traffic_series.mode().iloc[0])
    distance_span = distance_max - distance_min if distance_max > distance_min else 1.0
    qx0 = np.clip((pd.to_numeric(raw_by_id["haversine_distance_km"]).to_numpy(dtype=np.float64) - distance_min) / distance_span, 0.0, 1.0)
    traffic_filled = traffic_series.fillna(global_traffic_mode).to_numpy(dtype=np.float64)
    qx1 = np.clip(
        (traffic_filled / 3.0) * 0.60
        + pd.to_numeric(raw_by_id["is_rush_hour"]).to_numpy(dtype=np.float64) * 0.25
        + pd.to_numeric(raw_by_id["is_adverse_weather"]).to_numpy(dtype=np.float64) * 0.15,
        0.0, 1.0,
    )
    q_scaler = {
        "version": "india-vqc-2feature-train-only-v1",
        "fit_partition": "original train covariates only",
        "features_order": ["vqc_x0_distance", "vqc_x1_congestion"],
        "vqc_x0_distance": {"source": "haversine_distance_km", "min_km": distance_min, "max_km": distance_max, "output_range": [0.0, 1.0], "clip": True},
        "vqc_x1_congestion": {
            "source": ["traffic_density_ordinal", "is_rush_hour", "is_adverse_weather"],
            "traffic_missing_mode": global_traffic_mode,
            "formula": "clip((traffic_density_ordinal/3)*0.60 + is_rush_hour*0.25 + is_adverse_weather*0.15,0,1)",
            "output_range": [0.0, 1.0],
        },
        "fit_row_count": int(len(raw_train)),
        "fit_group_count": int(raw_train["Delivery_person_ID"].nunique()),
    }
    train_q_check = pd.DataFrame({"ID": tr["ID"], "q0": qx0, "q1": qx1}).merge(
        tr[["ID", "vqc_x0_distance", "vqc_x1_congestion"]], on="ID", validate="one_to_one",
    )
    if not np.allclose(train_q_check["q0"], train_q_check["vqc_x0_distance"], rtol=0, atol=1e-12) or not np.allclose(train_q_check["q1"], train_q_check["vqc_x1_congestion"], rtol=0, atol=1e-12):
        raise ValueError("Recomputed single train-only QML scaler disagrees with parent VQC mapping")
    q_scaler_path = RELEASE / "quantum_feature_scaler.json"
    q_scaler_path.write_text(json.dumps(q_scaler, indent=2), encoding="utf-8")

    y_train = tr[TARGET].to_numpy(dtype=np.float64)
    groups = tr["Delivery_person_ID"].astype(str).to_numpy()
    X_base = tr.loc[:, feature_names].astype(np.float32)
    oof = np.full(len(tr), np.nan, dtype=np.float64)
    fold_ids = np.full(len(tr), -1, dtype=np.int32)
    fold_records: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=N_SPLITS)
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    with threadpool_limits(limits=4):
        for fold, (fit_idx, heldout_idx) in enumerate(splitter.split(X_base, y_train, groups)):
            fit_groups = sorted(set(groups[fit_idx]))
            heldout_groups = sorted(set(groups[heldout_idx]))
            if set(fit_groups) & set(heldout_groups):
                raise AssertionError("Courier overlap detected in OOF fold")
            X_fold, transform_audit = fold_transform(
                tr, raw_train, fit_idx, feature_names, category_schema,
            )
            model = HistGradientBoostingRegressor(**{k: v for k, v in MODEL_CONFIG.items() if k != "class"})
            model.fit(X_fold.iloc[fit_idx], y_train[fit_idx])
            oof[heldout_idx] = model.predict(X_fold.iloc[heldout_idx])
            fold_ids[heldout_idx] = fold
            fold_checkpoint = RUNTIME / "fold_models" / f"fold_{fold}.joblib"
            fold_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "feature_names": feature_names, "preprocessing": transform_audit}, fold_checkpoint)
            fit_ids = sorted(tr.iloc[fit_idx]["ID"].astype(str).tolist())
            heldout_ids = sorted(tr.iloc[heldout_idx]["ID"].astype(str).tolist())
            fold_records.append({
                "fold": fold,
                "fit_rows": int(len(fit_idx)),
                "heldout_rows": int(len(heldout_idx)),
                "fit_row_ids_sha256": canonical_sha(fit_ids),
                "heldout_row_ids_sha256": canonical_sha(heldout_ids),
                "fit_group_ids": fit_groups,
                "heldout_group_ids": heldout_groups,
                "fit_group_count": len(fit_groups),
                "heldout_group_count": len(heldout_groups),
                "preprocessing": transform_audit,
                "preprocessing_settings_sha256": canonical_sha(transform_audit),
                "fit_group_membership_matches_training_rows": transform_audit["fit_groups"] == fit_groups,
                "checkpoint_path": str(fold_checkpoint.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint_sha256": sha256(fold_checkpoint),
            })
    if not np.isfinite(oof).all() or (fold_ids < 0).any() or not np.isfinite(qx0).all() or not np.isfinite(qx1).all():
        raise AssertionError("OOF model or feature map failed to cover every training row")

    # Validation is predicted only with the already-fitted v3 estimator.
    y_val = va[TARGET].to_numpy(dtype=np.float64)
    X_val = va.loc[:, feature_names].astype(np.float32)
    yhat_val = checkpoint["model"].predict(X_val)
    val_vqc = va[["vqc_x0_distance", "vqc_x1_congestion"]].to_numpy(dtype=np.float64)
    if not np.isfinite(val_vqc).all() or val_vqc.min() < 0.0 or val_vqc.max() > 1.0:
        raise ValueError("Validation VQC features do not match the train-only [0,1] scaler")
    val_groups = va["Delivery_person_ID"].astype(str).to_numpy()
    val_indices = np.arange(len(va))
    role_es_idx, role_fusion_idx = next(
        GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=20261029).split(val_indices, groups=val_groups)
    )
    es_groups = sorted(set(val_groups[role_es_idx]))
    fusion_groups = sorted(set(val_groups[role_fusion_idx]))
    es_ids = sorted(va.iloc[role_es_idx]["ID"].astype(str).tolist())
    fusion_ids = sorted(va.iloc[role_fusion_idx]["ID"].astype(str).tolist())
    if set(es_groups) & set(fusion_groups) or set(es_ids) & set(fusion_ids):
        raise AssertionError("Validation role split leaked courier/row membership")
    validation_roles = {
        "method": "GroupShuffleSplit within original validation courier groups",
        "seed": 20261029,
        "train_size": 0.5,
        "vqc_early_stop_group_ids": es_groups,
        "fusion_selection_group_ids": fusion_groups,
        "vqc_early_stop_row_count": len(es_ids),
        "fusion_selection_row_count": len(fusion_ids),
        "vqc_early_stop_row_ids_sha256": canonical_sha(es_ids),
        "fusion_selection_row_ids_sha256": canonical_sha(fusion_ids),
        "all_validation_row_ids_sha256": canonical_sha(sorted(va["ID"].astype(str).tolist())),
        "all_validation_group_ids": sorted(set(val_groups)),
    }
    oof_rows = pd.DataFrame({
        "ID": tr["ID"].to_numpy(),
        "Delivery_person_ID": tr["Delivery_person_ID"].to_numpy(),
        "split_partition": "train",
        "fold": fold_ids,
        "development_role": "train_oof",
        "y_true_minutes": y_train,
        "gbdt_pred_minutes": oof,
        "residual_minutes": y_train - oof,
        "vqc_x0_distance": qx0,
        "vqc_x1_congestion": qx1,
    })
    val_rows = pd.DataFrame({
        "ID": va["ID"].to_numpy(),
        "Delivery_person_ID": va["Delivery_person_ID"].to_numpy(),
        "split_partition": "val",
        "fold": pd.array([None] * len(va), dtype="Int64"),
        "development_role": np.where(np.isin(val_indices, role_es_idx), "vqc_early_stop", "fusion_selection"),
        "y_true_minutes": y_val,
        "gbdt_pred_minutes": yhat_val,
        "residual_minutes": y_val - yhat_val,
        "vqc_x0_distance": val_vqc[:, 0],
        "vqc_x1_congestion": val_vqc[:, 1],
    })
    development = pd.concat([oof_rows, val_rows], ignore_index=True)
    development_path = RUNTIME / "classical_development_predictions.parquet"
    development.to_parquet(development_path, index=False, engine="fastparquet")
    fold_path = RELEASE / "crossfit_manifest.json"
    fold_path.write_text(json.dumps({
        "schema_version": "group-crossfit-v3-fold-local-preprocessing",
        "folds": fold_records,
        "development_roles": validation_roles,
        "quantum_feature_scaler_path": str(q_scaler_path.relative_to(ROOT)).replace("\\", "/"),
        "quantum_feature_scaler_sha256": sha256(q_scaler_path),
    }, indent=2), encoding="utf-8")

    manifest = dict(parent_manifest)
    manifest.update({
        "schema_version": "run2-india-gbdt-recovery-v4-crossfit-correction",
        "run_id": RUN_ID,
        "created_at_utc": now(),
        "supersedes_run_id": parent_manifest.get("run_id"),
        "model": dict(parent_manifest["model"]),
        "crossfit": {
            "method": "GroupKFold(n_splits=5) restricted to original train",
            "fold_manifest_path": str(fold_path.relative_to(ROOT)).replace("\\", "/"),
            "fold_manifest_sha256": sha256(fold_path),
            "prediction_path": str(development_path.relative_to(ROOT)).replace("\\", "/"),
            "prediction_sha256": sha256(development_path),
            "rows": int(len(development)),
            "train_oof_rows": int(len(oof_rows)),
            "validation_rows": int(len(val_rows)),
            "contains_test_rows": False,
            "residual_identity": "residual_minutes = y_true_minutes - gbdt_pred_minutes",
            "folds": fold_records,
            "oof_train_metrics_minutes": metric_summary(y_train, oof),
            "validation_metrics_minutes": metric_summary(y_val, yhat_val),
            "fold_crossfit_wall_seconds": time.perf_counter() - start_wall,
            "fold_crossfit_cpu_seconds": time.process_time() - start_cpu,
            "preprocessing_scope": "GBDT imputers and categorical modes fit within each fold training couriers; quantum scaler is a single mapping fit on original-train covariates only and reused for all OOF rows and validation",
            "train_y_oof_only": True,
            "development_roles": validation_roles,
            "quantum_feature_scaler_path": str(q_scaler_path.relative_to(ROOT)).replace("\\", "/"),
            "quantum_feature_scaler_sha256": sha256(q_scaler_path),
        },
        "run_history": [
            {"run_id": "luna_hgbr_recovery_20261029T131809Z", "status": "aborted_before_fit", "reason": "pandas rowgroup-only filters returned test rows; no metric/model fit; target-bearing table was materialized transiently before partition assertion; no test outcome inspected or used"},
            {"run_id": "luna_hgbr_recovery_20261029T131809Z_v2", "status": "aborted_before_fit", "reason": "fastparquet row_filter could not read selected nullable raw clean columns; no estimator fit"},
            {"run_id": "luna_hgbr_recovery_20261029T131809Z_v3", "status": "superseded_oof_preprocessing", "reason": "fixed GBDT recovery fit/checkpoint retained; OOF features initially reused outer train imputation; v4 regenerates only fixed-config OOF folds with fold-local transforms"},
        ],
        "test_partition_read_or_used": False,
        "v4_protocol_path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
        "v4_protocol_sha256": sha256(protocol_path),
        "v4_fold_manifest_path": str(fold_path.relative_to(ROOT)).replace("\\", "/"),
        "v4_fold_manifest_sha256": sha256(fold_path),
        "quantum_feature_scaler_path": str(q_scaler_path.relative_to(ROOT)).replace("\\", "/"),
        "quantum_feature_scaler_sha256": sha256(q_scaler_path),
    })
    manifest_path = RELEASE / "recovered_gbdt_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "OK",
        "run_id": RUN_ID,
        "manifest": str(manifest_path),
        "development_table": str(development_path),
        "development_sha256": sha256(development_path),
        "reused_final_checkpoint": str(checkpoint_path),
        "reused_final_checkpoint_sha256": sha256(checkpoint_path),
        "train_oof_metrics_minutes": manifest["crossfit"]["oof_train_metrics_minutes"],
        "validation_metrics_minutes": manifest["crossfit"]["validation_metrics_minutes"],
        "folds": N_SPLITS,
        "test_labels_loaded": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
