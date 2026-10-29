"""Run the one authorized, bounded recovery fit for Indian ETA.

This script never reads sealed test rows. It trains one fixed HistGradientBoosting
configuration, creates five courier-group out-of-fold predictions inside the
original train partition, and then fits one release candidate on all original
train rows for validation/final inference.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
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
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "luna_hgbr_recovery_20261029T131809Z_v3"
RUNTIME = ROOT / ".runtime" / "run2" / RUN_ID
RELEASE = ROOT / "artifacts" / "run2_india" / RUN_ID
TARGET = "delivery_duration_min"
FEATURE_PATH = ROOT / "data" / "feature_schema.json"
SPLIT_PATH = ROOT / "data" / "split_manifest.json"
PREPROCESS_PATH = ROOT / "data" / "preprocessing_artifacts.json"
RAW_PATH = ROOT / "data" / "train.csv"
BASE_COMMIT = "40bd7fad0166b35b40bbc116e997cbac474e2c22"

GBDT_CONFIG = {
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
N_SPLITS = 5


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae_minutes": float(mean_absolute_error(y_true, y_pred)),
        "rmse_minutes": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_absolute_error_minutes": float(median_absolute_error(y_true, y_pred)),
    }


def fit_params() -> dict[str, Any]:
    return dict(GBDT_CONFIG)


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**{k: v for k, v in GBDT_CONFIG.items() if k != "class"})


def read_development_parquet(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read only train/val rows; ParquetFile row_filter=True enforces row-level filtering."""
    parquet = ParquetFile(path)
    filters = [[("split_partition", "in", ["train", "val"])]]
    partitions = parquet.to_pandas(columns=["split_partition"], filters=filters, row_filter=True)
    found = set(partitions["split_partition"].dropna().astype(str).unique())
    if found != {"train", "val"}:
        raise ValueError(f"strict development row filter expected train/val only, found {sorted(found)}")
    frame = parquet.to_pandas(columns=columns, filters=filters, row_filter=True)
    if set(frame["split_partition"].dropna().astype(str).unique()) != {"train", "val"}:
        raise ValueError("row-level target-bearing Parquet load included an unexpected partition")
    return frame


def main() -> int:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    RELEASE.mkdir(parents=True, exist_ok=True)
    for sentinel in (RUNTIME / "TEST-EVALUATION-OPENED.json", RUNTIME / "final_test_predictions.parquet"):
        if sentinel.exists():
            raise RuntimeError(f"Final test evaluation already opened; refusing a second test run: {sentinel}")

    schema = json.loads(FEATURE_PATH.read_text(encoding="utf-8-sig"))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8-sig"))
    preprocessing = json.loads(PREPROCESS_PATH.read_text(encoding="utf-8-sig"))
    feature_names = schema["gbdt_view"]["features"]
    if schema["target"]["name"] != TARGET or schema["target"]["unit"] != "minutes":
        raise ValueError("Unexpected ETA target schema; expected delivery_duration_min in minutes")

    protocol_path = RELEASE / "recovery_protocol.json"
    if protocol_path.exists():
        raise RuntimeError(f"Protocol already exists for immutable run {RUN_ID}; refusing a second fit")
    protocol = {
        "run_id": RUN_ID,
        "protocol_written_at_utc": utc_now(),
        "target_name": TARGET,
        "target_unit": "minutes",
        "fit": "one explicitly authorized recovery fit because the frozen Indian ETA checkpoint and saved OOF predictions were absent",
        "model_configuration": fit_params(),
        "feature_names": feature_names,
        "feature_schema_sha256": file_sha256(FEATURE_PATH),
        "split_manifest_sha256": file_sha256(SPLIT_PATH),
        "crossfit": {
            "method": "GroupKFold",
            "folds": N_SPLITS,
            "group": "Delivery_person_ID",
            "eligible_partition": "original train only",
            "fold_preprocessing": "parent train-only imputation/one-hot transform reused unchanged for all OOF folds; no fold-heldout outcomes enter model fitting",
        },
        "final_baseline_fit_partition": "original train only",
        "validation_role": "baseline reporting and subsequent residual architecture/alpha selection; no GBDT hyperparameter search",
        "sealed_test_role": "excluded from this script and all selection; separate single post-freeze evaluation",
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    # Only train/validation target-bearing rows are loaded here. The sealed test
    # partition is absent from all fitting, cross-fitting, and selection steps.
    allowed = ["train", "val"]
    gbdt_cols = ["ID", "Delivery_person_ID", "split_partition", TARGET, *feature_names]
    vqc_cols = ["ID", "Delivery_person_ID", "split_partition", "vqc_x0_distance", "vqc_x1_congestion"]
    gbdt = read_development_parquet(ROOT / "data" / "indian_delivery_gbdt.parquet", gbdt_cols)
    vqc = read_development_parquet(ROOT / "data" / "indian_delivery_vqc.parquet", vqc_cols)
    if set(gbdt["split_partition"].unique()) != {"train", "val"}:
        raise ValueError("Development load must contain exactly train and val; test is prohibited")
    if gbdt["ID"].duplicated().any() or vqc["ID"].duplicated().any():
        raise ValueError("Duplicate IDs found in a source partition")
    rows = gbdt.merge(vqc, on=["ID", "Delivery_person_ID", "split_partition"], validate="one_to_one", sort=False)
    if len(rows) != 29592 + 5989:
        raise ValueError(f"Unexpected development row count: {len(rows)}")
    train_mask = rows["split_partition"].eq("train").to_numpy()
    val_mask = rows["split_partition"].eq("val").to_numpy()
    tr = rows.loc[train_mask].reset_index(drop=True)
    va = rows.loc[val_mask].reset_index(drop=True)
    y_train = tr[TARGET].to_numpy(dtype=np.float64)
    y_val = va[TARGET].to_numpy(dtype=np.float64)
    groups = tr["Delivery_person_ID"].astype(str).to_numpy()
    X_train_full = tr.loc[:, feature_names].astype(np.float32)
    X_val_full = va.loc[:, feature_names].astype(np.float32)

    train_start = time.perf_counter()
    train_cpu_start = time.process_time()
    oof = np.full(len(tr), np.nan, dtype=np.float64)
    fold_audit: list[dict[str, Any]] = []
    fold_assignments = np.full(len(tr), -1, dtype=np.int32)
    splitter = GroupKFold(n_splits=N_SPLITS)
    with threadpool_limits(limits=4):
        for fold, (fit_idx, heldout_idx) in enumerate(splitter.split(X_train_full, y_train, groups)):
            if set(groups[fit_idx]).intersection(groups[heldout_idx]):
                raise AssertionError(f"Courier group leakage in fold {fold}")
            fold_imputer_X = X_train_full
            fold_imputation = {
                "source": "parent `data/preprocessing_artifacts.json` and `data/feature_schema.json`",
                "fit_partition": "original train only",
                "reuse_for_group_oof": "fixed target-independent preprocessing across folds; heldout fold labels are excluded from fit",
            }
            fold_model = make_model()
            fold_model.fit(fold_imputer_X.iloc[fit_idx], y_train[fit_idx])
            oof[heldout_idx] = fold_model.predict(fold_imputer_X.iloc[heldout_idx])
            fold_assignments[heldout_idx] = fold

            train_ids = sorted(tr.iloc[fit_idx]["ID"].astype(str).tolist())
            heldout_ids = sorted(tr.iloc[heldout_idx]["ID"].astype(str).tolist())
            train_groups = sorted(set(groups[fit_idx]))
            heldout_groups = sorted(set(groups[heldout_idx]))
            fold_path = RUNTIME / "fold_models" / f"fold_{fold}.joblib"
            fold_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": fold_model, "imputation": fold_imputation, "feature_names": feature_names}, fold_path)
            fold_audit.append({
            "fold": fold,
            "fit_rows": int(len(fit_idx)),
            "heldout_rows": int(len(heldout_idx)),
            "fit_row_ids_sha256": json_sha256(train_ids),
            "heldout_row_ids_sha256": json_sha256(heldout_ids),
            "fit_group_count": len(train_groups),
            "heldout_group_count": len(heldout_groups),
            "fit_group_ids": train_groups,
            "heldout_group_ids": heldout_groups,
            "fit_preprocessing": fold_imputation,
            "model_checkpoint_path": str(fold_path.relative_to(ROOT)).replace("\\", "/"),
            "model_checkpoint_sha256": file_sha256(fold_path),
        })
            partial = pd.DataFrame({
            "ID": tr.iloc[np.flatnonzero(np.isfinite(oof))]["ID"].to_numpy(),
            "Delivery_person_ID": tr.iloc[np.flatnonzero(np.isfinite(oof))]["Delivery_person_ID"].to_numpy(),
            "fold": fold_assignments[np.isfinite(oof)],
            "y_true_minutes": y_train[np.isfinite(oof)],
            "gbdt_pred_minutes": oof[np.isfinite(oof)],
        })
            partial["residual_minutes"] = partial["y_true_minutes"] - partial["gbdt_pred_minutes"]
            partial.to_parquet(RUNTIME / "oof_checkpoint.parquet", index=False, engine="fastparquet")

    if not np.isfinite(oof).all() or (fold_assignments < 0).any():
        raise AssertionError("Every original-train row must receive exactly one group-heldout prediction")
    crossfit_seconds = time.perf_counter() - train_start
    crossfit_cpu_seconds = time.process_time() - train_cpu_start

    # Fixed one-shot full train fit; no hyperparameter sweep or test use.
    final_model = make_model()
    with threadpool_limits(limits=4):
        final_model.fit(X_train_full, y_train)
        val_prediction = final_model.predict(X_val_full)
    validation_metrics = metrics(y_val, val_prediction)

    checkpoint_path = RELEASE / "recovered_eta_gbdt.joblib"
    checkpoint = {
        "model": final_model,
        "feature_names": list(feature_names),
        "target_name": TARGET,
        "target_unit": "minutes",
        "recovery_fit_authorized_missing_frozen_checkpoint": True,
        "model_configuration": fit_params(),
        "preprocessing_artifacts": preprocessing,
        "preprocessing_json_sha256": file_sha256(PREPROCESS_PATH),
        "feature_schema_sha256": file_sha256(FEATURE_PATH),
    }
    joblib.dump(checkpoint, checkpoint_path)
    reloaded = joblib.load(checkpoint_path)
    reload_prediction = reloaded["model"].predict(X_val_full)
    reload_max_abs_error = float(np.max(np.abs(reload_prediction - val_prediction)))
    if reload_max_abs_error != 0.0:
        raise AssertionError(f"Checkpoint reload changed validation predictions: {reload_max_abs_error}")

    train_rows = tr.copy()
    train_rows["fold"] = fold_assignments
    train_rows["y_true_minutes"] = y_train
    train_rows["gbdt_pred_minutes"] = oof
    train_rows["residual_minutes"] = y_train - oof
    train_rows = train_rows[[
        "ID", "Delivery_person_ID", "split_partition", "fold", "y_true_minutes",
        "gbdt_pred_minutes", "residual_minutes", "vqc_x0_distance", "vqc_x1_congestion",
    ]]
    val_rows = va[["ID", "Delivery_person_ID", "split_partition", "vqc_x0_distance", "vqc_x1_congestion"]].copy()
    val_rows["fold"] = pd.Series([None] * len(val_rows), dtype="Int64")
    val_rows["y_true_minutes"] = y_val
    val_rows["gbdt_pred_minutes"] = val_prediction
    val_rows["residual_minutes"] = y_val - val_prediction
    val_rows = val_rows[train_rows.columns]
    development = pd.concat([train_rows, val_rows], ignore_index=True)
    if development["split_partition"].eq("test").any() or len(development) != 35581:
        raise AssertionError("Development prediction artifact must contain only 29,592 train and 5,989 val rows")
    if not np.allclose(development["residual_minutes"], development["y_true_minutes"] - development["gbdt_pred_minutes"], rtol=0, atol=1e-12):
        raise AssertionError("Residual target identity failed")
    dev_path = RUNTIME / "classical_development_predictions.parquet"
    development.to_parquet(dev_path, index=False, engine="fastparquet")

    # Provenance audit: current working CSV differs from the parent Git blob only
    # by line terminators. Keep both byte hashes and verify identical parsed rows.
    current_raw = RAW_PATH.read_bytes()
    git_raw = subprocess.check_output(["git", "show", f"{BASE_COMMIT}:data/train.csv"], cwd=ROOT)
    current_csv = list(csv.reader(current_raw.decode("utf-8-sig").splitlines()))
    git_csv = list(csv.reader(git_raw.decode("utf-8-sig").splitlines()))
    normalized_bytes_equal = current_raw.replace(b"\r\n", b"\n") == git_raw
    semantic_cells_equal = current_csv == git_csv
    if not normalized_bytes_equal or not semantic_cells_equal:
        raise ValueError("Raw source differs from parent Git object beyond CRLF/LF normalization")

    manifest = {
        "schema_version": "run2-india-gbdt-recovery-v1",
        "run_id": RUN_ID,
        "created_at_utc": utc_now(),
        "protocol_path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": file_sha256(protocol_path),
        "recovery_fit_authorized_missing_frozen_checkpoint": True,
        "historical_frozen_eta_checkpoint_search": "No trusted frozen India ETA checkpoint or saved leakage-safe prediction artifact was found; this is an explicitly authorized one-time recovery fit, not the recovered original/Gemini model.",
        "target_name": TARGET,
        "target_unit": "minutes",
        "base_git_commit": BASE_COMMIT,
        "parent_split_manifest_path": "data/split_manifest.json",
        "parent_split_manifest_sha256": file_sha256(SPLIT_PATH),
        "parent_feature_schema_path": "data/feature_schema.json",
        "parent_feature_schema_sha256": file_sha256(FEATURE_PATH),
        "parent_preprocessing_path": "data/preprocessing_artifacts.json",
        "parent_preprocessing_sha256": file_sha256(PREPROCESS_PATH),
        "transformed_data_sha256": {
            name: file_sha256(ROOT / "data" / name)
            for name in ("indian_delivery_clean.parquet", "indian_delivery_gbdt.parquet", "indian_delivery_vqc.parquet")
        },
        "raw_csv_provenance": {
            "working_path": "data/train.csv",
            "working_file_sha256": file_sha256(RAW_PATH),
            "parent_git_object": f"{BASE_COMMIT}:data/train.csv",
            "parent_git_blob_sha256": hashlib.sha256(git_raw).hexdigest(),
            "byte_identical": current_raw == git_raw,
            "line_ending_normalization": "CRLF working copy converted to LF matches parent blob byte-for-byte",
            "normalized_bytes_equal": normalized_bytes_equal,
            "semantic_csv_cells_equal": semantic_cells_equal,
            "parsed_csv_row_count_including_header": len(current_csv),
        },
        "split": {
            "method": split["split_method"],
            "seed": split["random_seed"],
            "original_counts": split["split_counts"],
            "original_courier_counts": split["courier_counts"],
            "train_grouped_oof_folds": N_SPLITS,
        },
        "model": {
            "label": "recovered_one_time_eta_gbdt",
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "class": GBDT_CONFIG["class"],
            "configuration": fit_params(),
            "features": feature_names,
            "feature_count": len(feature_names),
            "fit_partition": "original train only",
            "test_partition_read_or_used": False,
            "validation_metrics_minutes": validation_metrics,
            "reload_validation_max_abs_prediction_error_minutes": reload_max_abs_error,
        },
        "crossfit": {
            "method": "GroupKFold(n_splits=5) restricted to original train partition",
            "prediction_path": str(dev_path.relative_to(ROOT)).replace("\\", "/"),
            "prediction_sha256": file_sha256(dev_path),
            "rows": int(len(development)),
            "train_oof_rows": int(train_mask.sum()),
            "validation_rows": int(val_mask.sum()),
            "contains_test_rows": False,
            "residual_identity": "residual_minutes = y_true_minutes - gbdt_pred_minutes",
            "folds": fold_audit,
            "oof_train_metrics_minutes": metrics(y_train, oof),
            "validation_metrics_minutes": validation_metrics,
            "crossfit_wall_seconds": crossfit_seconds,
            "crossfit_cpu_seconds": crossfit_cpu_seconds,
            "feature_preprocessing": "parent preprocessing constants and one-hot vocabulary are fitted on the complete original train partition and reused for OOF folds; heldout fold labels are excluded from GBDT fitting; all IDs, partition, and target columns are excluded from input features",
        },
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "parquet_engine": "fastparquet 2024.11.0",
            "device": "CPU",
            "test_evaluation_performed": False,
        },
        "independent_loader": {
            "contract": "joblib.load(checkpoint_path) returns dict with model, feature_names, target_name/unit, config and preprocessing metadata; model.predict requires columns in feature_names order and outputs minutes.",
            "trusted_local_artifact_only": True,
        },
    }
    fold_manifest_path = RELEASE / "crossfit_manifest.json"
    fold_manifest_path.write_text(json.dumps({"schema_version": "group-crossfit-v1", "folds": fold_audit}, indent=2), encoding="utf-8")
    manifest["crossfit"]["fold_manifest_path"] = str(fold_manifest_path.relative_to(ROOT)).replace("\\", "/")
    manifest["crossfit"]["fold_manifest_sha256"] = file_sha256(fold_manifest_path)
    manifest_path = RELEASE / "recovered_gbdt_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "OK",
        "manifest": str(manifest_path),
        "checkpoint_sha256": manifest["model"]["checkpoint_sha256"],
        "development_predictions": str(dev_path),
        "development_predictions_sha256": manifest["crossfit"]["prediction_sha256"],
        "train_oof_metrics_minutes": manifest["crossfit"]["oof_train_metrics_minutes"],
        "validation_metrics_minutes": validation_metrics,
        "folds": N_SPLITS,
        "test_rows_loaded": False,
        "raw_source_byte_different_semantically_equal": True,
        "crossfit_wall_seconds": crossfit_seconds,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
