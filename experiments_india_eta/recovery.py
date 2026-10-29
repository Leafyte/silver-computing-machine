"""Bounded HistGradientBoostingRegressor recovery for India ETA.

Only the committed GBDT feature view is used for model fitting. The VQC view is
joined by stable ID/courier/split keys and contributes only four explicit
candidate features to prediction records for the separate residual worker.
The sealed test partition is not returned by the loader or scored here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits


EXPECTED = {
    "source_csv_git_sha256": "4baa699e1c09d286f06790010a8c0b284ed3376cbde7e346e19d0ded16a240e3",
    "gbdt_parquet_sha256": "bd03b47c3d3e23050329cefc69d752aad607031cacdb9696151d237e3534a627",
    "vqc_parquet_sha256": "a633677472e6e0e874c2797a66eb7b011e3a307285553f8fa00c6ee664df4b4e",
    "clean_parquet_sha256": "9034d24389c7162f0ea0e275e5b4cc9ec16f04bba75c0f239dbcef8535a1f7c1",
}
EXPECTED_COUNTS = {"train": 29592, "val": 5989, "test": 5941}
VQC_FEATURES = [
    "vqc_x0_distance",
    "vqc_x1_congestion",
    "vqc_x2_multidelivery",
    "vqc_x3_preptime",
]
KEYS = ["ID", "Delivery_person_ID", "split_partition"]


@dataclass
class DataBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    feature_names: list[str]
    vqc_feature_names: list[str]
    provenance: dict[str, Any]
    validation_roles: dict[str, str]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _normalized_git_csv_sha(path: Path) -> tuple[str, int, int]:
    """Hash canonical LF bytes while retaining the actual checkout digest elsewhere."""
    b = path.read_bytes()
    crlf = b.count(b"\r\n")
    bare_cr = b.count(b"\r") - crlf
    if bare_cr:
        raise ValueError(f"CSV contains {bare_cr} bare CR bytes; refusing normalization")
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest(), len(b), crlf


def load_verified_data(root: str | Path) -> DataBundle:
    root = Path(root).resolve()
    data_dir = root / "data"
    paths = {
        "source_csv": data_dir / "train.csv",
        "gbdt_parquet": data_dir / "indian_delivery_gbdt.parquet",
        "vqc_parquet": data_dir / "indian_delivery_vqc.parquet",
        "clean_parquet": data_dir / "indian_delivery_clean.parquet",
    }
    absent = [str(p) for p in paths.values() if not p.is_file()]
    if absent:
        raise FileNotFoundError("Missing committed input(s): " + ", ".join(absent))
    actual = {k: _sha256(p) for k, p in paths.items()}
    normalized_sha, raw_size, crlf_count = _normalized_git_csv_sha(paths["source_csv"])
    if normalized_sha != EXPECTED["source_csv_git_sha256"]:
        raise ValueError(
            "Canonical LF-normalized data/train.csv digest does not match committed source hash: "
            f"{normalized_sha} != {EXPECTED['source_csv_git_sha256']}"
        )
    for name in ("gbdt_parquet", "vqc_parquet", "clean_parquet"):
        if actual[name] != EXPECTED[name + "_sha256"]:
            raise ValueError(f"{name} SHA256 mismatch: {actual[name]} != {EXPECTED[name + '_sha256']}")

    schema_path = data_dir / "feature_schema.json"
    split_path = data_dir / "split_manifest.json"
    preprocessing_path = data_dir / "preprocessing_artifacts.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    expected_features = list(schema["gbdt_view"]["features"])
    if split.get("source_sha256", "").lower() != EXPECTED["source_csv_git_sha256"]:
        raise ValueError("split_manifest source SHA does not match the verified canonical Git source SHA")
    if split.get("split_counts", {}) != {**EXPECTED_COUNTS, "excluded": 4071}:
        raise ValueError(f"Split manifest counts differ from frozen contract: {split.get('split_counts')}")
    expected_manifest_digests = {
        "gbdt_parquet_sha256": actual["gbdt_parquet"],
        "vqc_parquet_sha256": actual["vqc_parquet"],
        "clean_parquet_sha256": actual["clean_parquet"],
    }
    for key, digest in expected_manifest_digests.items():
        if split.get("file_digests", {}).get(key, "").lower() != digest:
            raise ValueError(f"Split manifest {key} does not match actual committed file")
    if len(expected_features) != 59:
        raise ValueError(f"Expected 59 frozen GBDT features, got {len(expected_features)}")
    if schema["target"]["name"] != "delivery_duration_min" or schema["target"]["unit"] != "minutes":
        raise ValueError("Unexpected regression target contract")

    gbdt_columns = [*KEYS, *expected_features, "delivery_duration_min"]
    # Push a split filter to the Parquet reader; test rows are immediately excluded.
    # fastparquet is available in the configured isolated RUN2 dependency paths.
    gbdt = pd.read_parquet(
        paths["gbdt_parquet"], engine="fastparquet", columns=gbdt_columns,
        filters=[("split_partition", "in", ["train", "val"])],
    )
    vqc = pd.read_parquet(
        paths["vqc_parquet"], engine="fastparquet", columns=[*KEYS, *VQC_FEATURES],
        filters=[("split_partition", "in", ["train", "val"])],
    )
    # fastparquet's row-level filter may decode the single-file row group before
    # applying the predicate; drop all test records immediately and never return
    # their features or targets to model-selection/training code.
    gbdt = gbdt.loc[gbdt.split_partition.isin(["train", "val"])].copy()
    vqc = vqc.loc[vqc.split_partition.isin(["train", "val"])].copy()
    if gbdt.columns.tolist() != [*KEYS, *expected_features, "delivery_duration_min"]:
        raise ValueError("GBDT view columns differ from frozen key/feature/target order")
    if gbdt[KEYS].isna().any().any() or gbdt["ID"].duplicated().any():
        raise ValueError("GBDT stable record IDs/keys are missing or duplicated")
    if vqc[KEYS].isna().any().any() or vqc["ID"].duplicated().any():
        raise ValueError("VQC stable record IDs/keys are missing or duplicated")
    joined = gbdt.merge(vqc, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not (joined["_merge"] == "both").all():
        raise ValueError("GBDT/VQC view IDs, courier IDs, and split labels do not match one-to-one")
    joined = joined.drop(columns="_merge")
    counts = joined["split_partition"].value_counts().to_dict()
    if counts != {"train": EXPECTED_COUNTS["train"], "val": EXPECTED_COUNTS["val"]}:
        raise ValueError(f"Train/validation counts differ from manifest after keyed join: {counts}")
    if set(gbdt["Delivery_person_ID"]) & set(vqc["Delivery_person_ID"]) == set():
        raise ValueError("Unexpected empty courier ID join")
    train = joined.loc[joined.split_partition == "train"].copy().reset_index(drop=True)
    validation = joined.loc[joined.split_partition == "val"].copy().reset_index(drop=True)
    train_couriers = set(train.Delivery_person_ID.astype(str))
    validation_couriers = set(validation.Delivery_person_ID.astype(str))
    if train_couriers & validation_couriers:
        raise ValueError("Source train/validation courier groups overlap")
    if not np.isfinite(train[expected_features].to_numpy(dtype=np.float64)).all():
        raise ValueError("Training features contain non-finite values")
    if not np.isfinite(validation[expected_features].to_numpy(dtype=np.float64)).all():
        raise ValueError("Validation features contain non-finite values")
    if not np.isfinite(train.delivery_duration_min.to_numpy(dtype=np.float64)).all() or not np.isfinite(
        validation.delivery_duration_min.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Train/validation targets contain non-finite values")

    ranked_couriers = sorted(
        validation_couriers,
        key=lambda courier: (hashlib.sha256(b"india-eta-val-role-v1|20261029|" + courier.encode("utf-8")).digest(), courier),
    )
    calibration_couriers = set(ranked_couriers[: len(ranked_couriers) // 3])
    roles = {
        courier: ("fusion_calibration" if courier in calibration_couriers else "model_selection")
        for courier in sorted(validation_couriers)
    }
    role_set = set(roles.values())
    if role_set != {"model_selection", "fusion_calibration"}:
        raise ValueError(f"Validation courier assignment failed to populate both roles: {role_set}")
    val_roles = validation.Delivery_person_ID.astype(str).map(roles)
    if val_roles.isna().any():
        raise AssertionError("Validation role mapping left unmapped couriers")
    val_role_counts = val_roles.value_counts().to_dict()
    val_role_courier_counts = validation.assign(_role=val_roles.to_numpy()).groupby("_role").Delivery_person_ID.nunique().to_dict()

    prov = {
        "source_commit": "7ffb99e85e01f775772783c472c9d24f39f9d868",
        "worktree_head": "40bd7fad0166b35b40bbc116e997cbac474e2c22",
        "source_csv": {
            "path": "data/train.csv",
            "worktree_sha256_crlf": actual["source_csv"],
            "canonical_git_sha256_lf": normalized_sha,
            "expected_git_sha256_lf": EXPECTED["source_csv_git_sha256"],
            "bytes": raw_size,
            "crlf_count": crlf_count,
            "normalization": "CRLF-to-LF only; byte-level normalization equals the committed Git blob",
        },
        "files_sha256": actual,
        "schema_sha256": _sha256(schema_path),
        "split_manifest_sha256": _sha256(split_path),
        "preprocessing_artifacts_sha256": _sha256(preprocessing_path),
        "gbdt_feature_count": len(expected_features),
        "target": "delivery_duration_min",
        "target_unit": "minutes",
        "split_counts_expected": EXPECTED_COUNTS,
        "split_counts_loaded_train_val": counts,
        "courier_counts_train_val": {"train": len(train_couriers), "val": len(validation_couriers)},
        "validation_role_counts": val_role_counts,
        "validation_role_courier_counts": val_role_courier_counts,
        "validation_role_algorithm": "Sort unique validation courier IDs by (SHA256(b'india-eta-val-role-v1|20261029|' + courier_id UTF-8) raw digest bytes, courier_id); first floor(N/3) groups=fusion_calibration, remaining groups=model_selection",
        "validation_role_assignment_sha256": hashlib.sha256(json.dumps(roles, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "prediction_join_key": ["ID", "Delivery_person_ID", "split_partition"],
        "vqc_features": VQC_FEATURES,
        "test_partition": "not loaded into DataBundle, not scored, not written to prediction artifacts",
    }
    return DataBundle(train, validation, expected_features, VQC_FEATURES, prov, roles)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {
        "mae_min": float(np.mean(np.abs(err))),
        "rmse_min": float(np.sqrt(np.mean(np.square(err)))),
        "median_absolute_error_min": float(np.median(np.abs(err))),
    }


def _model(config: dict[str, Any], seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=float(config["learning_rate"]),
        max_iter=int(config["max_iter"]),
        max_leaf_nodes=int(config["max_leaf_nodes"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        l2_regularization=float(config["l2_regularization"]),
        early_stopping=False,
        random_state=seed,
    )


def _model_complexity(model: HistGradientBoostingRegressor) -> dict[str, int]:
    trees = model._predictors
    return {
        "iterations_fitted": len(trees),
        "trees_per_iteration": int(model.n_trees_per_iteration_),
        "total_trees": sum(len(it) for it in trees),
        "total_leaf_nodes": sum(
            int(np.count_nonzero(pred[0].nodes["is_leaf"]))
            for pred in trees
        ),
        "feature_count": int(model.n_features_in_),
    }


def _val_role_array(frame: pd.DataFrame, role_by_courier: dict[str, str]) -> np.ndarray:
    return frame["Delivery_person_ID"].astype(str).map(role_by_courier).to_numpy()


def run_recovery(root: str | Path, output_dir: str | Path | None = None, *, seed: int = 20261029) -> dict[str, Any]:
    root = Path(root).resolve()
    out = Path(output_dir).resolve() if output_dir else root / "artifacts" / "india_eta_recovery"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gbdt").mkdir(exist_ok=True)
    data = load_verified_data(root)
    features = data.feature_names
    train = data.train
    val = data.validation
    x_train = train[features].to_numpy(dtype=np.float64)
    y_train = train.delivery_duration_min.to_numpy(dtype=np.float64)
    groups = train.Delivery_person_ID.astype(str).to_numpy()
    x_val = val[features].to_numpy(dtype=np.float64)
    y_val = val.delivery_duration_min.to_numpy(dtype=np.float64)
    roles = _val_role_array(val, data.validation_roles)
    select_mask = roles == "model_selection"
    calibration_mask = roles == "fusion_calibration"
    if select_mask.sum() < 500 or calibration_mask.sum() < 500:
        raise ValueError("Validation group roles are too small for bounded selection and calibration")

    baseline_value = float(np.median(y_train))
    baseline_predictions = np.full(y_val.shape, baseline_value, dtype=np.float64)
    configs = [
        {"name": "hgb_small", "max_iter": 100, "learning_rate": 0.08, "max_leaf_nodes": 15, "min_samples_leaf": 30, "l2_regularization": 1.0},
        {"name": "hgb_medium", "max_iter": 160, "learning_rate": 0.06, "max_leaf_nodes": 31, "min_samples_leaf": 25, "l2_regularization": 1.0},
        {"name": "hgb_regularized", "max_iter": 140, "learning_rate": 0.06, "max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 4.0},
    ]
    trials: list[dict[str, Any]] = []
    for config in configs:
        model = _model(config, seed)
        started = time.perf_counter()
        with threadpool_limits(limits=6):
            model.fit(x_train, y_train)
        elapsed = time.perf_counter() - started
        with threadpool_limits(limits=6):
            pred = model.predict(x_val)
        metrics = _metrics(y_val[select_mask], pred[select_mask])
        trials.append({
            "config": config,
            "selection_metrics": metrics,
            "selection_rows": int(select_mask.sum()),
            "calibration_rows_not_used_for_selection": int(calibration_mask.sum()),
            "training_seconds": float(elapsed),
            "complexity": _model_complexity(model),
        })
    best_idx = min(range(len(trials)), key=lambda i: trials[i]["selection_metrics"]["mae_min"])
    chosen_config = configs[best_idx]

    # OOF train predictions: each courier is held out from the fold model that predicts it.
    splitter = GroupKFold(n_splits=5)
    oof = np.full(len(train), np.nan, dtype=np.float64)
    fold_manifest: dict[str, Any] = {}
    fold_training_seconds: list[float] = []
    for fold, (fit_idx, held_idx) in enumerate(splitter.split(x_train, y_train, groups), start=0):
        fit_groups = sorted(set(groups[fit_idx]))
        held_groups = sorted(set(groups[held_idx]))
        if set(fit_groups) & set(held_groups):
            raise AssertionError(f"Courier leakage in OOF fold {fold}")
        fm = _model(chosen_config, seed + fold + 1)
        start = time.perf_counter()
        with threadpool_limits(limits=6):
            fm.fit(x_train[fit_idx], y_train[fit_idx])
        fold_training_seconds.append(time.perf_counter() - start)
        with threadpool_limits(limits=6):
            oof[held_idx] = fm.predict(x_train[held_idx])
        fold_manifest[str(fold)] = {
            "train_row_count": int(len(fit_idx)),
            "heldout_row_count": int(len(held_idx)),
            "train_courier_count": len(fit_groups),
            "heldout_courier_count": len(held_groups),
            "train_couriers": fit_groups,
            "heldout_couriers": held_groups,
            "train_courier_ids_sha256": hashlib.sha256("\n".join(fit_groups).encode()).hexdigest(),
            "heldout_courier_ids_sha256": hashlib.sha256("\n".join(held_groups).encode()).hexdigest(),
        }
    if not np.isfinite(oof).all():
        raise AssertionError("OOF prediction coverage incomplete")

    # Refit the frozen estimator on the full permitted training partition only.
    final_model = _model(chosen_config, seed)
    fit_start = time.perf_counter()
    with threadpool_limits(limits=6):
        final_model.fit(x_train, y_train)
    final_training_seconds = time.perf_counter() - fit_start
    with threadpool_limits(limits=6):
        val_pred = final_model.predict(x_val)
    val_median_seconds = []
    with threadpool_limits(limits=6):
        for _ in range(31):
            t0 = time.perf_counter()
            final_model.predict(x_val[:1])
            val_median_seconds.append(time.perf_counter() - t0)
    inference_latency_us = float(statistics.median(val_median_seconds) * 1e6)

    oof_frame = train[["ID", "Delivery_person_ID", "split_partition", *data.vqc_feature_names]].copy()
    oof_frame = oof_frame.rename(columns={"ID": "delivery_record_id", "Delivery_person_ID": "courier_id"})
    oof_frame["fold"] = -1
    for fold, info in fold_manifest.items():
        ids = set(info["heldout_couriers"])
        mask = oof_frame.courier_id.astype(str).isin(ids)
        oof_frame.loc[mask, "fold"] = int(fold)
    oof_frame["y_true_min"] = y_train
    oof_frame["gbdt_pred_min"] = oof
    val_frame = val[["ID", "Delivery_person_ID", "split_partition", *data.vqc_feature_names]].copy()
    val_frame = val_frame.rename(columns={"ID": "delivery_record_id", "Delivery_person_ID": "courier_id"})
    val_frame["validation_role"] = val_frame.courier_id.astype(str).map(data.validation_roles)
    val_frame["fold"] = pd.NA
    val_frame["y_true_min"] = y_val
    val_frame["gbdt_pred_min"] = val_pred
    if len(oof_frame) != EXPECTED_COUNTS["train"] or len(val_frame) != EXPECTED_COUNTS["val"]:
        raise AssertionError("Prediction row count changed")
    if oof_frame.fold.isna().any() or (oof_frame.fold < 0).any():
        raise AssertionError("OOF fold provenance missing for at least one row")

    gbdt_dir = out / "gbdt"
    model_path = gbdt_dir / "india_eta_hgb_regressor.joblib"
    joblib.dump(final_model, model_path, compress=3)
    oof_path = out / "oof_predictions.parquet"
    val_path = out / "validation_predictions.parquet"
    oof_frame.to_parquet(oof_path, index=False, engine="fastparquet")
    val_frame.to_parquet(val_path, index=False, engine="fastparquet")
    fold_path = out / "oof_fold_manifest.json"
    fold_path.write_text(json.dumps({"folds": fold_manifest, "n_splits": 5, "group_column": "Delivery_person_ID"}, indent=2), encoding="utf-8")
    role_counts = val_frame.validation_role.value_counts().to_dict()
    role_map_path = out / "validation_role_assignment.json"
    role_map_payload = {
        "algorithm": data.provenance["validation_role_algorithm"],
        "courier_to_role": data.validation_roles,
        "assignment_sha256": data.provenance["validation_role_assignment_sha256"],
        "role_courier_counts": data.provenance["validation_role_courier_counts"],
        "role_record_counts": role_counts,
        "record_id_to_role": dict(zip(val_frame.delivery_record_id.astype(str), val_frame.validation_role)),
    }
    role_map_payload["record_assignment_sha256"] = hashlib.sha256(
        json.dumps(role_map_payload["record_id_to_role"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    role_map_path.write_text(json.dumps(role_map_payload, indent=2), encoding="utf-8")
    schema_doc = {
        "estimator_identity": "sklearn.ensemble.HistGradientBoostingRegressor",
        "objective": "squared_error regression",
        "target": "delivery_duration_min",
        "target_unit": "minutes",
        "feature_names": features,
        "feature_count": len(features),
        "feature_order_source": "data/feature_schema.json:gbdt_view.features",
        "forbidden_model_inputs": ["ID", "Delivery_person_ID", "split_partition", "delivery_duration_min"],
        "categorical_handling": "one-hot numeric feature matrix as committed; no refit preprocessing",
        "preprocessing_artifacts_source_path": "data/preprocessing_artifacts.json",
        "preprocessing_artifacts_sha256": data.provenance["preprocessing_artifacts_sha256"],
        "preprocessing_artifacts_path": "artifacts/india_eta_recovery/gbdt/preprocessing_artifacts.json",
        "model_config": chosen_config,
        "seed": seed,
    }
    schema_path = gbdt_dir / "feature_schema.json"
    schema_path.write_text(json.dumps(schema_doc, indent=2), encoding="utf-8")
    preprocessing_copy = gbdt_dir / "preprocessing_artifacts.json"
    shutil.copyfile(root / "data" / "preprocessing_artifacts.json", preprocessing_copy)
    report = {
        "status": "trained",
        "run_type": "bounded committed-data reconstruction",
        "data_provenance": data.provenance,
        "estimator_identity": "sklearn.ensemble.HistGradientBoostingRegressor",
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "python_version": platform.python_version(),
        "fastparquet_version": __import__("fastparquet").__version__,
        "training_code_sha256": _sha256(Path(__file__).resolve()),
        "existing_checkpoint_search": "No valid India ETA regression checkpoint found in scoped local artifacts; this is an authorized bounded reconstruction.",
        "model_selection": {"criterion": "minimum validation model_selection MAE", "trials": trials, "chosen_config": chosen_config},
        "training_rows": len(train),
        "training_courier_groups": len(set(groups)),
        "training_record_ids_sha256": hashlib.sha256("\n".join(sorted(train.ID.astype(str))).encode("utf-8")).hexdigest(),
        "training_courier_ids_sha256": hashlib.sha256("\n".join(sorted(set(groups))).encode("utf-8")).hexdigest(),
        "validation_rows": len(val),
        "validation_courier_groups": len(data.validation_roles),
        "validation_role_counts_rows": role_counts,
        "validation_role_courier_counts": data.provenance["validation_role_courier_counts"],
        "validation_role_algorithm": data.provenance["validation_role_algorithm"],
        "validation_role_assignment_sha256": data.provenance["validation_role_assignment_sha256"],
        "baseline": {"type": "training-set median", "prediction_min": baseline_value, "metrics_all_validation": _metrics(y_val, baseline_predictions), "metrics_model_selection": _metrics(y_val[select_mask], baseline_predictions[select_mask]), "metrics_fusion_calibration": _metrics(y_val[calibration_mask], baseline_predictions[calibration_mask])},
        "frozen_model_metrics": {"all_validation_descriptive": _metrics(y_val, val_pred), "model_selection": _metrics(y_val[select_mask], val_pred[select_mask]), "fusion_calibration": _metrics(y_val[calibration_mask], val_pred[calibration_mask])},
        "cpu_training_seconds": {"selection_full_train_runs": [t["training_seconds"] for t in trials], "oof_fold_runs": fold_training_seconds, "final_refit_full_train": final_training_seconds},
        "inference_latency": {"measurement": "31 single-row predict calls; median elapsed time", "median_microseconds_per_row": inference_latency_us, "measured_on": platform.processor() or platform.machine()},
        "model_complexity": _model_complexity(final_model),
        "prediction_artifacts": {"oof": {"path": str(oof_path.relative_to(root)), "rows": len(oof_frame), "sha256": _sha256(oof_path)}, "validation": {"path": str(val_path.relative_to(root)), "rows": len(val_frame), "sha256": _sha256(val_path)}, "oof_fold_manifest": {"path": str(fold_path.relative_to(root)), "sha256": _sha256(fold_path)}, "validation_role_assignment": {"path": str(role_map_path.relative_to(root)), "sha256": _sha256(role_map_path)}},
        "model_artifacts": {"model": {"path": str(model_path.relative_to(root)), "sha256": _sha256(model_path)}, "feature_schema": {"path": str(schema_path.relative_to(root)), "sha256": _sha256(schema_path)}, "preprocessing_artifacts": {"path": str(preprocessing_copy.relative_to(root)), "sha256": _sha256(preprocessing_copy)}},
        "sealed_test": "not loaded by DataBundle, not used in fitting, selection, calibration, or written to artifacts",
    }
    report_path = root / "INDIA_ETA_GBDT_TRAINING_REPORT.md"
    report_path.write_text(_render_report(report), encoding="utf-8")
    manifest_path = out / "recovery_manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _render_report(report: dict[str, Any]) -> str:
    b, m = report["baseline"], report["frozen_model_metrics"]
    c = report["model_complexity"]
    sec = report["cpu_training_seconds"]
    inf = report["inference_latency"]
    p = report["data_provenance"]
    return f"""# India ETA GBDT recovery report

Status: **{report['status']}**. This is a bounded reconstruction from the committed feature view, not a recovered historical checkpoint.

## Data and leakage controls

- Canonical Git source CSV SHA-256: `{p['source_csv']['canonical_git_sha256_lf']}` (verified after CRLF-to-LF normalization; worktree bytes SHA-256: `{p['source_csv']['worktree_sha256_crlf']}`).
- GBDT Parquet SHA-256: `{p['files_sha256']['gbdt_parquet']}`; VQC Parquet SHA-256: `{p['files_sha256']['vqc_parquet']}`.
    - Input rows: {report['training_rows']:,} train / {report['validation_rows']:,} validation; target `delivery_duration_min` in minutes; {report['data_provenance']['gbdt_feature_count']} numeric features.
- Model inputs exclude stable ID, courier ID, split partition, and target. VQC features were keyed by ID, courier, and split and are retained only in aligned prediction files.
- Training OOF uses five `GroupKFold` folds by courier. Validation courier groups are deterministically separated into model-selection and fusion-calibration roles; calibration rows do not choose the GBDT configuration.
- Sealed test partition was not loaded into the DataBundle, scored, or serialized.

## Estimator and bounded selection

Estimator: `{report['estimator_identity']}`, objective `squared_error`, configuration `{json.dumps(report['model_selection']['chosen_config'], sort_keys=True)}`. Selected by minimum MAE on the courier-isolated `model_selection` subset among three bounded configurations. CPU training time: {sum(sec['selection_full_train_runs']) + sum(sec['oof_fold_runs']) + sec['final_refit_full_train']:.2f} s total across all fits; final full-training refit {sec['final_refit_full_train']:.2f} s.

## Validation results (minutes)

| Partition | Model | MAE | RMSE | Median absolute error |
|---|---|---:|---:|---:|
| All validation (descriptive) | Training-median baseline | {b['metrics_all_validation']['mae_min']:.4f} | {b['metrics_all_validation']['rmse_min']:.4f} | {b['metrics_all_validation']['median_absolute_error_min']:.4f} |
| All validation (descriptive) | Frozen HGB regressor | {m['all_validation_descriptive']['mae_min']:.4f} | {m['all_validation_descriptive']['rmse_min']:.4f} | {m['all_validation_descriptive']['median_absolute_error_min']:.4f} |
| Model-selection groups | Training-median baseline | {b['metrics_model_selection']['mae_min']:.4f} | {b['metrics_model_selection']['rmse_min']:.4f} | {b['metrics_model_selection']['median_absolute_error_min']:.4f} |
| Model-selection groups | Frozen HGB regressor | {m['model_selection']['mae_min']:.4f} | {m['model_selection']['rmse_min']:.4f} | {m['model_selection']['median_absolute_error_min']:.4f} |
| Fusion-calibration groups | Training-median baseline | {b['metrics_fusion_calibration']['mae_min']:.4f} | {b['metrics_fusion_calibration']['rmse_min']:.4f} | {b['metrics_fusion_calibration']['median_absolute_error_min']:.4f} |
| Fusion-calibration groups | Frozen HGB regressor | {m['fusion_calibration']['mae_min']:.4f} | {m['fusion_calibration']['rmse_min']:.4f} | {m['fusion_calibration']['median_absolute_error_min']:.4f} |

## Complexity and inference

- Fitted boosting iterations: {c['iterations_fitted']}; total trees: {c['total_trees']}; total leaf nodes: {c['total_leaf_nodes']}; input features: {c['feature_count']}.
- Median single-row inference latency over 31 routine prediction calls: {inf['median_microseconds_per_row']:.2f} microseconds per row on `{inf['measured_on']}`.

## Artifacts

- Model: `{report['model_artifacts']['model']['path']}` (SHA-256 `{report['model_artifacts']['model']['sha256']}`).
- Feature schema: `{report['model_artifacts']['feature_schema']['path']}`.
- Train OOF: `{report['prediction_artifacts']['oof']['path']}` ({report['prediction_artifacts']['oof']['rows']:,} rows).
- Validation predictions: `{report['prediction_artifacts']['validation']['path']}` ({report['prediction_artifacts']['validation']['rows']:,} rows).
- Fold provenance: `{report['prediction_artifacts']['oof_fold_manifest']['path']}`; per-fold courier lists and hashes are recorded.
- Reproducibility manifest: `artifacts/india_eta_recovery/recovery_manifest.json`.

The model remains a separate continuous ETA regressor and does not replace the Solomon/Dabia route-risk classifier.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    result = run_recovery(args.root, args.output_dir, seed=args.seed)
    print(json.dumps({"status": result["status"], "model": result["model_artifacts"]["model"], "validation": result["frozen_model_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
