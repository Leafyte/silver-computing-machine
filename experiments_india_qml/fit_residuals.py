"""Fit and select RUN2 India residual models from pre-freeze development rows.

This module intentionally opens only the train-OOF/validation development parquet.
The sealed test artifact is never accepted as input to this workflow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments_india_qml.quantum import (
    CircuitConfig,
    QuantumResidualRegressor,
    hybrid_predict_minutes,
    profile_cpu,
    profile_gradient_cpu,
    qpiai_expectations,
    validate_batched_forward,
)


FEATURE_NAMES = ("vqc_x0_distance", "vqc_x1_congestion")
RUN_ID = "luna_hgbr_recovery_20261029T131809Z_v5"
DEFAULT_DEVELOPMENT_TABLE = Path(".runtime/run2") / RUN_ID / "classical_development_predictions.parquet"
DEFAULT_MANIFEST = Path("artifacts/run2_india") / RUN_ID / "recovered_gbdt_manifest.json"
DEFAULT_ARTIFACTS_DIR = Path("artifacts/run2_india") / RUN_ID
REQUIRED_COLUMNS = {
    "ID", "Delivery_person_ID", "split_partition", "fold",
    "y_true_minutes", "gbdt_pred_minutes", "residual_minutes", *FEATURE_NAMES,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_parquet(path: Path) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas and a Parquet engine are required in the task-local runtime") from exc
    try:
        # First read only the partition column. If a sealed test partition is
        # present, abort before loading any target, prediction, or VQC columns.
        partitions = pd.read_parquet(path, columns=["split_partition"])
        splits = set(partitions["split_partition"].dropna().astype(str).unique())
        if splits != {"train", "val"}:
            raise ValueError(f"development input must contain train and val only; found {sorted(splits)}")
        return pd.read_parquet(path, columns=sorted(REQUIRED_COLUMNS))
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"cannot read the pre-freeze development Parquet with the installed engine ({type(exc).__name__})"
        ) from exc


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    absolute = np.abs(error)
    return {
        "mae_minutes": float(np.mean(absolute)),
        "rmse_minutes": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error_minutes": float(np.median(absolute)),
    }


def _alpha_on_validation(
    y_true: np.ndarray,
    classical_eta: np.ndarray,
    residual_prediction: np.ndarray,
) -> float:
    """Minimize clipped hybrid RMSE on validation, with alpha restricted to [0, 1]."""
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    eta = np.asarray(classical_eta, dtype=np.float64).reshape(-1)
    residual = np.asarray(residual_prediction, dtype=np.float64).reshape(-1)
    if not (y.shape == eta.shape == residual.shape) or not np.isfinite(y).all() \
            or not np.isfinite(eta).all() or not np.isfinite(residual).all():
        raise ValueError("validation targets, baseline ETA, and residual predictions must align and be finite")
    boundaries = [0.0, 1.0]
    nonzero = np.abs(residual) > 1e-15
    if np.any(nonzero):
        crossings = -eta[nonzero] / residual[nonzero]
        boundaries.extend(float(value) for value in crossings if 0.0 < value < 1.0)
    boundaries = sorted(set(boundaries))
    candidates = set(boundaries)
    # On each interval the rows clipped to zero are fixed, so the RMSE is a
    # quadratic in alpha. Add its exact stationary point if it lies in-interval.
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        midpoint = (left + right) / 2.0
        active = eta + midpoint * residual > 0.0
        denom = float(residual[active] @ residual[active])
        if denom > 1e-18:
            optimum = float(((y[active] - eta[active]) @ residual[active]) / denom)
            candidates.add(min(right, max(left, optimum)))
    return min(
        candidates,
        key=lambda alpha: (float(np.mean((np.maximum(0.0, eta + alpha * residual) - y) ** 2)), alpha),
    )


def _select_gate_on_validation(
    y_true: np.ndarray,
    classical_eta: np.ndarray,
    residual_prediction: np.ndarray,
    calibration_features: np.ndarray,
) -> dict[str, Any]:
    """Select an input-only congestion gate and alpha on calibration rows."""
    x = np.asarray(calibration_features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or len(x) != len(residual_prediction):
        raise ValueError("calibration features must align with residual predictions")
    congestion = x[:, FEATURE_NAMES.index("vqc_x1_congestion")]
    thresholds = np.quantile(congestion, [0.25, 0.5, 0.75]).tolist()
    configs: list[tuple[str, str | None, float | None]] = [("all", None, None)]
    for threshold in sorted(set(float(value) for value in thresholds)):
        configs.extend((
            (">=", ">=", threshold),
            ("<=", "<=", threshold),
        ))
    candidates: list[dict[str, Any]] = []
    for strategy, operator, threshold in configs:
        if strategy == "all":
            mask = np.ones(len(x), dtype=bool)
        elif operator == ">=":
            mask = congestion >= float(threshold)
        else:
            mask = congestion <= float(threshold)
        gated = np.where(mask, residual_prediction, 0.0)
        alpha = _alpha_on_validation(y_true, classical_eta, gated)
        hybrid = hybrid_predict_minutes(classical_eta, gated, alpha)
        candidates.append({
            "gate_strategy": "all rows" if strategy == "all" else f"vqc_x1_congestion {operator} {threshold:.17g}",
            "gate_feature": None if strategy == "all" else "vqc_x1_congestion",
            "gate_operator": strategy,
            "gate_threshold": threshold,
            "alpha_validation_only": float(alpha),
            "quantum_invocation_rate_calibration": float(np.mean(mask)),
            "quantum_invocation_rows_calibration": int(np.sum(mask)),
            "hybrid_calibration_metrics": _metrics(y_true, hybrid),
            "gate_mask": mask,
            "gated_residual_minutes": gated,
            "hybrid_predictions_minutes": hybrid,
        })
    return min(
        candidates,
        key=lambda item: (
            item["hybrid_calibration_metrics"]["rmse_minutes"],
            item["quantum_invocation_rate_calibration"],
            item["gate_threshold"] if item["gate_threshold"] is not None else 0.0,
        ),
    )


def _ridge_predict(
    x_train: np.ndarray,
    residual_train: np.ndarray,
    x_eval: np.ndarray,
    penalty: float,
) -> np.ndarray:
    design_eval = np.column_stack((np.ones(len(x_eval)), x_eval))
    coefficients = _ridge_coefficients(x_train, residual_train, penalty)
    return design_eval @ coefficients


def _ridge_coefficients(x_train: np.ndarray, residual_train: np.ndarray, penalty: float) -> np.ndarray:
    design_train = np.column_stack((np.ones(len(x_train)), x_train))
    regularizer = np.diag([0.0, penalty, penalty])  # unpenalized intercept
    coefficients = np.linalg.solve(design_train.T @ design_train + regularizer,
                                   design_train.T @ residual_train)
    return coefficients


def _validate_development_rows(frame: Any) -> tuple[Any, Any, dict[str, Any]]:
    columns = set(frame.columns)
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        raise ValueError(f"development parquet is missing required columns: {missing}")
    splits = set(frame["split_partition"].dropna().astype(str).unique())
    if splits != {"train", "val"}:
        raise ValueError(f"development input must contain train and val only; found {sorted(splits)}")
    if frame["ID"].isna().any() or frame["ID"].duplicated().any():
        raise ValueError("development row IDs must be non-null and unique")
    train = frame.loc[frame["split_partition"].astype(str) == "train"].copy()
    val = frame.loc[frame["split_partition"].astype(str) == "val"].copy()
    if train.empty or val.empty:
        raise ValueError("development parquet needs non-empty train and validation partitions")
    if train["Delivery_person_ID"].isna().any() or val["Delivery_person_ID"].isna().any():
        raise ValueError("courier IDs must be present to verify group-safe train/validation separation")
    fold_values = train["fold"].dropna().to_numpy(dtype=np.float64)
    if train["fold"].isna().any() or not np.isfinite(fold_values).all() \
            or not np.equal(fold_values, np.floor(fold_values)).all():
        raise ValueError("train rows must have integer OOF fold IDs")
    train_folds = set(fold_values.astype(int).tolist())
    if train_folds != {0, 1, 2, 3, 4}:
        raise ValueError(f"train rows must have OOF fold IDs 0..4; found {sorted(train_folds)}")
    if val["fold"].notna().any():
        raise ValueError("validation rows must use the single frozen estimator (fold must be null)")
    train_groups = set(train["Delivery_person_ID"].dropna().astype(str))
    val_groups = set(val["Delivery_person_ID"].dropna().astype(str))
    overlap = train_groups & val_groups
    if overlap:
        raise ValueError(f"courier groups cross train/validation boundary ({len(overlap)} overlaps)")
    groups_by_fold = {
        int(fold): set(rows["Delivery_person_ID"].astype(str))
        for fold, rows in train.groupby("fold", sort=True)
    }
    fold_overlaps = [
        (left, right)
        for left in sorted(groups_by_fold)
        for right in sorted(groups_by_fold)
        if left < right and groups_by_fold[left] & groups_by_fold[right]
    ]
    if fold_overlaps:
        raise ValueError(f"courier groups cross OOF folds: {fold_overlaps}")

    numeric_cols = [*FEATURE_NAMES, "y_true_minutes", "gbdt_pred_minutes", "residual_minutes"]
    for col in numeric_cols:
        values = train[col].to_numpy(dtype=np.float64)
        val_values = val[col].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.isfinite(val_values).all():
            raise ValueError(f"column {col!r} contains non-finite values")
    for name, rows in (("train", train), ("val", val)):
        features = rows.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
        if np.any(features < -1e-12) or np.any(features > 1.0 + 1e-12):
            raise ValueError(f"{name} VQC features must be train-bound scaled to [0,1]")
        residual = rows["residual_minutes"].to_numpy(dtype=np.float64)
        computed = rows["y_true_minutes"].to_numpy(dtype=np.float64) - rows["gbdt_pred_minutes"].to_numpy(dtype=np.float64)
        max_error = float(np.max(np.abs(residual - computed)))
        if max_error > 1e-6:
            raise ValueError(f"{name} residual_minutes != y_true_minutes - gbdt_pred_minutes (max error {max_error})")
    summary = {
        "train_oof_rows": len(train),
        "validation_rows": len(val),
        "train_fold_counts": {str(int(k)): int(v) for k, v in train["fold"].value_counts().sort_index().items()},
        "train_unique_couriers": len(train_groups),
        "validation_unique_couriers": len(val_groups),
        "group_overlap_count": 0,
        "oof_fold_group_overlap_pairs": len(fold_overlaps),
        "test_rows_present": False,
        "input_partitions": sorted(splits),
    }
    return train, val, summary


def _validate_recovery_manifest(
    manifest_path: Path | None,
    development_path: Path,
    summary: dict[str, Any],
) -> dict[str, Any] | None:
    if manifest_path is None:
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("recovery_fit_authorized_missing_frozen_checkpoint") is not True:
        raise ValueError("recovery manifest does not record the authorized missing-checkpoint rebuild")
    if manifest.get("target_name") != "delivery_duration_min" or str(manifest.get("target_unit", "")).lower() != "minutes":
        raise ValueError("recovery manifest target must be delivery_duration_min in minutes")
    parents = {
        key: manifest.get(key)
        for key in (
            "parent_split_manifest_sha256",
            "parent_feature_schema_sha256",
            "parent_preprocessing_sha256",
        )
    }


    transformed_hashes = manifest.get("transformed_data_sha256")
    if any(not isinstance(value, str) or len(value) != 64 for value in parents.values()):
        raise ValueError("recovery manifest must include SHA-256 hashes for split, schema, and preprocessing inputs")
    if not isinstance(transformed_hashes, dict) or not transformed_hashes \
            or any(not isinstance(value, str) or len(value) != 64 for value in transformed_hashes.values()):
        raise ValueError("recovery manifest must include SHA-256 hashes for transformed source views")

    project_root = Path(__file__).resolve().parents[1]

    def verify_artifact(relative: Any, expected_hash: Any, label: str) -> Path:
        if not isinstance(relative, (str, Path)) or not isinstance(expected_hash, str):
            raise ValueError(f"recovery manifest is missing path/hash for {label}")
        artifact_path = Path(relative)
        if not artifact_path.is_absolute():
            artifact_path = project_root / artifact_path
        if not artifact_path.is_file() or _sha256(artifact_path) != expected_hash.lower():
            raise ValueError(f"recovery manifest {label} is missing or has a SHA-256 mismatch")
        return artifact_path

    for path_key, hash_key in (
        ("protocol_path", "protocol_sha256"),
        ("parent_split_manifest_path", "parent_split_manifest_sha256"),
        ("parent_feature_schema_path", "parent_feature_schema_sha256"),
        ("parent_preprocessing_path", "parent_preprocessing_sha256"),
    ):
        verify_artifact(manifest.get(path_key), manifest.get(hash_key), path_key)
    for path_key, hash_key in (
        ("v4_protocol_path", "v4_protocol_sha256"),
        ("v4_fold_manifest_path", "v4_fold_manifest_sha256"),
        ("quantum_feature_scaler_path", "quantum_feature_scaler_sha256"),
    ):
        verify_artifact(manifest.get(path_key), manifest.get(hash_key), path_key)
    for file_name, digest in transformed_hashes.items():
        verify_artifact(Path("data") / file_name, digest, f"transformed source {file_name}")
    raw_provenance = manifest.get("raw_csv_provenance", {})
    verify_artifact(raw_provenance.get("working_path"), raw_provenance.get("working_file_sha256"), "raw CSV provenance")
    model = manifest.get("model", {})
    if model.get("fit_partition") != "original train only" or model.get("test_partition_read_or_used") is not False:
        raise ValueError("recovered GBDT manifest must be fitted only on original train and exclude test")
    checkpoint_path = verify_artifact(model.get("checkpoint_path"), model.get("checkpoint_sha256"), "recovered GBDT checkpoint")
    crossfit = manifest.get("crossfit")
    if not isinstance(crossfit, dict):
        raise ValueError("recovery manifest is missing crossfit metadata")
    if crossfit.get("contains_test_rows") is not False:
        raise ValueError("crossfit manifest must explicitly state contains_test_rows=false")
    if crossfit.get("prediction_sha256", "").lower() != _sha256(development_path):
        raise ValueError("development table SHA-256 does not match manifest crossfit prediction hash")
    prediction_path = crossfit.get("prediction_path")
    resolved_prediction_path = Path(prediction_path)
    if not resolved_prediction_path.is_absolute():
        resolved_prediction_path = project_root / resolved_prediction_path
    if resolved_prediction_path.resolve() != development_path.resolve():
        raise ValueError("manifest crossfit prediction path does not match the selected development table")
    counts = {
        "rows": summary["train_oof_rows"] + summary["validation_rows"],
        "train_oof_rows": summary["train_oof_rows"],
        "validation_rows": summary["validation_rows"],
    }
    for key, observed in counts.items():
        if crossfit.get(key) != observed:
            raise ValueError(f"manifest crossfit {key}={crossfit.get(key)!r} does not match table count {observed}")
    if crossfit.get("method", "").find("GroupKFold") < 0:
        raise ValueError("crossfit manifest must use group-preserving folds")
    fold_rows = crossfit.get("folds")
    if not isinstance(fold_rows, list):
        raise ValueError("crossfit manifest must list fold-level provenance")
    fold_ids = set()
    for fold_row in fold_rows:
        if not isinstance(fold_row, dict) or "fold" not in fold_row:
            raise ValueError("invalid fold provenance entry in recovery manifest")
        fold_ids.add(int(fold_row["fold"]))
    if fold_ids != {0, 1, 2, 3, 4}:
        raise ValueError(f"manifest must describe OOF folds 0..4; found {sorted(fold_ids)}")
    fold_manifest_relative = crossfit.get("fold_manifest_path")
    fold_manifest_hash = crossfit.get("fold_manifest_sha256")
    if not isinstance(fold_manifest_relative, str) or not isinstance(fold_manifest_hash, str):
        raise ValueError("crossfit manifest must point to its fold audit file and SHA-256")
    fold_manifest_path = Path(fold_manifest_relative)
    if not fold_manifest_path.is_absolute():
        fold_manifest_path = project_root / fold_manifest_path
    if not fold_manifest_path.is_file() or _sha256(fold_manifest_path) != fold_manifest_hash.lower():
        raise ValueError("crossfit fold-audit file is missing or its SHA-256 does not match the recovery manifest")
    return {
        "schema_version": manifest.get("schema_version"),
        "recovery_fit_authorized_missing_frozen_checkpoint": True,
        "target_name": manifest["target_name"],
        "target_unit": manifest["target_unit"],
        "parent_hash_names": sorted([*parents, *transformed_hashes]),
        "quantum_feature_scaler_path": manifest.get("quantum_feature_scaler_path"),
        "quantum_feature_scaler_sha256": manifest.get("quantum_feature_scaler_sha256"),
        "crossfit": {
            **counts,
            "contains_test_rows": False,
            "method": crossfit["method"],
            "fold_ids": sorted(fold_ids),
            "fold_manifest_path": str(fold_manifest_path.resolve()),
            "fold_manifest_sha256": fold_manifest_hash,
            "prediction_path": str(resolved_prediction_path.resolve()),
            "prediction_sha256": crossfit["prediction_sha256"],
            "recovered_checkpoint_path": str(checkpoint_path.resolve()),
            "recovered_checkpoint_sha256": model["checkpoint_sha256"],
        },
    }


def _split_validation_roles(
    validation: Any,
    manifest_path: Path,
    protocol_path: Path,
    *,
    development_sha256: str,
    manifest_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Use the immutable courier-disjoint development roles in the recovery manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    roles = manifest.get("crossfit", {}).get("development_roles")
    if not isinstance(roles, dict) or roles.get("method") != "GroupShuffleSplit within original validation courier groups":
        raise ValueError("recovery manifest lacks the frozen GroupShuffleSplit validation roles")
    stop_groups = set(map(str, roles.get("vqc_early_stop_group_ids", [])))
    calibration_groups = set(map(str, roles.get("fusion_selection_group_ids", [])))
    if not stop_groups or not calibration_groups or stop_groups & calibration_groups:
        raise ValueError("manifest validation role groups are empty or overlapping")
    observed_groups = set(validation["Delivery_person_ID"].astype(str))
    if stop_groups | calibration_groups != observed_groups:
        raise ValueError("manifest validation role groups do not exactly cover the development validation groups")
    group_values = validation["Delivery_person_ID"].astype(str).to_numpy()
    stop_indices = np.flatnonzero(np.isin(group_values, list(stop_groups)))
    calibration_indices = np.flatnonzero(np.isin(group_values, list(calibration_groups)))
    if not len(stop_indices) or not len(calibration_indices):
        raise ValueError("manifest validation roles must contain rows in both groups")
    role_labels = np.full(len(validation), "fusion_alpha_gate_calibration", dtype="U32")
    role_labels[stop_indices] = "qml_earlystop_architecture"
    protocol = {
        "schema_version": "run2-validation-role-split-v1",
        "assignment_method": roles["method"],
        "seed": int(roles.get("seed", -1)),
        "train_size": float(roles.get("train_size", 0.5)),
        "source": "recovered_gbdt_manifest.crossfit.development_roles; assignments not recomputed",
        "scope": "original validation partition only",
        "selection_roles": {
            "qml_earlystop_architecture": {
                "rows": int(len(stop_indices)),
                "courier_groups": len(stop_groups),
                "group_sha256": sorted(hashlib.sha256(g.encode("utf-8")).hexdigest() for g in stop_groups),
            },
            "fusion_alpha_gate_calibration": {
                "rows": int(len(calibration_indices)),
                "courier_groups": len(calibration_groups),
                "group_sha256": sorted(hashlib.sha256(g.encode("utf-8")).hexdigest() for g in calibration_groups),
            },
        },
        "courier_group_overlap_count": 0,
        "row_assignments": [
            {
                "row_id_sha256": hashlib.sha256(str(row_id).encode("utf-8")).hexdigest(),
                "courier_id_sha256": hashlib.sha256(str(group).encode("utf-8")).hexdigest(),
                "role": str(role),
            }
            for row_id, group, role in zip(validation["ID"].tolist(), group_values, role_labels.tolist())
        ],
        "development_data_sha256": development_sha256,
        "recovery_manifest_sha256": manifest_sha256,
        "written_before_qml_fitting": True,
    }
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return stop_indices, calibration_indices, protocol

def run_experiment(
    development_path: Path,
    artifacts_dir: Path,
    *,
    manifest_path: Path | None = None,
    train_rows: int = 4000,
    seed: int = 20261029,
    max_updates: int = 120,
    batch_size: int = 128,
    learning_rate: float = 0.01,
    patience: int = 15,
    profile_repeats: int = 32,
) -> dict[str, Any]:
    if not development_path.is_file():
        raise FileNotFoundError(f"pre-freeze development table not found: {development_path}")
    if manifest_path is None:
        raise ValueError("the recovered GBDT/crossfit manifest is required for QML fitting")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"development manifest not found: {manifest_path}")
    frame = _read_parquet(development_path)
    train, val, partition_summary = _validate_development_rows(frame)
    manifest_summary = _validate_recovery_manifest(manifest_path, development_path, partition_summary)
    development_hash = _sha256(development_path)
    manifest_hash = _sha256(manifest_path) if manifest_path else ""

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    role_path = artifacts_dir / "validation_role_protocol.json"
    stop_idx, calibration_idx, role_protocol = _split_validation_roles(
        val, manifest_path, role_path, development_sha256=development_hash,
        manifest_sha256=manifest_hash,
    )
    stop = val.iloc[stop_idx]
    calibration = val.iloc[calibration_idx]
    x_stop = stop.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    residual_stop = stop["residual_minutes"].to_numpy(dtype=np.float64)
    x_cal = calibration.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    residual_cal = calibration["residual_minutes"].to_numpy(dtype=np.float64)
    y_cal = calibration["y_true_minutes"].to_numpy(dtype=np.float64)
    eta_cal = calibration["gbdt_pred_minutes"].to_numpy(dtype=np.float64)
    x_val = val.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_val = val["y_true_minutes"].to_numpy(dtype=np.float64)
    eta_val = val["gbdt_pred_minutes"].to_numpy(dtype=np.float64)

    x_train_full = train.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    residual_train_full = train["residual_minutes"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    chosen_count = min(int(train_rows), len(train))
    if chosen_count < 32:
        raise ValueError("train_rows must select at least 32 OOF rows")
    chosen = np.sort(rng.choice(len(train), size=chosen_count, replace=False))
    x_train, residual_train = x_train_full[chosen], residual_train_full[chosen]

    # The role protocol above is durable before either circuit is fitted.
    base_cal_metrics = _metrics(y_cal, np.maximum(0.0, eta_cal))
    quantum: dict[str, Any] = {}
    models: dict[str, QuantumResidualRegressor] = {}
    for architecture in ("one_block", "reupload"):
        model = QuantumResidualRegressor(
            CircuitConfig(architecture, FEATURE_NAMES, residual_scale_minutes=None, seed=seed),
            seed=seed,
        )
        checkpoint = artifacts_dir / f"quantum_{architecture}.npz"
        fit_info = model.fit(
            x_train, residual_train, x_stop, residual_stop,
            max_updates=max_updates, batch_size=batch_size,
            learning_rate=learning_rate, patience=patience,
            checkpoint_path=checkpoint,
        )
        restored = QuantumResidualRegressor.load(checkpoint)
        stop_residual = restored.predict_minutes(x_stop)
        stop_metrics = _metrics(residual_stop, stop_residual)
        if abs(stop_metrics["rmse_minutes"] - fit_info["best_validation_rmse_minutes"]) > 1e-9:
            raise RuntimeError("early-stop RMSE does not match selected checkpoint on its frozen role")

        # This offline calibration prediction is used to tune a gate; deployment
        # inference below executes the circuit only on the feature-gated subset.
        q_residual_cal = restored.predict_minutes(x_cal)
        gate = _select_gate_on_validation(y_cal, eta_cal, q_residual_cal, x_cal)
        if gate["hybrid_calibration_metrics"]["rmse_minutes"] >= base_cal_metrics["rmse_minutes"]:
            gate = {
                **gate,
                "gate_strategy": "disabled; alpha=0 classical fallback",
                "gate_feature": None,
                "gate_operator": "disabled",
                "gate_threshold": None,
                "alpha_validation_only": 0.0,
                "quantum_invocation_rate_calibration": 0.0,
                "quantum_invocation_rows_calibration": 0,
                "hybrid_calibration_metrics": base_cal_metrics,
                "gate_mask": np.zeros(len(calibration), dtype=bool),
                "gated_residual_minutes": np.zeros(len(calibration), dtype=np.float64),
                "hybrid_predictions_minutes": np.maximum(0.0, eta_cal),
            }

        threshold = 0.0 if gate["gate_threshold"] is None else float(gate["gate_threshold"])
        operator = gate["gate_operator"] if gate["gate_operator"] in (">=", "<=") else ">="
        full_inference = restored.predict_hybrid_minutes(
            eta_val, x_val, float(gate["alpha_validation_only"]),
            gate_feature="vqc_x1_congestion", gate_operator=operator,
            gate_threshold=threshold,
        )
        full_metrics = _metrics(y_val, full_inference["eta_minutes"])
        full_residual = full_inference["quantum_residual_minutes"]
        full_mask = full_inference["quantum_invoked"]

        # Verify exact checkpoint reload parity on the actual gated inference path.
        original_inference = model.predict_hybrid_minutes(
            eta_val, x_val, float(gate["alpha_validation_only"]),
            gate_feature="vqc_x1_congestion", gate_operator=operator,
            gate_threshold=threshold,
        )
        reload_delta = float(np.max(np.abs(original_inference["eta_minutes"] - full_inference["eta_minutes"])))
        if reload_delta > 1e-12:
            raise RuntimeError(f"checkpoint reload inference drift {reload_delta} exceeds 1e-12 minutes")
        quantum[architecture] = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "fit_info": fit_info,
            "architecture_selection_role": "qml_earlystop_architecture",
            "earlystop_residual_validation_metrics": stop_metrics,
            "gate_strategy": gate["gate_strategy"],
            "gate_feature": gate["gate_feature"],
            "gate_operator": gate["gate_operator"],
            "gate_threshold": gate["gate_threshold"],
            "alpha_calibration_role": "fusion_alpha_gate_calibration",
            "alpha_validation_only": float(gate["alpha_validation_only"]),
            "quantum_invocation_rate_calibration": float(gate["quantum_invocation_rate_calibration"]),
            "quantum_invocation_rows_calibration": int(gate["quantum_invocation_rows_calibration"]),
            "calibration_residual_metrics_when_gated": _metrics(residual_cal, gate["gated_residual_minutes"]),
            "hybrid_calibration_metrics": gate["hybrid_calibration_metrics"],
            "validation_quantum_invocation_rate": float(np.mean(full_mask)),
            "validation_quantum_invocation_rows": int(np.sum(full_mask)),
            "validation_residual_predictions_minutes": full_residual.tolist(),
            "validation_gate_mask": full_mask.tolist(),
            "validation_hybrid_metrics": full_metrics,
            "checkpoint_reload_max_abs_eta_delta_minutes": reload_delta,
        }
        models[architecture] = restored

    # Architecture is selected only on the courier-disjoint early-stop role.
    selected_arch = min(
        quantum,
        key=lambda name: quantum[name]["earlystop_residual_validation_metrics"]["rmse_minutes"],
    )

    # Fit the classical residual control on all leakage-safe train-OOF residuals;
    # select ridge penalty and alpha on fusion-calibration couriers only.
    ridge: dict[str, Any] = {}
    for penalty in (0.01, 0.1, 1.0, 10.0):
        r_cal = _ridge_predict(x_train_full, residual_train_full, x_cal, penalty)
        alpha = _alpha_on_validation(y_cal, eta_cal, r_cal)
        calibration_hybrid = hybrid_predict_minutes(eta_cal, r_cal, alpha)
        ridge[str(penalty)] = {
            "ridge_penalty": penalty,
            "alpha_calibration_only": alpha,
            "hybrid_calibration_metrics": _metrics(y_cal, calibration_hybrid),
        }
    selected_ridge_key = min(ridge, key=lambda k: ridge[k]["hybrid_calibration_metrics"]["rmse_minutes"])
    selected_ridge = ridge[selected_ridge_key]
    if selected_ridge["hybrid_calibration_metrics"]["rmse_minutes"] >= base_cal_metrics["rmse_minutes"]:
        selected_ridge = {**selected_ridge, "alpha_calibration_only": 0.0}
    ridge_penalty = float(selected_ridge_key)
    selected_ridge_residual_val = _ridge_predict(x_train_full, residual_train_full, x_val, ridge_penalty)
    selected_ridge_coefficients = _ridge_coefficients(x_train_full, residual_train_full, ridge_penalty)
    selected_ridge_hybrid_val = hybrid_predict_minutes(
        eta_val, selected_ridge_residual_val, float(selected_ridge["alpha_calibration_only"]),
    )
    ridge["selected_penalty_key"] = selected_ridge_key
    ridge["selected"] = {
        **selected_ridge,
        "coefficients_by_order_intercept_then_feature_order": selected_ridge_coefficients.tolist(),
        "coefficient_feature_order": ["intercept", *FEATURE_NAMES],
        "validation_residual_metrics": _metrics(val["residual_minutes"].to_numpy(dtype=np.float64), selected_ridge_residual_val),
        "validation_hybrid_metrics": _metrics(y_val, selected_ridge_hybrid_val),
    }

    selected_candidate = quantum[selected_arch]
    selected_alpha = float(selected_candidate["alpha_validation_only"])
    selected_threshold = selected_candidate["gate_threshold"]
    selected_operator = selected_candidate["gate_operator"]
    selected_gate_feature = selected_candidate["gate_feature"]
    selected_model = models[selected_arch]
    final_val = selected_model.predict_hybrid_minutes(
        eta_val, x_val, selected_alpha,
        gate_feature="vqc_x1_congestion",
        gate_operator=selected_operator if selected_operator in (">=", "<=") else ">=",
        gate_threshold=0.0 if selected_threshold is None else float(selected_threshold),
    )
    selected_gate = {
        "architecture": selected_arch,
        "checkpoint_path": selected_candidate["checkpoint_path"],
        "checkpoint_sha256": selected_candidate["checkpoint_sha256"],
        "alpha": selected_alpha,
        "gate_strategy": selected_candidate["gate_strategy"],
        "gate_feature": selected_gate_feature,
        "gate_operator": selected_operator,
        "gate_threshold": selected_threshold,
        "gate_selection_role": "fusion_alpha_gate_calibration",
        "architecture_selection_role": "qml_earlystop_architecture",
        "quantum_invocation_rate_calibration": selected_candidate["quantum_invocation_rate_calibration"],
        "quantum_invocation_rate_validation": final_val["invocation_count"] / len(val),
        "quantum_invocation_rows_validation": final_val["invocation_count"],
        "fusion_gate_enabled": bool(selected_alpha > 0.0),
        "fallback_reason": (
            "alpha=0 classical fallback selected because hybrid did not beat baseline on the fusion-calibration role"
            if selected_alpha == 0.0 else None
        ),
    }

    # Compare identical full original-validation rows after selection; this is
    # reporting only and does not alter architecture, alpha, or gate.
    base_metrics_full = _metrics(y_val, np.maximum(0.0, eta_val))
    selection = {
        "evidence_status": "development-only selection with group-disjoint validation roles; sealed test not read",
        "development_table_path": str(development_path.resolve()),
        "development_table_sha256": development_hash,
        "manifest_path": str(manifest_path.resolve()) if manifest_path else None,
        "manifest_sha256": manifest_hash,
        "recovery_manifest_summary": manifest_summary,
        "feature_scaler_path": manifest_summary["quantum_feature_scaler_path"],
        "feature_scaler_sha256": manifest_summary["quantum_feature_scaler_sha256"],
        "classical_gbdt_checkpoint_path": manifest_summary["crossfit"]["recovered_checkpoint_path"],
        "classical_gbdt_checkpoint_sha256": manifest_summary["crossfit"]["recovered_checkpoint_sha256"],
        "split_and_group_checks": partition_summary,
        "validation_role_protocol_path": str(role_path),
        "validation_role_protocol_sha256": _sha256(role_path),
        "validation_role_protocol": role_protocol,
        "feature_order": list(FEATURE_NAMES),
        "target_unit": "minutes",
        "residual_definition": "y_true_minutes - gbdt_pred_minutes",
        "quantum_fit": {
            "train_oof_available_rows": len(train),
            "train_rows_used": len(x_train),
            "subset_seed": seed,
            "train_subset_fold_counts": {
                str(int(k)): int(v) for k, v in train.iloc[chosen]["fold"].value_counts().sort_index().items()
            },
            "max_updates": max_updates,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "patience": patience,
            "checkpointing": "best early-stop-role checkpoint saved on each improvement and at fit completion",
        },
        "classical_baseline_validation_metrics_reporting_only": base_metrics_full,
        "ridge_residual_control": ridge,
        "quantum_candidates": quantum,
        "selected_release_gate": selected_gate,
        "architecture_selection": "lowest residual RMSE on qml_earlystop_architecture role; disjoint from fusion calibration",
        "fusion_selection": "alpha and pre-QML congestion input gate chosen on fusion_alpha_gate_calibration role only",
        "test_evaluation": "not run; only pre-freeze train-OOF and original validation rows were read",
    }

    selected_residual = final_val["quantum_residual_minutes"]
    selected_mask = final_val["quantum_invoked"]
    validation_prediction_path = artifacts_dir / "validation_predictions.npz"
    validation_ids_hashed = np.asarray([
        hashlib.sha256(str(value).encode("utf-8")).hexdigest() for value in val["ID"].tolist()
    ])
    np.savez_compressed(
        validation_prediction_path,
        row_id_sha256=validation_ids_hashed,
        vqc_features=x_val,
        y_true_minutes=y_val,
        gbdt_eta_minutes=eta_val,
        quantum_residual_minutes=selected_residual,
        quantum_gate_mask=selected_mask,
        quantum_gate_threshold=np.asarray(np.nan if selected_threshold is None else selected_threshold),
        quantum_gate_feature=np.asarray("" if selected_gate_feature is None else selected_gate_feature),
        quantum_gate_operator=np.asarray(selected_operator),
        quantum_hybrid_eta_minutes=final_val["eta_minutes"],
        ridge_residual_minutes=selected_ridge_residual_val,
        ridge_alpha=np.asarray(selected_ridge["alpha_calibration_only"]),
        ridge_hybrid_eta_minutes=selected_ridge_hybrid_val,
        selected_architecture=np.asarray(selected_arch),
        selected_alpha=np.asarray(selected_alpha),
    )
    selection["validation_predictions_artifact"] = {
        "path": str(validation_prediction_path),
        "sha256": _sha256(validation_prediction_path),
        "row_ids": "SHA-256 hashed; no raw IDs stored",
        "rows": len(val),
    }

    profile_rows = np.sort(rng.choice(len(x_train_full), size=min(512, len(x_train_full)), replace=False))
    profile_x, profile_y = x_train_full[profile_rows], residual_train_full[profile_rows]
    quantum_profile: dict[str, Any] = {
        "evidence_label": "actual India train-OOF rows; throughput only, no fit/selection/test scoring",
        "development_data_sha256": development_hash,
        "rows_used_for_profile": len(profile_x),
        "batch_sizes": [128, 256, 512],
        "results": {},
    }
    for architecture, profile_model in models.items():
        numerical = validate_batched_forward(profile_model, profile_x)
        gradient = profile_model.gradient_check(profile_x[:32], profile_y[:32], h=1e-5)
        try:
            sdk_expectations = qpiai_expectations(profile_x[:8], profile_model.theta, architecture)
            local_expectations = profile_model.forward_expectations(profile_x[:8])
            qpiai = {
                "status": "local_statevector_executed",
                "examples": 8,
                "max_abs_expectation_error": float(np.max(np.abs(sdk_expectations - local_expectations))),
            }
        except Exception as exc:
            qpiai = {"status": "unavailable_in_fit_runtime", "error_type": type(exc).__name__}
        cpu_forward = profile_cpu(profile_model, profile_x, repeats=profile_repeats)
        cpu_gradient = profile_gradient_cpu(
            profile_model, profile_x, profile_y, repeats=profile_repeats,
            workload_label="actual India train-OOF rows; gradient kernel timing only",
        )
        quantum_profile["results"][architecture] = {
            "numerical_correctness_32": numerical,
            "gradient_check_32": gradient,
            "qpiai_local_parity": qpiai,
            "cpu_forward_profile": cpu_forward,
            "cpu_forward_and_gradient_profile": cpu_gradient,
            "fit_wall_seconds": quantum[architecture]["fit_info"]["wall_seconds"],
            "fit_cpu_seconds": quantum[architecture]["fit_info"]["cpu_seconds"],
            "fit_circuit_batch_evaluations": quantum[architecture]["fit_info"]["circuit_batch_evaluations"],
            "fit_circuit_example_evaluations": quantum[architecture]["fit_info"]["circuit_example_evaluations"],
        }
    profile_path = artifacts_dir / "quantum_profile.json"
    profile_path.write_text(json.dumps(quantum_profile, indent=2) + "\n", encoding="utf-8")

    selection_path = artifacts_dir / "residual_selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return {"selection_path": selection_path, "profile_path": profile_path, "selection": selection}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-table", type=Path,
        default=DEFAULT_DEVELOPMENT_TABLE,
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--train-rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--max-updates", type=int, default=120)
    parser.add_argument("--batch-size", type=int, choices=(128, 256, 512), default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--profile-repeats", type=int, default=32)
    args = parser.parse_args()
    result = run_experiment(
        args.development_table,
        args.artifacts_dir,
        manifest_path=args.manifest,
        train_rows=args.train_rows,
        seed=args.seed,
        max_updates=args.max_updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        profile_repeats=args.profile_repeats,
    )
    print(json.dumps({
        "selection_path": str(result["selection_path"].resolve()),
        "profile_path": str(result["profile_path"].resolve()),
        "selected_release_gate": result["selection"]["selected_release_gate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
