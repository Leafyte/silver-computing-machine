"""Leakage-safe residual fitting from Worker A's frozen GBDT predictions.

Only the train OOF and development-validation prediction artifacts are read.
This module does not open or accept a test prediction/table path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

import numpy as np

from experiments_india_qml.torch_residual import (
    BASE_FEATURES,
    REUPLOAD_FEATURES,
    TorchCircuitConfig,
    TorchResidualVQC,
    benchmark_torch_backends,
    qpiai_residual_parity,
)
from experiments_india_qml.portable_inference import predict_residual_minutes as predict_numpy_minutes


TARGET = "y_true_min"
PREDICTION = "gbdt_pred_min"
ID_COLUMNS = ("delivery_record_id", "courier_id", "split_partition")
VALIDATION_ROLES = ("model_selection", "fusion_calibration")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _metrics(y: np.ndarray, pred: np.ndarray, *, started: float | None = None) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    if len(y) != len(p) or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("metric arrays must be non-empty, finite, and aligned")
    error = p - y
    return {
        "n": int(len(y)), "mae_minutes": float(np.mean(np.abs(error))),
        "rmse_minutes": float(np.sqrt(np.mean(error ** 2))),
        "median_absolute_error_minutes": float(np.median(np.abs(error))),
        "inference_seconds": float(time.perf_counter() - started) if started is not None else None,
        "inference_microseconds_per_row": (
            float((time.perf_counter() - started) * 1e6 / len(y)) if started is not None else None
        ),
    }


def _validate_predictions(oof: Any, validation: Any, fold_manifest: Mapping[str, Any]) -> tuple[Any, Any]:
    """Check identities, split membership, labels, feature order, and groups."""
    required = set((*ID_COLUMNS, TARGET, PREDICTION, "fold", *REUPLOAD_FEATURES))
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"OOF prediction file is missing columns: {sorted(missing)}")
    val_required = set((*ID_COLUMNS, TARGET, PREDICTION, "validation_role", *REUPLOAD_FEATURES))
    val_missing = val_required - set(validation.columns)
    if val_missing:
        raise ValueError(f"validation prediction file is missing columns: {sorted(val_missing)}")
    if len(oof) != 29_592 or len(validation) != 5_989:
        raise ValueError(f"unexpected eligible prediction counts: OOF={len(oof)}, validation={len(validation)}")
    for name, frame, partition in (("OOF", oof, "train"), ("validation", validation, "val")):
        if frame[list(ID_COLUMNS)].isna().any().any():
            raise ValueError(f"{name} stable key contains null values")
        if frame.duplicated(list(ID_COLUMNS)).any() or frame["delivery_record_id"].duplicated().any():
            raise ValueError(f"{name} stable record IDs must be unique")
        if set(frame["split_partition"].astype(str)) != {partition}:
            raise ValueError(f"{name} contains rows outside split {partition}")
        for column in (TARGET, PREDICTION, *REUPLOAD_FEATURES):
            values = frame[column].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"{name}.{column} contains non-finite values")
        features = frame[list(REUPLOAD_FEATURES)].to_numpy(dtype=np.float64)
        if np.any(features < -1e-12) or np.any(features > 1.0 + 1e-12):
            raise ValueError(f"{name} VQC inputs must use committed [0,1] transforms")
    if oof["fold"].isna().any() or oof["fold"].nunique() < 2:
        raise ValueError("OOF predictions require multiple documented courier folds")
    folds = fold_manifest.get("folds")
    if not isinstance(folds, Mapping) or len(folds) != int(fold_manifest.get("n_splits", -1)):
        raise ValueError("OOF fold manifest does not match the declared split count")
    if str(fold_manifest.get("group_column")) not in {"Delivery_person_ID", "courier_id"}:
        raise ValueError("OOF fold manifest must document courier-group isolation")
    oof_couriers = set(oof["courier_id"].astype(str))
    assigned_couriers: set[str] = set()
    for fold_key, detail in folds.items():
        fold_number = int(fold_key)
        heldout = set(map(str, detail.get("heldout_couriers", [])))
        trained = set(map(str, detail.get("train_couriers", [])))
        if not heldout or heldout & trained or trained != oof_couriers - heldout:
            raise ValueError(f"fold {fold_key} has empty or overlapping courier groups")
        if assigned_couriers & heldout:
            raise ValueError("an OOF courier is assigned to multiple held-out folds")
        assigned_couriers.update(heldout)
        rows = oof[oof["fold"].astype(int) == fold_number]
        row_groups = set(rows["courier_id"].astype(str))
        if row_groups != heldout or row_groups & trained:
            raise ValueError(f"OOF rows for fold {fold_key} do not match held-out courier provenance")
        if len(rows) != int(detail.get("heldout_row_count", -1)):
            raise ValueError(f"OOF row count disagrees with fold manifest for fold {fold_key}")
        if int(detail.get("train_row_count", -1)) + len(rows) != len(oof):
            raise ValueError(f"fold {fold_key} training/held-out row counts do not sum to train rows")
    if assigned_couriers != set(oof["courier_id"].astype(str)):
        raise ValueError("fold manifest does not assign every OOF courier exactly once")
    train_couriers = oof_couriers
    val_couriers = set(validation["courier_id"].astype(str))
    if train_couriers & val_couriers:
        raise ValueError("train OOF and validation courier groups overlap")
    if set(validation["validation_role"].astype(str)) != set(VALIDATION_ROLES):
        raise ValueError("validation must have model_selection and fusion_calibration roles")
    couriers = sorted(set(validation["courier_id"].astype(str)))
    ranked = sorted(
        couriers,
        key=lambda courier: (
            hashlib.sha256(b"india-eta-val-role-v1|20261029|" + courier.encode("utf-8")).digest(),
            courier,
        ),
    )
    calibration_couriers = set(ranked[: len(ranked) // 3])
    def role_for(courier: Any) -> str:
        return "fusion_calibration" if str(courier) in calibration_couriers else "model_selection"
    expected_roles = validation["courier_id"].map(role_for).astype(str)
    if not np.array_equal(expected_roles.to_numpy(), validation["validation_role"].astype(str).to_numpy()):
        raise ValueError("validation_role does not match the frozen courier-hash 2/3:1/3 assignment")
    groups = validation.groupby("courier_id", dropna=False)["validation_role"].nunique()
    if (groups != 1).any():
        raise ValueError("a courier appears in both validation roles")
    return oof, validation


def _ridge_residual_fit(x: np.ndarray, y: np.ndarray, regularization: float = 10.0) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) != len(y) or x.ndim != 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ridge inputs must be finite and aligned")
    mean = x.mean(axis=0)
    centered = x - mean
    design = np.column_stack((np.ones(len(x)), centered))
    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0.0
    coeff = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {"model": "fixed_ridge_residual_control", "regularization": regularization,
            "feature_mean": mean.tolist(), "coefficients_with_intercept": coeff.tolist(),
            "feature_names": list(REUPLOAD_FEATURES), "target_unit": "minutes"}


def _ridge_predict(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    center = np.asarray(model["feature_mean"], dtype=np.float64)
    coef = np.asarray(model["coefficients_with_intercept"], dtype=np.float64)
    return coef[0] + (x - center) @ coef[1:]


def _calibrate_alpha(base: np.ndarray, residual: np.ndarray, y: np.ndarray) -> float:
    """One-parameter dev-only least-squares calibration, bounded to [0,1]."""
    d = np.asarray(residual, dtype=np.float64).reshape(-1)
    error = np.asarray(y, dtype=np.float64).reshape(-1) - np.asarray(base, dtype=np.float64).reshape(-1)
    denominator = float(d @ d)
    if denominator <= 1e-15:
        return 0.0
    return float(np.clip((d @ error) / denominator, 0.0, 1.0))


def _calibrate_selective(
    base: np.ndarray, residual: np.ndarray, y: np.ndarray,
) -> dict[str, float]:
    """Select a residual-magnitude gate and alpha on calibration couriers only."""
    base = np.asarray(base, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    candidates = [0.0, *np.quantile(np.abs(residual), [0.50, 0.75, 0.90]).tolist()]
    results = []
    for threshold in sorted(set(float(v) for v in candidates)):
        gate = (np.abs(residual) >= threshold).astype(np.float64)
        gated_residual = residual * gate
        alpha = _calibrate_alpha(base, gated_residual, y)
        pred = base + alpha * gated_residual
        results.append({"threshold_minutes": threshold, "alpha": alpha,
                        "mae_minutes": float(np.mean(np.abs(pred - y))),
                        "coverage": float(np.mean(gate > 0.0))})
    chosen = min(results, key=lambda row: (row["mae_minutes"], -row["coverage"], row["threshold_minutes"]))
    return {**chosen, "selection_role": "fusion_calibration", "candidate_count": len(results)}


def run_residual_training(
    *, oof_path: Path, validation_path: Path, output_dir: Path,
    fold_manifest_path: Path,
    source_paths: Mapping[str, Path] | None = None,
    max_epochs: int = 120, batch_size: int = 256, patience: int = 15,
    learning_rate: float = 0.01, seed: int = 20261029,
    device: str = "cpu",
) -> Path:
    """Fit and freeze both VQC variants, controls, and validation fusion policy."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas/pyarrow are required in the isolated RUN2 environment") from exc
    oof_path, validation_path, output_dir, fold_manifest_path = map(
        Path, (oof_path, validation_path, output_dir, fold_manifest_path)
    )
    if not oof_path.is_file() or not validation_path.is_file() or not fold_manifest_path.is_file():
        raise FileNotFoundError("Worker A's verified OOF and validation prediction artifacts are required")
    fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
    oof, validation = _validate_predictions(pd.read_parquet(oof_path), pd.read_parquet(validation_path), fold_manifest)
    train_x4 = oof[list(REUPLOAD_FEATURES)].to_numpy(dtype=np.float64)
    train_y = oof[TARGET].to_numpy(dtype=np.float64) - oof[PREDICTION].to_numpy(dtype=np.float64)
    val_x4 = validation[list(REUPLOAD_FEATURES)].to_numpy(dtype=np.float64)
    val_base = validation[PREDICTION].to_numpy(dtype=np.float64)
    val_y = validation[TARGET].to_numpy(dtype=np.float64)
    sel_mask = validation["validation_role"].astype(str).to_numpy() == "model_selection"
    cal_mask = validation["validation_role"].astype(str).to_numpy() == "fusion_calibration"
    if not sel_mask.any() or not cal_mask.any():
        raise ValueError("both validation roles need at least one row")
    output_dir.mkdir(parents=True, exist_ok=False)
    model_dir = output_dir / "models"
    model_dir.mkdir()
    checkpoint_paths = {}
    progress_paths = {}
    models: dict[str, TorchResidualVQC] = {}
    fit_info: dict[str, Any] = {}

    # Benchmark both backends on the same real 512-row OOF residual slice
    # before selecting the fastest verified backend for each architecture.
    backend_profile: dict[str, Any] = {}
    selected_devices: dict[str, str] = {}
    architecture_specs = (
        ("one_block_2feature", BASE_FEATURES, train_x4[:512, :2], train_y[:512], "vqc_one_block"),
        ("two_stage_reupload_4feature", REUPLOAD_FEATURES, train_x4[:512], train_y[:512], "vqc_reupload"),
    )
    for architecture, names, x_profile, y_profile, tag in architecture_specs:
        cfg = TorchCircuitConfig(architecture, tuple(names), seed=seed)
        profile = benchmark_torch_backends(cfg, x_profile, y_profile, repetitions=30, warmup=5)
        backend_profile[tag] = profile
        if device == "auto":
            selected_devices[tag] = min(profile["results"], key=lambda row: row["wall_seconds"])["device"]
        elif device in {"cpu", "cuda"}:
            selected_devices[tag] = device
            if device == "cuda" and not profile["cuda_available"]:
                raise RuntimeError("CUDA was selected but no verified CUDA device is available")
        else:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    _write_json(output_dir / "backend_comparison.json", backend_profile)

    # A small fixed-ridge control is fit to the exact same leakage-safe OOF
    # residuals, without accessing validation/test labels for coefficients.
    ridge = _ridge_residual_fit(train_x4, train_y, regularization=10.0)
    _write_json(model_dir / "classical_residual_ridge.json", ridge)

    for architecture, names, x_tr, x_sel, tag in (
        ("one_block_2feature", BASE_FEATURES, train_x4[:, :2], val_x4[sel_mask, :2], "vqc_one_block"),
        ("two_stage_reupload_4feature", REUPLOAD_FEATURES, train_x4, val_x4[sel_mask], "vqc_reupload"),
    ):
        config = TorchCircuitConfig(architecture, tuple(names), seed=seed)
        model = TorchResidualVQC(config, device=selected_devices[tag])
        checkpoint = model_dir / f"{tag}.json"
        progress = output_dir / f"{tag}_epochs.jsonl"
        info = model.fit(
            x_tr, train_y, x_sel,
            val_y[sel_mask] - val_base[sel_mask],
            max_epochs=max_epochs, batch_size=batch_size, learning_rate=learning_rate,
            patience=patience, seed=seed, checkpoint_path=checkpoint, progress_path=progress,
        )
        models[tag] = model
        fit_info[tag] = info
        checkpoint_paths[tag] = checkpoint
        progress_paths[tag] = progress

    pred_resid: dict[str, np.ndarray] = {}
    for tag, model in models.items():
        features = val_x4[:, :2] if tag == "vqc_one_block" else val_x4
        start = time.perf_counter()
        pred_resid[tag] = model.predict_residual_minutes(features)
        fit_info[tag]["validation_full_inference_wall_seconds"] = time.perf_counter() - start

    # Choose the architecture using only validation model-selection couriers.
    architecture_scores = {
        tag: _metrics(val_y[sel_mask], val_base[sel_mask] + pred_resid[tag][sel_mask])
        for tag in models
    }
    selected_tag = min(architecture_scores, key=lambda key: architecture_scores[key]["mae_minutes"])

    # Fit the fusion coefficient and optional selective gate on distinct
    # validation couriers. The test partition is not read anywhere in this path.
    alpha_all = _calibrate_alpha(val_base[cal_mask], pred_resid[selected_tag][cal_mask], val_y[cal_mask])
    selective = _calibrate_selective(val_base[cal_mask], pred_resid[selected_tag][cal_mask], val_y[cal_mask])
    raw_fusion = val_base + alpha_all * pred_resid[selected_tag]
    gate = (np.abs(pred_resid[selected_tag]) >= selective["threshold_minutes"]).astype(np.float64)
    selective_fusion = val_base + selective["alpha"] * pred_resid[selected_tag] * gate
    cal_baseline_mae = float(np.mean(np.abs(val_base[cal_mask] - val_y[cal_mask])))
    ridge_pred = val_base + _ridge_predict(ridge, val_x4)
    cal_ridge_mae = float(np.mean(np.abs(ridge_pred[cal_mask] - val_y[cal_mask])))
    cal_baseline_rmse = float(np.sqrt(np.mean((val_base[cal_mask] - val_y[cal_mask]) ** 2)))
    cal_ridge_rmse = float(np.sqrt(np.mean((ridge_pred[cal_mask] - val_y[cal_mask]) ** 2)))
    all_vqc_cal_mae = float(np.mean(np.abs(raw_fusion[cal_mask] - val_y[cal_mask])))
    selective_vqc_cal_mae = float(np.mean(np.abs(selective_fusion[cal_mask] - val_y[cal_mask])))
    all_vqc_cal_rmse = float(np.sqrt(np.mean((raw_fusion[cal_mask] - val_y[cal_mask]) ** 2)))
    selective_vqc_cal_rmse = float(np.sqrt(np.mean((selective_fusion[cal_mask] - val_y[cal_mask]) ** 2)))
    activation_mae_barrier = min(cal_baseline_mae, cal_ridge_mae)
    activation_rmse_barrier = min(cal_baseline_rmse, cal_ridge_rmse)
    qualified_fusions = []
    if all_vqc_cal_mae < activation_mae_barrier - 1e-9 and all_vqc_cal_rmse <= activation_rmse_barrier + 1e-9:
        qualified_fusions.append(("all_rows", all_vqc_cal_mae, alpha_all, None))
    if selective_vqc_cal_mae < activation_mae_barrier - 1e-9 and selective_vqc_cal_rmse <= activation_rmse_barrier + 1e-9:
        qualified_fusions.append(("selective", selective_vqc_cal_mae, selective["alpha"], selective["threshold_minutes"]))
    if not qualified_fusions:
        operational_policy = {"selected_model": "gbdt_fallback", "quantum_alpha": 0.0,
                              "selective_gate_threshold_minutes": None,
                              "reason": "VQC failed calibration-courier MAE/RMSE activation barriers versus both frozen GBDT and four-feature ridge control; operational correction forced to zero"}
    else:
        fusion_mode, _, alpha, threshold = min(qualified_fusions, key=lambda row: row[1])
        operational_policy = {"selected_model": selected_tag, "fusion_mode": fusion_mode,
                              "quantum_alpha": alpha, "selective_gate_threshold_minutes": threshold,
                              "reason": "selected on validation calibration couriers after beating GBDT and the same-feature ridge control under the frozen MAE/RMSE rule"}
    predictions = validation[list(ID_COLUMNS) + ["validation_role", TARGET, PREDICTION, *REUPLOAD_FEATURES]].copy()
    predictions["gbdt_only_pred_min"] = val_base
    predictions["ridge_residual_pred_min"] = val_base + _ridge_predict(ridge, val_x4)
    for tag in models:
        predictions[f"{tag}_raw_pred_min"] = val_base + pred_resid[tag]
    predictions[f"vqc_calibrated_pred_min"] = raw_fusion
    predictions[f"vqc_selective_pred_min"] = selective_fusion
    if operational_policy["selected_model"] == "gbdt_fallback":
        predictions["operational_pred_min"] = val_base
    elif operational_policy.get("fusion_mode") == "selective":
        predictions["operational_pred_min"] = selective_fusion
    else:
        predictions["operational_pred_min"] = raw_fusion
    predictions.to_parquet(output_dir / "validation_dev_predictions.parquet", index=False)

    dev_metrics: dict[str, Any] = {}
    for role, mask in (("model_selection", sel_mask), ("fusion_calibration", cal_mask)):
        role_rows = {}
        for tag in ("gbdt_only_pred_min", "ridge_residual_pred_min",
                    "vqc_one_block_raw_pred_min", "vqc_reupload_raw_pred_min",
                    "vqc_calibrated_pred_min", "vqc_selective_pred_min", "operational_pred_min"):
            values = predictions[tag].to_numpy(dtype=np.float64)
            role_rows[tag] = _metrics(val_y[mask], values[mask])
        dev_metrics[role] = role_rows
    _write_json(output_dir / "dev_metrics.json", dev_metrics)
    _write_json(output_dir / "fusion_policy.json", {
        "selected_vqc_architecture": selected_tag,
        "alpha_all": alpha_all, "selective_calibration": selective,
        "calibration_gbdt_mae_minutes": cal_baseline_mae,
        "calibration_ridge_mae_minutes": cal_ridge_mae,
        "calibration_gbdt_rmse_minutes": cal_baseline_rmse,
        "calibration_ridge_rmse_minutes": cal_ridge_rmse,
        "calibration_all_vqc_mae_minutes": all_vqc_cal_mae,
        "calibration_all_vqc_rmse_minutes": all_vqc_cal_rmse,
        "calibration_selective_vqc_mae_minutes": selective_vqc_cal_mae,
        "calibration_selective_vqc_rmse_minutes": selective_vqc_cal_rmse,
        "activation_rule": "calibration MAE must beat both GBDT-only and same-four-feature train-OOF ridge control; calibration RMSE must be no worse than either; otherwise alpha=0",
        "operational_policy": operational_policy,
        "target_unit": "delivery_duration_min / residual minutes",
        "fit_inputs": "train OOF residual only",
        "architecture_selection_split": "validation model_selection courier groups",
        "fusion_calibration_split": "validation fusion_calibration courier groups",
        "test_used": False,
    })

    parity = {}
    gradient_checks = {}
    portable_parity = {}
    for tag, model in models.items():
        model_features = train_x4[:16, :2] if tag == "vqc_one_block" else train_x4[:16]
        gradient_checks[tag] = model.gradient_check(model_features, train_y[:16], h=1e-5)
        if not gradient_checks[tag]["passed_abs_1e-5"]:
            raise RuntimeError(f"autograd finite-difference check failed for {tag}")
        parity_rows = val_x4[: min(8, len(val_x4))]
        parity_input = parity_rows[:, :2] if tag == "vqc_one_block" else parity_rows
        parity[tag] = qpiai_residual_parity(model, parity_input)
        if not parity[tag]["passed"]:
            raise RuntimeError(f"QpiAI local parity failed for {tag}")
        payload = json.loads(checkpoint_paths[tag].read_text(encoding="utf-8"))
        portable_prediction = predict_numpy_minutes(payload, parity_input)
        torch_prediction = model.predict_residual_minutes(parity_input)
        portable_error = np.abs(portable_prediction - torch_prediction)
        portable_parity[tag] = {
            "model_sha256": model.parameter_sha256(),
            "rows": int(len(parity_input)),
            "max_absolute_error_minutes": float(portable_error.max(initial=0.0)),
            "tolerance_minutes": 1e-10,
            "passed": bool(portable_error.max(initial=0.0) <= 1e-10),
            "execution": "portable batched NumPy CPU reference versus PyTorch model",
        }
        if not portable_parity[tag]["passed"]:
            raise RuntimeError(f"portable CPU prediction parity failed for {tag}")
    _write_json(output_dir / "qpiai_local_parity.json", parity)
    _write_json(output_dir / "autograd_gradient_checks.json", gradient_checks)
    _write_json(output_dir / "portable_inference_parity.json", portable_parity)

    provenance = {
        "oof_predictions": {"path": str(oof_path), "sha256": _sha256(oof_path), "rows": len(oof)},
        "validation_predictions": {"path": str(validation_path), "sha256": _sha256(validation_path), "rows": len(validation)},
        "oof_fold_manifest": {"path": str(fold_manifest_path), "sha256": _sha256(fold_manifest_path),
                              "fold_count": int(fold_manifest["n_splits"]),
                              "group_column": fold_manifest["group_column"]},
        "source_files": {
            name: {"path": str(path), "sha256": _sha256(Path(path))}
            for name, path in sorted((source_paths or {}).items())
        },
    }
    models_hash = {path.name: _sha256(path) for path in sorted(model_dir.glob("*.json"))}
    code_paths = [Path(__file__).resolve(), Path(__file__).with_name("torch_residual.py").resolve(),
                  Path(__file__).with_name("portable_inference.py").resolve(),
                  Path(__file__).with_name("quantum.py").resolve()]
    freeze_payload = {
        "schema_version": "run2_india_models_frozen_v1",
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_artifact_sha256": models_hash,
        "fusion_policy_sha256": _sha256(output_dir / "fusion_policy.json"),
        "oof_predictions_sha256": _sha256(oof_path),
        "validation_predictions_sha256": _sha256(validation_path),
        "fold_manifest_sha256": _sha256(fold_manifest_path),
        "selected_vqc_architecture": selected_tag,
        "operational_policy": operational_policy,
        "test_status": "sealed; not read, predicted, or evaluated during training/model selection",
    }
    _write_json(output_dir / "models_frozen.json", freeze_payload)
    _write_json(output_dir / "TEST-SEAL.json", {
        "schema_version": "run2_india_test_seal_v1",
        "models_frozen_sha256": _sha256(output_dir / "models_frozen.json"),
        "frozen_utc": freeze_payload["frozen_utc"],
        "test_opened": False,
        "test_ids_or_labels_read": False,
        "only_final_evaluator_may_open_test": True,
    })
    manifest = {
        "schema_version": "run2_india_residual_training_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_seed": seed,
        "execution": {"device_request": device, "selected_device_by_architecture": selected_devices,
                      "framework": str(_torch_version()), "cuda_available": _cuda_available(),
                      "cpu_vs_cuda_profiles": backend_profile},
        "source_provenance": provenance,
        "split_contract": {
            "train_oof_rows": len(oof), "train_oof_fold_count": int(oof["fold"].nunique()),
            "train_courier_count": int(oof["courier_id"].nunique()),
            "validation_rows": len(validation),
            "validation_selection_couriers": int(validation.loc[sel_mask, "courier_id"].nunique()),
            "validation_calibration_couriers": int(validation.loc[cal_mask, "courier_id"].nunique()),
        "validation_role_algorithm": "sort unique couriers by (SHA256(b'india-eta-val-role-v1|20261029|' + courier_id UTF-8), courier_id); first floor(N/3) calibration, remainder model_selection",
        "validation_role_map_sha256": hashlib.sha256(json.dumps(
            {str(k): str(v) for k, v in sorted(validation.drop_duplicates("courier_id").set_index("courier_id")["validation_role"].items())},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "validation_courier_overlap": sorted(set(validation.loc[sel_mask, "courier_id"].astype(str)) &
                                                  set(validation.loc[cal_mask, "courier_id"].astype(str))),
            "test_status": "sealed; no test file/path accepted or read by training entrypoint",
        },
        "feature_contract": {"baseline": list(BASE_FEATURES), "reupload": list(REUPLOAD_FEATURES),
                             "range": "preprocessed committed [0,1] inputs",
                             "forbidden_inputs": ["delivery_record_id", "courier_id", "split_partition", "target"]},
        "residual_target": "y_true_min - gbdt_pred_min; train targets use courier-grouped OOF predictions only",
        "fit_settings": {"max_epochs": max_epochs, "batch_size": batch_size, "patience": patience,
                         "learning_rate": learning_rate, "seed": seed},
        "fit_info": fit_info, "validation_model_selection": architecture_scores,
        "validation_metrics_path": "dev_metrics.json", "fusion_policy_path": "fusion_policy.json",
        "qpiai_local_parity_path": "qpiai_local_parity.json",
        "autograd_gradient_checks": gradient_checks,
        "portable_inference_parity": portable_parity,
        "models_frozen_path": "models_frozen.json",
        "test_seal_path": "TEST-SEAL.json",
        "model_artifact_sha256": models_hash,
        "code_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])).replace("\\", "/"):
                         _sha256(path) for path in code_paths},
        "test_labels_not_read": True,
    }
    artifact_hashes = {
        path.relative_to(output_dir).as_posix(): _sha256(path)
        for path in sorted(output_dir.rglob("*")) if path.is_file() and path.name not in {"run_manifest.json", "artifact_hashes.json"}
    }
    _write_json(output_dir / "artifact_hashes.json", artifact_hashes)
    manifest["artifact_hashes"] = artifact_hashes
    manifest["artifact_hashes_sha256"] = _sha256(output_dir / "artifact_hashes.json")
    _write_json(output_dir / "run_manifest.json", manifest)
    return output_dir


def _torch_version() -> str:
    from experiments_india_qml.torch_residual import _require_torch
    return str(_require_torch().__version__)


def _cuda_available() -> bool:
    from experiments_india_qml.torch_residual import _require_torch
    return bool(_require_torch().cuda.is_available())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=Path("artifacts/india_eta_recovery/oof_predictions.parquet"))
    parser.add_argument("--validation", type=Path, default=Path("artifacts/india_eta_recovery/validation_predictions.parquet"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/run2_india/training"))
    parser.add_argument("--fold-manifest", type=Path, default=Path("artifacts/india_eta_recovery/oof_fold_manifest.json"))
    parser.add_argument("--gbdt-manifest", type=Path, default=Path("artifacts/india_eta_recovery/recovery_manifest.json"))
    parser.add_argument("--gbdt-checkpoint", type=Path, default=Path("artifacts/india_eta_recovery/gbdt/india_eta_hgb_regressor.joblib"))
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    source_paths: dict[str, Path] = {
        "raw_csv": Path("data/train.csv"),
        "clean_parquet": Path("data/indian_delivery_clean.parquet"),
        "gbdt_parquet": Path("data/indian_delivery_gbdt.parquet"),
        "vqc_parquet": Path("data/indian_delivery_vqc.parquet"),
        "feature_schema": Path("data/feature_schema.json"),
        "split_manifest": Path("data/split_manifest.json"),
        "preprocessing_artifacts": Path("data/preprocessing_artifacts.json"),
        "gbdt_checkpoint": args.gbdt_checkpoint,
        "gbdt_manifest": args.gbdt_manifest,
    }
    result = run_residual_training(
        oof_path=args.oof, validation_path=args.validation, output_dir=args.output,
        fold_manifest_path=args.fold_manifest,
        source_paths=source_paths, max_epochs=args.max_epochs,
        batch_size=args.batch_size, patience=args.patience,
        learning_rate=args.learning_rate, seed=args.seed, device=args.device,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
