"""Independent pretest audit for the classical India ETA residual control."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

FEATURES = [
    "vqc_x0_distance", "vqc_x1_congestion",
    "vqc_x2_multidelivery", "vqc_x3_preptime",
]
KEYS = ["delivery_record_id", "courier_id", "split_partition"]
ALPHAS = (0.01, 1.0, 100.0)
EXPECTED = {
    "oof_predictions.parquet": "78a5461c89387e8630421008b92dfffcc84864e9e25bea7476eefa2c8d48032f",
    "validation_predictions.parquet": "3d9485c513e6a997516baea1a4a6c497c6021240f9cbf64d524037e9e8753f1a",
    "india_eta_hgb_regressor.joblib": "f1f94fbcc932f3519aaafcff7dc99d65b7951c0240730e54c320259152bbebdf",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {
        "mae_min": float(np.mean(np.abs(error))),
        "rmse_min": float(np.sqrt(np.mean(error * error))),
        "median_absolute_error_min": float(np.median(np.abs(error))),
    }


def _assert_metrics_close(actual: dict[str, float], expected: dict[str, float], label: str, tol: float = 1e-10) -> None:
    for key in ("mae_min", "rmse_min", "median_absolute_error_min"):
        if key not in expected or not math.isclose(actual[key], float(expected[key]), rel_tol=tol, abs_tol=tol):
            raise ValueError(f"{label} {key} mismatch: recomputed={actual[key]}, recorded={expected.get(key)}")


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    centered_x = x - x_mean
    centered_y = y - y_mean
    beta = np.linalg.solve(centered_x.T @ centered_x + alpha * np.eye(x.shape[1]), centered_x.T @ centered_y)
    intercept = y_mean - float(x_mean @ beta)
    return beta, intercept


def _keyed_records(frame) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        key = tuple(str(row[name]) for name in KEYS)
        if key in index:
            raise ValueError(f"duplicate control join key {key}")
        index[key] = row
    return index


def audit_residual_control(repo: str | Path) -> dict[str, Any]:
    """Recompute the experimental ridge control using training OOF only.

    No test records or test labels are opened. Validation labels are used only
    to reproduce the predeclared model-selection comparison; calibration rows
    are diagnostic and do not select the alpha.
    """
    root = Path(repo).resolve()
    gbdt_dir = root / "artifacts" / "india_eta_recovery"
    control_dir = root / "artifacts" / "india_eta_residual_control_v1"
    oof_path = gbdt_dir / "oof_predictions.parquet"
    val_path = gbdt_dir / "validation_predictions.parquet"
    control_manifest_path = control_dir / "recovery_manifest.json"
    checkpoint_path = control_dir / "residual_ridge.json"
    control_val_path = control_dir / "validation_predictions.parquet"
    role_map_path = gbdt_dir / "validation_role_assignment.json"
    gbdt_manifest_path = gbdt_dir / "recovery_manifest.json"

    for name, path in (("oof_predictions.parquet", oof_path), ("validation_predictions.parquet", val_path),
                       ("india_eta_hgb_regressor.joblib", gbdt_dir / "gbdt" / "india_eta_hgb_regressor.joblib")):
        actual = sha256_file(path)
        if actual != EXPECTED[name]:
            raise ValueError(f"frozen upstream {name} hash changed: {actual}")

    summary = json.loads(control_manifest_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    gbdt_manifest = json.loads(gbdt_manifest_path.read_text(encoding="utf-8"))
    role_manifest = json.loads(role_map_path.read_text(encoding="utf-8"))
    role_entry = gbdt_manifest["prediction_artifacts"]["validation_role_assignment"]
    if Path(role_entry["path"]).name != role_map_path.name or sha256_file(role_map_path) != role_entry["sha256"]:
        raise ValueError("frozen validation role-map file hash differs from recovery manifest")
    if summary.get("status") != "frozen_experimental_classical_control_not_operationally_activated":
        raise ValueError("residual control status does not keep it experimental/inactive")
    if checkpoint.get("status") != summary.get("status") or checkpoint.get("format") != "india-eta-ridge-residual-control-v1":
        raise ValueError("control checkpoint and summary format/status differ")
    if checkpoint.get("test_partition") != "not loaded into this control, not scored, and not serialized" or summary.get("no_test_rows") is not True:
        raise ValueError("control manifest does not assert test exclusion")
    if checkpoint.get("target") != "delivery_duration_min minus courier-held-out GBDT OOF prediction" or checkpoint.get("target_unit") != "minutes":
        raise ValueError("control residual target/unit contract differs")
    if checkpoint.get("feature_names") != FEATURES or checkpoint.get("join_keys") != KEYS:
        raise ValueError("control input feature order or stable join keys differ")
    if checkpoint.get("alpha_candidates") != list(ALPHAS):
        raise ValueError("alpha candidate set differs from frozen pretest protocol")
    if summary.get("gbdt_model_sha256") != EXPECTED["india_eta_hgb_regressor.joblib"]:
        raise ValueError("control is not bound to the frozen ETA GBDT checkpoint")
    if checkpoint.get("gbdt_oof_predictions_sha256") != EXPECTED["oof_predictions.parquet"] or checkpoint.get("gbdt_validation_predictions_sha256") != EXPECTED["validation_predictions.parquet"]:
        raise ValueError("control checkpoint is not bound to the frozen GBDT prediction artifacts")
    if checkpoint.get("validation_role_assignment_sha256") != role_manifest.get("assignment_sha256"):
        raise ValueError("control uses a different validation courier role map")
    if checkpoint.get("gbdt_recovery_manifest_sha256") != sha256_file(gbdt_manifest_path):
        raise ValueError("control checkpoint is not bound to the frozen GBDT recovery manifest")
    source_hashes = checkpoint.get("source_parquet_sha256", {})
    if source_hashes.get("gbdt") != gbdt_manifest["data_provenance"]["files_sha256"]["gbdt_parquet"] or source_hashes.get("vqc") != gbdt_manifest["data_provenance"]["files_sha256"]["vqc_parquet"]:
        raise ValueError("control source Parquet provenance differs from frozen GBDT manifest")
    if summary.get("validation_role_assignment_sha256") != role_manifest.get("assignment_sha256"):
        raise ValueError("summary validation role map hash mismatch")
    if checkpoint.get("split_manifest_sha256") != gbdt_manifest["data_provenance"]["split_manifest_sha256"]:
        raise ValueError("control split manifest hash mismatch")

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas + fastparquet are required for independent control acceptance") from exc
    oof = pd.read_parquet(oof_path, engine="fastparquet")
    val = pd.read_parquet(val_path, engine="fastparquet")
    val_control = pd.read_parquet(control_val_path, engine="fastparquet")
    if len(oof) != 29_592 or len(val) != 5_989 or len(val_control) != 5_989:
        raise ValueError("control inputs/output have unexpected row counts")
    if set(oof.split_partition.astype(str)) != {"train"} or set(val.split_partition.astype(str)) != {"val"} or set(val_control.split_partition.astype(str)) != {"val"}:
        raise ValueError("control rows contain a partition outside train OOF or validation")
    for frame, part in ((oof, "train"), (val, "val"), (val_control, "val")):
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError(f"{part} control rows have missing or duplicate stable join keys")
        if not all(name in frame.columns for name in FEATURES + ["y_true_min", "gbdt_pred_min"]):
            raise ValueError(f"{part} control rows lack required features/targets")
        numeric = frame[FEATURES + ["y_true_min", "gbdt_pred_min"]].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{part} inputs contain non-finite numbers")
        features = frame[FEATURES].to_numpy(dtype=np.float64)
        if np.any(features < -1e-12) or np.any(features > 1 + 1e-12):
            raise ValueError(f"{part} four candidate features are outside the committed [0,1] scale")

    oof_index = _keyed_records(oof)
    val_index = _keyed_records(val)
    control_val_index = _keyed_records(val_control)
    if set(control_val_index) != set(val_index):
        raise ValueError("control validation output IDs/courier/split do not match frozen validation records")
    if set(oof["delivery_record_id"].astype(str)) & set(val["delivery_record_id"].astype(str)):
        raise ValueError("train OOF and validation record IDs overlap")
    role_by_id = {str(k): str(v) for k, v in role_manifest.get("record_id_to_role", {}).items()}
    courier_by_id = {str(k): str(v) for k, v in role_manifest.get("courier_to_role", {}).items()}
    if set(role_by_id) != set(val["delivery_record_id"].astype(str)):
        raise ValueError("frozen ID-keyed role map does not exactly cover validation IDs")
    if set(courier_by_id) != set(val["courier_id"].astype(str)):
        raise ValueError("frozen courier role map does not exactly cover validation couriers")
    for row in val.to_dict(orient="records"):
        rid, courier, role = str(row["delivery_record_id"]), str(row["courier_id"]), str(row["validation_role"])
        if role_by_id.get(rid) != role or courier_by_id.get(courier) != role:
            raise ValueError(f"validation record {rid} role differs from frozen ID/courier map")
    selection_mask = val.validation_role.astype(str).to_numpy() == "model_selection"
    calibration_mask = val.validation_role.astype(str).to_numpy() == "fusion_calibration"
    selection_couriers = set(val.loc[selection_mask, "courier_id"].astype(str))
    calibration_couriers = set(val.loc[calibration_mask, "courier_id"].astype(str))
    if selection_mask.sum() != 4_124 or calibration_mask.sum() != 1_865 or len(selection_couriers) != 117 or len(calibration_couriers) != 58:
        raise ValueError("frozen selection/calibration record or courier counts differ")
    if selection_couriers & calibration_couriers:
        raise ValueError("selection/calibration courier groups overlap")

    x_train = oof[FEATURES].to_numpy(dtype=np.float64)
    y_train = oof.y_true_min.to_numpy(dtype=np.float64) - oof.gbdt_pred_min.to_numpy(dtype=np.float64)
    residual_hash = hashlib.sha256(y_train.astype("<f8", copy=False).tobytes()).hexdigest()
    if residual_hash != checkpoint.get("training_oof_residual_sha256"):
        raise ValueError("control training residual target hash is not y_true minus train OOF prediction")
    record_id_hash = hashlib.sha256("\n".join(sorted(oof.delivery_record_id.astype(str))).encode("utf-8")).hexdigest()
    if record_id_hash != checkpoint.get("training_record_ids_sha256") or checkpoint.get("training_rows") != len(oof):
        raise ValueError("control training ID/count provenance mismatch")
    if checkpoint.get("training_courier_groups") != int(oof.courier_id.nunique()):
        raise ValueError("control training courier group count mismatch")

    x_val = val[FEATURES].to_numpy(dtype=np.float64)
    y_val = val.y_true_min.to_numpy(dtype=np.float64)
    gbdt_val = val.gbdt_pred_min.to_numpy(dtype=np.float64)
    candidates: dict[float, dict[str, Any]] = {}
    trials: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        beta, intercept = _ridge_fit(x_train, y_train, alpha)
        correction = x_val @ beta + intercept
        prediction = gbdt_val + correction
        selection_metrics = _metrics(y_val[selection_mask], prediction[selection_mask])
        candidates[alpha] = {"beta": beta, "intercept": intercept, "correction": correction, "prediction": prediction}
        trials.append({"alpha": alpha, "model_selection_metrics": selection_metrics, "model_selection_rows": int(selection_mask.sum())})
    winner = min(range(len(trials)), key=lambda i: (trials[i]["model_selection_metrics"]["mae_min"], trials[i]["alpha"]))
    selected_alpha = float(trials[winner]["alpha"])
    if selected_alpha != float(checkpoint["alpha"]) or selected_alpha != float(summary["alpha"]):
        raise ValueError("recorded selected alpha is not the selection-only minimum MAE/tie-break result")
    beta = np.asarray(checkpoint["coefficients_min_per_unit"], dtype=np.float64)
    intercept = float(checkpoint["intercept_min"])
    if beta.shape != (len(FEATURES),) or not np.isfinite(beta).all() or not math.isfinite(intercept):
        raise ValueError("frozen ridge coefficients/intercept invalid")
    if not np.allclose(beta, candidates[selected_alpha]["beta"], rtol=1e-10, atol=1e-12) or not math.isclose(intercept, candidates[selected_alpha]["intercept"], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("frozen coefficients/intercept do not recompute from train OOF residuals")
    for actual, expected in zip(trials, checkpoint["selection_trials"]):
        if actual["alpha"] != float(expected["alpha"]):
            raise ValueError("selection trial alpha order differs")
        _assert_metrics_close(actual["model_selection_metrics"], expected["model_selection_metrics"], f"alpha={actual['alpha']} selection")

    chosen = candidates[selected_alpha]
    selected_metrics = _metrics(y_val[selection_mask], chosen["prediction"][selection_mask])
    baseline_metrics = _metrics(y_val[selection_mask], gbdt_val[selection_mask])
    calibration_metrics = _metrics(y_val[calibration_mask], chosen["prediction"][calibration_mask])
    all_val_metrics = _metrics(y_val, chosen["prediction"])
    _assert_metrics_close(selected_metrics, checkpoint["selection_selected_control_metrics"], "selected control/model-selection")
    _assert_metrics_close(baseline_metrics, checkpoint["selection_baseline_gbdt_only_metrics"], "GBDT/model-selection")
    _assert_metrics_close(calibration_metrics, checkpoint["calibration_selected_control_metrics_diagnostic_only"], "control/calibration diagnostic")
    _assert_metrics_close(all_val_metrics, checkpoint["all_validation_selected_control_metrics_descriptive_only"], "control/all-validation descriptive")

    # Keyed compare every saved validation row and output against independent
    # recomputation. The zip-based merge is by stable key, never row position.
    max_prediction_error = 0.0
    max_correction_error = 0.0
    for key, source_row in val_index.items():
        out_row = control_val_index[key]
        expected_role = str(source_row["validation_role"])
        if str(out_row["validation_role"]) != expected_role:
            raise ValueError(f"validation output role mismatch at {key}")
        source_target = float(source_row["y_true_min"])
        source_gbdt = float(source_row["gbdt_pred_min"])
        if source_target != float(out_row["y_true_min"]) or source_gbdt != float(out_row["gbdt_pred_min"]):
            raise ValueError(f"saved validation control target/base prediction mismatch at {key}")
        idx = val_index_key_to_row_index(val, key)
        expected_pred = float(chosen["prediction"][idx])
        expected_correction = float(chosen["correction"][idx])
        max_prediction_error = max(max_prediction_error, abs(float(out_row["ridge_eta_pred_min"]) - expected_pred))
        max_correction_error = max(max_correction_error, abs(float(out_row["residual_correction_min"]) - expected_correction))
        if float(out_row["selected_alpha"]) != selected_alpha:
            raise ValueError("saved validation rows disagree on the selected alpha")
    if max_prediction_error > 1e-10 or max_correction_error > 1e-10:
        raise ValueError("saved validation control predictions differ from independent ridge recomputation")

    control_manifest_sha = sha256_file(control_manifest_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    val_output_sha = sha256_file(control_val_path)
    if summary.get("checkpoint_sha256") != checkpoint_sha or summary.get("validation_predictions_sha256") != val_output_sha:
        raise ValueError("control summary artifact hashes do not match outputs")
    return {
        "status": "PASS",
        "scope": "experimental classical residual control only; no QML or test evaluation",
        "test_targets_read": False,
        "training_rows": len(oof),
        "training_oof_residual_target_sha256": residual_hash,
        "training_record_ids_sha256": record_id_hash,
        "selection_rows": int(selection_mask.sum()),
        "selection_couriers": len(selection_couriers),
        "calibration_rows": int(calibration_mask.sum()),
        "calibration_couriers": len(calibration_couriers),
        "selected_alpha_by_independent_selection_only_recompute": selected_alpha,
        "model_selection_metrics_recomputed_min": {
            "gbdt_only": baseline_metrics,
            "ridge_candidates": trials,
            "selected_ridge": selected_metrics,
        },
        "calibration_metrics_diagnostic_only_min": calibration_metrics,
        "all_validation_metrics_descriptive_only_min": all_val_metrics,
        "saved_validation_prediction_max_abs_error_min": max_prediction_error,
        "saved_validation_correction_max_abs_error_min": max_correction_error,
        "checkpoint_sha256": checkpoint_sha,
        "validation_predictions_sha256": val_output_sha,
        "control_manifest_sha256": control_manifest_sha,
        "status_is_experimental_not_activated": True,
        "limitations": ["control is not a VQC and demonstrates no quantum improvement", "test evaluation remains held"],
    }


def write_residual_control_acceptance(repo: str | Path, output: str | Path | None = None) -> Path:
    root = Path(repo).resolve()
    result = audit_residual_control(root)
    result["scope"] = "independent pretest audit of frozen classical residual control using train OOF and validation only"
    destination = Path(output).resolve() if output else root / "artifacts" / "run2_india" / "acceptance_residual_control.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return destination


def val_index_key_to_row_index(frame, key: tuple[str, str, str]) -> int:
    # Use the dataframe's stable composite keys to recover an aligned array index.
    match = (
        frame.delivery_record_id.astype(str).eq(key[0])
        & frame.courier_id.astype(str).eq(key[1])
        & frame.split_partition.astype(str).eq(key[2])
    )
    indices = np.flatnonzero(match.to_numpy())
    if len(indices) != 1:
        raise ValueError(f"validation key {key} does not identify exactly one row")
    return int(indices[0])
