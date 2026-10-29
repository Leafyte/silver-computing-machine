"""Acceptance checks for the bounded classical residual comparison control."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "india_eta_residual_control_v1"
RECOVERY = ROOT / "artifacts" / "india_eta_recovery"
FEATURES = ["vqc_x0_distance", "vqc_x1_congestion", "vqc_x2_multidelivery", "vqc_x3_preptime"]


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {
        "mae_min": float(np.mean(np.abs(error))),
        "rmse_min": float(np.sqrt(np.mean(error * error))),
        "median_absolute_error_min": float(np.median(np.abs(error))),
    }


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    beta = np.linalg.solve((x - x_mean).T @ (x - x_mean) + alpha * np.eye(x.shape[1]), (x - x_mean).T @ (y - y_mean))
    return beta, y_mean - float(x_mean @ beta)


def test_residual_control_selection_and_predictions_reproduce_from_train_oof() -> None:
    manifest_path = ARTIFACTS / "recovery_manifest.json"
    checkpoint_path = ARTIFACTS / "residual_ridge.json"
    predictions_path = ARTIFACTS / "validation_predictions.parquet"
    summary = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == summary["checkpoint_sha256"]
    assert hashlib.sha256(predictions_path.read_bytes()).hexdigest() == summary["validation_predictions_sha256"]
    assert checkpoint["status"] == "frozen_experimental_classical_control_not_operationally_activated"
    assert checkpoint["feature_names"] == FEATURES
    assert checkpoint["alpha_candidates"] == [0.01, 1.0, 100.0]
    assert checkpoint["gbdt_model_sha256"] == "f1f94fbcc932f3519aaafcff7dc99d65b7951c0240730e54c320259152bbebdf"

    oof_path = RECOVERY / "oof_predictions.parquet"
    gbdt_val_path = RECOVERY / "validation_predictions.parquet"
    oof = pd.read_parquet(oof_path, engine="fastparquet")
    gbdt_val = pd.read_parquet(gbdt_val_path, engine="fastparquet")
    result = pd.read_parquet(predictions_path, engine="fastparquet")
    assert len(oof) == checkpoint["training_rows"] == 29592
    assert len(result) == 5989
    assert set(oof.split_partition) == {"train"}
    assert set(gbdt_val.split_partition) == {"val"}
    assert set(result.split_partition) == {"val"}
    assert oof.delivery_record_id.is_unique and result.delivery_record_id.is_unique
    assert oof.duplicated(["delivery_record_id", "courier_id", "split_partition"]).sum() == 0
    assert result.duplicated(["delivery_record_id", "courier_id", "split_partition"]).sum() == 0
    assert result.delivery_record_id.tolist() == gbdt_val.delivery_record_id.tolist()

    # The training target is exactly the OOF residual, never a fitted-row error.
    train_residual = oof.y_true_min.to_numpy(dtype=np.float64) - oof.gbdt_pred_min.to_numpy(dtype=np.float64)
    residual_hash = hashlib.sha256(train_residual.astype("<f8", copy=False).tobytes()).hexdigest()
    assert residual_hash == checkpoint["training_oof_residual_sha256"]
    assert hashlib.sha256("\n".join(sorted(oof.delivery_record_id.astype(str))).encode()).hexdigest() == checkpoint["training_record_ids_sha256"]

    roles = result.validation_role.to_numpy()
    select = roles == "model_selection"
    calibration = roles == "fusion_calibration"
    assert int(select.sum()) == 4124
    assert int(calibration.sum()) == 1865
    assert set(result.loc[select, "courier_id"]).isdisjoint(set(result.loc[calibration, "courier_id"]))
    role_manifest = json.loads((RECOVERY / "validation_role_assignment.json").read_text(encoding="utf-8"))
    assert role_manifest["assignment_sha256"] == checkpoint["validation_role_assignment_sha256"]
    assert dict(zip(result.delivery_record_id.astype(str), result.validation_role)) == role_manifest["record_id_to_role"]

    x_train = oof[FEATURES].to_numpy(dtype=np.float64)
    x_val = result[FEATURES].to_numpy(dtype=np.float64)
    y_val = result.y_true_min.to_numpy(dtype=np.float64)
    gbdt_pred = result.gbdt_pred_min.to_numpy(dtype=np.float64)
    trials = []
    for alpha in [0.01, 1.0, 100.0]:
        beta, intercept = _ridge(x_train, train_residual, alpha)
        combined = gbdt_pred + x_val @ beta + intercept
        trials.append((alpha, _metrics(y_val[select], combined[select])))
    expected_choice = min(trials, key=lambda item: (item[1]["mae_min"], item[0]))[0]
    assert expected_choice == checkpoint["alpha"]

    beta = np.asarray(checkpoint["coefficients_min_per_unit"], dtype=np.float64)
    correction = x_val @ beta + checkpoint["intercept_min"]
    reconstructed = gbdt_pred + correction
    np.testing.assert_array_equal(correction, result.residual_correction_min.to_numpy(dtype=np.float64))
    np.testing.assert_array_equal(reconstructed, result.ridge_eta_pred_min.to_numpy(dtype=np.float64))


def test_alpha_selection_metrics_match_model_selection_only() -> None:
    checkpoint = json.loads((ARTIFACTS / "residual_ridge.json").read_text(encoding="utf-8"))
    result = pd.read_parquet(ARTIFACTS / "validation_predictions.parquet", engine="fastparquet")
    select = result.validation_role.to_numpy() == "model_selection"
    calibration = result.validation_role.to_numpy() == "fusion_calibration"
    baseline = _metrics(result.y_true_min.to_numpy()[select], result.gbdt_pred_min.to_numpy()[select])
    chosen = _metrics(result.y_true_min.to_numpy()[select], result.ridge_eta_pred_min.to_numpy()[select])
    np.testing.assert_allclose(list(baseline.values()), list(checkpoint["selection_baseline_gbdt_only_metrics"].values()), rtol=0, atol=1e-14)
    np.testing.assert_allclose(list(chosen.values()), list(checkpoint["selection_selected_control_metrics"].values()), rtol=0, atol=1e-14)
    calibration_metrics = _metrics(result.y_true_min.to_numpy()[calibration], result.ridge_eta_pred_min.to_numpy()[calibration])
    np.testing.assert_allclose(list(calibration_metrics.values()), list(checkpoint["calibration_selected_control_metrics_diagnostic_only"].values()), rtol=0, atol=1e-14)
    assert checkpoint["validation_calibration_rows_not_used_for_choice"] == int(calibration.sum())
    assert checkpoint["test_partition"] == "not loaded into this control, not scored, and not serialized"
