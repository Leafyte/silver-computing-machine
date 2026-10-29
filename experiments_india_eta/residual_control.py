"""Bounded linear residual control on top of frozen India ETA OOF predictions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .recovery import VQC_FEATURES, _metrics


GBDT_MODEL_SHA256 = "f1f94fbcc932f3519aaafcff7dc99d65b7951c0240730e54c320259152bbebdf"
OOF_SHA256 = "78a5461c89387e8630421008b92dfffcc84864e9e25bea7476eefa2c8d48032f"
VALIDATION_SHA256 = "3d9485c513e6a997516baea1a4a6c497c6021240f9cbf64d524037e9e8753f1a"
ALPHAS = (0.01, 1.0, 100.0)
JOIN_KEYS = ["delivery_record_id", "courier_id", "split_partition"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Ridge with train-only feature/target centering and unpenalized intercept."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    centered_x = x - x_mean
    centered_y = y - y_mean
    beta = np.linalg.solve(centered_x.T @ centered_x + alpha * np.eye(x.shape[1]), centered_x.T @ centered_y)
    intercept = y_mean - float(x_mean @ beta)
    return beta, intercept


def _correction(x: np.ndarray, beta: np.ndarray, intercept: float) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) @ np.asarray(beta, dtype=np.float64) + float(intercept)


def run_control(root: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    artifact_dir = root / "artifacts" / "india_eta_recovery"
    out = Path(output_dir).resolve() if output_dir else root / "artifacts" / "india_eta_residual_control_v1"
    out.mkdir(parents=True, exist_ok=True)
    oof_path = artifact_dir / "oof_predictions.parquet"
    val_path = artifact_dir / "validation_predictions.parquet"
    gbdt_path = artifact_dir / "gbdt" / "india_eta_hgb_regressor.joblib"
    gbdt_manifest_path = artifact_dir / "recovery_manifest.json"
    role_map_path = artifact_dir / "validation_role_assignment.json"
    for path, expected in (
        (oof_path, OOF_SHA256),
        (val_path, VALIDATION_SHA256),
        (gbdt_path, GBDT_MODEL_SHA256),
    ):
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen GBDT input hash mismatch for {path.name}: {actual} != {expected}")

    gbdt_manifest = json.loads(gbdt_manifest_path.read_text(encoding="utf-8"))
    role_manifest = json.loads(role_map_path.read_text(encoding="utf-8"))
    provenance = gbdt_manifest["data_provenance"]
    if provenance["validation_role_assignment_sha256"] != role_manifest["assignment_sha256"]:
        raise ValueError("Validation role mapping differs from canonical GBDT recovery mapping")
    if provenance["prediction_join_key"] != ["ID", "Delivery_person_ID", "split_partition"]:
        raise ValueError("Frozen GBDT predictions do not record the required stable join key")
    if provenance["vqc_features"] != VQC_FEATURES:
        raise ValueError("Frozen prediction artifacts do not record the four expected VQC features")

    oof = pd.read_parquet(oof_path, engine="fastparquet")
    val = pd.read_parquet(val_path, engine="fastparquet")
    for frame, expected_split in ((oof, "train"), (val, "val")):
        if frame["delivery_record_id"].isna().any() or frame[JOIN_KEYS].isna().any().any():
            raise ValueError(f"Missing ID/courier/split key in {expected_split} prediction rows")
        if frame["delivery_record_id"].duplicated().any():
            raise ValueError(f"Duplicate record ID in {expected_split} predictions")
        if set(frame["split_partition"]) != {expected_split}:
            raise ValueError(f"Unexpected split labels in {expected_split} predictions")

    # The frozen GBDT recovery artifacts already contain the verified keyed join
    # to the committed GBDT/VQC views. Validate each composite key and role map
    # without reopening the source tables or touching sealed test values.
    for frame, expected_rows, split in ((oof, 29592, "train"), (val, 5989, "val")):
        if len(frame) != expected_rows or frame.duplicated(JOIN_KEYS).any():
            raise ValueError(f"Frozen {split} artifact does not have the required unique keyed row count")
        if any(feature not in frame.columns for feature in VQC_FEATURES):
            raise ValueError(f"Frozen {split} artifact is missing one of the VQC candidate features")
        values = frame[VQC_FEATURES + ["y_true_min", "gbdt_pred_min"]].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Frozen {split} artifact contains non-finite values")
    if set(oof["split_partition"]) != {"train"} or set(val["split_partition"]) != {"val"}:
        raise ValueError("Residual control inputs contain an unexpected split label")
    record_to_role = role_manifest.get("record_id_to_role", {})
    courier_to_role = role_manifest.get("courier_to_role", {})
    if set(record_to_role) != set(val.delivery_record_id.astype(str)):
        raise ValueError("Role map IDs do not equal the validation prediction IDs")
    if not (val.delivery_record_id.astype(str).map(record_to_role).to_numpy() == val.validation_role.astype(str).to_numpy()).all():
        raise ValueError("Validation row roles do not match the frozen ID-keyed role map")
    if not (val.courier_id.astype(str).map(courier_to_role).to_numpy() == val.validation_role.astype(str).to_numpy()).all():
        raise ValueError("Validation courier roles do not match the frozen courier role map")
    x_train = oof[VQC_FEATURES].to_numpy(dtype=np.float64)
    y_residual_train = oof["y_true_min"].to_numpy(dtype=np.float64) - oof["gbdt_pred_min"].to_numpy(dtype=np.float64)
    x_val = val[VQC_FEATURES].to_numpy(dtype=np.float64)
    y_val = val["y_true_min"].to_numpy(dtype=np.float64)
    gbdt_val = val["gbdt_pred_min"].to_numpy(dtype=np.float64)
    roles = val["validation_role"].astype(str).to_numpy()
    selection_mask = roles == "model_selection"
    calibration_mask = roles == "fusion_calibration"
    if selection_mask.sum() != 4124 or calibration_mask.sum() != 1865:
        raise ValueError("Validation role record counts differ from frozen split contract")
    if set(val.loc[selection_mask, "courier_id"]) & set(val.loc[calibration_mask, "courier_id"]):
        raise ValueError("Validation model-selection and calibration courier groups overlap")

    trials = []
    candidates = []
    for alpha in ALPHAS:
        beta, intercept = _ridge_fit(x_train, y_residual_train, alpha)
        corrections = _correction(x_val, beta, intercept)
        combined = gbdt_val + corrections
        trials.append({
            "alpha": alpha,
            "model_selection_metrics": _metrics(y_val[selection_mask], combined[selection_mask]),
            "model_selection_rows": int(selection_mask.sum()),
        })
        candidates.append((alpha, beta, intercept, corrections, combined))
    winner = min(range(len(trials)), key=lambda i: (trials[i]["model_selection_metrics"]["mae_min"], trials[i]["alpha"]))
    alpha, beta, intercept, corrections, combined = candidates[winner]

    # Calibration scores are reported only for later diagnostics; they never select alpha.
    baseline_metrics = _metrics(y_val[selection_mask], gbdt_val[selection_mask])
    selected_metrics = _metrics(y_val[selection_mask], combined[selection_mask])
    calibration_metrics = _metrics(y_val[calibration_mask], combined[calibration_mask])
    all_val_metrics = _metrics(y_val, combined)
    val_out = val[["delivery_record_id", "courier_id", "split_partition", "validation_role", *VQC_FEATURES, "y_true_min", "gbdt_pred_min"]].copy()
    val_out["residual_correction_min"] = corrections
    val_out["ridge_eta_pred_min"] = combined
    val_out["selected_alpha"] = float(alpha)
    val_out_path = out / "validation_predictions.parquet"
    val_out.to_parquet(val_out_path, index=False, engine="fastparquet")

    gbdt_manifest_sha = _sha256(gbdt_manifest_path)
    checkpoint = {
        "format": "india-eta-ridge-residual-control-v1",
        "status": "frozen_experimental_classical_control_not_operationally_activated",
        "estimator": "closed-form centered Ridge with unpenalized intercept",
        "target": "delivery_duration_min minus courier-held-out GBDT OOF prediction",
        "target_unit": "minutes",
        "feature_names": VQC_FEATURES,
        "feature_scaling": "Use the committed VQC candidate features as stored in [0,1]; no refit or additional scaling.",
        "alpha": float(alpha),
        "alpha_candidates": list(ALPHAS),
        "coefficients_min_per_unit": [float(x) for x in beta],
        "intercept_min": float(intercept),
        "training_rows": len(oof),
        "training_record_ids_sha256": hashlib.sha256("\n".join(sorted(oof.delivery_record_id.astype(str))).encode("utf-8")).hexdigest(),
        "training_courier_groups": int(oof.courier_id.nunique()),
        "training_oof_residual_sha256": hashlib.sha256(y_residual_train.astype("<f8", copy=False).tobytes()).hexdigest(),
        "join_keys": JOIN_KEYS,
        "gbdt_model_sha256": GBDT_MODEL_SHA256,
        "gbdt_recovery_manifest_sha256": gbdt_manifest_sha,
        "gbdt_oof_predictions_sha256": OOF_SHA256,
        "gbdt_validation_predictions_sha256": VALIDATION_SHA256,
        "validation_role_assignment_sha256": role_manifest["assignment_sha256"],
        "split_manifest_sha256": provenance["split_manifest_sha256"],
        "source_parquet_sha256": {
            "gbdt": provenance["files_sha256"]["gbdt_parquet"],
            "vqc": provenance["files_sha256"]["vqc_parquet"],
        },
        "validation_model_selection_couriers": 117,
        "validation_model_selection_rows": int(selection_mask.sum()),
        "validation_calibration_couriers": 58,
        "validation_calibration_rows_not_used_for_choice": int(calibration_mask.sum()),
        "selection_trials": trials,
        "selection_baseline_gbdt_only_metrics": baseline_metrics,
        "selection_selected_control_metrics": selected_metrics,
        "calibration_selected_control_metrics_diagnostic_only": calibration_metrics,
        "all_validation_selected_control_metrics_descriptive_only": all_val_metrics,
        "test_partition": "not loaded into this control, not scored, and not serialized",
        "activation_policy": "Experimental comparison control only; do not activate without separate Opus review and frozen quantum comparison.",
    }
    checkpoint_path = out / "residual_ridge.json"
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, allow_nan=False), encoding="utf-8")
    summary = {
        "status": checkpoint["status"],
        "checkpoint": str(checkpoint_path.relative_to(root)),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "validation_predictions": str(val_out_path.relative_to(root)),
        "validation_predictions_sha256": _sha256(val_out_path),
        "training_oof_rows": len(oof),
        "validation_rows": len(val),
        "alpha": alpha,
        "selection_trials": trials,
        "selection_baseline_gbdt_only_metrics": baseline_metrics,
        "selection_selected_control_metrics": selected_metrics,
        "calibration_selected_control_metrics_diagnostic_only": calibration_metrics,
        "all_validation_selected_control_metrics_descriptive_only": all_val_metrics,
        "gbdt_model_sha256": GBDT_MODEL_SHA256,
        "gbdt_oof_predictions_sha256": OOF_SHA256,
        "gbdt_validation_predictions_sha256": VALIDATION_SHA256,
        "validation_role_assignment_sha256": role_manifest["assignment_sha256"],
        "split_manifest_sha256": provenance["split_manifest_sha256"],
        "no_test_rows": True,
    }
    manifest_path = out / "recovery_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    report_path = root / "INDIA_ETA_RESIDUAL_CONTROL_REPORT.md"
    report_path.write_text(_render_report(summary, checkpoint), encoding="utf-8")
    return summary


def _render_report(summary: dict[str, Any], checkpoint: dict[str, Any]) -> str:
    baseline = summary["selection_baseline_gbdt_only_metrics"]
    chosen = summary["selection_selected_control_metrics"]
    calib = summary["calibration_selected_control_metrics_diagnostic_only"]
    lines = [
        "# India ETA classical residual control",
        "",
        "This is a small comparison control trained on courier-held-out GBDT OOF residuals. It is experimental and is not operationally activated.",
        "",
        "## Protocol",
        "",
        f"- Training target: `y_true_min - gbdt_pred_min` from {checkpoint['training_rows']:,} training-only OOF records; features are the four committed VQC candidate inputs in their stored [0,1] range.",
        f"- Stable join keys verified one-to-one against committed views: `{', '.join(JOIN_KEYS)}`.",
        f"- Ridge alpha candidates: `{list(ALPHAS)}`; centered features/target from OOF training only, unpenalized intercept; no feature scaling.",
        f"- Alpha selection used only the 4,124 model-selection validation rows from 117 couriers. The 1,865 fusion-calibration rows from 58 couriers did not choose alpha.",
        "- Test records were not loaded, scored, used for choice, or written to outputs.",
        "- Base checkpoint, OOF, validation, split, source and role-map hashes are recorded in `residual_ridge.json`.",
        "",
        "## Model-selection validation metrics (minutes)",
        "",
        "| Candidate | Alpha | MAE | RMSE | Median absolute error |",
        "|---|---:|---:|---:|---:|",
        f"| GBDT only | — | {baseline['mae_min']:.6f} | {baseline['rmse_min']:.6f} | {baseline['median_absolute_error_min']:.6f} |",
    ]
    for trial in checkpoint["selection_trials"]:
        m = trial["model_selection_metrics"]
        lines.append(f"| Ridge residual control | {trial['alpha']:g} | {m['mae_min']:.6f} | {m['rmse_min']:.6f} | {m['median_absolute_error_min']:.6f} |")
    lines.extend([
        f"| Selected alpha {checkpoint['alpha']:g} | {checkpoint['alpha']:g} | {chosen['mae_min']:.6f} | {chosen['rmse_min']:.6f} | {chosen['median_absolute_error_min']:.6f} |",
        "",
        "## Held-out calibration diagnostics (choice not made here)",
        "",
        f"Selected alpha {checkpoint['alpha']:g}: MAE {calib['mae_min']:.6f}, RMSE {calib['rmse_min']:.6f}, median absolute error {calib['median_absolute_error_min']:.6f} minutes. These rows remain reserved for later calibration/fusion review; no parameter was selected from them.",
        "",
        "## Files",
        "",
        f"- Coefficients/intercept checkpoint: `{summary['checkpoint']}` (SHA-256 `{summary['checkpoint_sha256']}`).",
        f"- Validation comparison predictions: `{summary['validation_predictions']}` ({summary['validation_rows']:,} rows; SHA-256 `{summary['validation_predictions_sha256']}`).",
        f"- Recovery manifest: `artifacts/india_eta_residual_control_v1/recovery_manifest.json`.",
        "",
        "No quantum code, QPU, test data, retraining of the GBDT, or operational fusion choice was used.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run_control(args.root, args.output_dir), indent=2))
