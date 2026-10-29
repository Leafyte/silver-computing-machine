"""Independent development-only audit for frozen RUN2 4Q weights.

The module intentionally never opens a test split or test prediction file. It
checks source/checkpoint hashes, group roles, development metrics, checkpoint
reload parity, local QpiAI parity and numerical rank on train-only features.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments_india_vqc4.circuit import (
    ANGLE_COUNT,
    FEATURE_NAMES,
    NumpyFourQubitResidualVQC,
    jacobian_rank_report,
    numpy_expectations,
    qpiai_expectations,
)
from experiments_india_vqc4.circuit import TorchFourQubitVQC


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "india_eta_vqc4_run2"
REPRO = ROOT / "artifacts" / "india_eta_vqc4_run2_repro_metrics"
RECOVERY = ROOT / "artifacts" / "india_eta_recovery"
OOF = RECOVERY / "oof_predictions.parquet"
VAL_INPUT = RECOVERY / "validation_predictions.parquet"
VAL_OUTPUT = RUN / "validation_predictions.parquet"
MANIFEST_PATH = RUN / "run_manifest.json"
PROTOCOL_PATH = ROOT / "experiments_india_vqc4" / "retrospective_test_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual.lower() != expected.lower():
        raise AssertionError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return actual


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {
        "mae_min": float(np.mean(np.abs(err))),
        "rmse_min": float(np.sqrt(np.mean(err * err))),
        "median_absolute_error_min": float(np.median(np.abs(err))),
    }


def _close_metrics(actual: dict[str, float], expected: dict[str, float], label: str) -> None:
    if set(actual) != {"mae_min", "rmse_min", "median_absolute_error_min"}:
        raise AssertionError(f"{label} metric fields are incomplete")
    for name, value in actual.items():
        if not math.isfinite(value) or abs(value - float(expected[name])) > 2e-5:
            raise AssertionError(f"{label}.{name} mismatch: recomputed={value}, manifest={expected[name]}")


def run_acceptance() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_models_selection_only_no_test_read":
        raise AssertionError("training manifest is not marked frozen and test-free")
    if manifest.get("test_partition") != "not read, scored, or used; final evaluation is separate after freeze":
        raise AssertionError("training manifest does not preserve test isolation status")
    manifest_sha = sha256(MANIFEST_PATH)
    hashes = manifest["source_hashes"]
    pinned_sources = {
        "gbdt_checkpoint": "f1f94fbcc932f3519aaafcff7dc99d65b7951c0240730e54c320259152bbebdf",
        "oof_predictions": "78a5461c89387e8630421008b92dfffcc84864e9e25bea7476eefa2c8d48032f",
        "validation_predictions": "3d9485c513e6a997516baea1a4a6c497c6021240f9cbf64d524037e9e8753f1a",
        "validation_role_assignment": "631dbb800d8759858ccbed05345d95549940782fa8c81390e9a4affeb0ba6425",
        "ridge_control": "5203fe9844a9721cfed1f26411a32bb0ebd872cafa8f64e661c73c2afc8189e2",
    }
    if hashes != pinned_sources:
        raise AssertionError(f"unexpected source artifact pins: {hashes}")
    _assert_hash(RECOVERY / "gbdt" / "india_eta_hgb_regressor.joblib", hashes["gbdt_checkpoint"], "frozen GBDT")
    _assert_hash(OOF, hashes["oof_predictions"], "frozen train OOF")
    _assert_hash(VAL_INPUT, hashes["validation_predictions"], "frozen validation source")
    recovery_manifest_path = RECOVERY / "recovery_manifest.json"
    _assert_hash(recovery_manifest_path, "abe44ec210312bb70e45659f0a286d5898ed46681d1005edb98b4bd4d8358520", "recovery manifest")
    recovery_manifest = json.loads(recovery_manifest_path.read_text(encoding="utf-8"))
    role_path = RECOVERY / "validation_role_assignment.json"
    role_file_hash = sha256(role_path)
    role_file_record = recovery_manifest.get("prediction_artifacts", {}).get("validation_role_assignment", {})
    if role_file_record.get("sha256") != role_file_hash:
        raise AssertionError("raw validation-role file hash does not match the recovery artifact inventory")
    role_payload = json.loads(role_path.read_text(encoding="utf-8"))
    canonical_map = json.dumps(role_payload.get("courier_to_role"), sort_keys=True,
                               separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if hashlib.sha256(canonical_map).hexdigest() != hashes["validation_role_assignment"]:
        raise AssertionError("canonical courier-to-role map hash does not match the training manifest")
    _assert_hash(ROOT / "artifacts" / "india_eta_residual_control_v1" / "residual_ridge.json", hashes["ridge_control"], "Ridge control")

    circuit_path = ROOT / "experiments_india_vqc4" / "circuit.py"
    trainer_path = ROOT / "experiments_india_vqc4" / "train.py"
    circuit_sha = _assert_hash(circuit_path, manifest["circuit_code_sha256"], "circuit source")
    trainer_sha = _assert_hash(trainer_path, manifest["training_code_sha256"], "trainer source")
    _assert_hash(RUN / "training_progress.jsonl", manifest["training_log_sha256"], "epoch/optimizer log")
    _assert_hash(VAL_OUTPUT, manifest["validation_predictions_sha256"], "frozen validation output")
    _assert_hash(Path(ROOT / manifest["backend_profile_path"]), manifest["backend_profile_sha256"], "CPU/CUDA profile")
    repro_path = REPRO / "reproduction_metrics.json"
    repro_sha = _assert_hash(repro_path, "d4b040187601229b9c472a0fbe990613b79242d6781959964297ea01ca0bd2bd", "deterministic resource reproduction")
    repro = json.loads(repro_path.read_text(encoding="utf-8"))
    if repro.get("status") != "both_frozen_checkpoints_reproduced_bitwise" or repro.get("test_partition_read") is not False:
        raise AssertionError("deterministic resource reproduction is not a test-free bitwise match")
    if repro.get("selection_or_model_tuning_performed") is not False:
        raise AssertionError("resource reproduction unexpectedly changed model selection")
    for name in ("baseline60", "extended76"):
        if repro["reproduced_checkpoint_sha256"][name] != manifest["stages"][name]["checkpoint_sha256"]:
            raise AssertionError(f"resource reproduction weights differ for {name}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("test_status", "").find("not pristine") < 0 or protocol.get("interpretation", {}).get("may_use_test_to_change_policy") is not False:
        raise AssertionError("retrospective test qualification is missing or allows test-driven tuning")
    protocol_sha = sha256(PROTOCOL_PATH)

    role_map = role_payload["courier_to_role"]
    ordered_couriers = sorted(
        role_map,
        key=lambda courier: (
            hashlib.sha256(f"india-eta-val-role-v1|20261029|{courier}".encode("utf-8")).digest(),
            courier,
        ),
    )
    calibration_count = len(ordered_couriers) // 3
    expected_roles = {courier: ("fusion_calibration" if i < calibration_count else "model_selection")
                      for i, courier in enumerate(ordered_couriers)}
    if role_map != expected_roles or len(role_map) != 175 or calibration_count != 58:
        raise AssertionError("frozen validation role map does not match its hash-ranked courier rule")

    # Only the precomputed train-OOF target columns are used for residual scale.
    oof = pd.read_parquet(OOF, columns=["y_true_min", "gbdt_pred_min"], engine="fastparquet")
    if len(oof) != 29592:
        raise AssertionError("unexpected OOF row count")
    train_residual = (oof["y_true_min"] - oof["gbdt_pred_min"]).to_numpy(dtype=np.float32)
    expected_scale = max(float(np.std(train_residual.astype(np.float64), ddof=0)), 1.0)
    if abs(expected_scale - float(manifest["residual_scale_minutes"])) > 1e-10:
        raise AssertionError("residual scale did not match the frozen train-OOF residual population std")
    del oof, train_residual

    val = pd.read_parquet(VAL_OUTPUT, engine="fastparquet")
    if len(val) != 5989 or not val["split_partition"].eq("val").all():
        raise AssertionError("validation output has wrong split or row count")
    keys = ["delivery_record_id", "courier_id", "split_partition"]
    if val.duplicated(keys).any() or val["delivery_record_id"].duplicated().any():
        raise AssertionError("validation output has duplicate composite or bare record keys")
    if set(val["courier_id"].unique()) != set(role_map):
        raise AssertionError("validation courier coverage differs from frozen role map")
    if any(role_map[courier] != role for courier, role in val[["courier_id", "validation_role"]].drop_duplicates().itertuples(index=False, name=None)):
        raise AssertionError("validation output changed a courier's frozen selection/calibration role")
    selection = val["validation_role"].eq("model_selection").to_numpy()
    calibration = val["validation_role"].eq("fusion_calibration").to_numpy()
    if int(selection.sum()) != 4124 or int(calibration.sum()) != 1865 or np.any(selection & calibration):
        raise AssertionError("validation role row counts or disjointness failed")
    x_val = val[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    if not np.isfinite(x_val).all() or np.any(x_val < 0.0) or np.any(x_val > 1.0):
        raise AssertionError("validation VQC feature contract failed")

    checkpoint_hashes = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    models: dict[str, NumpyFourQubitResidualVQC] = {}
    for name, variant, expected_params in (("baseline60", "baseline60", 65), ("extended76", "extended76", 81)):
        path = RUN / f"{name}_best.json"
        record = manifest["stages"][name]
        checkpoint_hashes[name] = _assert_hash(path, record["checkpoint_sha256"], f"{name} checkpoint")
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = NumpyFourQubitResidualVQC.from_payload(payload)
        if model.variant != variant or model.parameter_count != expected_params:
            raise AssertionError(f"{name} checkpoint architecture/count is invalid")
        if model.config.feature_names != tuple(FEATURE_NAMES):
            raise AssertionError(f"{name} feature order is invalid")
        if abs(model.residual_scale_minutes - expected_scale) > 1e-10:
            raise AssertionError(f"{name} residual scale differs from train-only scale")
        models[name] = model
        checkpoints[name] = payload
        y = val["y_true_min"].to_numpy(dtype=np.float64)
        base = val["gbdt_pred_min"].to_numpy(dtype=np.float64)
        correction = model.predict_residual_minutes(x_val)
        predicted = np.maximum(0.0, base + correction)
        saved = val[f"{name}_hybrid_eta_min"].to_numpy(dtype=np.float64)
        if not np.allclose(predicted, saved, rtol=0.0, atol=2e-5):
            raise AssertionError(f"{name} JSON checkpoint reload does not reproduce saved validation predictions")
        _close_metrics(_metrics(y[selection], predicted[selection]), manifest["selection_metrics"][name], f"selection {name}")
        _close_metrics(_metrics(y[calibration], predicted[calibration]), manifest["calibration_metrics_after_freeze"][name], f"calibration {name}")

    y_all = val["y_true_min"].to_numpy(dtype=np.float64)
    base_all = val["gbdt_pred_min"].to_numpy(dtype=np.float64)
    # Baseline and Ridge policy metrics are recomputed independently from the saved
    # GBDT predictions and the already frozen four-feature Ridge weights.
    _close_metrics(_metrics(y_all[selection], base_all[selection]), manifest["selection_metrics"]["gbdt_only"], "selection GBDT")
    _close_metrics(_metrics(y_all[calibration], base_all[calibration]), manifest["calibration_metrics_after_freeze"]["gbdt_only"], "calibration GBDT")
    ridge = json.loads((ROOT / "artifacts" / "india_eta_residual_control_v1" / "residual_ridge.json").read_text(encoding="utf-8"))
    qfeatures = val[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    ridge_correction = qfeatures @ np.asarray(ridge["coefficients_min_per_unit"], dtype=np.float64) + float(ridge["intercept_min"])
    ridge_pred = np.maximum(0.0, base_all + ridge_correction)
    _close_metrics(_metrics(y_all[selection], ridge_pred[selection]), manifest["selection_metrics"]["ridge_control"], "selection Ridge")
    _close_metrics(_metrics(y_all[calibration], ridge_pred[calibration]), manifest["calibration_metrics_after_freeze"]["ridge_control"], "calibration Ridge")

    # Independently verify that the predeclared deployment rule yields alpha=0.
    base_m = manifest["selection_metrics"]["gbdt_only"]
    ridge_m = manifest["selection_metrics"]["ridge_control"]
    candidate_metrics = {name: manifest["selection_metrics"][name] for name in models}
    passing = [name for name, m in candidate_metrics.items()
               if m["mae_min"] < base_m["mae_min"] and m["mae_min"] < ridge_m["mae_min"]
               and m["rmse_min"] <= base_m["rmse_min"] and m["rmse_min"] <= ridge_m["rmse_min"]]
    expected_choice = passing[0] if passing else "alpha0_gbdt_only"
    expected_alpha = 1.0 if passing else 0.0
    if manifest.get("selection_choice") != expected_choice or float(manifest.get("selected_alpha")) != expected_alpha:
        raise AssertionError("frozen operational policy does not match the predeclared dev-only gate")

    # Frozen checkpoint rank/sensitivity on train-only features (no outcomes).
    rank_frame = pd.read_parquet(OOF, columns=list(FEATURE_NAMES), engine="fastparquet")
    rows = rank_frame.iloc[np.linspace(0, len(rank_frame) - 1, 64, dtype=int)].to_numpy(dtype=np.float64)
    del rank_frame
    base_angles = np.asarray(checkpoints["baseline60"]["quantum_angles"], dtype=np.float64)
    ext_angles = np.asarray(checkpoints["extended76"]["quantum_angles"], dtype=np.float64)
    fitted_rank = jacobian_rank_report(rows, base_angles, extended_angles=ext_angles,
                                       step=1e-5, relative_tolerance=1e-8)

    # Numerical and actual SDK parity on eight validation feature rows only.
    parity_x = val.iloc[np.linspace(0, len(val) - 1, 8, dtype=int)][list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    sdk = {}
    import torch
    torch_checks = {}
    with torch.no_grad():
        torch_rows = torch.as_tensor(parity_x, dtype=torch.float32, device="cpu")
        for name, payload in checkpoints.items():
            torch_model = TorchFourQubitVQC.from_payload(payload, device="cpu").eval()
            torch_prediction = torch_model(torch_rows).detach().cpu().numpy().astype(np.float64)
            numpy_prediction = models[name].predict_residual_minutes(parity_x)
            error = float(np.max(np.abs(torch_prediction - numpy_prediction)))
            if error > 2e-5:
                raise AssertionError(f"{name} trained checkpoint Torch/NumPy reload parity failed: {error}")
            torch_checks[name] = {"rows": 8, "max_abs_residual_minutes_error": error,
                                  "runtime": torch.__version__, "device": "CPU"}
    for name, model in models.items():
        numpy_z = numpy_expectations(parity_x, model.quantum_angles, model.variant)
        sdk_z = qpiai_expectations(parity_x, model.quantum_angles, model.variant)
        zerr = float(np.max(np.abs(numpy_z - sdk_z)))
        numpy_resid = model.predict_residual_minutes(parity_x)
        sdk_resid = model.residual_scale_minutes * (sdk_z @ model.readout[:4] + model.readout[-1])
        rerr = float(np.max(np.abs(numpy_resid - sdk_resid)))
        if zerr > 2e-12 or rerr > 2e-12:
            raise AssertionError(f"{name} actual checkpoint QpiAI parity failed")
        sdk[name] = {"rows": 8, "max_abs_z_error": zerr, "max_abs_residual_minutes_error": rerr,
                     "backend": "local QpiAI StatevectorSimulator; shots=1 exact statevector; no cloud/QPU"}

    progress = [json.loads(line) for line in (RUN / "training_progress.jsonl").read_text(encoding="utf-8").splitlines() if line]
    training = {}
    for name in ("baseline60", "extended76"):
        entries = [row for row in progress if row["stage"] == name]
        best = min(entries, key=lambda row: row["selection"]["mae_min"])
        stage = manifest["stages"][name]
        if len(entries) != stage["epochs_run"] or sum(row["optimizer_steps"] for row in entries) != stage["optimizer_steps"]:
            raise AssertionError(f"{name} actual update/epoch counts differ from run manifest")
        if best["epoch"] != stage["best_epoch"]:
            raise AssertionError(f"{name} best epoch does not match validation log")
        training[name] = {
            "epochs_completed": len(entries),
            "best_epoch": best["epoch"],
            "optimizer_updates": sum(row["optimizer_steps"] for row in entries),
            "sum_epoch_seconds": float(sum(row["epoch_seconds"] for row in entries)),
            "best_logged_selection_mae_min": float(best["selection"]["mae_min"]),
        }

    report = {
        "format": "india_eta_vqc4_independent_frozen_acceptance_v1",
        "status": "PASS_DEVELOPMENT_CHECKS_TEST_TARGETS_NOT_ACCESSED_FEATURE_ACCESS_INCIDENT",
        "run_manifest_sha256": manifest_sha,
        "retrospective_test_protocol_sha256": protocol_sha,
        "source_hashes": hashes,
        "raw_role_file_sha256": role_file_hash,
        "code_hashes": {"circuit.py": circuit_sha, "train.py": trainer_sha},
        "checkpoint_hashes": checkpoint_hashes,
        "training": training,
        "deterministic_resource_reproduction": {
            "path": "artifacts/india_eta_vqc4_run2_repro_metrics/reproduction_metrics.json",
            "sha256": repro_sha,
            "checkpoint_hashes_bitwise_equal_to_frozen_run": True,
            "optimizer_updates": repro["total_optimizer_steps"],
            "wall_seconds": repro["wall_seconds"],
            "cpu_process_seconds": repro["cpu_process_seconds"],
            "windows_peak_working_set_bytes": repro["process_peak_rss_bytes"],
            "forward_calls": repro["forward_counts"]["total"]["forward_calls"],
            "forward_input_rows": repro["forward_counts"]["total"]["forward_input_rows"],
            "statevector_amplitude_elements": repro["forward_counts"]["total"]["statevector_amplitude_elements"],
            "extension_gradient_l2_min_mean_max": [
                repro["gradient_stats"]["extended76"]["extension16_gradient_l2_by_step_min"],
                repro["gradient_stats"]["extended76"]["extension16_gradient_l2_by_step_mean"],
                repro["gradient_stats"]["extended76"]["extension16_gradient_l2_by_step_max"],
            ],
            "calibration_or_test_read": False,
        },
        "roles": {"model_selection_couriers": 117, "selection_rows": 4124,
                  "fusion_calibration_couriers": 58, "calibration_rows": 1865,
                  "courier_group_disjoint": True, "role_map_independently_recomputed": True},
        "input_feature_contract": {
            "feature_order": list(FEATURE_NAMES),
            "selection_and_validation_rows_are_finite_and_in_0_1": True,
            "stable_record_key_unique_in_validation": True,
            "raw_59_to_4_cross_source_formula_join": "not accepted in this audit; fastparquet filters returned all split rows, so no test labels were requested and the cross-source check was dropped",
        },
        "residual_scale_minutes": expected_scale,
        "selection_metrics_recomputed": manifest["selection_metrics"],
        "calibration_metrics_recomputed_after_choice": manifest["calibration_metrics_after_freeze"],
        "selection_choice": expected_choice,
        "selected_alpha": expected_alpha,
        "fitted_weight_jacobian": fitted_rank,
        "actual_checkpoint_torch_parity": torch_checks,
        "actual_checkpoint_qpiai_parity": sdk,
        "test_partition_feature_materialization_incident": {
            "occurred": True,
            "reason": "fastparquet pandas filters silently returned every partition during a feature-only cross-source check",
            "test_feature_rows_materialized": 5941,
            "target_or_synthetic_label_columns_requested_or_read": False,
            "test_features_used_for_model_selection_or_metrics": False,
        },
        "test_target_columns_requested_or_read_by_this_audit": False,
        "prior_test_exposure_disclosure": "The original test partition was previously accessed and scored in the prior RUN2 two-qubit workflow; any new 4Q score is retrospective, not a pristine holdout result.",
    }
    out = RUN / "independent_frozen_acceptance.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run_acceptance()
    print(json.dumps({k: v for k, v in result.items() if k not in ("fitted_weight_jacobian",)}, indent=2, sort_keys=True))
