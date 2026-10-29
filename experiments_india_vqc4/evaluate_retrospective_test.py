"""One-shot retrospective RUN2 4Q evaluation after an explicit freeze record.

The original India test partition was previously opened by a separate 2Q run;
this script is therefore descriptive only and can never select a checkpoint or
policy. The scorer writes a start sentinel before opening source rows and
refuses a second invocation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "india_eta_vqc4_run2"
RELEASE = ROOT / "artifacts" / "run2_india" / "india_eta_vqc4_cpu_release_v1"
RELEASE_ZIP = ROOT / "artifacts" / "run2_india" / "india_eta_vqc4_cpu_release_v1.zip"
DATA_GBDT = ROOT / "data" / "indian_delivery_gbdt.parquet"
SPLIT_MANIFEST = ROOT / "data" / "split_manifest.json"
PRETEST_FREEZE = RUN / "PRETEST_FREEZE.json"
START_SENTINEL = RUN / "TEST-EVALUATION-STARTED-vqc4.json"
OUTPUT_PREDICTIONS = RUN / "retrospective_test_predictions.parquet"
OUTPUT_METRICS = RUN / "retrospective_test_metrics.json"
OUTPUT_COMPLETE = RUN / "TEST-EVALUATION-COMPLETE-vqc4.json"

EXPECTED_TEST_ROWS = 5941
EXPECTED_TEST_COURIERS = 176
EXPECTED_GBDT_TABLE_SHA = "bd03b47c3d3e23050329cefc69d752aad607031cacdb9696151d237e3534a627"
EXPECTED_SPLIT_SHA = "10bee51eebac51bf68813a35a42b06adeb2e30588accdc582bbe9b0d8136c1bb"
EXPECTED_ZIP_SHA = "18d5d4859e53d3d59063e60cdcdd45cb6eff6757bba6474f024cdde91aa06b80"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_values(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return {
        "MAE_minutes": float(np.mean(np.abs(error))),
        "RMSE_minutes": float(np.sqrt(np.mean(error * error))),
        "median_absolute_error_minutes": float(np.median(np.abs(error))),
    }


def _verify_pretest_freeze() -> tuple[dict[str, Any], str]:
    freeze = json.loads(PRETEST_FREEZE.read_text(encoding="utf-8"))
    if freeze.get("format") != "india_eta_vqc4_pretest_freeze_v1" or freeze.get("status") != "FROZEN_BEFORE_RETROSPECTIVE_TEST_SCORE":
        raise RuntimeError("pretest freeze record missing or invalid")
    if freeze.get("test_qualification", {}).get("pristine_or_unseen") is not False:
        raise RuntimeError("the original test partition must be explicitly marked previously exposed")
    if freeze.get("fusion_policy", {}).get("alpha") != 0.0 or freeze["fusion_policy"].get("selected_variant") is not None:
        raise RuntimeError("operational fusion policy is not the frozen alpha=0 fallback")
    expected_files = freeze["hashes"]
    checks = {
        "run_manifest": RUN / "run_manifest.json",
        "independent_acceptance": RUN / "independent_frozen_acceptance.json",
        "protocol": ROOT / "experiments_india_vqc4" / "retrospective_test_protocol.json",
        "circuit_code": ROOT / "experiments_india_vqc4" / "circuit.py",
        "trainer_code": ROOT / "experiments_india_vqc4" / "train.py",
        "scorer_code": ROOT / "experiments_india_vqc4" / "evaluate_retrospective_test.py",
        "portable_code": ROOT / "experiments_india_vqc4" / "portable.py",
        "package_builder_code": ROOT / "experiments_india_vqc4" / "package_release.py",
        "runtime_status": RUN / "runtime_status.json",
        "reproduction_metrics": ROOT / "artifacts" / "india_eta_vqc4_run2_repro_metrics" / "reproduction_metrics.json",
        "baseline60_checkpoint": RUN / "baseline60_best.json",
        "extended76_checkpoint": RUN / "extended76_best.json",
        "release_manifest": RELEASE / "release_manifest.json",
        "release_archive": RELEASE_ZIP,
    }
    if set(expected_files) != set(checks):
        raise RuntimeError("pretest freeze hash inventory is incomplete or contains unrecognized files")
    actual = {key: sha256(path) for key, path in checks.items()}
    if actual != expected_files:
        mismatch = {key: {"actual": actual[key], "frozen": expected_files[key]}
                    for key in actual if actual[key] != expected_files.get(key)}
        raise RuntimeError(f"a frozen source/model/policy/release artifact changed after freeze: {mismatch}")
    if actual["release_archive"] != EXPECTED_ZIP_SHA:
        raise RuntimeError("portable archive hash differs from the independently verified release")
    return freeze, sha256(PRETEST_FREEZE)


def evaluate_once() -> dict[str, Any]:
    if START_SENTINEL.exists() or OUTPUT_METRICS.exists() or OUTPUT_PREDICTIONS.exists() or OUTPUT_COMPLETE.exists():
        raise RuntimeError("a vqc4 retrospective test attempt already exists; refusing to rescore")
    freeze, freeze_sha = _verify_pretest_freeze()
    # A durable start record binds the single evaluation attempt to the exact
    # pretest state. It is written before any source split/target rows are read.
    start_record = {
        "format": "india_eta_vqc4_test_evaluation_start_v1",
        "status": "STARTED_NO_SELECTION_ALLOWED",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pretest_freeze_sha256": freeze_sha,
        "release_archive_sha256": EXPECTED_ZIP_SHA,
        "qualification": "previously exposed test partition; retrospective only",
        "test_target_read": False,
    }
    START_SENTINEL.write_text(json.dumps(start_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Load the already-verified portable runtime. No fit, selection, or fusion
    # update is possible in this module.
    spec = importlib.util.spec_from_file_location("india_vqc4_frozen_release_predictor", RELEASE / "predictor.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("portable release predictor cannot be imported")
    import sys
    if str(RELEASE) not in sys.path:
        sys.path.insert(0, str(RELEASE))
    package_predictor = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package_predictor
    spec.loader.exec_module(package_predictor)
    runtime = package_predictor.load_release(RELEASE)
    schema = json.loads((RELEASE / "feature_schema.json").read_text(encoding="utf-8"))
    features = schema["feature_names"]
    requested = ["ID", "Delivery_person_ID", "split_partition", "delivery_duration_min", *features]
    source_sha = sha256(DATA_GBDT)
    split_sha = sha256(SPLIT_MANIFEST)
    if source_sha != EXPECTED_GBDT_TABLE_SHA or split_sha != EXPECTED_SPLIT_SHA:
        raise RuntimeError("test source or original split manifest hash mismatch")

    # This Parquet table is one row group and fastparquet's row filter is not
    # applied within it. The scorer records the actual materialized row count,
    # then immediately selects only split_partition == test for predictions and
    # metric calculations; non-test targets are never included in scores.
    source = pd.read_parquet(
        DATA_GBDT,
        columns=requested,
        filters=[("split_partition", "==", "test")],
        engine="fastparquet",
    )
    rows_materialized = int(len(source))
    test = source.loc[source["split_partition"].eq("test"), requested].copy()
    del source
    if len(test) != EXPECTED_TEST_ROWS:
        raise RuntimeError(f"expected {EXPECTED_TEST_ROWS} test records, found {len(test)}")
    test = test.rename(columns={"ID": "delivery_record_id", "Delivery_person_ID": "courier_id"})
    test = test.sort_values("delivery_record_id", kind="mergesort").reset_index(drop=True)
    keys = ["delivery_record_id", "courier_id", "split_partition"]
    if not test["split_partition"].eq("test").all() or test.duplicated(keys).any() or test["delivery_record_id"].duplicated().any():
        raise RuntimeError("test record keys/split are invalid")
    if test["courier_id"].nunique() != EXPECTED_TEST_COURIERS:
        raise RuntimeError("unexpected number of test courier groups")
    matrix = test[features].to_numpy(dtype=np.float64)
    truth = test["delivery_duration_min"].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all() or not np.isfinite(truth).all():
        raise RuntimeError("test input/target contains non-finite values")

    # Compute each frozen pipeline once on the same keyed rows, timing only
    # inference and excluding source data load/metric calculation.
    mapping_start = time.perf_counter()
    quantum_features = runtime._vqc_features(matrix)
    mapping_seconds = time.perf_counter() - mapping_start

    baseline_start = time.perf_counter()
    baseline_raw = np.asarray(runtime.gbdt.predict(matrix), dtype=np.float64)
    baseline_seconds = time.perf_counter() - baseline_start
    if baseline_raw.shape != truth.shape or not np.isfinite(baseline_raw).all():
        raise RuntimeError("frozen GBDT returned invalid predictions")
    baseline = np.maximum(0.0, baseline_raw)

    ridge_start = time.perf_counter()
    ridge_correction = quantum_features @ np.asarray(runtime.ridge["coefficients_min_per_unit"], dtype=np.float64)
    ridge_correction = ridge_correction + float(runtime.ridge["intercept_min"])
    ridge_prediction = np.maximum(0.0, baseline_raw + ridge_correction)
    ridge_increment_seconds = time.perf_counter() - ridge_start

    vqc_predictions: dict[str, np.ndarray] = {}
    vqc_seconds: dict[str, float] = {}
    for name in ("baseline60", "extended76"):
        started = time.perf_counter()
        residual = np.asarray(runtime.models[name].predict_residual_minutes(quantum_features), dtype=np.float64)
        vqc_predictions[name] = np.maximum(0.0, baseline_raw + residual)
        vqc_seconds[name] = time.perf_counter() - started
        if residual.shape != truth.shape or not np.isfinite(vqc_predictions[name]).all():
            raise RuntimeError(f"{name} returned invalid residual predictions")

    alpha = float(runtime.fusion["alpha"])
    selected_variant = runtime.fusion.get("selected_variant")
    if alpha != 0.0 or selected_variant is not None:
        raise RuntimeError("development-frozen policy changed; stop before test scoring")
    operational = np.maximum(0.0, baseline_raw)

    predictions = {
        "gbdt_only": baseline,
        "ridge_control": ridge_prediction,
        "baseline60_candidate": vqc_predictions["baseline60"],
        "extended76_candidate": vqc_predictions["extended76"],
        "operational_alpha0": operational,
    }
    if not np.array_equal(operational, np.maximum(0.0, baseline)):
        raise AssertionError("alpha=0 operational policy did not fall back to the baseline")
    metrics = {name: _metric_values(truth, pred) for name, pred in predictions.items()}
    prediction_frame = test[keys].copy()
    prediction_frame["y_true_min"] = truth
    for name, pred in predictions.items():
        prediction_frame[f"{name}_pred_min"] = pred
    prediction_frame.to_parquet(OUTPUT_PREDICTIONS, index=False, engine="fastparquet")

    total_times = {
        "gbdt_only": baseline_seconds,
        "ridge_control": baseline_seconds + mapping_seconds + ridge_increment_seconds,
        "baseline60_candidate": baseline_seconds + mapping_seconds + vqc_seconds["baseline60"],
        "extended76_candidate": baseline_seconds + mapping_seconds + vqc_seconds["extended76"],
        "operational_alpha0": baseline_seconds,
    }
    report = {
        "format": "india_eta_vqc4_retrospective_test_metrics_v1",
        "status": "COMPLETED_RETROSPECTIVE_TEST_EVALUATION_NO_SELECTION",
        "qualification": "The original test partition was previously accessed/scored in a RUN2 2-qubit evaluation. The 4Q acceptance audit also materialized test feature columns during a row-filter failure but did not read target or synthetic-label columns. These results are retrospective, not a pristine or unseen holdout.",
        "pretest_freeze_sha256": freeze_sha,
        "release_archive_sha256": EXPECTED_ZIP_SHA,
        "source_hashes": {"gbdt_table": source_sha, "split_manifest": split_sha},
        "partition": "test",
        "rows": len(test),
        "unique_record_ids": int(test["delivery_record_id"].nunique()),
        "courier_groups": int(test["courier_id"].nunique()),
        "row_group_behavior": {"source_rows_materialized_by_fastparquet": rows_materialized,
                                "test_rows_used_for_prediction_and_metrics": len(test),
                                "non_test_targets_used_for_scoring": False,
                                "why_materialized_count_exceeds_test_count": "fastparquet ignored row-level filters for the single mixed-split row group; records were filtered in memory before inference/scoring"},
        "target": "delivery_duration_min",
        "unit": "minutes",
        "metrics": metrics,
        "inference": {
            "device": "CPU; portable NumPy statevector and frozen HistGradientBoostingRegressor",
            "test_records_per_pipeline": len(test),
            "python_batch_evaluations": 1,
            "row_level_model_calls": {
                "GBDT": {"calls": 1, "rows": len(test)},
                "Ridge residual": {"calls": 1, "rows": len(test)},
                "baseline60 candidate": {"calls": 1, "rows": len(test)},
                "extended76 candidate": {"calls": 1, "rows": len(test)},
                "operational alpha=0": {"additional_VQC_calls": 0, "rows": len(test)},
            },
            "elapsed_seconds": total_times,
            "vqc_residual_forward_seconds": vqc_seconds,
            "shared_feature_mapping_seconds": mapping_seconds,
            "shared_gbdt_seconds": baseline_seconds,
            "ridge_correction_increment_seconds": ridge_increment_seconds,
            "latency_convention": "one un-repeated batch call per candidate; excludes input parsing and metrics; shared GBDT/feature mapping is included in each hybrid pipeline's total",
        },
        "policy": {"selected_variant": None, "alpha": 0.0, "choice_source": "pretest validation-only freeze", "test_used_for_selection": False},
        "prediction_artifact": {"path": str(OUTPUT_PREDICTIONS.relative_to(ROOT)), "sha256": sha256(OUTPUT_PREDICTIONS), "rows": len(prediction_frame)},
        "started_utc": start_record["started_utc"],
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT_METRICS.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    complete = {
        "format": "india_eta_vqc4_test_evaluation_complete_v1",
        "status": report["status"],
        "pretest_freeze_sha256": freeze_sha,
        "metrics_sha256": sha256(OUTPUT_METRICS),
        "predictions_sha256": sha256(OUTPUT_PREDICTIONS),
        "test_target_read": True,
        "test_score_changed_model_or_policy": False,
        "completed_utc": report["finished_utc"],
    }
    OUTPUT_COMPLETE.write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    start_record["status"] = "COMPLETED_NO_SELECTION"
    start_record["test_target_read"] = True
    start_record["metrics_sha256"] = complete["metrics_sha256"]
    start_record["predictions_sha256"] = complete["predictions_sha256"]
    START_SENTINEL.write_text(json.dumps(start_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate_once(), indent=2, sort_keys=True, allow_nan=False))
