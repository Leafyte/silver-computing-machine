"""Freeze and evaluate the Luna RUN2 India residual model release once."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from experiments_india_qml.quantum import QuantumResidualRegressor, hybrid_predict_minutes


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "luna_hgbr_recovery_20261029T131809Z_v5"
ARTIFACTS = ROOT / "artifacts" / "run2_india" / RUN_ID
RUNTIME = ROOT / ".runtime" / "run2" / RUN_ID
MANIFEST = ARTIFACTS / "recovered_gbdt_manifest.json"
SELECTION = ARTIFACTS / "residual_selection.json"
DEVELOPMENT = RUNTIME / "classical_development_predictions.parquet"
RELEASE = ARTIFACTS / "release_manifest.json"
RIDGE = ARTIFACTS / "ridge_residual_control.npz"
TEST_START = RUNTIME / "sealed_test_evaluation_started.json"
TEST_RETRY_START = RUNTIME / "sealed_test_evaluation_retry.json"
FAILED_ATTEMPT = ARTIFACTS / "sealed_test_attempt1_aborted.json"
TEST_OUTPUT = RUNTIME / "final_test_predictions.parquet"
TEST_METRICS = ARTIFACTS / "final_test_metrics.json"
BUNDLE = ARTIFACTS / "portable_bundle"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    e = np.asarray(prediction, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {
        "mae_minutes": float(np.mean(np.abs(e))),
        "rmse_minutes": float(np.sqrt(np.mean(e * e))),
        "median_absolute_error_minutes": float(np.median(np.abs(e))),
    }


def _build_portable_bundle(
    baseline_path: Path,
    candidates: dict[str, Any],
    feature_names: list[str],
    scaler_path: Path,
    selected_arch: str,
) -> dict[str, Any]:
    if BUNDLE.exists():
        raise FileExistsError(f"portable bundle already exists: {BUNDLE}")
    package = BUNDLE / "experiments_india_qml"
    package.mkdir(parents=True)
    shutil.copy2(ROOT / "experiments_india_qml" / "quantum.py", package / "quantum.py")
    (package / "__init__.py").write_text("\"\"\"Portable India residual inference package.\"\"\"\n", encoding="utf-8")
    shutil.copy2(ROOT / "experiments_india_qml" / "portable_inference_v5.py", BUNDLE / "inference.py")
    shutil.copy2(baseline_path, BUNDLE / "recovered_eta_gbdt.joblib")
    shutil.copy2(scaler_path, BUNDLE / "quantum_feature_scaler.json")
    shutil.copy2(ROOT / "data" / "feature_schema.json", BUNDLE / "feature_schema.json")
    shutil.copy2(ROOT / "data" / "preprocessing_artifacts.json", BUNDLE / "preprocessing_artifacts.json")
    candidate_files: dict[str, Any] = {}
    for arch, item in candidates.items():
        source = _path(item["checkpoint_path"])
        name = f"quantum_{arch}.npz"
        shutil.copy2(source, BUNDLE / name)
        candidate_files[arch] = {"path": name, "sha256": sha256(BUNDLE / name)}
    bundle_manifest = {
        "schema_version": "run2-india-portable-inference-v1",
        "target_unit": "minutes",
        "gbdt_feature_order": feature_names,
        "q_feature_order": ["vqc_x0_distance", "vqc_x1_congestion"],
        "q_feature_scaler_file": "quantum_feature_scaler.json",
        "operational_alpha": 0.0,
        "selected_candidate_architecture": selected_arch,
        "validation_selected_quantum_candidate": {
            "architecture": selected_arch,
            "alpha": float(candidates[selected_arch]["alpha"]),
            "gate_feature": candidates[selected_arch].get("gate_feature"),
            "gate_operator": candidates[selected_arch]["gate_operator"],
            "gate_threshold": candidates[selected_arch].get("gate_threshold"),
        },
        "files": {
            "recovered_gbdt": {"path": "recovered_eta_gbdt.joblib", "sha256": sha256(BUNDLE / "recovered_eta_gbdt.joblib")},
            "quantum_candidates": candidate_files,
            "quantum_feature_scaler": {"path": "quantum_feature_scaler.json", "sha256": sha256(BUNDLE / "quantum_feature_scaler.json")},
            "feature_schema": {"path": "feature_schema.json", "sha256": sha256(BUNDLE / "feature_schema.json")},
            "preprocessing_artifacts": {"path": "preprocessing_artifacts.json", "sha256": sha256(BUNDLE / "preprocessing_artifacts.json")},
            "loader": {"path": "inference.py", "sha256": sha256(BUNDLE / "inference.py")},
            "quantum_module": {"path": "experiments_india_qml/quantum.py", "sha256": sha256(package / "quantum.py")},
        },
        "runtime": {"python": "3.12", "numpy": "1.26.4", "pandas": "2.2.3", "scikit_learn": "1.4.2", "joblib": "1.4.2"},
        "input_contract": "Supply the transformed GBDT features in gbdt_feature_order and the two [0,1] Q features in q_feature_order. The bundle does not include the source dataset.",
    }
    (BUNDLE / "bundle_manifest.json").write_text(json.dumps(bundle_manifest, indent=2) + "\n", encoding="utf-8")
    rng = np.random.default_rng(7341)
    fixture_x = rng.uniform(0.0, 1.0, (8, len(feature_names))).astype(np.float32)
    fixture_q = rng.uniform(0.0, 1.0, (8, 2)).astype(np.float64)
    np.savez_compressed(BUNDLE / "portable_smoke_fixture.npz", gbdt_features=fixture_x, q_features=fixture_q)
    # Store expected predictions only, never labels. This is a synthetic API fixture.
    sys.path.insert(0, str(BUNDLE))
    try:
        from inference import load_bundle, predict_minutes
        loaded = load_bundle(BUNDLE)
        expected = {
            mode: predict_minutes(loaded, fixture_x, fixture_q, mode=mode)["eta_minutes"]
            for mode in ("operational_release", "candidate_hybrid", "candidate_selective")
        }
    finally:
        sys.path.remove(str(BUNDLE))
        sys.modules.pop("inference", None)
    np.savez_compressed(BUNDLE / "portable_smoke_expected.npz", **expected)
    (BUNDLE / "README.md").write_text(
        "# RUN2 India portable inference bundle\n\n"
        "This bundle contains the recovered GBDT checkpoint, both trained two-qubit residual checkpoints, the train-only Q feature scaler, the feature schema and preprocessing artifact, and an inference loader. It contains no source dataset.\n\n"
        "Install the runtime versions in `bundle_manifest.json`. Add this folder to `PYTHONPATH`, then use `from inference import load_bundle, predict_minutes`. The operational policy returns the GBDT fallback (`alpha=0`); candidate modes are retained for scientific comparison. Inputs must follow the exact feature orders in `bundle_manifest.json`.\n\n"
        "Run `python portable_smoke.py` from this directory to verify portable checkpoint loading and deterministic API parity. The fixture is synthetic and has no labels.\n",
        encoding="utf-8",
    )
    (BUNDLE / "portable_smoke.py").write_text(
        "import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parent))\n"
        "import numpy as np\nfrom inference import load_bundle, predict_minutes\n"
        "bundle=load_bundle('.')\n"
        "with np.load('portable_smoke_fixture.npz',allow_pickle=False) as x, np.load('portable_smoke_expected.npz',allow_pickle=False) as e:\n"
        "    for mode in ('operational_release','candidate_hybrid','candidate_selective'):\n"
        "        got=predict_minutes(bundle,x['gbdt_features'],x['q_features'],mode=mode)['eta_minutes']\n"
        "        assert np.allclose(got,e[mode],rtol=0,atol=1e-12),mode\n"
        "print('portable inference smoke passed')\n",
        encoding="utf-8",
    )
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="portable-smoke-", dir=RUNTIME) as temporary:
        moved_bundle = Path(temporary) / "bundle"
        shutil.copytree(BUNDLE, moved_bundle)
        subprocess.run(
            [sys.executable, "-I", str(moved_bundle / "portable_smoke.py")],
            cwd=moved_bundle, check=True, capture_output=True, text=True,
        )
    files = {}
    for path in sorted(BUNDLE.rglob("*")):
        if path.is_file() and path.name != "bundle_manifest.json":
            files[str(path.relative_to(BUNDLE)).replace("\\", "/")] = sha256(path)
    return {
        "root": str(BUNDLE.relative_to(ROOT)).replace("\\", "/"),
        "files_sha256": files,
        "bundle_manifest_sha256": sha256(BUNDLE / "bundle_manifest.json"),
        "file_count": len(files),
        "move_to_temporary_directory_smoke": "passed",
    }


def _path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def prepare_release() -> dict[str, Any]:
    """Freeze source hashes, selected models, gate, and OOF-trained ridge control."""
    if RELEASE.exists() or TEST_START.exists() or TEST_OUTPUT.exists() or TEST_METRICS.exists():
        raise FileExistsError("release/test artifacts already exist; this run is immutable")
    import pandas as pd

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8-sig"))
    if selection.get("test_evaluation") != "not run; only pre-freeze train-OOF and original validation rows were read":
        raise ValueError("selection artifact must explicitly state the sealed test was not evaluated")
    if selection.get("development_table_sha256") != sha256(DEVELOPMENT):
        raise ValueError("selected candidate does not reference the v5 development table hash")
    if manifest.get("crossfit", {}).get("contains_test_rows") is not False:
        raise ValueError("development manifest does not exclude test rows")
    if selection.get("selected_release_gate") is None:
        raise ValueError("no validation-selected release gate is available")

    frame = pd.read_parquet(DEVELOPMENT, engine="fastparquet")
    if set(frame["split_partition"].astype(str).unique()) != {"train", "val"}:
        raise ValueError("pre-freeze data contains a partition other than train/val")
    train = frame.loc[frame["split_partition"].astype(str).eq("train")]
    if len(train) != 29592:
        raise ValueError("unexpected OOF training row count")
    x = train[["vqc_x0_distance", "vqc_x1_congestion"]].to_numpy(dtype=np.float64)
    r = train["residual_minutes"].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(r).all():
        raise ValueError("nonfinite OOF rows in ridge control fit")
    ridge = selection["ridge_residual_control"]["selected"]
    penalty = float(ridge["ridge_penalty"])
    design = np.column_stack((np.ones(len(x)), x))
    regularizer = np.diag([0.0, penalty, penalty])
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ r)
    np.savez_compressed(RIDGE, coefficients=coefficients, penalty=np.asarray(penalty), feature_names=np.asarray(["intercept", "vqc_x0_distance", "vqc_x1_congestion"]))

    candidates: dict[str, Any] = {}
    for arch in ("one_block", "reupload"):
        candidate = selection["quantum_candidates"][arch]
        checkpoint = _path(candidate["checkpoint_path"])
        if not checkpoint.is_file() or sha256(checkpoint) != candidate["checkpoint_sha256"]:
            raise ValueError(f"{arch} checkpoint missing or hash mismatch")
        # Load before freeze to validate model file integrity and expected architecture.
        q = QuantumResidualRegressor.load(checkpoint)
        if q.config.architecture != arch:
            raise ValueError(f"{arch} checkpoint architecture mismatch")
        candidates[arch] = {
            "checkpoint_path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": sha256(checkpoint),
            "alpha": float(candidate["alpha_validation_only"]),
            "gate_feature": candidate.get("gate_feature"),
            "gate_operator": candidate["gate_operator"],
            "gate_threshold": candidate.get("gate_threshold"),
            "calibration_invocation_rate": candidate["quantum_invocation_rate_calibration"],
        }
    candidate = selection["selected_release_gate"]
    scaler_path = _path(manifest["quantum_feature_scaler_path"])
    if not scaler_path.is_file():
        raise ValueError("train-only Q feature scaler is missing")
    baseline = manifest["model"]
    baseline_path = _path(baseline["checkpoint_path"])
    if not baseline_path.is_file() or sha256(baseline_path) != baseline["checkpoint_sha256"]:
        raise ValueError("recovered GBDT checkpoint missing or hash mismatch")
    baseline_bundle = joblib.load(baseline_path)
    bundle_info = _build_portable_bundle(
        baseline_path, candidates, list(baseline_bundle["feature_names"]), scaler_path,
        candidate["architecture"],
    )
    packaged_baseline_path = BUNDLE / "recovered_eta_gbdt.joblib"
    # The recovered manifest references its parent v3 protocol. Include the
    # current v5 protocol and every transformed input hash separately.
    protocol_path = _path(manifest["v4_protocol_path"])
    fold_path = _path(manifest["crossfit"]["fold_manifest_path"])
    release = {
        "schema_version": "run2-india-portable-hybrid-release-v1",
        "run_id": RUN_ID,
        "freeze_state": "frozen_before_test",
        "freeze_timestamp_utc": now(),
        "selection_partition": "val",
        "model_selection_source": "validation",
        "selected_architecture": candidate["architecture"],
        "selected_alpha": 0.0,
        "operational_release": {
            "alpha": 0.0,
            "gate_enabled": False,
            "policy": "classical GBDT fallback; validation-only full-validation comparison found the selected residual hybrid slightly worse than baseline RMSE",
            "reason": {
                "gbdt_validation_rmse_minutes": selection["classical_baseline_validation_metrics_reporting_only"]["rmse_minutes"],
                "candidate_hybrid_validation_rmse_minutes": selection["quantum_candidates"][candidate["architecture"]]["validation_hybrid_metrics"]["rmse_minutes"],
            },
        },
        "validation_selected_quantum_candidate": {
            "architecture": candidate["architecture"],
            "alpha": float(candidate["alpha"]),
            "gate_feature": candidate.get("gate_feature"),
            "gate_operator": candidate["gate_operator"],
            "gate_threshold": candidate.get("gate_threshold"),
            "deployment_invocation_rate_validation": candidate["quantum_invocation_rate_validation"],
            "candidate_kept_for_one_time_test_comparison": True,
        },
        "target": {"name": "delivery_duration_min", "unit": "minutes"},
        "feature_order": ["vqc_x0_distance", "vqc_x1_congestion"],
        "model_artifacts": {
            "recovered_gbdt": {"path": str(packaged_baseline_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(packaged_baseline_path), "original_reference": str(baseline_path.relative_to(ROOT)).replace("\\", "/")},
            "quantum_candidates": candidates,
            "ridge_control": {"path": str(RIDGE.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(RIDGE), "penalty": penalty, "alpha": float(ridge["alpha_calibration_only"])},
        },
        "portable_bundle": bundle_info,
        "data_artifacts": {
            "development_table": {"path": str(DEVELOPMENT.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(DEVELOPMENT)},
            "recovery_manifest": {"path": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(MANIFEST)},
            "selection": {"path": str(SELECTION.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(SELECTION)},
            "feature_scaler": {"path": str(scaler_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(scaler_path)},
            "crossfit_manifest": {"path": str(fold_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(fold_path)},
            "protocol": {"path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(protocol_path)},
        },
        "validation_role_protocol_sha256": selection["validation_role_protocol_sha256"],
        "development_roles": manifest["crossfit"]["development_roles"],
        "selected_gate_source": "candidate threshold chosen on courier-disjoint fusion calibration validation role; operational release disables QML and uses alpha=0 classical fallback",
        "test_evaluation_timestamp_utc": None,
        "test_metrics": None,
        "test_predictions_path": None,
        "portable_command": "python -m experiments_india_qml.portable_release_v5 --evaluate",
        "acceptance": "pretest checks must pass before the evaluation command is invoked",
    }
    RELEASE.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    return release


def evaluate_test_once(*, resume_failed_evaluation: bool = False) -> dict[str, Any]:
    """One sealed evaluation. Writes the start sentinel before target-bearing read."""
    if not RELEASE.is_file() or TEST_OUTPUT.exists() or TEST_METRICS.exists():
        raise RuntimeError("release missing or sealed test outputs already exist; refusing duplicate evaluation")
    retry = TEST_START.exists()
    if retry and not resume_failed_evaluation:
        raise RuntimeError("a sealed evaluation attempt is already recorded; use the explicit failed-attempt recovery only")
    if not retry and resume_failed_evaluation:
        raise RuntimeError("failed-attempt recovery requested without a prior recorded attempt")
    if retry:
        if not FAILED_ATTEMPT.is_file() or TEST_RETRY_START.exists():
            raise RuntimeError("no valid recorded failed attempt, or recovery attempt already exists")
        first_attempt = json.loads(TEST_START.read_text(encoding="utf-8"))
        failed = json.loads(FAILED_ATTEMPT.read_text(encoding="utf-8"))
        if failed.get("status") != "aborted_before_metrics_persisted" or failed.get("test_outcomes_inspected_or_used_for_selection") is not False:
            raise RuntimeError("recorded first attempt is not eligible for code-only recovery")
    elif TEST_RETRY_START.exists():
        raise RuntimeError("unexpected recovery sentinel exists without first attempt")
    import pandas as pd
    from fastparquet import ParquetFile

    release = json.loads(RELEASE.read_text(encoding="utf-8-sig"))
    if release.get("freeze_state") != "frozen_before_test" or release.get("test_metrics") is not None:
        raise RuntimeError("release is not frozen in pre-test state")
    # Hash-check all frozen artifacts and model files before touching test labels.
    for section in ("model_artifacts", "data_artifacts"):
        if section == "model_artifacts":
            entries = [release[section]["recovered_gbdt"], release[section]["ridge_control"], *release[section]["quantum_candidates"].values()]
        else:
            entries = list(release[section].values())
        for entry in entries:
            relative = entry.get("path", entry.get("checkpoint_path"))
            if relative is None:
                raise RuntimeError(f"frozen artifact has no path field in {section}")
            path = _path(relative)
            expected_hash = entry.get("sha256", entry.get("checkpoint_sha256"))
            if expected_hash is None or not path.is_file() or sha256(path) != expected_hash:
                raise RuntimeError(f"frozen artifact hash mismatch: {relative}")
    attempt_start_path = TEST_RETRY_START if retry else TEST_START
    attempt_start_path.write_text(json.dumps({
        "run_id": RUN_ID,
        "attempt": 2 if retry else 1,
        "started_at_utc": now(),
        "release_manifest_sha256": sha256(RELEASE),
        "evaluator_source_sha256": sha256(Path(__file__).resolve()),
        "test_read_boundary": "sentinel written before target-bearing Parquet read",
        "resumes_failed_attempt": bool(retry),
        "prior_attempt_marker": str(TEST_START.relative_to(ROOT)).replace("\\", "/") if retry else None,
    }, indent=2) + "\n", encoding="utf-8")

    model_bundle = joblib.load(_path(release["model_artifacts"]["recovered_gbdt"]["path"]))
    gbdt_cols = ["ID", "Delivery_person_ID", "split_partition", "delivery_duration_min", *model_bundle["feature_names"]]
    vqc_cols = ["ID", "Delivery_person_ID", "split_partition", "vqc_x0_distance", "vqc_x1_congestion"]
    gbdt_file = ROOT / "data" / "indian_delivery_gbdt.parquet"
    vqc_file = ROOT / "data" / "indian_delivery_vqc.parquet"
    filters = [[("split_partition", "in", ["test"])]]
    # Explicit row_filter=True is required: parquet filters alone only prune row groups.
    gbdt = ParquetFile(gbdt_file).to_pandas(columns=gbdt_cols, filters=filters, row_filter=True)
    if set(gbdt["split_partition"].astype(str).unique()) != {"test"}:
        raise RuntimeError("test target-bearing table contains unexpected partitions")
    vqc = ParquetFile(vqc_file).to_pandas(columns=vqc_cols, filters=filters, row_filter=True)
    if set(vqc["split_partition"].astype(str).unique()) != {"test"}:
        raise RuntimeError("test VQC table contains unexpected partitions")
    test = gbdt.merge(vqc, on=["ID", "Delivery_person_ID", "split_partition"], validate="one_to_one", sort=False)
    if len(test) != 5941 or test["ID"].isna().any() or test["ID"].duplicated().any():
        raise RuntimeError("unexpected final test row count or ID integrity failure")
    y = test["delivery_duration_min"].to_numpy(dtype=np.float64)
    X = test[release["feature_order"]].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all() or not np.isfinite(X).all() or X.min() < 0.0 or X.max() > 1.0:
        raise RuntimeError("test target or VQC features are not finite/in-range")
    eta = np.maximum(0.0, model_bundle["model"].predict(test[model_bundle["feature_names"]].astype(np.float32)))
    residual_target = y - eta
    with np.load(_path(release["model_artifacts"]["ridge_control"]["path"]), allow_pickle=False) as z:
        coef = z["coefficients"].astype(np.float64)
    ridge_resid = np.column_stack((np.ones(len(X)), X)) @ coef
    ridge_alpha = float(release["model_artifacts"]["ridge_control"]["alpha"])
    ridge_hybrid = hybrid_predict_minutes(eta, ridge_resid, ridge_alpha)

    predictions: dict[str, np.ndarray] = {"classical_gbdt": eta, "classical_plus_ridge": ridge_hybrid}
    residuals: dict[str, np.ndarray] = {"classical_plus_ridge": ridge_resid}
    invocation: dict[str, int] = {}
    candidate_unfused: dict[str, np.ndarray] = {}
    for arch, candidate in release["model_artifacts"]["quantum_candidates"].items():
        q = QuantumResidualRegressor.load(_path(candidate["checkpoint_path"]))
        raw_resid = q.predict_minutes(X)
        candidate_unfused[arch] = raw_resid
        candidate_unfused_hybrid = hybrid_predict_minutes(eta, raw_resid, float(candidate["alpha"]))
        predictions[f"quantum_{arch}_hybrid"] = candidate_unfused_hybrid
        residuals[f"quantum_{arch}"] = raw_resid
        gate_feature = candidate.get("gate_feature")
        threshold = candidate.get("gate_threshold")
        if float(candidate["alpha"]) > 0 and gate_feature and threshold is not None:
            gated = q.predict_hybrid_minutes(eta, X, float(candidate["alpha"]), gate_feature=gate_feature, gate_operator=candidate["gate_operator"], gate_threshold=float(threshold))
        else:
            gated = {"eta_minutes": eta.copy(), "quantum_residual_minutes": np.zeros(len(eta)), "quantum_invoked": np.zeros(len(eta), dtype=bool), "invocation_count": 0}
        predictions[f"quantum_{arch}_selective"] = gated["eta_minutes"]
        residuals[f"quantum_{arch}_selective"] = gated["quantum_residual_minutes"]
        invocation[arch] = gated["invocation_count"]

    selected_arch = release["selected_architecture"]
    selected = release["model_artifacts"]["quantum_candidates"][selected_arch]
    predictions["quantum_hybrid"] = predictions[f"quantum_{selected_arch}_hybrid"]
    predictions["selective_quantum"] = predictions[f"quantum_{selected_arch}_selective"]
    predictions["alpha_zero_fallback"] = eta.copy()
    predictions["operational_release"] = eta.copy()
    residuals["selected_quantum"] = residuals[f"quantum_{selected_arch}"]
    residuals["selected_quantum_selective"] = residuals[f"quantum_{selected_arch}_selective"]

    metrics_by_model = {name: metrics(y, pred) for name, pred in predictions.items()}
    residual_metrics = {name: metrics(residual_target, pred) for name, pred in residuals.items()}
    execution_work = {
        "implementation": "local NumPy statevector simulator; no QPU",
        "all_row_candidate_diagnostic_examples_by_architecture": {
            "one_block": len(test),
            "reupload": len(test),
        },
        "selective_reupload_candidate_examples": invocation.get("reupload", 0),
        "total_statevector_example_evaluations_in_scoring_run": 2 * len(test) + invocation.get("reupload", 0),
        "deployment_policy_expected_invocations": {
            "operational_alpha_zero": 0,
            "selective_reupload_candidate": invocation.get("reupload", 0),
        },
        "note": "The scorer ran all-row candidate diagnostics for C and both architectures, then the input-gated path for D. The selective invocation rate is the frozen deployment policy rate, separate from the total diagnostic simulator work above.",
    }
    record = {
        "evaluation_attempt": 2 if retry else 1,
        "prior_attempt_aborted_without_metrics": bool(retry),
        "prior_attempt_evidence_path": str(FAILED_ATTEMPT.relative_to(ROOT)).replace("\\", "/") if retry else None,
        "evaluation_start_sentinel_path": str(attempt_start_path.relative_to(ROOT)).replace("\\", "/"),
        "test_row_count": len(test),
        "unique_test_ids": int(test["ID"].nunique()),
        "test_target_unit": "minutes",
        "metrics": metrics_by_model,
        "residual_target_metrics": residual_metrics,
        "quantum_invocations": invocation,
        "quantum_invocation_rates": {name: count / len(test) for name, count in invocation.items()},
        "quantum_execution_work": execution_work,
        "selected_architecture": selected_arch,
        "selected_alpha": release["selected_alpha"],
        "validation_candidate_alpha": selected["alpha"],
        "operational_release_policy": "alpha=0 GBDT fallback, selected before test because the candidate hybrid was slightly worse on full validation RMSE",
        "selection_partition": "val",
        "model_selection_source": "validation",
        "evaluated_at_utc": now(),
    }
    output = pd.DataFrame({
        "ID": test["ID"].to_numpy(),
        "Delivery_person_ID": test["Delivery_person_ID"].to_numpy(),
        "y_true_minutes": y,
        **{f"prediction_{name}_minutes": pred for name, pred in predictions.items()},
        "target_residual_minutes": residual_target,
        **{f"predicted_residual_{name}_minutes": pred for name, pred in residuals.items()},
    })
    output.to_parquet(TEST_OUTPUT, index=False, engine="fastparquet")
    record["prediction_artifact"] = {"path": str(TEST_OUTPUT.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(TEST_OUTPUT)}
    TEST_METRICS.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    release["test_evaluation_timestamp_utc"] = record["evaluated_at_utc"]
    release["test_metrics"] = metrics_by_model
    release["test_residual_metrics"] = residual_metrics
    release["test_predictions_path"] = record["prediction_artifact"]
    release["test_evaluation_record_path"] = str(TEST_METRICS.relative_to(ROOT)).replace("\\", "/")
    release["test_evaluation_record_sha256"] = sha256(TEST_METRICS)
    RELEASE.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    group.add_argument("--resume-failed-evaluation", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        result = prepare_release()
    else:
        result = evaluate_test_once(resume_failed_evaluation=args.resume_failed_evaluation)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
