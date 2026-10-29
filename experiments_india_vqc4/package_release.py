"""Build a self-contained CPU release from frozen RUN2 artifacts only.

This packaging command reads the frozen development manifest/checkpoints and
existing release metadata. It never opens the raw dataset or test partition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "artifacts" / "india_eta_vqc4_run2"
CLASSICAL_RELEASE = ROOT / "artifacts" / "india_eta_classical_release_v1"
RIDGE_DIR = ROOT / "artifacts" / "india_eta_residual_control_v1"
REPRO_DIR = ROOT / "artifacts" / "india_eta_vqc4_run2_repro_metrics"
OUT_DEFAULT = ROOT / "artifacts" / "run2_india" / "india_eta_vqc4_cpu_release_v1"
REQUIRED = {
    "README.md", "circuit.py", "baseline60.json", "extended76.json",
    "feature_schema.json", "fusion_policy.json", "gbdt_model.joblib",
    "preprocessing_artifacts.json", "reference_fixture.json",
    "requirements.txt", "ridge_control.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _copy(source: Path, dest: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copyfile(source, dest)


def _make_policy(run: dict[str, Any]) -> dict[str, Any]:
    choice = run.get("selection_choice")
    alpha = float(run.get("selected_alpha", float("nan")))
    if alpha not in (0.0, 1.0):
        raise ValueError("expected a frozen discrete alpha policy (0 or 1)")
    if alpha == 0.0:
        if choice != "alpha0_gbdt_only":
            raise ValueError("alpha 0 must explicitly select the GBDT operational fallback")
        variant = None
    else:
        if choice not in ("baseline60", "extended76"):
            raise ValueError("nonzero alpha must identify the selected 4Q architecture")
        variant = choice
    return {
        "format": "india_eta_vqc4_fusion_policy_v1",
        "selected_variant": variant,
        "alpha": alpha,
        "correction": "max(0, gbdt_minutes + alpha * vqc_residual_minutes)",
        "selection_role": "117 courier groups / 4,124 validation rows only",
        "fusion_calibration_rows_used_for_choice": False,
        "fallback_rule": "alpha=0 unless the selected VQC strictly improves MAE and is no worse in RMSE than both GBDT and the frozen Ridge control on model-selection validation rows",
        "test_used_for_policy": False,
        "test_qualification": "original test partition was previously exposed by a prior 2Q evaluation; any new score is retrospective only",
        "pretest_feature_access_incident": "the independent audit materialized test feature columns via a fastparquet filter that returned all rows; target and synthetic label columns were never requested/read",
        "selection_metrics": run.get("selection_metrics"),
        "source_manifest_sha256": sha256(RUN_DIR / "run_manifest.json"),
        "source_checkpoint_sha256": {
            "baseline60": sha256(RUN_DIR / "baseline60_best.json"),
            "extended76": sha256(RUN_DIR / "extended76_best.json"),
        },
    }


def build(output: Path = OUT_DEFAULT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite release output: {output}")
    run_path = RUN_DIR / "run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("status") != "frozen_models_selection_only_no_test_read":
        raise ValueError("training manifest is not frozen without test use")
    if run.get("train_rows") != 29592 or run.get("selection_rows") != 4124 or run.get("calibration_rows_held_until_freeze") != 1865:
        raise ValueError("unexpected frozen training/validation-role row counts")
    expected_sources = {
        "gbdt_checkpoint": "f1f94fbcc932f3519aaafcff7dc99d65b7951c0240730e54c320259152bbebdf",
        "oof_predictions": "78a5461c89387e8630421008b92dfffcc84864e9e25bea7476eefa2c8d48032f",
        "ridge_control": "5203fe9844a9721cfed1f26411a32bb0ebd872cafa8f64e661c73c2afc8189e2",
    }
    if run.get("source_hashes", {}) != {**run.get("source_hashes", {}), **expected_sources}:
        raise ValueError("frozen source hashes do not match the accepted classical inputs")
    repro_path = REPRO_DIR / "reproduction_metrics.json"
    if sha256(repro_path) != "d4b040187601229b9c472a0fbe990613b79242d6781959964297ea01ca0bd2bd":
        raise ValueError("bitwise checkpoint resource reproduction hash does not match the audited copy")
    repro = json.loads(repro_path.read_text(encoding="utf-8"))
    if repro.get("status") != "both_frozen_checkpoints_reproduced_bitwise" or repro.get("test_partition_read") is not False:
        raise ValueError("resource reproduction is not a test-free bitwise checkpoint match")
    for name in ("baseline60_best.json", "extended76_best.json"):
        expected = run.get("stages", {}).get(name.removesuffix("_best.json"), {}).get("checkpoint_sha256")
        if not expected or sha256(RUN_DIR / name) != expected:
            raise ValueError(f"checkpoint is not bound by the frozen training manifest: {name}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    _copy(CLASSICAL_RELEASE / "model.joblib", output / "gbdt_model.joblib")
    _copy(CLASSICAL_RELEASE / "feature_schema.json", output / "feature_schema.json")
    _copy(CLASSICAL_RELEASE / "preprocessing_artifacts.json", output / "preprocessing_artifacts.json")
    _copy(ROOT / "experiments_india_vqc4" / "circuit.py", output / "circuit.py")
    _copy(RUN_DIR / "baseline60_best.json", output / "baseline60.json")
    _copy(RUN_DIR / "extended76_best.json", output / "extended76.json")
    _copy(RIDGE_DIR / "residual_ridge.json", output / "ridge_control.json")
    _copy(ROOT / "experiments_india_vqc4" / "portable.py", output / "predictor.py")
    _write_json(output / "fusion_policy.json", _make_policy(run))
    _write_json(output / "requirements.txt", {
        "note": "This bundle requires Python 3.12.10 and the pinned dependencies below.",
        "requirements": ["numpy==1.26.4", "scikit-learn==1.4.2", "joblib==1.6.0"],
        "torch_required_for_inference": False,
        "qpu_or_cloud_required": False,
    })
    (output / "README.md").write_text(
        "# India ETA 4Q CPU release\n\n"
        "This bundle predicts delivery duration in minutes from the committed 59 numeric GBDT features. "
        "It derives the four frozen VQC inputs from the supplied feature matrix. It includes both trained 4Q residual candidates, "
        "the frozen operational selection/fallback policy, and the Ridge comparison control.\n\n"
        "The NumPy loader does not require Torch, QpiAI, cloud access, or a QPU. The original test partition was exposed in an earlier 2Q run, "
        "so any 4Q test scores are retrospective and are not a pristine holdout claim. No raw dataset, OOF table, or validation rows are included.\n\n"
        "Run `python predictor.py --verify-reference` to check the synthetic-only fixture, or pass a JSON file containing `feature_names` and `rows` to `--input-json`.\n",
        encoding="utf-8",
    )
    # Synthetic reference inputs reuse no source row or label. The expected
    # outputs are materialized after loading the candidate bundle in isolation.
    fixture_source = json.loads((CLASSICAL_RELEASE / "reference_fixture.json").read_text(encoding="utf-8"))
    fixture = {
        "format": "india_eta_vqc4_synthetic_reference_v1",
        "synthetic_only": True,
        "not_from_dataset": True,
        "feature_names": fixture_source["feature_names"],
        "rows": fixture_source["rows"],
        "expected_predictions_min": {},
        "note": "Expected outputs are generated from this frozen release's own weights; this fixture validates packaging/reload parity, not model quality.",
    }
    _write_json(output / "reference_fixture.json", fixture)
    (output / "requirements.txt").write_text(
        "numpy==1.26.4\nscikit-learn==1.4.2\njoblib==1.6.0\n",
        encoding="utf-8",
    )

    # Materialize synthetic expectations from the exact copied checkpoint set
    # without loading any dataset or prediction table.
    import joblib
    import numpy as np
    from experiments_india_vqc4.circuit import NumpyFourQubitResidualVQC
    from experiments_india_vqc4.portable import IndiaETARelease

    feature_schema = json.loads((output / "feature_schema.json").read_text(encoding="utf-8"))
    preprocessing = json.loads((output / "preprocessing_artifacts.json").read_text(encoding="utf-8"))
    ridge = json.loads((output / "ridge_control.json").read_text(encoding="utf-8"))
    fusion = json.loads((output / "fusion_policy.json").read_text(encoding="utf-8"))
    models = {
        name: NumpyFourQubitResidualVQC.from_payload(json.loads((output / f"{name}.json").read_text(encoding="utf-8")))
        for name in ("baseline60", "extended76")
    }
    runtime = IndiaETARelease(
        output, {}, feature_schema["feature_names"], preprocessing,
        joblib.load(output / "gbdt_model.joblib"), ridge, fusion, models,
    )
    fixture["expected_predictions_min"] = runtime.predict(fixture["rows"], fixture["feature_names"])
    _write_json(output / "reference_fixture.json", fixture)

    # Seed a manifest with all non-code bundle artifacts. The loader embeds this
    # manifest digest, so a changed manifest cannot redirect deserialization.
    artifacts = {}
    for name in sorted(REQUIRED):
        path = output / name
        artifacts[name] = {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
    feature_schema = json.loads((output / "feature_schema.json").read_text(encoding="utf-8"))
    manifest = {
        "format": "india_eta_vqc4_cpu_release_v1",
        "release_name": "India ETA 4-qubit residual VQC CPU release",
        "release_version": "1.0.0",
        "test_status": "retrospective_exposed_partition",
        "feature_count": 59,
        "feature_names": feature_schema["feature_names"],
        "input_contract": "59 ordered committed numeric features; no identifiers, split labels, or target",
        "target": "delivery_duration_min",
        "target_unit": "minutes",
        "inference": {"device": "CPU", "torch_required": False, "qpu": False, "cloud": False},
        "training_manifest_sha256": sha256(run_path),
        "independent_acceptance_sha256": sha256(RUN_DIR / "independent_frozen_acceptance.json"),
        "retrospective_test_protocol_sha256": sha256(ROOT / "experiments_india_vqc4" / "retrospective_test_protocol.json"),
        "resource_reproduction_sha256": sha256(repro_path),
        "reproduced_frozen_checkpoints_bitwise": True,
        "resource_reproduction": {"wall_seconds": repro["wall_seconds"], "cpu_process_seconds": repro["cpu_process_seconds"],
                                  "windows_peak_working_set_bytes": repro["process_peak_rss_bytes"],
                                  "optimizer_steps": repro["total_optimizer_steps"]},
        "training_code_sha256": run.get("training_code_sha256"),
        "circuit_sha256": sha256(ROOT / "experiments_india_vqc4" / "circuit.py"),
        "source_hashes": run.get("source_hashes"),
        "selection_choice": run.get("selection_choice"),
        "selected_alpha": run.get("selected_alpha"),
        "artifacts": artifacts,
    }
    _write_json(output / "release_manifest.json", manifest)
    manifest_sha = sha256(output / "release_manifest.json")
    loader = (output / "predictor.py").read_text(encoding="utf-8")
    if loader.count("__MANIFEST_SHA256__") != 1:
        raise ValueError("portable loader must contain one manifest hash placeholder")
    (output / "predictor.py").write_text(loader.replace("__MANIFEST_SHA256__", manifest_sha), encoding="utf-8")

    zip_path = output.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(output.iterdir(), key=lambda p: p.name):
            if not path.is_file():
                raise ValueError(f"unexpected non-file release member: {path.name}")
            info = zipfile.ZipInfo(path.name, date_time=(2026, 10, 29, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    zip_sha = sha256(zip_path)
    zip_path.with_suffix(zip_path.suffix + ".sha256").write_text(f"{zip_sha}  {zip_path.name}\n", encoding="ascii")
    return {
        "status": "portable_release_built",
        "release_dir": str(output),
        "zip_path": str(zip_path),
        "zip_sha256": zip_sha,
        "manifest_sha256": manifest_sha,
        "artifact_count": len(REQUIRED),
        "bundle_files": sorted(path.name for path in output.iterdir()),
        "builder_python": sys.version.split()[0],
        "test_rows_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
