"""Focused integrity tests for the read-only India ETA preflight."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments_india_qml.preflight import (
    RECOVERY_MANIFEST_RELATIVE,
    _git_blob,
    main,
    source_identity,
    verify_recovery_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _artifact(root: Path, relative: str, content: bytes) -> dict[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": relative, "sha256": hashlib.sha256(content).hexdigest()}


def _valid_recovery_fixture(root: Path) -> Path:
    files = {
        "model": _artifact(root, "artifacts/eta/model.joblib", b"model placeholder; deliberately not deserialized"),
        "feature_schema": _artifact(root, "artifacts/eta/feature_schema.json", b"{}"),
        "preprocessing_artifacts": _artifact(root, "artifacts/eta/preprocessing.json", b"{}"),
        "oof": {**_artifact(root, "artifacts/eta/oof.parquet", b"opaque oof bytes"), "rows": 29_592},
        "validation": {**_artifact(root, "artifacts/eta/validation.parquet", b"opaque validation bytes"), "rows": 5_989},
        "oof_fold_manifest": _artifact(root, "artifacts/eta/folds.json", b"{}"),
        "validation_role_assignment": _artifact(root, "artifacts/eta/roles.json", b"{}"),
    }
    manifest = {
        "status": "trained",
        "estimator_identity": "sklearn.ensemble.HistGradientBoostingRegressor",
        "data_provenance": {"target": "delivery_duration_min", "target_unit": "minutes",
                             "gbdt_feature_count": 59},
        "sealed_test": "not loaded by DataBundle, not used in fitting, selection, calibration, or written to artifacts",
        "model_artifacts": {key: files[key] for key in ("model", "feature_schema", "preprocessing_artifacts")},
        "prediction_artifacts": {key: files[key] for key in
                                  ("oof", "validation", "oof_fold_manifest", "validation_role_assignment")},
    }
    path = root / "artifacts/india_eta_recovery/recovery_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_local_csv_crlf_normalization_matches_split_hash_and_git_blob_without_writing() -> None:
    source = ROOT / "data/train.csv"
    original_bytes = source.read_bytes()
    split = json.loads((ROOT / "data/split_manifest.json").read_text(encoding="utf-8"))
    committed = _git_blob(ROOT, "data/train.csv")
    assert committed is not None
    identity = source_identity(original_bytes, committed, split["source_sha256"])
    assert identity["matches"] is True
    assert identity["canonical_lf_sha256"] == "4baa699e1c09d286f06790010a8c0b284ed3376cbde7e346e19d0ded16a240e3"
    assert identity["raw_sha256"] == "d293b6fc7157aef30fb34301216a3432ec895d407ce2ea9edddf942b416cdde1"
    assert source.read_bytes() == original_bytes


def test_changed_csv_content_does_not_pass_canonical_or_git_blob_check() -> None:
    source = ROOT / "data/train.csv"
    split = json.loads((ROOT / "data/split_manifest.json").read_text(encoding="utf-8"))
    committed = _git_blob(ROOT, "data/train.csv")
    assert committed is not None
    corrupted = bytearray(source.read_bytes())
    corrupted[len(corrupted) // 2] ^= 1
    result = source_identity(bytes(corrupted), committed, split["source_sha256"])
    assert result["matches"] is False
    assert result["canonical_matches_expected"] is False
    assert result["canonical_matches_git_blob"] is False


def test_known_recovery_manifest_and_all_artifact_hashes_verify_without_loading(tmp_path: Path) -> None:
    manifest = _valid_recovery_fixture(tmp_path)
    report, problems = verify_recovery_manifest(tmp_path, manifest)
    assert problems == []
    assert report["verified"] is True
    assert report["feature_count"] == 59
    assert report["oof_rows"] == 29_592
    assert report["validation_rows"] == 5_989


def test_missing_and_tampered_manifest_artifacts_fail_hash_verification(tmp_path: Path) -> None:
    manifest = _valid_recovery_fixture(tmp_path)
    parsed = json.loads(manifest.read_text(encoding="utf-8"))
    missing = tmp_path / parsed["prediction_artifacts"]["validation"]["path"]
    missing.unlink()
    report, problems = verify_recovery_manifest(tmp_path, manifest)
    assert report["verified"] is False
    assert any("prediction_artifacts.validation" in problem for problem in problems)

    # Restore exact bytes and then alter them, proving content is covered too.
    _artifact(tmp_path, parsed["prediction_artifacts"]["validation"]["path"], b"tampered validation bytes")
    report, problems = verify_recovery_manifest(tmp_path, manifest)
    assert report["verified"] is False
    assert any("prediction_artifacts.validation" in problem for problem in problems)


def test_manifest_path_traversal_is_rejected(tmp_path: Path) -> None:
    manifest = _valid_recovery_fixture(tmp_path)
    parsed = json.loads(manifest.read_text(encoding="utf-8"))
    parsed["model_artifacts"]["model"]["path"] = "../escape.joblib"
    manifest.write_text(json.dumps(parsed), encoding="utf-8")
    _, problems = verify_recovery_manifest(tmp_path, manifest)
    assert any("escapes repository root" in problem for problem in problems)


def test_real_default_preflight_reports_classical_ready_but_keeps_quantum_held(capsys) -> None:
    exit_code = main(["--repo", str(ROOT), "--json"])
    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "CLASSICAL_READY"
    assert report["recovery_manifest"]["verified"] is True
    assert report["quantum_training_authorized"] is False
    assert report["quantum_status"] == "pending_external_Opus_architecture_review"


def test_explicit_paths_must_match_the_manifest_and_canonical_predictions(capsys, tmp_path: Path) -> None:
    model = ROOT / "artifacts/india_eta_recovery/gbdt/india_eta_hgb_regressor.joblib"
    oof = ROOT / "artifacts/india_eta_recovery/oof_predictions.parquet"
    val = ROOT / "artifacts/india_eta_recovery/validation_predictions.parquet"
    ok = main(["--repo", str(ROOT), "--checkpoint", str(model), "--predictions", str(oof), "--predictions", str(val), "--json"])
    assert ok == 0
    capsys.readouterr()
    outsider = tmp_path / "other_predictions.parquet"
    outsider.write_bytes(b"not the trusted artifact")
    code = main(["--repo", str(ROOT), "--predictions", str(outsider), "--json"])
    assert code == 2
    blocked = json.loads(capsys.readouterr().out)
    assert blocked["status"] == "BLOCKED"
    assert any("not a manifest-verified" in item for item in blocked["blockers"])
