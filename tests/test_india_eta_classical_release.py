"""Portable ZIP/directory reference and pre-deserialization integrity checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts" / "india_eta_classical_release_v1"
ARCHIVE = ROOT / "artifacts" / "india_eta_classical_release_v1.zip"
EXPECTED_FILES = {
    "README.md",
    "feature_schema.json",
    "model.joblib",
    "predictor.py",
    "preprocessing_artifacts.json",
    "reference_fixture.json",
    "release_manifest.json",
    "requirements.txt",
}


def _verify(bundle: Path) -> list[float]:
    process = subprocess.run(
        [sys.executable, str(bundle / "predictor.py"), "--verify-reference"],
        cwd=bundle,
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    return json.loads(process.stdout)["predictions_min"]


def test_directory_and_unpacked_zip_reference_predictions_match_exactly(tmp_path: Path) -> None:
    expected_archive_sha = (ROOT / "artifacts" / "india_eta_classical_release_v1.zip.sha256").read_text().split()[0]
    actual_archive_sha = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    assert actual_archive_sha == expected_archive_sha
    with zipfile.ZipFile(ARCHIVE) as bundle_zip:
        names = set(bundle_zip.namelist())
        assert names == EXPECTED_FILES
        assert not any(name.endswith((".csv", ".parquet", ".pkl", ".npz")) for name in names)
        unpacked = tmp_path / "unpacked_classical_release"
        bundle_zip.extractall(unpacked)
    manifest = json.loads((RELEASE / "release_manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["artifacts"].items():
        path = RELEASE / name
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    directory_predictions = _verify(RELEASE)
    unpacked_predictions = _verify(unpacked)
    assert directory_predictions == unpacked_predictions
    fixture = json.loads((RELEASE / "reference_fixture.json").read_text(encoding="utf-8"))
    assert fixture["synthetic_only"] and fixture["not_from_dataset"]
    assert directory_predictions == fixture["expected_predictions_min"]


def test_tampered_model_is_rejected_before_joblib_deserialization(tmp_path: Path) -> None:
    tampered = tmp_path / "tampered_release"
    shutil.copytree(RELEASE, tampered)
    model_path = tampered / "model.joblib"
    contents = bytearray(model_path.read_bytes())
    contents[-1] ^= 0x01
    model_path.write_bytes(contents)
    process = subprocess.run(
        [sys.executable, str(tampered / "predictor.py"), "--verify-reference"],
        cwd=tampered,
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "SHA-256 mismatch for model.joblib" in process.stderr
    assert "UnpicklingError" not in process.stderr


def test_manifest_rejects_traversal_paths_before_deserialization(tmp_path: Path) -> None:
    modified = tmp_path / "path_traversal_release"
    shutil.copytree(RELEASE, modified)
    manifest_path = modified / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["README.md"]["path"] = "../outside.md"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    process = subprocess.run(
        [sys.executable, str(modified / "predictor.py"), "--verify-reference"],
        cwd=modified,
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "Unsafe artifact path for README.md" in process.stderr
    assert "UnpicklingError" not in process.stderr
