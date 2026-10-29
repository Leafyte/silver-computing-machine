"""CPU release loader tests; only the release's synthetic reference rows are used."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts" / "run2_india" / "india_eta_vqc4_cpu_release_v1"
ZIP_PATH = ROOT / "artifacts" / "run2_india" / "india_eta_vqc4_cpu_release_v1.zip"
EXPECTED_ZIP_SHA256 = "18d5d4859e53d3d59063e60cdcdd45cb6eff6757bba6474f024cdde91aa06b80"
EXPECTED_BUNDLE_FILES = {
    "README.md", "baseline60.json", "circuit.py", "extended76.json", "feature_schema.json",
    "fusion_policy.json", "gbdt_model.joblib", "predictor.py", "preprocessing_artifacts.json",
    "reference_fixture.json", "release_manifest.json", "requirements.txt", "ridge_control.json",
}


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module():
    if str(RELEASE) not in sys.path:
        sys.path.insert(0, str(RELEASE))
    spec = importlib.util.spec_from_file_location("india_vqc4_release_predictor", RELEASE / "predictor.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_cli(directory: Path) -> dict:
    env = dict(__import__("os").environ)
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    process = subprocess.run(
        [sys.executable, str(directory / "predictor.py"), "--verify-reference"],
        cwd=directory, env=env, text=True, capture_output=True, check=True,
    )
    return json.loads(process.stdout)


def test_directory_loader_checks_inventory_and_synthetic_predictions() -> None:
    predictor = _module()
    runtime = predictor.load_release(RELEASE)
    outputs = predictor.verify_reference(RELEASE)
    assert set(outputs) == {
        "gbdt_only_min", "ridge_control_min", "baseline60_candidate_min",
        "extended76_candidate_min", "operational_min",
    }
    assert outputs["operational_min"] == outputs["gbdt_only_min"]
    assert all(np.isfinite(values).all() for values in map(np.asarray, outputs.values()))
    assert _run_cli(RELEASE)["status"] == "pass"
    # Importing predictor normally creates a Python cache; the next validation
    # and ordinary CLI invocation still accept the exact release inventory.
    assert predictor.load_release(RELEASE).fusion["alpha"] == runtime.fusion["alpha"]
    assert _run_cli(RELEASE)["status"] == "pass"


def test_archive_hash_and_isolated_zip_extraction_reload(tmp_path: Path) -> None:
    assert _sha(ZIP_PATH) == EXPECTED_ZIP_SHA256
    extracted = tmp_path / "release"
    extracted.mkdir()
    with zipfile.ZipFile(ZIP_PATH) as archive:
        assert set(archive.namelist()) == EXPECTED_BUNDLE_FILES
        assert all(Path(name).name == name for name in archive.namelist())
        archive.extractall(extracted)
    assert {item.name for item in extracted.iterdir()} == EXPECTED_BUNDLE_FILES
    result = _run_cli(extracted)
    assert result["status"] == "pass"
    # Test the documented ordinary Python import path in an extracted bundle.
    code = "import json, predictor; print(json.dumps({'status':'pass','outputs':predictor.verify_reference()}))"
    env = dict(__import__("os").environ)
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    first = subprocess.run([sys.executable, "-c", code], cwd=extracted, env=env,
                           text=True, capture_output=True, check=True)
    second = subprocess.run([sys.executable, "-c", code], cwd=extracted, env=env,
                            text=True, capture_output=True, check=True)
    assert json.loads(first.stdout)["status"] == "pass"
    assert json.loads(second.stdout)["status"] == "pass"


def test_model_tamper_rejected_before_deserialization_and_extra_file_rejected(tmp_path: Path, monkeypatch) -> None:
    predictor = _module()
    tampered = tmp_path / "tampered"
    shutil.copytree(RELEASE, tampered)
    path = tampered / "gbdt_model.joblib"
    path.write_bytes(path.read_bytes() + b"tamper")
    calls = []
    monkeypatch.setattr(predictor.joblib, "load", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="artifact integrity mismatch"):
        predictor.load_release(tampered)
    assert calls == []

    extra = tmp_path / "extra"
    shutil.copytree(RELEASE, extra)
    (extra / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="missing, extra, or nested"):
        predictor.load_release(extra)


def test_input_contract_rejects_bool_object_and_unsafe_paths() -> None:
    predictor = _module()
    runtime = predictor.load_release(RELEASE)
    with pytest.raises(ValueError, match="real numeric dtype"):
        runtime.predict(np.ones((1, 59), dtype=bool))
    with pytest.raises(ValueError, match="real numeric dtype"):
        runtime.predict(np.ones((1, 59), dtype=object))
    with pytest.raises(ValueError, match="feature order"):
        runtime.predict(np.ones((1, 59), dtype=float), list(reversed(runtime.feature_names)))
    with pytest.raises(ValueError, match="safe basenames"):
        predictor._safe_name("../outside.json")
