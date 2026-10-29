"""Independent audit of the portable classical ETA release directory and ZIP."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
from types import ModuleType
import tempfile
import zipfile
from typing import Any


FORBIDDEN_RELEASE_SUFFIXES = {".csv", ".parquet", ".feather", ".arrow"}
FORBIDDEN_RELEASE_NAMES = {
    "train.csv", "indian_delivery_clean.parquet", "indian_delivery_gbdt.parquet",
    "indian_delivery_vqc.parquet", "oof_predictions.parquet", "validation_predictions.parquet",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"unsafe release path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe release path: {value!r}")
    return path


def _load_and_verify_directory(release_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    root = release_dir.resolve(strict=True)
    manifest_path = root / "release_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("release_manifest.json must be a regular in-bundle file")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("created_from_frozen_checkpoint") or manifest.get("training_performed_during_packaging") is not False:
        raise ValueError("release is not declared as a frozen, inference-only package")
    inventory = manifest.get("artifacts")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("manifest artifact inventory must be a non-empty object")
    verified: dict[str, str] = {}
    for key, entry in inventory.items():
        key_path = _safe_relative(key)
        rel = _safe_relative(entry.get("path"))
        if str(rel) != str(key_path):
            raise ValueError(f"artifact key/path mismatch: {key} != {entry.get('path')}")
        candidate = root.joinpath(*rel.parts)
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"release artifact is missing or symlinked: {key}")
        try:
            candidate.resolve(strict=True).relative_to(root)
        except ValueError as exc:
            raise ValueError(f"release artifact escapes bundle root: {key}") from exc
        digest = sha256_file(candidate)
        if digest != str(entry.get("sha256", "")).lower():
            raise ValueError(f"release artifact SHA-256 mismatch: {key}")
        if candidate.stat().st_size != int(entry.get("bytes", -1)):
            raise ValueError(f"release artifact byte count mismatch: {key}")
        if candidate.name.lower() in FORBIDDEN_RELEASE_NAMES or candidate.suffix.lower() in FORBIDDEN_RELEASE_SUFFIXES:
            raise ValueError(f"raw data or large intermediate is in release: {key}")
        verified[key] = digest

    actual_files = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
    }
    expected_files = set(inventory) | {"release_manifest.json"}
    if actual_files != expected_files:
        raise ValueError(f"release inventory differs from files on disk: unlisted={sorted(actual_files - expected_files)}, missing={sorted(expected_files - actual_files)}")
    if any(item.is_symlink() for item in root.rglob("*")):
        raise ValueError("release directory must not contain symlinks")
    if any(item.stat().st_size > 25 * 1024 * 1024 for item in root.rglob("*") if item.is_file()):
        raise ValueError("release contains a file larger than 25 MiB")
    return manifest, verified


def _verify_archive(archive_path: Path, release_dir: Path, inventory: dict[str, str]) -> dict[str, Any]:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    sha_file = archive_path.with_suffix(archive_path.suffix + ".sha256")
    fields = sha_file.read_text(encoding="utf-8").strip().split()
    if not fields or fields[0].lower() != sha256_file(archive_path):
        raise ValueError("external ZIP SHA-256 file does not match the archive")
    archive_digest = sha256_file(archive_path)
    expected_names = set(inventory) | {"release_manifest.json"}
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("release ZIP contains duplicate paths")
        for info in infos:
            rel = _safe_relative(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ValueError(f"release ZIP contains a symlink: {info.filename}")
            if info.is_dir():
                raise ValueError(f"release ZIP includes an explicit directory entry: {info.filename}")
            if rel.as_posix() != info.filename:
                raise ValueError(f"release ZIP path is not canonical: {info.filename}")
            if Path(info.filename).name.lower() in FORBIDDEN_RELEASE_NAMES or Path(info.filename).suffix.lower() in FORBIDDEN_RELEASE_SUFFIXES:
                raise ValueError(f"raw data/intermediate file is in release ZIP: {info.filename}")
        if set(names) != expected_names:
            raise ValueError(f"ZIP member inventory mismatch: extra={sorted(set(names) - expected_names)}, missing={sorted(expected_names - set(names))}")
        for name, digest in inventory.items():
            if sha256_bytes(archive.read(name)) != digest:
                raise ValueError(f"ZIP member differs from verified release directory: {name}")
        if sha256_bytes(archive.read("release_manifest.json")) != sha256_file(release_dir / "release_manifest.json"):
            raise ValueError("ZIP manifest differs from release directory manifest")
    return {"path": str(archive_path), "sha256": archive_digest, "members": len(expected_names), "passed": True}


def _load_predictor(path: Path):
    name = "_run2_eta_release_predictor_independent"
    # Execute verified source directly so the audit does not write __pycache__
    # into the exact-inventory release directory.
    module = ModuleType(name)
    module.__file__ = str(path)
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    exec(code, module.__dict__)
    return module


def _check_feature_and_reference_contract(release_dir: Path, manifest: dict[str, Any], source_schema_path: Path) -> dict[str, Any]:
    schema = json.loads((release_dir / "feature_schema.json").read_text(encoding="utf-8"))
    reference = json.loads((release_dir / "reference_fixture.json").read_text(encoding="utf-8"))
    source_schema = json.loads(source_schema_path.read_text(encoding="utf-8-sig"))
    expected_features = source_schema["gbdt_view"]["features"]
    features = schema.get("feature_names")
    forbidden = {"ID", "delivery_record_id", "Delivery_person_ID", "courier_id", "split_partition", "delivery_duration_min", "target_normalized", "is_sla_breach_synthetic"}
    if features != expected_features or len(features) != 59:
        raise ValueError("release schema differs from frozen 59-feature source order")
    if set(features) & forbidden or len(set(features)) != len(features):
        raise ValueError("release feature schema includes forbidden identity/target fields or duplicates")
    model_meta = manifest.get("model", {})
    if model_meta.get("feature_order") != features or model_meta.get("feature_count") != 59:
        raise ValueError("manifest feature order/count does not match release schema")
    if reference.get("synthetic_only") is not True or reference.get("not_from_dataset") is not True:
        raise ValueError("reference fixture is not explicitly marked synthetic")
    if reference.get("feature_names") != features or len(reference.get("rows", [])) != 3:
        raise ValueError("synthetic reference fixture shape/schema mismatch")
    if any(set(row) != set(features) for row in reference["rows"]):
        raise ValueError("synthetic reference row keys differ from feature schema")
    for row in reference["rows"]:
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in row.values()):
            raise ValueError("synthetic reference contains non-finite/non-numeric values")
    if reference["expected_predictions_min"] != manifest["synthetic_reference"]["expected_predictions_min"]:
        raise ValueError("manifest and fixture expected predictions differ")
    if len(reference["expected_predictions_min"]) != 3:
        raise ValueError("expected three synthetic reference outputs")
    return {"feature_count": len(features), "ordered_schema_matches_source": True, "synthetic_reference_rows": len(reference["rows"]), "feature_names": features}


def audit_classical_release(
    release_dir: str | Path,
    archive_path: str | Path | None = None,
    *,
    source_schema_path: str | Path,
    source_checkpoint_path: str | Path,
    extract_zip: bool = True,
) -> dict[str, Any]:
    """Hash-verify, reload and exercise a classical release using synthetic rows only."""
    root = Path(release_dir).resolve(strict=True)
    manifest, hashes = _load_and_verify_directory(root)
    model_meta = manifest.get("model", {})
    source_sha = sha256_file(source_checkpoint_path)
    model_sha = hashes.get("model.joblib")
    if source_sha != model_sha or model_meta.get("source_checkpoint_sha256") != source_sha:
        raise ValueError("packaged model differs from independently accepted frozen source checkpoint")
    if manifest.get("preprocessing_availability", {}).get("raw_record_feature_engineering_code_included") is not False:
        raise ValueError("release preprocessing boundary must state raw feature engineering is absent")
    reference_audit = _check_feature_and_reference_contract(root, manifest, Path(source_schema_path))
    archive = Path(archive_path) if archive_path else root.with_suffix(".zip")
    archive_audit = _verify_archive(archive, root, hashes)

    # The official loader is imported and executed from the bundle itself.
    predictor = _load_predictor(root / "predictor.py")
    predicted = predictor.verify_reference(root)
    fixture = json.loads((root / "reference_fixture.json").read_text(encoding="utf-8"))
    import numpy as np
    np.testing.assert_allclose(predicted, fixture["expected_predictions_min"], rtol=0, atol=1e-12)
    model, feature_order = predictor.load_release(root)
    if feature_order != reference_audit["feature_names"]:
        raise ValueError("portable loader returned the wrong feature order")

    # Independent trusted deserialization only after our source/checkpoint/hash
    # checks, followed by numerical recomputation on synthetic-only rows.
    import joblib
    trusted_model = joblib.load(root / "model.joblib")
    matrix = np.asarray([[row[name] for name in feature_order] for row in fixture["rows"]], dtype=np.float64)
    independent_predictions = np.asarray(trusted_model.predict(matrix), dtype=np.float64)
    if not np.isfinite(independent_predictions).all():
        raise ValueError("reloaded estimator produced non-finite reference output")
    np.testing.assert_allclose(independent_predictions, fixture["expected_predictions_min"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(predicted, dtype=np.float64), independent_predictions)

    # Negative input contract tests do not touch real delivery records.
    rows = fixture["rows"]
    with _expect_error(ValueError):
        predictor.predict_rows(rows, root, feature_order[:-1] + [feature_order[-2]])
    bad_row = dict(rows[0]); bad_row.pop(feature_order[0])
    with _expect_error(ValueError):
        predictor.predict_rows([bad_row], root)
    bad_row = dict(rows[0]); bad_row["unexpected"] = 1.0
    with _expect_error(ValueError):
        predictor.predict_rows([bad_row], root)
    bad_row = dict(rows[0]); bad_row[feature_order[0]] = float("nan")
    with _expect_error(ValueError):
        predictor.predict_rows([bad_row], root)
    bad_row = dict(rows[0]); bad_row[feature_order[0]] = True
    with _expect_error(ValueError):
        predictor.predict_rows([bad_row], root)

    tamper_audit = _verify_tamper_rejection(root)
    extracted_audit: dict[str, Any] | None = None
    if extract_zip:
        extracted_audit = _verify_extracted_zip(archive, archive_audit["sha256"], root, hashes, fixture["expected_predictions_min"])
    return {
        "status": "PASS",
        "scope": "classical CPU bundle only; no train/validation/test records loaded",
        "release_directory_sha256s": hashes,
        "zip": archive_audit,
        "model_matches_frozen_checkpoint": True,
        "portable_loader_reference_max_abs_error_min": float(np.max(np.abs(np.asarray(predicted, dtype=np.float64) - np.asarray(fixture["expected_predictions_min"], dtype=np.float64)))),
        "independent_joblib_reference_max_abs_error_min": float(np.max(np.abs(independent_predictions - np.asarray(fixture["expected_predictions_min"], dtype=np.float64)))),
        "feature_contract": reference_audit,
        "tamper_fail_closed": tamper_audit,
        "isolated_zip_extraction": extracted_audit,
        "real_dataset_rows_read": False,
        "sealed_test_opened": False,
    }


class _expect_error:
    def __init__(self, expected_type: type[BaseException]):
        self.expected_type = expected_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"expected {self.expected_type.__name__}, but no error was raised")
        if not issubclass(exc_type, self.expected_type):
            return False
        return True


def _verify_tamper_rejection(release_dir: Path) -> dict[str, Any]:
    scratch = release_dir.parents[1] / ".runtime"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="run2-release-tamper-", dir=scratch) as tmp:
        copy = Path(tmp) / "tampered"
        shutil.copytree(release_dir, copy)
        model_path = copy / "model.joblib"
        original = model_path.read_bytes()
        model_path.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))
        predictor = _load_predictor(copy / "predictor.py")
        called = False
        load = predictor.joblib.load
        def forbidden_deserialize(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("joblib.load must not run on tampered artifact")
        predictor.joblib.load = forbidden_deserialize
        try:
            with _expect_error(ValueError):
                predictor.load_release(copy)
        finally:
            predictor.joblib.load = load
        if called:
            raise AssertionError("tampered estimator was deserialized before integrity rejection")
    traversal_checks = 0
    for unsafe_path in ("../outside.txt", "..\\outside.txt", "C:/outside.txt"):
        with tempfile.TemporaryDirectory(prefix="run2-release-traversal-", dir=scratch) as tmp:
            copy = Path(tmp) / "tampered"
            shutil.copytree(release_dir, copy)
            manifest_path = copy / "release_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["README.md"]["path"] = unsafe_path
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            predictor = _load_predictor(copy / "predictor.py")
            called = False
            load = predictor.joblib.load
            def forbidden_deserialize(*args, **kwargs):
                nonlocal called
                called = True
                raise AssertionError("joblib.load must not run on manifest traversal")
            predictor.joblib.load = forbidden_deserialize
            try:
                with _expect_error(ValueError):
                    predictor.load_release(copy)
                with _expect_error(ValueError):
                    _load_and_verify_directory(copy)
            finally:
                predictor.joblib.load = load
            if called:
                raise AssertionError("path traversal reached model deserialization")
            traversal_checks += 1
    return {
        "tampered_model_rejected_before_deserialization": True,
        "manifest_path_traversal_variants_rejected": traversal_checks,
    }


def _verify_extracted_zip(archive_path: Path, expected_sha: str, source_dir: Path, inventory: dict[str, str], expected_predictions: list[float]) -> dict[str, Any]:
    scratch = source_dir.parents[1] / ".runtime"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="run2-release-extract-", dir=scratch) as tmp:
        target = Path(tmp) / "bundle"
        target.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            if sha256_file(archive_path) != expected_sha:
                raise ValueError("release ZIP changed before extraction")
            for info in archive.infolist():
                rel = _safe_relative(info.filename)
                destination = target.joinpath(*rel.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(info))
        manifest, extracted_hashes = _load_and_verify_directory(target)
        if extracted_hashes != inventory:
            raise ValueError("isolated ZIP extraction changed the verified artifact inventory")
        predictor = _load_predictor(target / "predictor.py")
        outputs = predictor.verify_reference(target)
        import numpy as np
        np.testing.assert_allclose(outputs, expected_predictions, rtol=0, atol=1e-12)
    return {"extract_to_fresh_tempdir": True, "loader_and_reference_passed": True}

