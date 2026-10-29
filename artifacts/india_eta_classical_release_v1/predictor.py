"""Hash-checked CPU inference for the India ETA classical release."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import joblib
import sklearn

MODEL_SHA256 = "f1f94fbcc932f3519aaafcff7dc99d65b7951c0240730e54c320259152bbebdf"
SCHEMA_SHA256 = "5743a95aadb02b0e6d1869316ded2b4c9da5bde04cc1737e46be56d6b2a4dcc9"
PREPROCESS_SHA256 = "df20646f12b77cb9fb03bf831b3373e5f8c9f69d6e5f0c92c35bbd46c682f351"
REFERENCE_SHA256 = "fd6d7da8e3d8dd8191fde10a483ad72ef896feabe61be9ee1862add9d28dd806"
SKLEARN_VERSION = "1.4.2"
NUMPY_VERSION = "1.26.4"
JOBLIB_VERSION = "1.6.0"
REQUIRED_ARTIFACT_NAMES = {
    "README.md", "feature_schema.json", "model.joblib", "predictor.py",
    "preprocessing_artifacts.json", "reference_fixture.json", "requirements.txt",
}
EXPECTED_FILES = {
    "model.joblib": MODEL_SHA256,
    "feature_schema.json": SCHEMA_SHA256,
    "preprocessing_artifacts.json": PREPROCESS_SHA256,
    "reference_fixture.json": REFERENCE_SHA256,
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load_release(release_dir: str | Path | None = None):
    base = Path(release_dir) if release_dir is not None else Path(__file__).resolve().parent
    base = base.resolve()
    manifest_path = base / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sklearn.__version__ != SKLEARN_VERSION or np.__version__ != NUMPY_VERSION or joblib.__version__ != JOBLIB_VERSION:
        raise RuntimeError(
            f"Dependency version mismatch: sklearn={sklearn.__version__}, numpy={np.__version__}, joblib={joblib.__version__}; "
            f"required={SKLEARN_VERSION}, {NUMPY_VERSION}, {JOBLIB_VERSION}"
        )
    # Validate the complete inventory and safe in-bundle paths before any model deserialization.
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != REQUIRED_ARTIFACT_NAMES:
        raise ValueError("Release manifest artifact inventory is incomplete or contains unknown members")
    actual_members = {entry.name for entry in base.iterdir() if entry.is_file()}
    if actual_members != REQUIRED_ARTIFACT_NAMES | {"release_manifest.json"}:
        raise ValueError("Release directory has missing or unlisted files")
    if any(not entry.is_file() for entry in base.iterdir()):
        raise ValueError("Release directory contains an unexpected subdirectory")
    for name in sorted(REQUIRED_ARTIFACT_NAMES):
        record = artifacts[name]
        relative = record.get("path")
        if relative != name or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Unsafe artifact path for {name}")
        path = base / name
        actual_size = path.stat().st_size
        actual = sha256(path)
        expected = EXPECTED_FILES.get(name)
        if expected is not None and actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {actual} != {expected}")
        if actual_size != record.get("bytes") or actual != record.get("sha256"):
            raise ValueError(f"Artifact inventory mismatch for {name}")
    if manifest.get("model", {}).get("source_checkpoint_sha256") != MODEL_SHA256:
        raise ValueError("Release manifest does not name the pinned frozen checkpoint")
    schema = json.loads((base / "feature_schema.json").read_text(encoding="utf-8"))
    features = schema.get("feature_names", [])
    if len(features) != 59 or len(set(features)) != 59:
        raise ValueError("Feature schema must contain 59 unique ordered names")
    forbidden = {"ID", "Delivery_person_ID", "split_partition", "delivery_duration_min"}
    if forbidden.intersection(features):
        raise ValueError("Feature schema contains an identifier, split, or target")
    if manifest.get("model", {}).get("feature_order") != features:
        raise ValueError("Feature order disagrees between manifest and schema")
    if sha256(base / "model.joblib") != MODEL_SHA256:
        raise ValueError("Model SHA mismatch; refusing to deserialize")
    model = joblib.load(base / "model.joblib")
    if model.__class__.__module__ != "sklearn.ensemble._hist_gradient_boosting.gradient_boosting" or model.n_features_in_ != 59:
        raise ValueError("Unexpected estimator identity or input dimension")
    return model, features

def predict_rows(rows, release_dir: str | Path | None = None, feature_names=None):
    model, expected_features = load_release(release_dir)
    if feature_names is not None and list(feature_names) != expected_features:
        raise ValueError("Input feature_names do not match the exact release order")
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty list of feature dictionaries")
    matrix = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(expected_features):
            missing = sorted(set(expected_features) - set(row if isinstance(row, dict) else []))
            extra = sorted(set(row if isinstance(row, dict) else []) - set(expected_features))
            raise ValueError(f"row {index} feature mismatch; missing={missing}, extra={extra}")
        values = []
        for name in expected_features:
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"row {index} feature {name} must be a finite number")
            values.append(float(value))
        matrix.append(values)
    predictions = np.asarray(model.predict(np.asarray(matrix, dtype=np.float64)), dtype=np.float64)
    if not np.isfinite(predictions).all():
        raise RuntimeError("Model produced a non-finite ETA")
    return [float(value) for value in predictions]

def verify_reference(release_dir: str | Path | None = None):
    base = Path(release_dir) if release_dir is not None else Path(__file__).resolve().parent
    fixture = json.loads((base / "reference_fixture.json").read_text(encoding="utf-8"))
    if fixture.get("synthetic_only") is not True or fixture.get("not_from_dataset") is not True:
        raise ValueError("Reference fixture is not marked synthetic-only")
    actual = predict_rows(fixture["rows"], base, fixture["feature_names"])
    expected = fixture["expected_predictions_min"]
    if not np.allclose(np.asarray(actual), np.asarray(expected, dtype=np.float64), rtol=0.0, atol=1e-12):
        raise AssertionError("Synthetic reference predictions differ from the release fixture")
    return actual

def main():
    parser = argparse.ArgumentParser(description="CPU-only India ETA inference")
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--verify-reference", action="store_true")
    args = parser.parse_args()
    if args.verify_reference:
        print(json.dumps({"status": "pass", "predictions_min": verify_reference()}))
        return
    if args.input_json is None:
        parser.error("provide --input-json or --verify-reference")
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    model, features = load_release()
    names = payload.get("feature_names")
    if names is not None and names != features:
        raise SystemExit("Input feature_names do not match release order")
    predictions = predict_rows(payload.get("rows"), feature_names=names or features)
    print(json.dumps({"prediction_unit": "minutes", "predictions": predictions}))

if __name__ == "__main__":
    main()
