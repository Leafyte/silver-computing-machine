"""Package and verify the already-frozen India ETA GBDT; does not train."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn


MODEL_SHA256 = "f1f94fbcc932f3519aaafcff7dc99d65b7951c0240730e54c320259152bbebdf"
SOURCE_SCHEMA_SHA256 = "a3b013ec731e5b6767f45a6c6a5d87c99cdffefe8a06976a02f46c97a966c7d5"
PREPROCESS_SHA256 = "df20646f12b77cb9fb03bf831b3373e5f8c9f69d6e5f0c92c35bbd46c682f351"
SKLEARN_VERSION = "1.4.2"
NUMPY_VERSION = "1.26.4"
JOBLIB_VERSION = "1.6.0"
FILES = [
    "README.md",
    "feature_schema.json",
    "model.joblib",
    "predictor.py",
    "preprocessing_artifacts.json",
    "reference_fixture.json",
    "release_manifest.json",
    "requirements.txt",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_reference_rows(feature_schema: dict) -> tuple[list[str], list[dict[str, float]]]:
    names = list(feature_schema["feature_names"])
    rows = []
    base_rows = [
        {
            "haversine_distance_km": 2.4, "departure_hour": 12, "day_of_week": 2, "is_weekend": 0,
            "is_rush_hour": 0, "is_lunch_rush": 1, "is_dinner_rush": 0,
            "delivery_person_age_imp": 29, "delivery_person_ratings_imp": 4.8,
            "order_to_pickup_min_imp": 7, "order_to_pickup_missing_flag": 0,
            "vehicle_condition": 1, "multiple_deliveries_imp": 0, "is_festival": 0,
            "traffic_density_ordinal_imp": 0, "is_adverse_weather": 0,
        },
        {
            "haversine_distance_km": 10.7, "departure_hour": 18, "day_of_week": 5, "is_weekend": 0,
            "is_rush_hour": 1, "is_lunch_rush": 0, "is_dinner_rush": 1,
            "delivery_person_age_imp": 37, "delivery_person_ratings_imp": 4.5,
            "order_to_pickup_min_imp": 11, "order_to_pickup_missing_flag": 0,
            "vehicle_condition": 2, "multiple_deliveries_imp": 2, "is_festival": 1,
            "traffic_density_ordinal_imp": 3, "is_adverse_weather": 1,
        },
        {
            "haversine_distance_km": 18.2, "departure_hour": 3, "day_of_week": 6, "is_weekend": 1,
            "is_rush_hour": 0, "is_lunch_rush": 0, "is_dinner_rush": 0,
            "delivery_person_age_imp": 45, "delivery_person_ratings_imp": 4.1,
            "order_to_pickup_min_imp": 15, "order_to_pickup_missing_flag": 1,
            "vehicle_condition": 0, "multiple_deliveries_imp": 3, "is_festival": 0,
            "traffic_density_ordinal_imp": 2, "is_adverse_weather": 1,
        },
    ]
    cats = [
        {"weather_clean_imp": "Sunny", "traffic_density_clean": "Low", "type_of_order_clean": "Snack", "type_of_vehicle_clean": "motorcycle", "city_type_clean_imp": "Urban", "city_code": "MUM"},
        {"weather_clean_imp": "Stormy", "traffic_density_clean": "Jam", "type_of_order_clean": "Buffet", "type_of_vehicle_clean": "bicycle", "city_type_clean_imp": "Semi-Urban", "city_code": "BANG"},
        {"weather_clean_imp": "Cloudy", "traffic_density_clean": "Medium", "type_of_order_clean": "Drinks", "type_of_vehicle_clean": "electric_scooter", "city_type_clean_imp": "Metropolitian", "city_code": "CHEN"},
    ]
    for values, chosen in zip(base_rows, cats):
        row = {name: 0.0 for name in names}
        row.update({k: float(v) for k, v in values.items()})
        for group, category in chosen.items():
            column = f"{group}_{category}"
            if column not in row:
                raise ValueError(f"Reference category is absent from model schema: {column}")
            row[column] = 1.0
        rows.append({name: float(row[name]) for name in names})
    return names, rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_dir = root / "artifacts" / "india_eta_recovery" / "gbdt"
    release = root / "artifacts" / "india_eta_classical_release_v1"
    model_src = source_dir / "india_eta_hgb_regressor.joblib"
    schema_src = source_dir / "feature_schema.json"
    preprocessing_src = source_dir / "preprocessing_artifacts.json"
    if sha256(model_src) != MODEL_SHA256:
        raise SystemExit("Frozen source model SHA mismatch; refusing to package")
    if sha256(schema_src) != SOURCE_SCHEMA_SHA256:
        raise SystemExit("Frozen feature schema SHA mismatch; refusing to package")
    if sha256(preprocessing_src) != PREPROCESS_SHA256:
        raise SystemExit("Frozen preprocessing artifact SHA mismatch; refusing to package")
    release.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_src, release / "model.joblib")
    shutil.copyfile(schema_src, release / "feature_schema.json")
    shutil.copyfile(preprocessing_src, release / "preprocessing_artifacts.json")

    schema_path = release / "feature_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    # Rewrite repository-relative metadata paths to portable bundle-local paths;
    # the 59 names and their order remain unchanged from the committed schema.
    schema["preprocessing_artifacts_source_path"] = "preprocessing_artifacts.json"
    schema["preprocessing_artifacts_path"] = "preprocessing_artifacts.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    packed_schema_sha = sha256(schema_path)
    names = list(schema["feature_names"])
    rows = make_reference_rows(schema)
    if rows[0] != names or len(names) != 59:
        raise ValueError("Portable feature schema must have exactly 59 ordered inputs")
    forbidden = {"ID", "Delivery_person_ID", "split_partition", "delivery_duration_min"}
    if forbidden & set(names):
        raise ValueError("Portable predictor schema includes forbidden identifiers or target")
    # Verify the pinned checkpoint digest before deserializing, including during package creation.
    if sha256(release / "model.joblib") != MODEL_SHA256:
        raise SystemExit("Copied model SHA mismatch; refusing to deserialize")
    if sklearn.__version__ != SKLEARN_VERSION or np.__version__ != NUMPY_VERSION or joblib.__version__ != JOBLIB_VERSION:
        raise SystemExit("Packaging environment differs from frozen model dependency versions")
    model = joblib.load(release / "model.joblib")
    reference_matrix = np.asarray([[row[name] for name in names] for row in rows[1]], dtype=np.float64)
    expected = np.asarray(model.predict(reference_matrix), dtype=np.float64)
    fixture = {
        "format": "india-eta-synthetic-reference-v1",
        "synthetic_only": True,
        "not_from_dataset": True,
        "feature_names": names,
        "rows": rows[1],
        "expected_predictions_min": [float(x) for x in expected],
    }
    fixture_path = release / "reference_fixture.json"
    fixture_path.write_text(json.dumps(fixture, indent=2, allow_nan=False), encoding="utf-8")

    ref_sha = sha256(fixture_path)
    (release / "predictor.py").write_text(_predictor_source(ref_sha, packed_schema_sha), encoding="utf-8")
    (release / "requirements.txt").write_text(
        "numpy==1.26.4\nscikit-learn==1.4.2\njoblib==1.6.0\n",
        encoding="utf-8",
    )
    recovery = json.loads((root / "artifacts" / "india_eta_recovery" / "recovery_manifest.json").read_text(encoding="utf-8"))
    (release / "README.md").write_text(_readme(recovery), encoding="utf-8")
    payloads = {
        name: {
            "path": name,
            "bytes": (release / name).stat().st_size,
            "sha256": sha256(release / name),
        }
        for name in FILES
        if name != "release_manifest.json"
    }
    manifest = {
        "release_name": "India ETA classical CPU release",
        "release_version": "1.0.0",
        "created_from_frozen_checkpoint": True,
        "training_performed_during_packaging": False,
        "model": {
            "estimator_identity": "sklearn.ensemble.HistGradientBoostingRegressor",
            "objective": "squared_error regression",
            "target": "delivery_duration_min",
            "target_unit": "minutes",
            "config": recovery["model_selection"]["chosen_config"],
            "feature_count": 59,
            "feature_order": names,
            "source_checkpoint_sha256": MODEL_SHA256,
            "validation_metrics_min": recovery["frozen_model_metrics"],
            "baseline_metrics_min": recovery["baseline"],
        },
        "data_provenance": {
            "source_commit": recovery["data_provenance"]["source_commit"],
            "canonical_source_csv_sha256": recovery["data_provenance"]["source_csv"]["canonical_git_sha256_lf"],
            "worktree_source_csv_sha256_crlf": recovery["data_provenance"]["source_csv"]["worktree_sha256_crlf"],
            "gbdt_parquet_sha256": recovery["data_provenance"]["files_sha256"]["gbdt_parquet"],
            "split_manifest_sha256": recovery["data_provenance"]["split_manifest_sha256"],
            "source_feature_schema_sha256": SOURCE_SCHEMA_SHA256,
            "test_metrics_included": False,
        },
        "dependencies": {
            "python": "3.12.10",
            "numpy": NUMPY_VERSION,
            "scikit_learn": SKLEARN_VERSION,
            "joblib": JOBLIB_VERSION,
            "inference_device": "CPU",
        },
        "preprocessing_availability": {
            "artifact_included": "preprocessing_artifacts.json",
            "artifact_sha256": PREPROCESS_SHA256,
            "raw_record_feature_engineering_code_included": False,
            "input_contract": "Caller supplies the committed 59 numeric GBDT features in exact schema order; the release does not include raw dataset files or the full raw-record preprocessing pipeline.",
        },
        "synthetic_reference": {
            "path": "reference_fixture.json",
            "sha256": ref_sha,
            "rows": len(rows[1]),
            "expected_predictions_min": [float(x) for x in expected],
        },
        "artifacts": payloads,
        "loader_security": "predictor.py checks hard-coded model, schema, preprocessing, and fixture SHA-256 pins before joblib.load; archive consumers should also verify the external ZIP SHA-256 file.",
    }
    manifest_path = release / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    names_for_zip = sorted(FILES)
    archive_path = root / "artifacts" / "india_eta_classical_release_v1.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name in names_for_zip:
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 29, 12, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (release / name).read_bytes())
    archive_hash = sha256(archive_path)
    (root / "artifacts" / "india_eta_classical_release_v1.zip.sha256").write_text(
        f"{archive_hash}  india_eta_classical_release_v1.zip\n", encoding="ascii"
    )
    print(json.dumps({
        "release_dir": str(release),
        "archive": str(archive_path),
        "archive_sha256": archive_hash,
        "model_sha256": MODEL_SHA256,
        "reference_predictions_min": [float(x) for x in expected],
        "bundle_files": names_for_zip,
    }, indent=2))


def _readme(recovery: dict) -> str:
    m = recovery["frozen_model_metrics"]["all_validation_descriptive"]
    return f"""# India ETA classical CPU release

This is an inference-only bundle for the already-frozen `HistGradientBoostingRegressor` ETA reconstruction. Packaging does not retrain the model. It predicts `delivery_duration_min` in minutes from 59 preprocessed numeric features.

## Load and predict

Requires Python 3.12 and the exact dependencies in `requirements.txt`. Verify the external ZIP digest before extraction:

```powershell
Get-FileHash ./india_eta_classical_release_v1.zip -Algorithm SHA256
python predictor.py --input-json ./input.json
```

The input JSON has exactly this shape, with the full ordered feature-name list and one or more rows keyed by those feature names:

```json
{{"feature_names": ["...59 names in feature_schema.json order..."], "rows": [{{"feature_a": 1.0, "...": 0.0}}]}}
```

The prediction loader checks hard-coded SHA-256 pins for the checkpoint, feature schema, preprocessing artifact, and synthetic reference fixture before it calls `joblib.load`. It then checks the recorded scikit-learn, NumPy, and joblib versions and validates that every row has exactly the 59 expected finite numeric features.

## Preprocessing boundary

`preprocessing_artifacts.json` contains the committed imputation and scaling parameters. The full raw-record cleaning, geospatial feature engineering, categorical one-hot encoding, and split pipeline is not included. The release contains no raw CSV, Parquet dataset, OOF file, or validation prediction file. A Dell deployment must provide already-engineered features matching `feature_schema.json`; this bundle does not accept raw delivery records.

## Reference check

`reference_fixture.json` contains three synthetic feature rows created for this release and their exact minute-valued outputs. They are not dataset records and are not accuracy evidence. Verify them with:

```powershell
python predictor.py --verify-reference
```

## Provenance and validation

- Frozen model SHA-256: `{MODEL_SHA256}`.
- Estimator: `sklearn.ensemble.HistGradientBoostingRegressor`, squared-error objective, `{json.dumps(recovery['model_selection']['chosen_config'], sort_keys=True)}`.
- Validation-only MAE/RMSE/median absolute error: {m['mae_min']:.6f} / {m['rmse_min']:.6f} / {m['median_absolute_error_min']:.6f} minutes. Sealed test results are not included.
- `release_manifest.json` inventories every bundle file and records source/split hashes, dependency versions, input order, preprocessing availability, and reference outputs.
"""


def _predictor_source(reference_sha: str, packed_schema_sha: str) -> str:
    return f'''"""Hash-checked CPU inference for the India ETA classical release."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import joblib
import sklearn

MODEL_SHA256 = "{MODEL_SHA256}"
SCHEMA_SHA256 = "{packed_schema_sha}"
PREPROCESS_SHA256 = "{PREPROCESS_SHA256}"
REFERENCE_SHA256 = "{reference_sha}"
SKLEARN_VERSION = "{SKLEARN_VERSION}"
NUMPY_VERSION = "{NUMPY_VERSION}"
JOBLIB_VERSION = "{JOBLIB_VERSION}"
REQUIRED_ARTIFACT_NAMES = {{
    "README.md", "feature_schema.json", "model.joblib", "predictor.py",
    "preprocessing_artifacts.json", "reference_fixture.json", "requirements.txt",
}}
EXPECTED_FILES = {{
    "model.joblib": MODEL_SHA256,
    "feature_schema.json": SCHEMA_SHA256,
    "preprocessing_artifacts.json": PREPROCESS_SHA256,
    "reference_fixture.json": REFERENCE_SHA256,
}}

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
            f"Dependency version mismatch: sklearn={{sklearn.__version__}}, numpy={{np.__version__}}, joblib={{joblib.__version__}}; "
            f"required={{SKLEARN_VERSION}}, {{NUMPY_VERSION}}, {{JOBLIB_VERSION}}"
        )
    # Validate the complete inventory and safe in-bundle paths before any model deserialization.
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != REQUIRED_ARTIFACT_NAMES:
        raise ValueError("Release manifest artifact inventory is incomplete or contains unknown members")
    actual_members = {{entry.name for entry in base.iterdir() if entry.is_file()}}
    if actual_members != REQUIRED_ARTIFACT_NAMES | {{"release_manifest.json"}}:
        raise ValueError("Release directory has missing or unlisted files")
    if any(not entry.is_file() for entry in base.iterdir()):
        raise ValueError("Release directory contains an unexpected subdirectory")
    for name in sorted(REQUIRED_ARTIFACT_NAMES):
        record = artifacts[name]
        relative = record.get("path")
        if relative != name or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Unsafe artifact path for {{name}}")
        path = base / name
        actual_size = path.stat().st_size
        actual = sha256(path)
        expected = EXPECTED_FILES.get(name)
        if expected is not None and actual != expected:
            raise ValueError(f"SHA-256 mismatch for {{name}}: {{actual}} != {{expected}}")
        if actual_size != record.get("bytes") or actual != record.get("sha256"):
            raise ValueError(f"Artifact inventory mismatch for {{name}}")
    if manifest.get("model", {{}}).get("source_checkpoint_sha256") != MODEL_SHA256:
        raise ValueError("Release manifest does not name the pinned frozen checkpoint")
    schema = json.loads((base / "feature_schema.json").read_text(encoding="utf-8"))
    features = schema.get("feature_names", [])
    if len(features) != 59 or len(set(features)) != 59:
        raise ValueError("Feature schema must contain 59 unique ordered names")
    forbidden = {{"ID", "Delivery_person_ID", "split_partition", "delivery_duration_min"}}
    if forbidden.intersection(features):
        raise ValueError("Feature schema contains an identifier, split, or target")
    if manifest.get("model", {{}}).get("feature_order") != features:
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
            raise ValueError(f"row {{index}} feature mismatch; missing={{missing}}, extra={{extra}}")
        values = []
        for name in expected_features:
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"row {{index}} feature {{name}} must be a finite number")
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
        print(json.dumps({{"status": "pass", "predictions_min": verify_reference()}}))
        return
    if args.input_json is None:
        parser.error("provide --input-json or --verify-reference")
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    model, features = load_release()
    names = payload.get("feature_names")
    if names is not None and names != features:
        raise SystemExit("Input feature_names do not match release order")
    predictions = predict_rows(payload.get("rows"), feature_names=names or features)
    print(json.dumps({{"prediction_unit": "minutes", "predictions": predictions}}))

if __name__ == "__main__":
    main()
'''


if __name__ == "__main__":
    main()
