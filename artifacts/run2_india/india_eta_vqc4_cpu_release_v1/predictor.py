"""CPU-only inference and integrity loader for a packaged RUN2 4Q release.

This module is copied as ``predictor.py`` into the standalone bundle.  It does
not read the source dataset, OOF predictions, validation rows, or test rows.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

# Keep imports from creating unlisted bytecode files inside the exact-inventory
# release directory when the CLI is run on a normal Python installation.
sys.dont_write_bytecode = True

try:
    from .circuit import NumpyFourQubitResidualVQC, FEATURE_NAMES
except ImportError:  # standalone release directory
    from circuit import NumpyFourQubitResidualVQC, FEATURE_NAMES


RELEASE_FORMAT = "india_eta_vqc4_cpu_release_v1"
MANIFEST_SHA256 = "1a0a2052b655a4306189ed0fe0d5f99be29b0379b0e322cf31aa514eb9200c5b"
SKLEARN_VERSION = "1.4.2"
NUMPY_VERSION = "1.26.4"
JOBLIB_VERSION = "1.6.0"
GBDT_FEATURE_COUNT = 59
EXPECTED_FILES = {
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


def _safe_name(name: Any) -> str:
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("release artifact paths must be safe basenames")
    if "/" in name or "\\" in name:
        raise ValueError("release artifact paths cannot contain directory separators")
    return name


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path.name}")
    return value


class IndiaETARelease:
    def __init__(self, base: Path, manifest: dict[str, Any], feature_names: list[str],
                 preprocessing: dict[str, Any], gbdt: Any, ridge: dict[str, Any],
                 fusion: dict[str, Any], models: dict[str, NumpyFourQubitResidualVQC]):
        self.base = base
        self.manifest = manifest
        self.feature_names = feature_names
        self.preprocessing = preprocessing
        self.gbdt = gbdt
        self.ridge = ridge
        self.fusion = fusion
        self.models = models

    def _matrix(self, rows: Any, feature_names: list[str] | tuple[str, ...] | None = None) -> np.ndarray:
        if feature_names is not None and list(feature_names) != self.feature_names:
            raise ValueError("input feature order does not match the release schema")
        if isinstance(rows, np.ndarray):
            raw = np.asarray(rows)
            if raw.dtype.kind not in "iuf":
                raise ValueError("input matrix must use a real numeric dtype (booleans and object arrays are rejected)")
            values = raw.astype(np.float64, copy=False)
        elif isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows):
            expected = set(self.feature_names)
            values_list = []
            for i, row in enumerate(rows):
                if set(row) != expected:
                    missing = sorted(expected - set(row))
                    extra = sorted(set(row) - expected)
                    raise ValueError(f"row {i} feature mismatch: missing={missing}, extra={extra}")
                vals = []
                for name in self.feature_names:
                    value = row[name]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        raise ValueError(f"row {i} feature {name} must be a finite number")
                    vals.append(float(value))
                values_list.append(vals)
            values = np.asarray(values_list, dtype=np.float64)
        else:
            raise ValueError("rows must be a non-empty matrix or list of feature dictionaries")
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != len(self.feature_names):
            raise ValueError(f"input matrix must have non-empty shape [rows,{len(self.feature_names)}]")
        if not np.isfinite(values).all():
            raise ValueError("all release inputs must be finite")
        return values

    def _vqc_features(self, x: np.ndarray) -> np.ndarray:
        index = {name: i for i, name in enumerate(self.feature_names)}
        preprocessing = self.preprocessing
        scale = preprocessing["scaling"]
        dmin = float(scale["haversine_distance_min"])
        dmax = float(scale["haversine_distance_max"])
        if not dmax > dmin:
            raise ValueError("invalid frozen distance scaling bounds")
        distance = np.clip((x[:, index["haversine_distance_km"]] - dmin) / (dmax - dmin), 0.0, 1.0)
        congestion = np.clip(
            0.60 * x[:, index["traffic_density_ordinal_imp"]] / 3.0
            + 0.25 * x[:, index["is_rush_hour"]]
            + 0.15 * x[:, index["is_adverse_weather"]],
            0.0, 1.0,
        )
        multidelivery = np.clip(x[:, index["multiple_deliveries_imp"]] / 3.0, 0.0, 1.0)
        preptime = np.clip((x[:, index["order_to_pickup_min_imp"]] - 5.0) / 10.0, 0.0, 1.0)
        return np.column_stack((distance, congestion, multidelivery, preptime))

    def predict(self, rows: Any, feature_names: list[str] | tuple[str, ...] | None = None) -> dict[str, list[float]]:
        """Return baseline, control, both candidates, and frozen operational ETA in minutes."""
        x = self._matrix(rows, feature_names)
        qx = self._vqc_features(x)
        baseline = np.asarray(self.gbdt.predict(x), dtype=np.float64)
        if baseline.shape != (len(x),) or not np.isfinite(baseline).all():
            raise RuntimeError("GBDT returned invalid predictions")
        control_weights = np.asarray(self.ridge["coefficients_min_per_unit"], dtype=np.float64)
        control = np.maximum(0.0, baseline + qx @ control_weights + float(self.ridge["intercept_min"]))
        candidates: dict[str, np.ndarray] = {}
        for name in ("baseline60", "extended76"):
            residual = np.asarray(self.models[name].predict_residual_minutes(qx), dtype=np.float64).reshape(-1)
            if residual.shape != (len(x),) or not np.isfinite(residual).all():
                raise RuntimeError(f"{name} returned invalid residual corrections")
            candidates[name] = np.maximum(0.0, baseline + residual)
        alpha = float(self.fusion["alpha"])
        selected = self.fusion.get("selected_variant")
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("fusion alpha must be finite and in [0,1]")
        if selected not in (None, "baseline60", "extended76"):
            raise ValueError("fusion selected_variant must be null or a known model")
        if alpha == 0.0 or selected is None:
            operational = np.maximum(0.0, baseline)
        else:
            correction = np.asarray(self.models[selected].predict_residual_minutes(qx), dtype=np.float64).reshape(-1)
            operational = np.maximum(0.0, baseline + alpha * correction)
        output = {
            "gbdt_only_min": np.maximum(0.0, baseline),
            "ridge_control_min": control,
            "baseline60_candidate_min": candidates["baseline60"],
            "extended76_candidate_min": candidates["extended76"],
            "operational_min": operational,
        }
        if any(not np.isfinite(v).all() for v in output.values()):
            raise RuntimeError("release produced non-finite predictions")
        return {key: values.astype(float).tolist() for key, values in output.items()}


def load_release(release_dir: str | Path | None = None) -> IndiaETARelease:
    """Verify the manifest and every member before deserializing the trusted estimator."""
    base = Path(release_dir) if release_dir is not None else Path(__file__).resolve().parent
    base = base.resolve()
    manifest_path = base / "release_manifest.json"
    if not manifest_path.is_file() or sha256(manifest_path) != MANIFEST_SHA256:
        raise ValueError("release manifest is missing or does not match the loader's pinned SHA-256")
    manifest = _read_json(manifest_path)
    if manifest.get("format") != RELEASE_FORMAT or manifest.get("test_status") != "retrospective_exposed_partition":
        raise ValueError("unsupported release format or missing exposed-test qualification")
    if (np.__version__, sklearn.__version__, joblib.__version__) != (NUMPY_VERSION, SKLEARN_VERSION, JOBLIB_VERSION):
        raise RuntimeError(
            f"runtime mismatch: numpy={np.__version__}, sklearn={sklearn.__version__}, joblib={joblib.__version__}; "
            f"required={NUMPY_VERSION}, {SKLEARN_VERSION}, {JOBLIB_VERSION}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != EXPECTED_FILES:
        raise ValueError("release artifact inventory is incomplete or contains unexpected files")
    expected_dir_files = EXPECTED_FILES | {"release_manifest.json", "predictor.py"}
    members = list(base.iterdir())
    files = {item.name for item in members if item.is_file() and not item.is_symlink()}
    directories = [item for item in members if item.is_dir()]
    if any(item.is_symlink() for item in members) or files != expected_dir_files or any(item.name != "__pycache__" for item in directories):
        raise ValueError("release directory has missing, extra, or nested members")
    for cache in directories:
        for cached in cache.iterdir():
            if cached.is_symlink() or not cached.is_file() or not cached.name.endswith(".pyc") or not cached.name.startswith(("circuit.", "predictor.")):
                raise ValueError("release __pycache__ contains unexpected or unsafe members")
    for key in sorted(EXPECTED_FILES):
        entry = artifacts[key]
        if not isinstance(entry, dict) or _safe_name(entry.get("path")) != key:
            raise ValueError(f"unsafe or invalid path for release artifact {key}")
        path = base / key
        actual = sha256(path)
        if path.stat().st_size != entry.get("bytes") or actual != entry.get("sha256"):
            raise ValueError(f"artifact integrity mismatch for {key}")
    schema = _read_json(base / "feature_schema.json")
    feature_names = schema.get("feature_names")
    if not isinstance(feature_names, list) or len(feature_names) != GBDT_FEATURE_COUNT or len(set(feature_names)) != GBDT_FEATURE_COUNT:
        raise ValueError("release must contain 59 unique ordered GBDT features")
    if set(feature_names) & {"ID", "Delivery_person_ID", "split_partition", "delivery_duration_min"}:
        raise ValueError("release feature schema contains an identifier, split, or target")
    if manifest.get("feature_names") != feature_names or manifest.get("feature_count") != GBDT_FEATURE_COUNT:
        raise ValueError("release feature order disagrees with the manifest")
    # Hashes and inventory are checked before this trusted local joblib is deserialized.
    gbdt = joblib.load(base / "gbdt_model.joblib")
    if gbdt.n_features_in_ != GBDT_FEATURE_COUNT or "HistGradientBoostingRegressor" not in str(type(gbdt)):
        raise ValueError("unexpected GBDT estimator identity or input dimension")
    preprocessing = _read_json(base / "preprocessing_artifacts.json")
    ridge = _read_json(base / "ridge_control.json")
    fusion = _read_json(base / "fusion_policy.json")
    models = {}
    for name in ("baseline60", "extended76"):
        payload = _read_json(base / f"{name}.json")
        model = NumpyFourQubitResidualVQC.from_payload(payload)
        if model.variant != name or tuple(model.config.feature_names) != tuple(FEATURE_NAMES):
            raise ValueError(f"{name} checkpoint configuration mismatch")
        models[name] = model
    return IndiaETARelease(base, manifest, feature_names, preprocessing, gbdt, ridge, fusion, models)


def verify_reference(release_dir: str | Path | None = None) -> dict[str, list[float]]:
    base = Path(release_dir) if release_dir is not None else Path(__file__).resolve().parent
    fixture = _read_json(base / "reference_fixture.json")
    if fixture.get("synthetic_only") is not True or fixture.get("not_from_dataset") is not True:
        raise ValueError("release reference rows must be explicitly synthetic")
    runtime = load_release(base)
    actual = runtime.predict(fixture.get("rows"), fixture.get("feature_names"))
    expected = fixture.get("expected_predictions_min")
    if not isinstance(expected, dict) or set(expected) != set(actual):
        raise ValueError("synthetic reference predictions are incomplete")
    for name, values in actual.items():
        if not np.allclose(values, expected[name], rtol=0.0, atol=1e-10):
            raise AssertionError(f"synthetic release parity failed for {name}")
    return actual


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Portable CPU-only India ETA release")
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--verify-reference", action="store_true")
    args = parser.parse_args()
    if args.verify_reference:
        print(json.dumps({"status": "pass", "predictions": verify_reference()}, allow_nan=False))
        return
    if args.input_json is None:
        parser.error("provide --input-json or --verify-reference")
    payload = _read_json(args.input_json)
    runtime = load_release()
    result = runtime.predict(payload.get("rows"), payload.get("feature_names"))
    print(json.dumps({"unit": "minutes", "predictions": result}, allow_nan=False))


if __name__ == "__main__":
    main()
