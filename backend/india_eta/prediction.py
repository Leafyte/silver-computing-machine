"""Safe JSON-only linear ETA fixture inference; no dynamic code or model paths."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .errors import EtaError, invalid

SCHEMA = "india_eta.prediction.request.v1"
FEATURE_SCHEMA_HASH = "sha256:25c0fe95ab8d1cb722a5c7f8c3b9d059c09004ddfee1ebd7bfd5529930838563"
_MODEL_ID = "india-eta-linear-fixture"
_MODEL_VERSION = "1.0.0"
_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "india_eta" / "prediction_model.json"
_EXPECTED_FIXTURE_SHA256 = "f644ed71adeecc4d8f2399910a087f57db6e5774ea40fe6f8534e2fa017a036a"
_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}$")


def _finite(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise invalid("invalid_timestamp", "Timestamp must be an ISO-8601 string with a timezone")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise invalid("invalid_timestamp", "Timestamp must be an ISO-8601 string with a timezone") from None
    if result.tzinfo is None or result.utcoffset() is None:
        raise invalid("invalid_timestamp", "Timestamp must include a timezone offset")
    return result


def _canonical_hash(release: Mapping[str, Any]) -> str:
    body = {k: v for k, v in release.items() if k != "content_sha256"}
    encoded = json.dumps(_plain(body), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def load_model_registry(root: str | Path, allowlist: Mapping[str, Mapping[str, str]]) -> Mapping[str, Any]:
    """Load only server-configured relative JSON paths pinned by external SHA-256 values.

    Never pass request data as `root` or `allowlist`; these are deployment config.
    The returned outer registry and nested releases are immutable snapshots.
    """
    try:
        root_path = Path(root).resolve(strict=True)
    except OSError:
        raise EtaError("model_unavailable", "Configured ETA model root is unavailable", 503) from None
    if not isinstance(allowlist, Mapping) or len(allowlist) > 32:
        raise EtaError("model_unavailable", "Configured ETA model allowlist is invalid", 503)
    loaded: dict[str, Any] = {}
    for key, config in allowlist.items():
        if not isinstance(key, str) or not _KEY.fullmatch(key) or "@" not in key:
            raise EtaError("model_unavailable", "Configured ETA model key is invalid", 503)
        if not isinstance(config, Mapping) or set(config) != {"relative_path", "sha256"}:
            raise EtaError("model_unavailable", "Configured ETA model entry is invalid", 503)
        relative = config["relative_path"]
        expected = config["sha256"]
        if not isinstance(relative, str) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise EtaError("model_unavailable", "Configured ETA model entry is invalid", 503)
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or relative_path.suffix.lower() != ".json":
            raise EtaError("model_unavailable", "Configured ETA model path must be relative JSON", 503)
        try:
            path = (root_path / relative_path).resolve(strict=True)
        except OSError:
            raise EtaError("model_unavailable", "Configured ETA model file is unavailable", 503) from None
        try:
            path.relative_to(root_path)
        except ValueError:
            raise EtaError("model_unavailable", "Configured ETA model path escapes its root", 503) from None
        if path.stat().st_size > 131_072:
            raise EtaError("model_unavailable", "Configured ETA model file exceeds the size limit", 503)
        try:
            release = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise EtaError("model_unavailable", "Configured ETA model JSON is invalid", 503) from None
        digest = _canonical_hash(release) if isinstance(release, Mapping) else ""
        if digest != expected or release.get("content_sha256") != digest:
            raise EtaError("model_integrity_error", "Configured ETA model failed integrity verification", 503)
        loaded[key] = MappingProxyType({"release": _freeze(release), "sha256": expected})
    return MappingProxyType(loaded)


def _load_release(registry: Mapping[str, Any] | None, model_id: str, model_version: str) -> tuple[Mapping[str, Any], str]:
    key = f"{model_id}@{model_version}"
    if registry is None:
        if key != f"{_MODEL_ID}@{_MODEL_VERSION}":
            raise EtaError("model_unavailable", "Requested ETA release is not allowlisted", 503)
        try:
            release = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise EtaError("model_unavailable", "Allowlisted ETA fixture release is unavailable", 503) from None
        digest = _canonical_hash(release)
        if digest != _EXPECTED_FIXTURE_SHA256 or release.get("content_sha256") != digest:
            raise EtaError("model_integrity_error", "Allowlisted ETA fixture failed integrity verification", 503)
        return _freeze(release), digest

    # Parent injection is a trusted server-configured registry: key -> {release, sha256}.
    if not isinstance(registry, Mapping) or len(registry) > 32:
        raise EtaError("model_unavailable", "Injected ETA registry is invalid", 503)
    entry = registry.get(key)
    if not isinstance(entry, Mapping) or set(entry) != {"release", "sha256"}:
        raise EtaError("model_unavailable", "Requested ETA release is not allowlisted", 503)
    release = entry["release"]
    if not isinstance(release, Mapping) or not isinstance(entry["sha256"], str):
        raise EtaError("model_unavailable", "Injected ETA release is invalid", 503)
    digest = _canonical_hash(release)
    if digest != entry["sha256"] or release.get("content_sha256") != digest:
        raise EtaError("model_integrity_error", "Injected ETA release failed integrity verification", 503)
    return _freeze(_plain(release)), digest


def _validate_release(release: Mapping[str, Any], model_id: str, model_version: str) -> None:
    required = {"model_id", "model_version", "prediction_target", "duration_unit", "feature_schema_hash",
                "feature_names", "coefficients", "intercept", "supported_geographies", "supported_vehicles",
                "model_kind", "provenance", "content_sha256"}
    if set(release) != required:
        raise EtaError("model_unavailable", "ETA release schema is unsupported", 503)
    if (release["model_id"], release["model_version"]) != (model_id, model_version):
        raise EtaError("model_unavailable", "ETA release is not allowlisted", 503)
    if release["model_kind"] not in {"fixture", "frozen_classical"}:
        raise EtaError("model_unavailable", "ETA release type is unsupported", 503)
    names, coeffs = release["feature_names"], release["coefficients"]
    if not isinstance(names, (list, tuple)) or list(names) != ["distance_km", "traffic_index"]:
        raise EtaError("model_unavailable", "ETA feature schema is unsupported", 503)
    if not isinstance(coeffs, (list, tuple)) or len(coeffs) != len(names) or not all(_finite(x) for x in coeffs):
        raise EtaError("model_unavailable", "ETA release parameters are invalid", 503)
    if not _finite(release["intercept"]) or release["duration_unit"] != "minute" or release["prediction_target"] != "travel_time":
        raise EtaError("model_unavailable", "ETA target or units are unsupported", 503)
    if release["feature_schema_hash"] != FEATURE_SCHEMA_HASH:
        raise EtaError("model_unavailable", "ETA feature schema is unsupported", 503)
    if not isinstance(release["supported_geographies"], (list, tuple)) or not isinstance(release["supported_vehicles"], (list, tuple)):
        raise EtaError("model_unavailable", "ETA support declaration is invalid", 503)
    if not isinstance(release["provenance"], Mapping):
        raise EtaError("model_unavailable", "ETA provenance is invalid", 503)
    if release["model_kind"] == "frozen_classical":
        provenance = release["provenance"]
        if (provenance.get("status") != "validated_source_domain" or
                not all(isinstance(provenance.get(k), str) and provenance[k] for k in ("source", "data_hash", "split_hash", "training_data_hash"))):
            raise EtaError("model_unavailable", "Frozen ETA provenance is incomplete", 503)


def _location(value: Any) -> None:
    if not isinstance(value, dict) or set(value) - {"id", "latitude", "longitude"} or "id" not in value:
        raise invalid("invalid_location", "Location schema is invalid")
    if not isinstance(value["id"], str) or not _KEY.fullmatch(value["id"]):
        raise invalid("invalid_location", "Location identifier is invalid")
    has_lat, has_lon = "latitude" in value, "longitude" in value
    if has_lat != has_lon:
        raise invalid("invalid_location", "Coordinates must include latitude and longitude")
    if has_lat:
        lat, lon = value["latitude"], value["longitude"]
        if not _finite(lat) or not _finite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise invalid("invalid_location", "Coordinates are invalid")


def predict_eta(payload: dict[str, Any], registry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Predict a travel-only duration from a hash-verified allowlisted JSON model."""
    started = time.perf_counter()
    required = {"schema_version", "request_id", "prediction_time", "source_domain", "geography",
                "origin", "destination", "vehicle_type", "features", "model_id", "model_version"}
    optional = {"feature_schema_hash", "qml_requested"}
    if not isinstance(payload, dict) or set(payload) - (required | optional) or not required <= set(payload):
        raise invalid("invalid_schema", "Prediction request schema is invalid")
    if "qml_requested" in payload and not isinstance(payload["qml_requested"], bool):
        raise invalid("invalid_qml_flag", "Quantum request flag must be a boolean")
    if payload["schema_version"] != SCHEMA:
        raise invalid("unsupported_schema", "Prediction request schema version is unsupported")
    if not isinstance(payload["request_id"], str) or not _KEY.fullmatch(payload["request_id"]):
        raise invalid("invalid_request_id", "Request identifier is invalid")
    as_of = _timestamp(payload["prediction_time"])
    if payload["source_domain"] != "india_food_delivery" or payload["geography"] != "IN":
        raise EtaError("unsupported_domain", "ETA release does not support this source domain or geography", 422)
    _location(payload["origin"])
    _location(payload["destination"])
    if payload["origin"]["id"] == payload["destination"]["id"]:
        raise invalid("invalid_route", "Origin and destination must differ")
    if not isinstance(payload["features"], dict) or set(payload["features"]) != {"distance_km", "traffic_index", "observed_at"}:
        raise invalid("invalid_features", "Feature schema is invalid")
    features = payload["features"]
    if not _finite(features["distance_km"]) or features["distance_km"] <= 0:
        raise invalid("invalid_features", "Distance must be finite and positive")
    if not _finite(features["traffic_index"]) or not 0 <= features["traffic_index"] <= 1:
        raise invalid("invalid_features", "Traffic index must be between zero and one")
    if not isinstance(features["observed_at"], dict) or set(features["observed_at"]) != {"distance_km", "traffic_index"}:
        raise invalid("invalid_features", "Feature observation timestamps are invalid")
    for stamp in features["observed_at"].values():
        if _timestamp(stamp).astimezone(as_of.tzinfo) > as_of:
            raise invalid("future_feature", "Feature observation must be no later than prediction time")
    if "feature_schema_hash" in payload and payload["feature_schema_hash"] != FEATURE_SCHEMA_HASH:
        raise EtaError("feature_schema_mismatch", "Request feature schema does not match ETA release", 422)
    if not isinstance(payload["model_id"], str) or not isinstance(payload["model_version"], str):
        raise EtaError("model_unavailable", "Requested ETA release is not allowlisted", 503)
    if not _KEY.fullmatch(payload["model_id"]) or not _KEY.fullmatch(payload["model_version"]):
        raise EtaError("model_unavailable", "Requested ETA release is not allowlisted", 503)
    release, digest = _load_release(registry, payload["model_id"], payload["model_version"])
    _validate_release(release, payload["model_id"], payload["model_version"])
    if payload["vehicle_type"] not in release["supported_vehicles"]:
        raise EtaError("unsupported_vehicle", "ETA release does not support this vehicle type", 422)
    if payload["geography"] not in release["supported_geographies"]:
        raise EtaError("unsupported_domain", "ETA release does not support this geography", 422)
    duration = float(release["intercept"] + sum(
        coefficient * float(features[name]) for name, coefficient in zip(release["feature_names"], release["coefficients"])
    ))
    if not math.isfinite(duration) or duration <= 0:
        raise EtaError("invalid_prediction", "ETA release produced a non-positive duration", 500)
    arrival = as_of + timedelta(minutes=duration)
    model_kind = release["model_kind"]
    is_fixture = model_kind == "fixture"
    return {
        "schema_version": "india_eta.prediction.response.v1", "request_id": payload["request_id"],
        "prediction_time": as_of.isoformat(),
        "predicted_duration_minutes": duration, "predicted_arrival": arrival.isoformat(),
        "prediction_target": "travel_time", "prediction_source": "classical_fixture" if is_fixture else "classical_frozen_model",
        "model_id": release["model_id"], "model_version": release["model_version"], "model_sha256": digest,
        "feature_schema_hash": FEATURE_SCHEMA_HASH, "data_hash": release["provenance"].get("data_hash"),
        "split_hash": release["provenance"].get("split_hash"),
        "provenance_status": release["provenance"].get("status", "unavailable"),
        "source_domain": payload["source_domain"], "geography": payload["geography"],
        "vehicle_type": payload["vehicle_type"],
        "condition_context": {"distance_km": float(features["distance_km"]), "traffic_index": float(features["traffic_index"])},
        "qml_requested": payload.get("qml_requested", False), "qml_executed": False, "quantum_residual_minutes": None,
        "hybrid_duration_minutes": None, "execution_type": "classical_fixture" if is_fixture else "classical_json_model", "backend": "local_json_linear",
        "runtime_ms": (time.perf_counter() - started) * 1000.0,
        "fallback_used": True, "fallback_reason": "duration_quantum_checkpoint_missing",
    }
