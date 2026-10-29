"""Explicit synthetic ETA-to-joint-travel adapter; never a real road transfer."""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import hashlib
import json
import math

from backend.domain import validate_scenario
from .compatibility import check_compatibility
from .errors import EtaError
from .prediction import predict_eta, _timestamp, _finite, _load_release, _validate_release

_UNBOUND = object()


def _reject(code="invalid_eta_update", message="Synthetic ETA update schema is invalid"):
    raise EtaError(code, message, 422)


def model_identity_for_version(version, *, registry=None, model_id=None):
    """Select an initial identity from trusted server releases, never client hashes."""
    if not isinstance(version, str) or (model_id is not None and not isinstance(model_id, str)):
        _reject()
    if registry is None:
        candidates = [("india-eta-linear-fixture", "1.0.0")] if version == "1.0.0" else []
    elif isinstance(registry, Mapping) and len(registry) <= 32:
        candidates = [(entry["release"]["model_id"], version) for entry in registry.values()
                      if isinstance(entry, Mapping) and isinstance(entry.get("release"), Mapping)
                      and entry["release"].get("model_version") == version]
    else:
        _reject("model_unavailable", "Server model registry is invalid")
    if model_id is not None:
        candidates = [item for item in candidates if item[0] == model_id]
    if not candidates:
        if model_id is not None:
            _reject("model_unavailable", "Requested initial model identity is not allowlisted")
        return None
    if len(candidates) != 1:
        _reject("ambiguous_model_identity", "Initial registration must select a unique allowlisted model identity")
    ident, model_version = candidates[0]
    release, digest = _load_release(registry, ident, model_version)
    _validate_release(release, ident, model_version)
    return {"model_id": ident, "model_version": model_version, "model_sha256": digest}


def resolve_eta_update(payload, *, registry=None, model_identity=_UNBOUND):
    """Resolve optional server-computed predictions before durable event admission.

    All residuals/worlds are caller-declared synthetic scenarios. This does not
    turn food-delivery observations into calibrated road-routing durations.
    Provenance excludes runtime measurements so identical requests are idempotent.
    """
    if not isinstance(payload, dict) or "eta_update" not in payload:
        return payload
    if payload.get("mode") != "synthetic_scenario":
        _reject("unsupported_domain_transfer", "ETA overlays require explicitly simulated scenario mode")
    try:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        _reject()
    if len(body.encode()) > 256_000:
        _reject("invalid_request", "ETA event exceeds local input budget")
    update = payload["eta_update"]
    if not isinstance(update, dict) or set(update) != {"schema_version", "arcs"} or update["schema_version"] != "india_eta.synthetic_update.v1":
        _reject()
    if not isinstance(update["arcs"], list) or not 1 <= len(update["arcs"]) <= 8:
        _reject()
    source = payload.get("scenario")
    if not validate_scenario(source)["valid"]:
        _reject("invalid_scenario", "Source travel scenario is invalid")
    worlds = payload.get("weighted_scenarios")
    if not isinstance(worlds, list) or not 1 <= len(worlds) <= 100:
        _reject()
    world_ids = []
    for row in worlds:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or row.get("source_label") != "synthetic_overlay"
                or row.get("prediction_target") != "travel_time" or "eta_source_provenance" in row):
            _reject("unsupported_domain_transfer", "ETA overlays require labeled synthetic travel-time worlds")
        world_ids.append(row["id"])
    if len(set(world_ids)) != len(world_ids):
        _reject()
    from .optimization import _validate_terms
    try:
        _validate_terms(source, worlds, payload.get("currency", "XTS"), payload.get("operating_rate", 1.0))
    except (TypeError, ValueError, OverflowError, KeyError):
        _reject("invalid_scenario", "Joint travel scenarios or financial terms are invalid")
    if model_identity is _UNBOUND:
        model_identity = model_identity_for_version(payload.get("model_version"), registry=registry)
    if model_identity is None:
        _reject("unbound_model_identity", "Active plan has no bound ETA model identity")
    event_time = _timestamp(payload.get("as_of"))
    resolved = deepcopy(payload)
    resolved.pop("eta_update")
    seen = set()
    for arc in update["arcs"]:
        required = {"origin", "destination", "prediction_request", "residual_minutes_by_scenario"}
        if not isinstance(arc, dict) or set(arc) != required:
            _reject()
        origin, destination = arc["origin"], arc["destination"]
        if not isinstance(origin, str) or not isinstance(destination, str) or origin == destination or (origin, destination) in seen:
            _reject()
        if origin not in source["travel"] or destination not in source["travel"][origin]:
            _reject("invalid_scenario", "ETA overlay arc is absent from the source travel matrix")
        seen.add((origin, destination))
        request = arc["prediction_request"]
        if (not isinstance(request, dict) or not isinstance(request.get("origin"), dict)
                or not isinstance(request.get("destination"), dict)
                or request["origin"].get("id") != origin or request["destination"].get("id") != destination):
            _reject("invalid_eta_update", "Prediction locations must match the explicit synthetic arc")
        prediction = predict_eta(request, registry=registry)
        actual_identity = {key: prediction[key] for key in ("model_id", "model_version", "model_sha256")}
        if actual_identity != model_identity:
            _reject("model_identity_mismatch", "Prediction model identity differs from the approved plan")
        if prediction["model_version"] != payload.get("model_version"):
            _reject("model_version_mismatch", "Prediction and active event model versions must match")
        if _timestamp(prediction["prediction_time"]) != event_time:
            _reject("stale_prediction", "Prediction time must match the event timestamp")
        compatibility = check_compatibility({
            "schema_version": "india_eta.compatibility.request.v1", "source_domain": "synthetic_overlay",
            "geography": prediction["geography"], "source_units": "minute",
            "prediction_target": prediction["prediction_target"], "road_routing_provenance": "synthetic_fixture",
            "vehicle_type": prediction["vehicle_type"], "features": list(prediction["condition_context"]),
            "time_semantics": "timezone_aware_service_start", "service_semantics": "travel_only",
            "horizon_minutes": prediction["predicted_duration_minutes"], "uncertainty_status": "synthetic_fixture",
            "duration_minutes": prediction["predicted_duration_minutes"],
        })
        if compatibility["decision"] != "scenario_only":
            _reject("unsupported_domain_transfer", "Prediction target cannot be used in this synthetic travel scenario")
        residuals = arc["residual_minutes_by_scenario"]
        if not isinstance(residuals, dict) or set(residuals) != set(world_ids):
            _reject("invalid_eta_update", "Synthetic residuals must cover every frozen joint scenario")
        provenance = {"schema_version": "india_eta.synthetic_source.v1", "source": "explicit_synthetic_overlay",
                      "model_id": prediction["model_id"], "model_version": prediction["model_version"],
                      "model_sha256": prediction["model_sha256"], "feature_schema_hash": prediction["feature_schema_hash"],
                      "prediction_time": prediction["prediction_time"], "origin": origin, "destination": destination,
                      "predicted_duration_minutes": prediction["predicted_duration_minutes"],
                      "prediction_request_sha256": hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                      "prediction_request": deepcopy(request),
                      "execution_type": prediction["execution_type"], "qml_executed": prediction["qml_executed"],
                      "compatibility": compatibility, "uncertainty_status": "synthetic_fixture"}
        resolved["scenario"]["travel"][origin][destination] = prediction["predicted_duration_minutes"]
        for row in resolved["weighted_scenarios"]:
            residual = residuals[row["id"]]
            if not _finite(residual):
                _reject()
            duration = prediction["predicted_duration_minutes"] + residual
            if not math.isfinite(duration) or duration < 0:
                _reject("invalid_eta_update", "Synthetic residual implies an invalid travel duration")
            if not isinstance(row.get("travel"), dict) or not isinstance(row["travel"].get(origin), dict) or destination not in row["travel"][origin]:
                _reject("invalid_scenario", "ETA overlay arc is absent from a joint travel world")
            row["travel"][origin][destination] = duration
            row.setdefault("eta_source_provenance", []).append({**provenance, "synthetic_residual_minutes": residual})
    return resolved
