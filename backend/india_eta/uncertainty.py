"""Weighted residual arithmetic with trusted frozen-distribution support."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Mapping

from .errors import EtaError, invalid
from .prediction import _canonical_hash, _finite, _timestamp

_HASH = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_FIELDS = {"distance_km", "traffic_index"}


def _unavailable(currency: str, status: str = "unavailable", reason: str = "trusted_residual_distribution_not_configured") -> dict[str, Any]:
    return {"schema_version": "india_eta.sla.response.v1", "status": status,
            "probability_late": None, "expected_late_minutes": None, "expected_penalty": None,
            "currency": currency, "scenario_count": 0, "reason": reason}


def _validated_artifact(registry: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if not isinstance(registry, Mapping) or len(registry) > 32:
        raise EtaError("uncertainty_unavailable", "Configured uncertainty registry is invalid", 503)
    entry = registry.get(key)
    if not isinstance(entry, Mapping) or set(entry) != {"artifact", "sha256"}:
        raise EtaError("uncertainty_unavailable", "Requested uncertainty release is not allowlisted", 503)
    artifact, expected = entry["artifact"], entry["sha256"]
    if not isinstance(artifact, Mapping) or not isinstance(expected, str) or not _HASH.fullmatch(expected):
        raise EtaError("uncertainty_unavailable", "Configured uncertainty release is invalid", 503)
    digest = _canonical_hash(artifact)
    if digest != expected or artifact.get("content_sha256") != digest:
        raise EtaError("uncertainty_integrity_error", "Configured uncertainty release failed integrity verification", 503)
    required = {"schema_version", "uncertainty_id", "uncertainty_version", "model_id", "model_version",
                "feature_schema_hash", "source_domain", "geography", "prediction_target", "status",
                "condition_support", "scenarios", "provenance", "content_sha256"}
    if set(artifact) != required or artifact.get("schema_version") != "india_eta.residual_distribution.v1":
        raise EtaError("uncertainty_unavailable", "Uncertainty release schema is unsupported", 503)
    if f"{artifact.get('uncertainty_id')}@{artifact.get('uncertainty_version')}" != key:
        raise EtaError("uncertainty_integrity_error", "Uncertainty registry key does not match its release", 503)
    string_fields = {"uncertainty_id", "uncertainty_version", "model_id", "model_version", "feature_schema_hash",
                     "source_domain", "geography", "prediction_target", "status"}
    if any(not isinstance(artifact[field], str) or not artifact[field] for field in string_fields):
        raise EtaError("uncertainty_unavailable", "Frozen residual release metadata is invalid", 503)
    if artifact["status"] not in {"validated_source_domain", "out_of_domain"}:
        raise EtaError("uncertainty_unavailable", "Frozen residual status is unsupported", 503)
    support = artifact["condition_support"]
    if not isinstance(support, Mapping) or set(support) - _CONDITION_FIELDS:
        raise EtaError("uncertainty_unavailable", "Frozen residual condition support is invalid", 503)
    for bounds in support.values():
        if (not isinstance(bounds, Mapping) or set(bounds) != {"minimum", "maximum"} or
                not _finite(bounds["minimum"]) or not _finite(bounds["maximum"]) or bounds["minimum"] > bounds["maximum"]):
            raise EtaError("uncertainty_unavailable", "Frozen residual condition support is invalid", 503)
    scenarios = artifact["scenarios"]
    if not isinstance(scenarios, (list, tuple)) or len(scenarios) > 100 or (artifact["status"] == "validated_source_domain" and not scenarios):
        raise EtaError("uncertainty_unavailable", "Frozen residual scenarios are invalid", 503)
    if artifact["status"] == "validated_source_domain":
        _validate_provenance(artifact)
        total = 0.0
        for scenario in scenarios:
            if not isinstance(scenario, Mapping) or set(scenario) != {"residual_minutes", "weight"}:
                raise EtaError("uncertainty_unavailable", "Frozen residual scenarios are invalid", 503)
            if not _finite(scenario["residual_minutes"]) or not _finite(scenario["weight"]) or scenario["weight"] <= 0:
                raise EtaError("uncertainty_unavailable", "Frozen residual scenarios are invalid", 503)
            total += float(scenario["weight"])
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise EtaError("uncertainty_unavailable", "Frozen residual weights must sum to one", 503)
    return artifact


def _validate_provenance(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = artifact.get("provenance")
    required = {"source_id", "data_hash", "calibration_split_hash", "evaluation_split_hash",
                "sample_count", "calibration_method", "calibration_artifact_hash"}
    if not isinstance(provenance, Mapping) or set(provenance) != required:
        raise EtaError("uncertainty_unavailable", "Frozen residual provenance is incomplete", 503)
    for key in required - {"sample_count"}:
        if not isinstance(provenance[key], str) or not provenance[key].strip():
            raise EtaError("uncertainty_unavailable", "Frozen residual provenance is incomplete", 503)
    for key in {"data_hash", "calibration_split_hash", "evaluation_split_hash", "calibration_artifact_hash"}:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", provenance[key]):
            raise EtaError("uncertainty_unavailable", "Frozen residual evidence hash is invalid", 503)
    count = provenance["sample_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise EtaError("uncertainty_unavailable", "Frozen residual sample count is invalid", 503)
    return provenance


def _conditions_supported(artifact: Mapping[str, Any], prediction: Mapping[str, Any]) -> bool:
    support = artifact["condition_support"]
    context = prediction.get("condition_context")
    if not isinstance(support, Mapping) or not isinstance(context, Mapping) or set(support) - _CONDITION_FIELDS:
        return False
    for field, bounds in support.items():
        low, high = bounds["minimum"], bounds["maximum"]
        value = context.get(field)
        if not _finite(low) or not _finite(high) or low > high or not _finite(value) or not low <= value <= high:
            return False
    return True


def _risk_result(prediction: Mapping[str, Any], deadline: datetime, rate: float, currency: str,
                 scenarios: list[Mapping[str, Any]], status: str, source: str,
                 provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    duration = float(prediction["predicted_duration_minutes"])
    arrival = _timestamp(prediction["predicted_arrival"])
    total_weight = 0.0
    weighted_late = 0.0
    weighted_late_minutes = 0.0
    for scenario in scenarios:
        if not isinstance(scenario, Mapping) or set(scenario) != {"residual_minutes", "weight"}:
            raise invalid("invalid_scenario", "Residual scenario schema is invalid")
        residual, weight = scenario["residual_minutes"], scenario["weight"]
        if not _finite(residual) or not _finite(weight) or weight <= 0:
            raise invalid("invalid_scenario", "Residuals must be finite and scenario weights positive")
        if duration + float(residual) < 0:
            raise invalid("invalid_scenario", "Residual scenario implies a negative duration")
        projected_arrival = arrival.fromtimestamp(arrival.timestamp() + 60 * float(residual), tz=arrival.tzinfo)
        lateness = max(0.0, (projected_arrival - deadline).total_seconds() / 60.0)
        total_weight += float(weight)
        weighted_late += float(weight) * (1.0 if lateness > 0 else 0.0)
        weighted_late_minutes += float(weight) * lateness
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise invalid("invalid_weights", "Scenario weights must sum to one")
    result: dict[str, Any] = {
        "schema_version": "india_eta.sla.response.v1", "status": status, "scenario_source": source,
        "probability_late": weighted_late, "expected_late_minutes": weighted_late_minutes,
        "expected_penalty": rate * weighted_late_minutes, "currency": currency,
        "scenario_count": len(scenarios), "deadline": deadline.isoformat(),
        "risk_claim": "synthetic_arithmetic_only" if status == "synthetic_fixture" else "frozen_validated_residual_distribution",
    }
    if provenance is not None:
        result["uncertainty_provenance"] = dict(provenance)
    return result


def _check_prediction(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], datetime, datetime, float]:
    prediction = payload["prediction"]
    if not isinstance(prediction, Mapping) or prediction.get("schema_version") != "india_eta.prediction.response.v1":
        raise invalid("invalid_prediction", "Prediction response schema is invalid")
    duration = prediction.get("predicted_duration_minutes")
    if not _finite(duration) or duration <= 0 or prediction.get("prediction_target") != "travel_time":
        raise invalid("invalid_prediction", "SLA requires a valid travel-time prediction")
    arrival = _timestamp(prediction.get("predicted_arrival"))
    prediction_time = _timestamp(prediction.get("prediction_time"))
    elapsed = (arrival - prediction_time).total_seconds() / 60.0
    if not math.isclose(elapsed, float(duration), rel_tol=0.0, abs_tol=1e-6):
        raise invalid("invalid_prediction", "Prediction arrival and duration are inconsistent")
    deadline = _timestamp(payload["deadline"])
    rate = payload["late_penalty_per_minute"]
    if not _finite(rate) or rate < 0:
        raise invalid("invalid_penalty", "Penalty rate must be finite and nonnegative")
    currency = payload["currency"]
    if not isinstance(currency, str) or not 3 <= len(currency) <= 8 or not currency.isalpha():
        raise invalid("invalid_currency", "Currency must be an alphabetic code")
    return prediction, deadline, arrival, float(rate)


def evaluate_sla(payload: dict[str, Any], registry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate weighted synthetic scenarios or a server-pinned residual release.

    `registry` is server configuration, never request data. Each key is
    uncertainty_id@uncertainty_version and each entry is {artifact, sha256}.
    """
    required = {"schema_version", "prediction", "deadline", "late_penalty_per_minute", "currency"}
    optional = {"scenarios", "scenario_source", "uncertainty_id", "uncertainty_version"}
    if not isinstance(payload, dict) or set(payload) - (required | optional) or not required <= set(payload):
        raise invalid("invalid_schema", "SLA request schema is invalid")
    if payload["schema_version"] != "india_eta.sla.request.v1":
        raise invalid("unsupported_schema", "SLA request schema version is unsupported")
    prediction, deadline, _, rate = _check_prediction(payload)
    currency = payload["currency"]

    has_id = "uncertainty_id" in payload
    has_version = "uncertainty_version" in payload
    if has_id != has_version:
        raise invalid("invalid_uncertainty_reference", "Both uncertainty ID and version are required")
    if has_id:
        if "scenarios" in payload or "scenario_source" in payload:
            raise invalid("invalid_schema", "Trusted residual selection cannot be mixed with caller scenarios")
        uncertainty_id, version = payload["uncertainty_id"], payload["uncertainty_version"]
        if not isinstance(uncertainty_id, str) or not isinstance(version, str):
            raise invalid("invalid_uncertainty_reference", "Uncertainty ID and version must be strings")
        if registry is None:
            return _unavailable(currency)
        artifact = _validated_artifact(registry, f"{uncertainty_id}@{version}")
        for artifact_key, prediction_key in (("model_id", "model_id"), ("model_version", "model_version"),
                                             ("feature_schema_hash", "feature_schema_hash"),
                                             ("source_domain", "source_domain"), ("geography", "geography"),
                                             ("prediction_target", "prediction_target")):
            if artifact.get(artifact_key) != prediction.get(prediction_key):
                raise EtaError("uncertainty_mismatch", "Residual release does not match the ETA prediction", 422)
        if artifact["status"] == "out_of_domain":
            return _unavailable(currency, "out_of_domain", "residual_release_marked_out_of_domain")
        if artifact["status"] != "validated_source_domain":
            raise EtaError("uncertainty_unavailable", "Frozen residual status is unsupported", 503)
        provenance = _validate_provenance(artifact)
        if not _conditions_supported(artifact, prediction):
            return _unavailable(currency, "out_of_domain", "prediction_conditions_outside_frozen_support")
        scenarios = artifact["scenarios"]
        if not isinstance(scenarios, (tuple, list)) or not 1 <= len(scenarios) <= 100:
            raise EtaError("uncertainty_unavailable", "Frozen residual scenarios are invalid", 503)
        return _risk_result(prediction, deadline, rate, currency, list(scenarios),
                            "validated_source_domain", "frozen_residual_registry", provenance)

    source = payload.get("scenario_source", "unavailable")
    scenarios = payload.get("scenarios")
    if not isinstance(source, str) or source not in {"synthetic_fixture", "unavailable"}:
        raise invalid("invalid_scenario_source", "Scenario source is not an accepted evidence label")
    if source == "unavailable":
        if scenarios is not None and scenarios != []:
            raise invalid("invalid_scenarios", "Unavailable uncertainty cannot include residual scenarios")
        return _unavailable(currency)
    if not isinstance(scenarios, list) or not 1 <= len(scenarios) <= 100:
        raise invalid("invalid_scenarios", "Synthetic scenario list must contain 1 to 100 rows")
    return _risk_result(prediction, deadline, rate, currency, scenarios, "synthetic_fixture", "synthetic_fixture")


def interval_coverage(observations: list[float], intervals: list[list[float]]) -> dict[str, Any]:
    """Descriptive interval coverage helper; it does not establish calibration."""
    if not isinstance(observations, list) or not isinstance(intervals, list) or len(observations) != len(intervals) or not observations:
        raise invalid("invalid_coverage_data", "Coverage inputs must be non-empty paired lists")
    covered = 0
    for value, bounds in zip(observations, intervals):
        if not _finite(value) or not isinstance(bounds, list) or len(bounds) != 2 or not all(_finite(x) for x in bounds):
            raise invalid("invalid_coverage_data", "Coverage values and interval bounds must be finite")
        if bounds[0] > bounds[1]:
            raise invalid("invalid_coverage_data", "Interval lower bound must not exceed upper bound")
        covered += bounds[0] <= value <= bounds[1]
    return {"covered": covered, "count": len(observations), "empirical_coverage": covered / len(observations),
            "status": "descriptive_only"}
