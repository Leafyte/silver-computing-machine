"""Explicit compatibility decisions for mapping ETA outputs into route arcs."""

from __future__ import annotations

import math
from typing import Any

from .errors import invalid

_SUPPORTED_FEATURES = {"distance_km", "traffic_index", "weather"}


def check_compatibility(payload: dict[str, Any]) -> dict[str, Any]:
    required = {"schema_version", "source_domain", "geography", "source_units", "prediction_target",
                "road_routing_provenance", "vehicle_type", "features", "time_semantics", "service_semantics",
                "horizon_minutes", "uncertainty_status"}
    optional = {"duration_minutes"}
    if not isinstance(payload, dict) or set(payload) - (required | optional) or not required <= set(payload):
        raise invalid("invalid_schema", "Compatibility request schema is invalid")
    if payload["schema_version"] != "india_eta.compatibility.request.v1":
        raise invalid("unsupported_schema", "Compatibility request schema version is unsupported")
    string_fields = {"source_domain", "geography", "source_units", "prediction_target", "road_routing_provenance",
                     "vehicle_type", "time_semantics", "service_semantics", "uncertainty_status"}
    if any(not isinstance(payload[key], str) for key in string_fields):
        raise invalid("invalid_field_type", "Compatibility categorical fields must be strings")
    reasons: list[str] = []
    target = payload["prediction_target"]
    if payload["source_domain"] not in {"india_food_delivery", "synthetic_overlay"}:
        reasons.append("unsupported_source_domain")
    if payload["geography"] != "IN":
        reasons.append("unsupported_geography")
    if not isinstance(payload["features"], list) or any(not isinstance(x, str) for x in payload["features"]):
        raise invalid("invalid_features", "Feature names must be a list of strings")
    if set(payload["features"]) - _SUPPORTED_FEATURES:
        reasons.append("unsupported_features")
    try:
        finite_horizon = math.isfinite(float(payload["horizon_minutes"]))
    except (TypeError, ValueError, OverflowError):
        finite_horizon = False
    if (not isinstance(payload["horizon_minutes"], (int, float)) or isinstance(payload["horizon_minutes"], bool)
            or not finite_horizon or payload["horizon_minutes"] <= 0):
        raise invalid("invalid_horizon", "Horizon must be positive minutes")
    if "duration_minutes" in payload:
        try:
            finite_duration = math.isfinite(float(payload["duration_minutes"]))
        except (TypeError, ValueError, OverflowError):
            finite_duration = False
        if (not isinstance(payload["duration_minutes"], (int, float)) or isinstance(payload["duration_minutes"], bool)
                or not finite_duration or payload["duration_minutes"] <= 0):
            raise invalid("invalid_duration", "Scenario duration must be positive finite minutes")
    if not isinstance(payload["vehicle_type"], str) or payload["vehicle_type"] not in {"two_wheeler", "car", "van"}:
        reasons.append("unsupported_vehicle")
    if payload["source_units"] != "minute":
        reasons.append("unsupported_duration_unit")
    if payload["time_semantics"] != "timezone_aware_service_start":
        reasons.append("unsupported_clock_semantics")
    if payload["service_semantics"] not in {"travel_only", "service_start"}:
        reasons.append("unsupported_service_semantics")
    if payload["uncertainty_status"] not in {"unavailable", "synthetic_fixture", "validated_source_domain", "out_of_domain"}:
        raise invalid("invalid_uncertainty_status", "Uncertainty status is unsupported")
    if target == "whole_delivery_time":
        reasons.append("whole_delivery_time_cannot_populate_road_arc")
    elif target != "travel_time":
        reasons.append("unsupported_prediction_target")
    routing = payload["road_routing_provenance"]
    synthetic_overlay = payload["source_domain"] == "synthetic_overlay"
    synthetic_route = synthetic_overlay and routing == "synthetic_fixture"
    scenario_reasons: list[str] = []
    if routing != "verified_route_duration" and not synthetic_route:
        reasons.append("verified_road_duration_unavailable")
    if synthetic_route:
        scenario_reasons.append("synthetic_route_duration_not_road_verified")
    if reasons:
        decision = "rejected"
    elif synthetic_route or payload["source_domain"] == "synthetic_overlay" or payload["uncertainty_status"] != "validated_source_domain":
        decision = "scenario_only"
        reasons.extend(scenario_reasons)
        if not synthetic_route:
            reasons.append("synthetic_or_unavailable_uncertainty_not_operational_support")
    else:
        # Caller labels are never evidence. No trusted residual/calibration registry is wired here.
        decision = "scenario_only"
        reasons.append("trusted_uncertainty_registry_not_configured")
    result = {"schema_version": "india_eta.compatibility.response.v1", "decision": decision,
            "supported": decision == "supported", "reasons": reasons,
            "road_arc_mapping_allowed": decision == "supported",
            "decision_basis": "declared_contract_fields_only"}
    if "duration_minutes" in payload:
        result["scenario_duration_minutes"] = float(payload["duration_minutes"])
    return result
