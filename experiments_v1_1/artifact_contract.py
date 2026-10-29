"""Versioned frozen model & optimization artifact contract (V1.1).

Defines the strict cross-team interface between the Lenovo Legion 5 Pro QML
training engine, the MacBook Quantum Optimization benchmark suite, and the
Antigravity production backend.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence


CONTRACT_SCHEMA_VERSION = "debugleaf.hybrid_artifact.v1.1"

REQUIRED_TOP_LEVEL_KEYS = (
    "schema_version",
    "model_and_dataset_version",
    "split_identifiers",
    "scenario_and_seed_Provenance",
    "risk_predictions_and_calibration",
    "candidate_routes_and_coefficients",
    "qubo_and_constraints",
    "execution_backend_and_config",
)


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


@dataclass(frozen=True)
class HybridDispatchArtifact:
    schema_version: str
    model_and_dataset_version: Dict[str, Any]
    split_identifiers: Dict[str, Any]
    scenario_and_seed_Provenance: Dict[str, Any]
    risk_predictions_and_calibration: Dict[str, Any]
    candidate_routes_and_coefficients: Dict[str, Any]
    qubo_and_constraints: Dict[str, Any]
    execution_backend_and_config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        body = asdict(self)
        body["artifact_sha256"] = canonical_sha256(body)
        return body


def validate_hybrid_artifact(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate versioned cross-team model/QUBO artifact without mutating inputs."""
    errors: List[str] = []
    if payload.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        errors.append(f"schema_version must equal {CONTRACT_SCHEMA_VERSION!r}")

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in payload:
            errors.append(f"missing required section: {key}")

    md = payload.get("model_and_dataset_version", {})
    for field in ("model_id", "model_version", "dataset_id", "dataset_sha256", "is_reference_fixture"):
        if field not in md:
            errors.append(f"model_and_dataset_version missing {field}")

    splits = payload.get("split_identifiers", {})
    for field in ("replicate_id", "train_split_sha256", "validation_split_sha256", "test_split_sha256", "evaluation_split"):
        if field not in splits:
            errors.append(f"split_identifiers missing {field}")

    prov = payload.get("scenario_and_seed_Provenance", {})
    for field in ("scenario_id", "family_seed", "split_seed", "qaoa_seed"):
        if field not in prov:
            errors.append(f"scenario_and_seed_Provenance missing {field}")

    risk = payload.get("risk_predictions_and_calibration", {})
    for field in ("route_probabilities", "readout_type", "brier_score_validation", "ece_validation"):
        if field not in risk:
            errors.append(f"risk_predictions_and_calibration missing {field}")
    if isinstance(risk.get("route_probabilities"), Mapping):
        for rid, p in risk["route_probabilities"].items():
            if not (0.0 <= float(p) <= 1.0):
                errors.append(f"probability out of [0, 1] for route {rid}")

    routes = payload.get("candidate_routes_and_coefficients", {})
    for field in ("candidate_route_ids", "transport_cost_usd", "labor_cost_usd", "expected_sla_penalty_usd", "total_route_coefficients_usd"):
        if field not in routes:
            errors.append(f"candidate_routes_and_coefficients missing {field}")

    qubo_meta = payload.get("qubo_and_constraints", {})
    for field in ("n_variables", "variable_order", "linear", "quadratic", "offset", "coverage_penalty", "vehicle_penalty", "required_stops"):
        if field not in qubo_meta:
            errors.append(f"qubo_and_constraints missing {field}")

    exec_meta = payload.get("execution_backend_and_config", {})
    for field in ("backend_name", "execution_mode", "is_simulated", "qpu_authorized", "layers", "shots"):
        if field not in exec_meta:
            errors.append(f"execution_backend_and_config missing {field}")

    if "artifact_sha256" in payload:
        copy_without_hash = {k: v for k, v in payload.items() if k != "artifact_sha256"}
        if canonical_sha256(copy_without_hash) != payload["artifact_sha256"]:
            errors.append("artifact_sha256 does not match canonical payload")

    return {"valid": len(errors) == 0, "errors": errors}

