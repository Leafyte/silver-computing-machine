"""Bounded static validation of the proposed combined API contract and examples."""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator, RefResolver


ROOT = Path(__file__).resolve().parent
openapi = json.loads((ROOT / "debugleaf-backend-openapi-v2.proposed.json").read_text(encoding="utf-8"))
examples = json.loads((ROOT / "debugleaf-backend-examples-v2.proposed.json").read_text(encoding="utf-8"))
schemas = openapi["components"]["schemas"]

assert openapi["openapi"] == "3.1.0"
assert len(openapi["paths"]) == 12
for schema in schemas.values():
    Draft202012Validator.check_schema(schema)

refs: list[str] = []


def gather_refs(node: object) -> None:
    if isinstance(node, dict):
        if isinstance(node.get("$ref"), str):
            refs.append(node["$ref"])
        for value in node.values():
            gather_refs(value)
    elif isinstance(node, list):
        for value in node:
            gather_refs(value)


gather_refs(openapi)
for ref in refs:
    assert ref.startswith("#/"), ref
    target: object = openapi
    for part in ref[2:].split("/"):
        target = target[part.replace("~1", "/").replace("~0", "~")]  # type: ignore[index]

resolver = RefResolver.from_schema(openapi)


def validate(schema_name: str, payload: object) -> None:
    Draft202012Validator(schemas[schema_name], resolver=resolver).validate(payload)


dispatch = examples["dispatch"]
eta = examples["prepared_eta"]
representative = [
    ("Session", dispatch["pairing_redeem_response"]),
    ("DriverState", dispatch["driver_state_response"]),
    ("CongestionEventRequest", dispatch["congestion_event_request"]),
    ("JobAccepted", dispatch["job_accepted_202"]),
    ("Job", dispatch["job_proposal_ready_200"]),
    ("Proposal", dispatch["proposal_200"]),
    ("ApprovalRequest", dispatch["approval_request"]),
    ("ApprovalResponse", dispatch["approval_success_200"]),
    ("DispatcherSnapshot", dispatch["dispatcher_snapshot_200"]),
    ("ChangeFeed", dispatch["change_feed_200"]),
    ("ChangeFeed", dispatch["expired_cursor_410"]),
    ("ErrorEnvelope", dispatch["unauthorized_401"]),
    ("ErrorEnvelope", dispatch["stale_revision_409"]),
    ("ErrorEnvelope", dispatch["invalid_request_422"]),
    ("ErrorEnvelope", dispatch["rate_limited_429"]),
    ("ErrorEnvelope", dispatch["service_unavailable_503"]),
    ("PreparedPredictionRequestV2", eta["request"]),
    ("PreparedPredictionResponseV2", eta["published_fixture_expected_response_not_executed"]),
]
for schema_name, payload in representative:
    validate(schema_name, payload)
for payload in eta["error_examples"].values():
    validate("Error", payload)

assert eta["request"]["include_diagnostics"] is True
assert eta["published_fixture_expected_response_not_executed"]["diagnostics"]["requested"] is True
assert eta["published_fixture_expected_response_not_executed"]["operational_policy"]["alpha"] == 0.0
assert len(eta["request"]["prepared_features"]) == 59
assert len(eta["request"]["feature_provenance"]["features"]) == 59

print(
    f"PASS: paths={len(openapi['paths'])}, refs={len(refs)}, schemas={len(schemas)}, "
    f"representative_payloads={len(representative) + len(eta['error_examples'])}; "
    "diagnostics opt-in, alpha=0, 59 features/provenance"
)
