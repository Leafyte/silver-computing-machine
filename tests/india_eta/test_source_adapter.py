from copy import deepcopy
import json
from pathlib import Path

import pytest

from backend.india_eta.errors import EtaError
from backend.india_eta.source_adapter import resolve_eta_update, model_identity_for_version
from .test_api import workflow


def eta_event():
    _, event = workflow()
    event["mode"] = "synthetic_scenario"
    event["model_version"] = "1.0.0"
    request = json.loads((Path(__file__).resolve().parents[2] / "fixtures/india_eta/prediction_request.json").read_text())
    request.update(origin={"id": "D"}, destination={"id": "A"}, prediction_time=event["as_of"])
    request["features"]["distance_km"] = 1.875  # 5 + 3.2*1.875 + 12*.75 = 20
    request["features"]["observed_at"] = {name: event["as_of"] for name in ("distance_km", "traffic_index")}
    event["eta_update"] = {"schema_version": "india_eta.synthetic_update.v1", "arcs": [
        {"origin": "D", "destination": "A", "prediction_request": request, "residual_minutes_by_scenario": {"joint1": 5}}]}
    return event


def test_server_prediction_is_connected_to_frozen_joint_cost_inputs():
    raw = eta_event()
    original = deepcopy(raw)
    resolved = resolve_eta_update(raw)
    assert raw == original  # Input baseline is immutable.
    assert "eta_update" not in resolved
    assert resolved["scenario"]["travel"]["D"]["A"] == 20
    world = resolved["weighted_scenarios"][0]
    assert world["travel"]["D"]["A"] == 25
    source = world["eta_source_provenance"][0]
    assert source["model_sha256"] and source["prediction_request_sha256"]
    assert source["prediction_request"] == raw["eta_update"]["arcs"][0]["prediction_request"]
    assert source["compatibility"]["decision"] == "scenario_only"
    assert source["compatibility"]["road_arc_mapping_allowed"] is False
    assert source["qml_executed"] is False
    assert resolve_eta_update(raw) == resolved  # Runtime never contaminates idempotency.


@pytest.mark.parametrize("case", ["live", "stale", "model", "origin", "negative", "missing_world", "unlabeled", "spoof", "unknown", "huge"])
def test_unsafe_or_incompatible_transfer_rejected_before_optimization(case):
    raw = eta_event()
    arc = raw["eta_update"]["arcs"][0]
    if case == "live": raw["mode"] = "actual"
    elif case == "stale": arc["prediction_request"]["prediction_time"] = "2027-01-01T10:00:00Z"
    elif case == "model": raw["model_version"] = "other"
    elif case == "origin": arc["prediction_request"]["origin"] = {"id": "B"}
    elif case == "negative": arc["residual_minutes_by_scenario"]["joint1"] = -100
    elif case == "missing_world": arc["residual_minutes_by_scenario"] = {}
    elif case == "unlabeled": raw["weighted_scenarios"][0]["source_label"] = "india_whole_delivery"
    elif case == "spoof": raw["weighted_scenarios"][0]["eta_source_provenance"] = [{"qml_executed": True}]
    elif case == "unknown": arc["unknown"] = "NON_SECRET_SENTINEL"
    elif case == "huge": arc["residual_minutes_by_scenario"]["joint1"] = 10**400
    with pytest.raises(EtaError) as exc:
        resolve_eta_update(raw)
    assert exc.value.status == 422
    assert "NON_SECRET_SENTINEL" not in str(exc.value)


@pytest.mark.parametrize("case", ["zero_weight", "wrong_sum", "missing_matrix_cell"])
def test_full_joint_scenario_semantics_are_checked_before_admission(case):
    raw = eta_event()
    row = raw["weighted_scenarios"][0]
    if case == "zero_weight": row["weight"] = 0
    elif case == "wrong_sum": row["weight"] = .25
    else: row["travel"]["B"].pop("A")
    with pytest.raises(EtaError) as exc:
        resolve_eta_update(raw)
    assert exc.value.code == "invalid_scenario"


def test_same_version_alternate_model_is_not_the_approved_identity():
    from backend.india_eta.prediction import _canonical_hash
    raw = eta_event()
    release = json.loads((Path(__file__).resolve().parents[2] / "fixtures/india_eta/prediction_model.json").read_text())
    other = deepcopy(release)
    other["model_id"] = "alternate-fixture"
    other["intercept"] = 50
    other["content_sha256"] = _canonical_hash(other)
    registry = {f"{r['model_id']}@{r['model_version']}": {"release": r, "sha256": r["content_sha256"]} for r in (release, other)}
    bound = model_identity_for_version("1.0.0", registry=registry, model_id=release["model_id"])
    with pytest.raises(EtaError) as ambiguous:
        model_identity_for_version("1.0.0", registry=registry)
    assert ambiguous.value.code == "ambiguous_model_identity"
    raw["eta_update"]["arcs"][0]["prediction_request"]["model_id"] = other["model_id"]
    with pytest.raises(EtaError) as mismatch:
        resolve_eta_update(raw, registry=registry, model_identity=bound)
    assert mismatch.value.code == "model_identity_mismatch"
