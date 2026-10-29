from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Barrier

import pytest

from backend.india_eta.controller import EtaController
from backend.india_eta.errors import EtaError
from backend.domain import generate_routes
from backend.validator import validate_plan


FIXTURES = Path(__file__).parents[2] / "fixtures" / "india_eta"


def workflow(name: str) -> dict:
    return json.loads((FIXTURES / f"workflow-{name}.json").read_text(encoding="utf-8"))


def controller(tmp_path: Path) -> EtaController:
    return EtaController(str(tmp_path / "eta.sqlite3"))


def register(c: EtaController, data: dict, *, plan_id: str = "workflow-plan") -> dict:
    payload = dict(data["register"])
    payload["plan_id"] = plan_id
    return c.register_plan(payload, {"subject": "dispatcher-01", "role": "dispatcher"})


def event(data: dict, *, event_id: str | None = None, offset: int = 0) -> dict:
    result = json.loads(json.dumps(data["event"]))
    if event_id:
        result["event_id"] = event_id
    return result


def approve(c: EtaController, result: dict) -> dict:
    proposal = result["proposal"]
    return c.approve_proposal(proposal["proposal_id"], {
        "base_version": proposal["base_version"], "event_id": proposal["event_id"],
        "model_version": proposal["model_version"], "context_hash": proposal["context_hash"],
    }, {"subject": "dispatcher-02", "role": "dispatcher"})


def test_keep_and_alert_workflows_do_not_change_active_plan(tmp_path: Path) -> None:
    keep = workflow("keep")
    c = controller(tmp_path)
    active = register(c, keep)
    result = c.process_event(event(keep))
    assert result["status"] == "KEEP"
    assert c.get_active(active["plan_id"])["version"] == 1

    alert = workflow("alert")
    c2 = EtaController(str(tmp_path / "alert.sqlite3"), optimizer=lambda *a, **k: _same_plan_result(alert))
    register(c2, alert)
    result = c2.process_event(event(alert))
    assert result["status"] == "ALERT"
    assert result["reason"] == "material_risk_without_material_cost_change"
    assert c2.get_active("workflow-plan")["version"] == 1


def test_hard_window_risk_with_zero_penalty_is_alert_not_keep(tmp_path: Path) -> None:
    data = workflow("hard-alert")
    def project(scenario, plan, state, weighted, **kwargs):
        violated = any(order.get("hard_window", False) for order in scenario["orders"])
        return {"status": "projected", "valid": True, "projection_mode": "synthetic_scenario",
                "evaluation": {"valid": True, "expected_cost": {"amount": 12.0, "currency": "XTS"},
                               "hard_window_violation_frequency": 1.0 if violated else 0.0},
                "expected_remaining_cost": {"late_penalty_cost": 0.0,
                                             "hard_window_violation_frequency": 1.0 if violated else 0.0},
                "plan": {"routes": plan["routes"]}}
    c = EtaController(str(tmp_path / "hard-risk.sqlite3"), projector=project,
                      optimizer=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no material cost change")))
    register(c, data, plan_id="hard-risk-plan")
    result = c.process_event(data["event"])
    assert result["status"] == "ALERT"
    assert result["reason"] == "material_risk_without_material_cost_change"


def _same_plan_result(data: dict) -> dict:
    plan = data["plan"]
    # The input scenario is frozen in the workflow fixture; production path is
    # exercised by the proposal workflow below. This deterministic result is
    # used only to prove risk-without-benefit classification.
    from backend.validator import validate_plan
    return {"plan": plan, "validation": validate_plan(data["register"]["scenario"], plan),
            "expected_cost": None, "quantum": {"status": "not_run"},
            "classical_reference": None, "fallback": {"used": True},
            "execution_type": "test_injected_classical_result", "coefficients": None}


def test_proposal_approval_restart_and_versioned_manifests(tmp_path: Path) -> None:
    data = workflow("propose")
    db = tmp_path / "eta.sqlite3"
    c = EtaController(str(db))
    register(c, data)
    result = c.process_event(event(data))
    assert result["status"] == "PROPOSE_REPLAN"
    assert c.get_active("workflow-plan")["version"] == 1
    proposal_id = result["proposal_id"]
    restarted = EtaController(str(db))
    duplicate = restarted.process_event(event(data))
    assert duplicate["proposal_id"] == proposal_id
    assert duplicate["status"] == "PROPOSE_REPLAN"
    approved = approve(restarted, result)
    assert approved["version"] == 2
    assert restarted.get_active("workflow-plan")["version"] == 2
    assert restarted.get_manifest("V1", 1)["plan_version"] == 1
    assert restarted.get_manifest("V1", 2)["approval"]["proposal_id"] == proposal_id


def test_model_identity_persists_across_approval_and_restart(tmp_path: Path) -> None:
    data = workflow("propose")
    identity = {"model_id": "eta-travel-vqc", "model_version": data["register"]["model_version"],
                "model_sha256": "a" * 64}
    data["register"]["eta_model_identity"] = identity
    db = tmp_path / "eta-model-identity.sqlite3"
    c = EtaController(str(db))
    registered = register(c, data)
    assert registered["eta_model_identity"] == identity
    result = c.process_event(event(data))
    approve(c, result)
    restarted = EtaController(str(db))
    assert restarted.get_active("workflow-plan")["eta_model_identity"] == identity
    assert restarted.get_active("workflow-plan")["version"] == 2


@pytest.mark.parametrize("identity", [
    {"model_id": "eta-travel-vqc", "model_version": "different", "model_sha256": "a" * 64},
    {"model_id": "eta-travel-vqc", "model_version": "v1", "model_sha256": "a" * 63},
    {"model_id": "eta-travel-vqc", "model_version": "v1", "model_sha256": "A" * 64},
    {"model_id": "eta-travel-vqc", "model_version": "v1", "model_sha256": "a" * 64, "extra": True},
    {"model_id": "eta-travel-vqc", "model_sha256": "a" * 64},
])
def test_malformed_model_identity_rejected(tmp_path: Path, identity: dict) -> None:
    data = workflow("keep")
    data["register"]["eta_model_identity"] = identity
    with pytest.raises(EtaError) as rejected:
        register(controller(tmp_path), data)
    assert rejected.value.code == "invalid_request"


def test_completed_prefix_approval_preserves_snapshot_and_completion_time(tmp_path: Path) -> None:
    data = workflow("propose")
    data["register"]["cooldown_seconds"] = 0
    state = data["event"]["observed_state"]
    state[1].update({"position": "B", "elapsed_minutes": 3.0, "available_at": 3.0,
                     "completed_prefix": [{"order_id": "B", "completed_at": 2.0}]})
    c = controller(tmp_path)
    register(c, data)
    result = c.process_event(data["event"])
    assert result["status"] == "PROPOSE_REPLAN"
    approve(c, result)
    manifest = c.get_manifest("V2", 2)
    first = manifest["ordered_stops"][0]
    assert first["order_id"] == "B"
    assert first["status"] == "completed"
    assert first["immutable"] is True
    assert first["completed_at"] == 2.0
    assert manifest["completed_stop_references"] == ["B"]


def test_approval_writes_revocation_manifest_for_departed_vehicle(tmp_path: Path) -> None:
    data = workflow("departed")
    c = EtaController(str(tmp_path / "eta.sqlite3"), optimizer=lambda *args, **kwargs: _candidate_result(data))
    register(c, data, plan_id="departed-plan")
    result = c.process_event(data["event"])
    assert result["status"] == "PROPOSE_REPLAN"
    approve(c, result)
    old_vehicle_manifest = c.get_manifest("V1")
    assert old_vehicle_manifest["status"] == "revoked"
    assert old_vehicle_manifest["ordered_stops"] == []
    assert old_vehicle_manifest["plan_version"] == 2
    assert c.get_manifest("V1", 1)["plan_version"] == 1


def _candidate_result(data: dict) -> dict:
    from backend.validator import validate_plan
    plan = data["candidate_plan"]
    validation = validate_plan(data["event"]["scenario"], plan)
    return {"plan": plan, "validation": validation, "expected_cost": None,
            "quantum": {"status": "not_run"}, "classical_reference": None,
            "fallback": {"used": True, "reason": "test injected route"},
            "execution_type": "test_injected_classical_result", "coefficients": None}


def test_stale_unauthorized_and_competing_approvals_are_rejected(tmp_path: Path) -> None:
    data = workflow("propose")
    c = controller(tmp_path)
    register(c, data)
    result = c.process_event(event(data))
    proposal = result["proposal"]
    correct = {"base_version": 1, "event_id": proposal["event_id"],
               "model_version": proposal["model_version"], "context_hash": proposal["context_hash"]}
    with pytest.raises(EtaError) as unauthorized:
        c.approve_proposal(proposal["proposal_id"], correct, {"subject": "reader", "role": "viewer"})
    assert unauthorized.value.code == "forbidden"
    stale = dict(correct, base_version=99)
    with pytest.raises(EtaError) as stale_error:
        c.approve_proposal(proposal["proposal_id"], stale, "dispatcher")
    assert stale_error.value.code == "stale_context"
    barrier = Barrier(2)

    def race() -> str:
        local = EtaController(str(tmp_path / "eta.sqlite3"))
        barrier.wait()
        try:
            approve(local, result)
            return "approved"
        except EtaError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: race(), range(2)))
    assert outcomes.count("approved") == 1
    assert c.get_active("workflow-plan")["version"] == 2


def test_cancel_invalidates_proposal_and_duplicate_body_conflict(tmp_path: Path) -> None:
    data = workflow("propose")
    c = controller(tmp_path)
    register(c, data)
    payload = event(data)
    result = c.process_event(payload)
    assert c.cancel_event(payload["event_id"])["status"] == "cancelled"
    with pytest.raises(EtaError) as cancelled:
        approve(c, result)
    assert cancelled.value.code in {"proposal_not_pending", "event_cancelled"}
    changed = dict(payload, event_type="eta_update")
    with pytest.raises(EtaError) as duplicate:
        c.process_event(changed)
    assert duplicate.value.code == "idempotency_conflict"


def test_cancel_during_optimizer_suppresses_publication(tmp_path: Path) -> None:
    data = workflow("propose")
    c = controller(tmp_path)
    register(c, data)
    with pytest.raises(EtaError) as cancelled:
        c.process_event(event(data), cancel_check=lambda: True)
    assert cancelled.value.code == "cancelled"
    assert c.get_active("workflow-plan")["version"] == 1
    assert c.store.read("SELECT status FROM eta_events WHERE event_id='proposal-event'")[0][0] == "cancelled"


def test_cooldown_persists_across_controller_restart(tmp_path: Path) -> None:
    data = workflow("propose")
    data["register"]["cooldown_seconds"] = 3600
    db = tmp_path / "eta.sqlite3"
    c = EtaController(str(db))
    register(c, data)
    first = c.process_event(event(data))
    assert first["status"] == "PROPOSE_REPLAN"
    later = event(data, event_id="cooldown-event-2")
    later["as_of"] = "2026-10-29T10:04:00Z"
    restarted = EtaController(str(db))
    result = restarted.process_event(later)
    assert result["status"] == "ALERT"
    assert result["reason"] == "proposal_cooldown_active"


def test_reverse_plan_requires_persisted_hysteresis_margin(tmp_path: Path) -> None:
    data = workflow("propose")
    data["register"]["cooldown_seconds"] = 0
    data["register"]["hysteresis_cost_threshold"] = 30.0
    c = controller(tmp_path)
    register(c, data)
    result = c.process_event(event(data))
    assert result["status"] == "PROPOSE_REPLAN"
    approve(c, result)
    reverse = event(data, event_id="reverse-event")
    reverse["plan_version"] = 2
    reverse["as_of"] = "2026-10-29T10:05:00Z"
    reverse["scenario"] = data["register"]["scenario"]
    reverse["weighted_scenarios"] = data["register"]["weighted_scenarios"]
    decision = c.process_event(reverse)
    assert decision["status"] == "ALERT"
    assert decision["reason"] == "reverse_plan_hysteresis_active"
    assert decision["required_net_benefit"] == 30.1


def test_expired_proposal_persists_terminal_status(tmp_path: Path) -> None:
    data = workflow("propose")
    now = [datetime(2026, 10, 29, 10, 0, tzinfo=timezone.utc)]
    c = EtaController(str(tmp_path / "eta.sqlite3"), clock=lambda: now[0])
    data["register"]["expiry_seconds"] = 1
    register(c, data)
    result = c.process_event(event(data))
    now[0] += timedelta(seconds=2)
    with pytest.raises(EtaError) as expired:
        approve(c, result)
    assert expired.value.code == "proposal_expired"
    assert c.get_proposal(result["proposal_id"])["status"] == "expired"


def test_async_job_status_guard_blocks_cancelled_event_approval(tmp_path: Path) -> None:
    data = workflow("propose")
    c = controller(tmp_path)
    register(c, data)
    result = c.process_event(event(data))
    with c.store.transaction() as conn:
        conn.execute("CREATE TABLE eta_async_jobs(event_id TEXT PRIMARY KEY,status TEXT NOT NULL)")
        conn.execute("INSERT INTO eta_async_jobs(event_id,status) VALUES(?,?)", (result["event_id"], "cancelled"))
    with pytest.raises(EtaError) as blocked:
        approve(c, result)
    assert blocked.value.code == "event_job_not_succeeded"


def test_tampered_candidate_cannot_reassign_completed_route_prefix(tmp_path: Path) -> None:
    data = workflow("propose")
    state = data["event"]["observed_state"]
    state[1]["position"] = "B"
    state[1]["elapsed_minutes"] = 3.0
    state[1]["available_at"] = 3.0
    state[1]["completed_prefix"] = [{"order_id": "B", "completed_at": 2.0}]
    c = controller(tmp_path)
    register(c, data)
    result = c.process_event(data["event"])
    proposal_id = result["proposal_id"]
    row = c.store.read("SELECT proposal_json FROM eta_proposals WHERE proposal_id=?", (proposal_id,))[0]
    proposal = json.loads(row["proposal_json"])
    # Still a statically feasible exact-cover route, but it moves completed B
    # from V2 to V1. Approval must re-project from the persisted state.
    scenario = proposal["scenario"]
    generated = generate_routes(scenario)
    reordered = [next(r for r in generated if r["vehicle_id"] == vid and r["order_ids"] == ids)
                 for vid, ids in (("V1", ["B", "A"]), ("V2", ["C"]))]
    proposal["plan"] = {"status": "feasible", "routes": reordered,
                        "objective": sum(r["objective"] for r in reordered),
                        "travel_cost": sum(r["travel_cost"] for r in reordered),
                        "lateness_exposure": sum(r["lateness_exposure"] for r in reordered)}
    assert validate_plan(scenario, proposal["plan"])["valid"]
    with c.store.transaction() as conn:
        conn.execute("UPDATE eta_proposals SET proposal_json=? WHERE proposal_id=?",
                     (json.dumps(proposal), proposal_id))
    with pytest.raises(EtaError) as rejected:
        approve(c, result)
    assert rejected.value.code == "validation_failure"
    assert c.get_active("workflow-plan")["version"] == 1


def test_forged_high_benefit_does_not_override_recomputed_cost(tmp_path: Path) -> None:
    data = workflow("propose")
    c = controller(tmp_path)
    register(c, data)
    result = c.process_event(event(data))
    proposal_id = result["proposal_id"]
    row = c.store.read("SELECT proposal_json FROM eta_proposals WHERE proposal_id=?", (proposal_id,))[0]
    proposal = json.loads(row["proposal_json"])
    proposal["plan"] = data["register"]["plan"]  # feasible old route, with no event-context benefit
    proposal["net_benefit"] = 1_000_000
    with c.store.transaction() as conn:
        conn.execute("UPDATE eta_proposals SET proposal_json=? WHERE proposal_id=?",
                     (json.dumps(proposal), proposal_id))
    with pytest.raises(EtaError) as rejected:
        approve(c, result)
    assert rejected.value.code == "validation_failure"
    assert c.get_active("workflow-plan")["version"] == 1


def test_completed_prefix_and_clock_cannot_regress(tmp_path: Path) -> None:
    data = workflow("propose")
    state = data["event"]["observed_state"]
    # V2's active route is B,C in this fixture.
    state[1]["position"] = "B"
    state[1]["elapsed_minutes"] = 3
    state[1]["available_at"] = 3
    state[1]["completed_prefix"] = [{"order_id": "B", "completed_at": 2}]
    c = controller(tmp_path)
    register(c, data)
    payload = event(data)
    c.process_event(payload)
    regressed = json.loads(json.dumps(payload))
    regressed["event_id"] = "later-regression"
    regressed["as_of"] = "2026-10-29T10:03:00Z"
    regressed["observed_state"][1]["completed_prefix"] = []
    with pytest.raises(EtaError) as exc:
        c.process_event(regressed)
    assert exc.value.code == "state_regression"


def test_registration_requires_trusted_actor_and_independent_cost(tmp_path: Path) -> None:
    data = workflow("keep")
    c = controller(tmp_path)
    with pytest.raises(EtaError) as forbidden:
        c.register_plan(data["register"], {"subject": "bad", "role": "driver"})
    assert forbidden.value.code == "forbidden"
    wrong = dict(data["register"], expected_cost={"total": 999999})
    with pytest.raises(EtaError) as mismatch:
        c.register_plan(wrong, "admin")
    assert mismatch.value.code == "validation_failure"
