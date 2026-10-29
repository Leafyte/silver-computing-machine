"""Real ASGI requests through middleware, schemas, router and durable jobs."""
from copy import deepcopy
import json
from pathlib import Path
import time
import threading

from fastapi.testclient import TestClient
import pytest

from backend.api import create_app
from backend.auth import AuthConfig, authorize_request
from backend.domain import generate_routes

ROOT = Path(__file__).resolve().parents[2]
TOKENS = {"dispatcher": "ETA_DISPATCHER_TEST_SENTINEL", "admin": "ETA_ADMIN_TEST_SENTINEL", "researcher": "ETA_RESEARCHER_TEST_SENTINEL"}


def request_fixture():
    value = json.loads((ROOT / "fixtures/india_eta/prediction_request.json").read_text())
    # Fixed model is 5 + 3.2*distance + 12*traffic; these inputs give 20 minutes.
    value["features"]["distance_km"] = 1.875
    return value


def workflow():
    scenario = {"depot": "D", "vehicles": [{"id": "V", "capacity": 2}],
                "orders": [{"id": "A", "demand": 1, "window": [0, 12], "service": 0, "penalty": 10},
                           {"id": "B", "demand": 1, "window": [0, 100], "service": 0, "penalty": 0}],
                "travel": {"D": {"D": 0, "A": 10, "B": 1}, "A": {"D": 1, "A": 0, "B": 1}, "B": {"D": 1, "A": 1, "B": 0}}}
    route = next(r for r in generate_routes(scenario) if r["order_ids"] == ["A", "B"])
    plan = {"status": "feasible", "routes": [route], "objective": route["objective"], "travel_cost": route["travel_cost"], "lateness_exposure": route["lateness_exposure"]}
    def weighted(s):
        return [{"id": "joint1", "weight": 1., "travel": deepcopy(s["travel"]), "source_label": "synthetic_overlay", "prediction_target": "travel_time"}]
    register = {"plan_id": "fleet1", "version": 1, "model_version": "fixture-v1", "as_of": "2026-10-29T10:00:00Z", "scenario": scenario,
                "weighted_scenarios": weighted(scenario), "plan": plan, "currency": "XTS", "operating_rate": 1., "cooldown_seconds": 30}
    changed = deepcopy(scenario)
    changed["travel"]["D"]["A"] = 20
    event = {"event_id": "traffic1", "plan_id": "fleet1", "plan_version": 1, "model_version": "fixture-v1", "as_of": "2026-10-29T10:01:00Z", "event_type": "delay",
             "scenario": changed, "weighted_scenarios": weighted(changed),
             "observed_state": [{"vehicle_id": "V", "position": "D", "position_type": "simulated", "elapsed_minutes": 1., "available_at": 1., "completed_prefix": []}]}
    return register, event


def wait_job(client, event_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = client.get(f"/api/v1/eta/events/{event_id}").json()
        if result["status"] not in ("queued", "running"):
            return result
        time.sleep(.01)
    raise AssertionError("API event job did not finish")


def test_prediction_sla_and_missing_uncertainty_through_real_api(tmp_path):
    with TestClient(create_app(eta_db_path=tmp_path / "eta.sqlite")) as client:
        response = client.post("/api/v1/eta/predictions", json=request_fixture())
        assert response.status_code == 200, response.text
        prediction = response.json()
        assert prediction["predicted_duration_minutes"] == 20
        assert prediction["qml_executed"] is False
        assert prediction["qml_requested"] is True
        request = {"schema_version": "india_eta.sla.request.v1", "prediction": prediction,
                   "deadline": "2026-01-01T10:25:00+05:30", "late_penalty_per_minute": 6, "currency": "XTS",
                   "scenario_source": "synthetic_fixture", "scenarios": [{"residual_minutes": r, "weight": 1/3} for r in [-5, 0, 10]]}
        sla = client.post("/api/v1/eta/sla", json=request)
        assert sla.status_code == 200, sla.text
        assert sla.json()["expected_penalty"] == pytest.approx(10)
        assert sla.json()["probability_late"] == pytest.approx(1/3)
        request.pop("scenarios"); request.pop("scenario_source")
        absent = client.post("/api/v1/eta/sla", json=request).json()
        assert absent["status"] == "unavailable" and absent["probability_late"] is None


def test_default_local_mode_cannot_register_or_approve_assignments(tmp_path):
    with TestClient(create_app(eta_db_path=tmp_path / "eta.sqlite")) as client:
        registered = client.post("/api/v1/eta/plans", json=workflow()[0])
        assert registered.status_code == 403
        approved = client.post("/api/v1/eta/proposals/fake/approve", json={})
        assert approved.status_code == 403


def test_role_policy_has_no_researcher_assignment_escalation():
    config = AuthConfig.from_mapping("token", TOKENS)
    auth = b"Bearer " + TOKENS["researcher"].encode()
    for path in ("/api/v1/eta/plans", "/api/v1/eta/events", "/api/v1/eta/proposals/id/approve", "/api/v1/eta/events/id/cancel"):
        assert authorize_request(config, "POST", path, auth)[0] == 403
    assert authorize_request(config, "GET", "/api/v1/eta/manifests/V", auth) is None
    assert authorize_request(config, "POST", "/api/v1/eta/unknown", b"Bearer " + TOKENS["admin"].encode())[0] == 403


def test_durable_event_proposal_approval_and_manifest_api(tmp_path):
    config = AuthConfig.from_mapping("token", TOKENS)
    header = {"Authorization": "Bearer " + TOKENS["dispatcher"]}
    db = tmp_path / "eta.sqlite"
    with TestClient(create_app(auth_config=config, eta_db_path=db)) as client:
        client.headers.update(header)
        register, event = workflow()
        response = client.post("/api/v1/eta/plans", json=register)
        assert response.status_code == 200, response.text
        submitted = client.post("/api/v1/eta/events", json=event)
        assert submitted.status_code == 202, submitted.text
        job = wait_job(client, "traffic1")
        assert job["status"] == "succeeded", job
        result = job["result"]
        assert result["status"] == "PROPOSE_REPLAN", result
        assert client.get("/api/v1/eta/plans/fleet1").json()["version"] == 1
        proposal = client.get("/api/v1/eta/proposals/" + result["proposal_id"]).json()
        approval = {"base_version": 1, "event_id": "traffic1", "model_version": "fixture-v1", "context_hash": proposal["proposal"]["context_hash"]}
        accepted = client.post("/api/v1/eta/proposals/" + result["proposal_id"] + "/approve", json=approval)
        assert accepted.status_code == 200, accepted.text
        assert client.get("/api/v1/eta/plans/fleet1").json()["version"] == 2
        manifest = client.get("/api/v1/eta/manifests/V").json()
        assert manifest["plan_version"] == 2
        assert [s["order_id"] for s in manifest["ordered_stops"]] == ["B", "A"]
        assert client.get("/api/v1/eta/manifests/V?version=1").json()["plan_version"] == 1
        assert client.post("/api/v1/eta/proposals/" + result["proposal_id"] + "/approve", json=approval).status_code == 409
        assert client.post("/api/v1/eta/events", json=event).json()["event_id"] == "traffic1"
    with TestClient(create_app(auth_config=config, eta_db_path=db)) as client:
        client.headers.update(header)
        assert client.get("/api/v1/eta/manifests/V").json()["plan_version"] == 2
        assert client.post("/api/v1/eta/events", json=event).json()["status"] == "succeeded"


def test_boundary_redaction_and_existing_health(tmp_path):
    sentinel = "NON_SECRET_BAD_INPUT_SENTINEL"
    with TestClient(create_app(eta_db_path=tmp_path / "eta.sqlite")) as client:
        assert client.get("/api/v1/health").status_code == 200
        for body in ({"unknown": sentinel}, [], {**request_fixture(), "unknown": sentinel}):
            response = client.post("/api/v1/eta/predictions", json=body)
            assert response.status_code == 422
            assert sentinel not in response.text
        response = client.post("/api/v1/eta/events", json={"unknown": sentinel})
        assert response.status_code == 422
        assert sentinel not in response.text


def test_input_budget_precedes_json_parsing_and_handles_chunked_bytes(tmp_path):
    with TestClient(create_app(eta_db_path=tmp_path / "eta.sqlite")) as client:
        oversized = b'{' + b'x' * 256_001
        response = client.post("/api/v1/eta/predictions", content=oversized)
        assert response.status_code == 413 and response.json()["error"]["code"] == "request_budget_exceeded"
        response = client.post("/api/v1/eta/predictions", content=iter([oversized[:120_000], oversized[120_000:]]))
        assert response.status_code == 413
        assert client.post("/api/v1/eta/predictions", content=b'{invalid').status_code == 422


def test_async_boundary_failure_before_reservation_becomes_terminal(tmp_path):
    config = AuthConfig.from_mapping("token", TOKENS)
    with TestClient(create_app(auth_config=config, eta_db_path=tmp_path / "eta.sqlite")) as client:
        client.headers.update({"Authorization": "Bearer " + TOKENS["dispatcher"]})
        _, event = workflow()
        assert client.post("/api/v1/eta/events", json=event).status_code == 202
        result = wait_job(client, event["event_id"])
        assert result["status"] == "failed"
        assert result["error"]["code"] == "not_found"


def test_eta_input_flows_through_source_gate_to_proposal_and_approval(tmp_path):
    from .test_source_adapter import eta_event
    config = AuthConfig.from_mapping("token", TOKENS)
    with TestClient(create_app(auth_config=config, eta_db_path=tmp_path / "eta.sqlite")) as client:
        client.headers.update({"Authorization": "Bearer " + TOKENS["dispatcher"]})
        register, _ = workflow()
        register["model_version"] = "1.0.0"
        assert client.post("/api/v1/eta/plans", json=register).status_code == 200
        event = eta_event()
        assert client.post("/api/v1/eta/events", json=event).status_code == 202
        job = wait_job(client, event["event_id"])
        assert job["status"] == "succeeded", job
        assert job["result"]["status"] == "PROPOSE_REPLAN"
        proposal = job["result"]["proposal"]
        world = proposal["weighted_scenarios"][0]
        assert world["travel"]["D"]["A"] == 25
        assert world["eta_source_provenance"][0]["predicted_duration_minutes"] == 20
        assert world["eta_source_provenance"][0]["compatibility"]["decision"] == "scenario_only"
        approval = {"base_version": 1, "event_id": event["event_id"], "model_version": "1.0.0", "context_hash": proposal["context_hash"]}
        approved = client.post(f"/api/v1/eta/proposals/{proposal['proposal_id']}/approve", json=approval)
        assert approved.status_code == 200, approved.text
        assert client.get("/api/v1/eta/manifests/V").json()["plan_version"] == 2
        assert client.post("/api/v1/eta/events", json=event).json()["request_sha256"] == job["request_sha256"]


@pytest.mark.parametrize("operation", ["cancel", "timeout"])
def test_api_running_event_cannot_publish_after_cancel_or_timeout(tmp_path, operation):
    from backend.india_eta.controller import EtaController
    from backend.india_eta.jobs import EtaJobs
    db = tmp_path / "eta.sqlite"
    controller = EtaController(db)
    entered, release = threading.Event(), threading.Event()
    def delayed(payload, *, cancel_check=None):
        entered.set()
        assert release.wait(3)
        return controller.process_event(payload, cancel_check=cancel_check)
    jobs = EtaJobs(controller, db, max_workers=1, timeout_seconds=.1 if operation == "timeout" else 5, worker=delayed)
    config = AuthConfig.from_mapping("token", TOKENS)
    try:
        with TestClient(create_app(auth_config=config, eta_controller=controller, eta_job_manager=jobs)) as client:
            client.headers.update({"Authorization": "Bearer " + TOKENS["dispatcher"]})
            register, event = workflow()
            assert client.post("/api/v1/eta/plans", json=register).status_code == 200
            assert client.post("/api/v1/eta/events", json=event).status_code == 202
            assert entered.wait(1)
            if operation == "cancel":
                result = client.post("/api/v1/eta/events/traffic1/cancel").json()
            else:
                time.sleep(.12)
                result = client.get("/api/v1/eta/events/traffic1").json()
            assert result["status"] == ("cancelled" if operation == "cancel" else "timed_out")
            release.set()
            jobs.close(wait=True)
            assert client.get("/api/v1/eta/plans/fleet1").json()["version"] == 1
            assert controller.store.read("SELECT proposal_id FROM eta_proposals WHERE status='pending'") == []
            assert client.get("/api/v1/eta/events/traffic1").json()["result"] is None
    finally:
        release.set()
        jobs.close(wait=True)
