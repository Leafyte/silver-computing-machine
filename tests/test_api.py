import json
import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from backend.api import create_app
from backend.domain import solve_exact
from backend.jobs import JobCapacityError, JobManager
from backend.validator import validate_plan


class _Response:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.text = body.decode("utf-8")

    def json(self):
        return json.loads(self.text)


class _ASGIClient:
    """Direct route adapter; this sandbox blocks Windows loopback socketpairs."""
    def __init__(self, app):
        self.app = app

    def _call(self, method, path, *, json_body=None):
        for route in self.app.routes:
            if not isinstance(route, APIRoute) or method not in route.methods:
                continue
            if route.path == path:
                result = route.endpoint(json_body) if method == "POST" and route.body_field else route.endpoint()
            elif "{job_id}" in route.path:
                prefix, suffix = route.path.split("{job_id}")
                if not path.startswith(prefix) or (suffix and not path.endswith(suffix)):
                    continue
                job_id = path[len(prefix):len(path) - len(suffix) if suffix else None]
                result = route.endpoint(job_id)
            else:
                continue
            if isinstance(result, JSONResponse):
                return _Response(result.status_code, result.body)
            return _Response(route.status_code or 200, json.dumps(result).encode())
        return _Response(404, b'{"error":{"code":"not_found"}}')

    def get(self, path):
        return self._call("GET", path)

    def post(self, path, json=None):
        return self._call("POST", path, json_body=json)


@pytest.fixture
def scenario():
    return json.loads((Path(__file__).parents[1] / "fixtures" / "dispatch_tiny.json").read_text())


@pytest.fixture
def manager(tmp_path):
    value = JobManager(tmp_path / "jobs.sqlite3", max_workers=2, timeout_seconds=2)
    yield value
    value.close(wait=True)


@pytest.fixture
def client(manager):
    return _ASGIClient(create_app(manager))


def test_versioned_health_validation_routes_and_plan_are_working(client, scenario):
    assert client.get("/api/v1/health").json()["api_version"] == "v1"
    valid = client.post("/api/v1/scenarios/validate", json=scenario)
    assert valid.status_code == 200 and valid.json()["valid"]
    routes = client.post("/api/v1/scenarios/routes", json=scenario)
    assert routes.status_code == 200 and routes.json()["count"] > 0
    response = client.post("/api/v1/scenarios/plan", json=scenario)
    assert response.status_code == 200
    assert response.json()["validation"]["valid"]


def test_exact_budget_is_stable_503_and_never_misreported_as_infeasible(client):
    response = client.post("/api/v1/scenarios/plan", json=_high_branching_assignment_case())
    assert response.status_code == 503
    assert response.json() == {"error": {
        "code": "solver_budget_exceeded",
        "message": "dispatch search budget exceeded; no plan was published"}}
    assert "plan" not in response.json()


def test_job_records_budget_exhaustion_as_failed_not_infeasible(manager):
    submitted = manager.submit(_high_branching_assignment_case())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        current = manager.get(submitted["id"])
        if current["status"] in {"failed", "infeasible", "succeeded", "timed_out"}:
            break
        time.sleep(0.005)
    assert current["status"] == "failed"
    assert current["error"]["code"] == "solver_budget_exceeded"
    assert current["result"] is None


def _two_order_one_vehicle(scenario):
    result = json.loads(json.dumps(scenario))
    result["vehicles"] = result["vehicles"][:1]
    result["orders"] = result["orders"][:2]
    locations = [result["depot"], *(order["id"] for order in result["orders"])]
    result["travel"] = {a: {b: scenario["travel"][a][b] for b in locations} for a in locations}
    return result


def _high_branching_assignment_case():
    orders = [f"O{i}" for i in range(8)]
    locations = ["D", *orders]
    travel = {source: {target: (0 if source == target or (source != "D" and target != "D") else 1)
                       for target in locations} for source in locations}
    return {"depot": "D",
            "vehicles": [{"id": f"V{i}", "capacity": 1} for i in range(8)],
            "orders": [{"id": order, "demand": 1, "window": [0, 100], "service": 0, "penalty": 0}
                       for order in orders],
            "travel": travel}


def _financial_tradeoff_scenario():
    return {
        "depot": "D",
        "vehicles": [{"id": "V1", "capacity": 2}],
        "orders": [
            {"id": "A", "demand": 1, "window": [0, 1], "service": 0, "penalty": 2},
            {"id": "B", "demand": 1, "window": [0, 100], "service": 2, "penalty": 0},
        ],
        "travel": {
            "D": {"D": 0, "A": 1, "B": 0},
            "A": {"D": 0, "A": 0, "B": 9},
            "B": {"D": 0, "A": 0.5, "B": 0},
        },
    }


def _financial_request(scenario, rate=1.0):
    return {"scenario": scenario, "currency": "USD",
            "operating_cost_per_travel_minute": rate,
            "penalty_rate_unit": "currency_per_late_minute"}


def _single_late_order_scenario(scenario, *, hard_window=False, close=1, outbound=2):
    value = json.loads(json.dumps(scenario))
    value["vehicles"] = value["vehicles"][:1]
    value["orders"] = value["orders"][:1]
    value["orders"][0]["window"] = [0, close]
    value["orders"][0]["service"] = 0
    value["orders"][0]["penalty"] = 4
    if hard_window:
        value["orders"][0]["hard_window"] = True
    value["travel"] = {"D": {"D": 0, "A": outbound}, "A": {"D": 0, "A": 0}}
    return value


def test_api_preserves_hard_window_flag_and_keeps_legacy_soft_default(client, scenario):
    soft = _single_late_order_scenario(scenario)
    soft_routes = client.post("/api/v1/scenarios/routes", json=soft)
    assert soft_routes.status_code == 200 and soft_routes.json()["count"] == 1
    soft_plan = client.post("/api/v1/scenarios/plan", json=soft)
    assert soft_plan.status_code == 200
    assert soft_plan.json()["plan"]["routes"][0]["stops"][0]["lateness"] == pytest.approx(1)

    hard = _single_late_order_scenario(scenario, hard_window=True)
    hard_routes = client.post("/api/v1/scenarios/routes", json=hard)
    assert hard_routes.status_code == 200 and hard_routes.json()["count"] == 0
    hard_plan = client.post("/api/v1/scenarios/plan", json=hard)
    assert hard_plan.status_code == 409
    assert hard_plan.json()["error"]["code"] == "infeasible"

    invalid = json.loads(json.dumps(hard))
    invalid["orders"][0]["hard_window"] = 1
    rejected = client.post("/api/v1/scenarios/validate", json=invalid)
    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "invalid_scenario"

    # The independent validation endpoint must receive the flag and reject a
    # legacy-soft plan that is late under this hard-window input.
    forged = client.post("/api/v1/plans/validate", json={"scenario": hard,
                             "plan": soft_plan.json()["plan"]})
    assert forged.status_code == 200 and not forged.json()["valid"]
    assert any("hard_window" in error["field"] for error in forged.json()["errors"])


def test_hard_windows_propagate_through_financial_risk_recovery_and_qaoa(client, scenario):
    hard_ok = _single_late_order_scenario(scenario, hard_window=True, close=3, outbound=1)
    original_plan = client.post("/api/v1/scenarios/plan", json=hard_ok).json()["plan"]
    assert validate_plan(hard_ok, original_plan)["valid"]

    hard_late = _single_late_order_scenario(scenario, hard_window=True, close=1, outbound=2)
    financial = client.post("/api/v1/scenarios/financial-plan", json=_financial_request(hard_late))
    assert financial.status_code == 409
    assert financial.json()["error"]["code"] == "infeasible"
    assert "plan" not in financial.json()

    risk_late = _single_late_order_scenario(scenario, hard_window=True, close=3, outbound=4)
    risk = client.post("/api/v1/risk/evaluate", json={
        "forecast_scenario": hard_ok, "nominal_scenario": hard_ok, "plan": original_plan,
        "realized_scenarios": [{"scenario_id": "hard-close-violation", "travel": risk_late["travel"]}]})
    assert risk.status_code == 200
    risk_result = risk.json()
    assert risk_result["status"] == "completed"
    assert risk_result["summary"]["empirical_probability_hard_window_violation"] == 1.0
    breach = risk_result["scenarios"][0]
    assert breach["feasible"] is False
    assert breach["hard_window_violation"] is True
    assert breach["validation"]["valid"] is False
    assert any("hard_window" in error for error in breach["validation"]["errors"])
    assert breach["routes"][0]["stops"][0]["lateness"] > 0

    recovery = client.post("/api/v1/recovery/plan", json={
        "original_scenario": hard_ok, "original_plan": original_plan,
        "updated_scenario": risk_late, "locked_prefixes": {}})
    assert recovery.status_code == 409
    assert recovery.json()["error"]["code"] == "recovery_infeasible"

    qaoa = client.post("/api/v1/scenarios/qaoa", json={
        "scenario": hard_late, "shots": 8, "angle_grid": 2})
    assert qaoa.status_code == 200
    assert qaoa.json()["status"] == "infeasible"
    assert qaoa.json()["execution_type"] == "classical"
    assert qaoa.json()["fallback"]["used"]


def test_vqc_supports_hard_windows_and_rejects_infeasible_hard_window(client, scenario):
    hard_ok = _single_late_order_scenario(scenario, hard_window=True, close=3, outbound=1)
    feasible = client.post("/api/v1/scenarios/vqc-plan", json={
        "forecast_scenario": hard_ok, "nominal_scenario": hard_ok,
        "theta": [0.31, -0.47], "model_version": "fixture-v1"})
    assert feasible.status_code == 200
    assert feasible.json()["validation"]["valid"]
    assert all(stop["lateness"] <= 0
               for route in feasible.json()["plan"]["routes"] for stop in route["stops"])

    hard_late = _single_late_order_scenario(scenario, hard_window=True, close=1, outbound=2)
    infeasible = client.post("/api/v1/scenarios/vqc-plan", json={
        "forecast_scenario": hard_late, "nominal_scenario": hard_late,
        "theta": [0.31, -0.47], "model_version": "fixture-v1"})
    assert infeasible.status_code == 409
    assert infeasible.json()["error"]["code"] == "infeasible"
    assert "fallback" not in infeasible.json()


def test_financial_plan_requires_explicit_terms_and_independently_validates(client):
    scenario = _financial_tradeoff_scenario()
    low = client.post("/api/v1/scenarios/financial-plan", json=_financial_request(scenario, 0.1))
    high = client.post("/api/v1/scenarios/financial-plan", json=_financial_request(scenario, 1.0))
    assert low.status_code == high.status_code == 200
    assert low.json()["financial_validation"]["valid"]
    assert high.json()["financial_validation"]["valid"]
    assert low.json()["financial_objective"]["total_cost"] == pytest.approx(1.0)
    assert high.json()["financial_objective"]["total_cost"] == pytest.approx(3.5)
    assert low.json()["plan"]["routes"][0]["order_ids"] == ["A", "B"]
    assert high.json()["plan"]["routes"][0]["order_ids"] == ["B", "A"]


def test_financial_endpoint_rejects_missing_terms_unknown_fields_and_secret_values(client):
    scenario = _financial_tradeoff_scenario()
    missing = _financial_request(scenario)
    del missing["currency"]
    response = client.post("/api/v1/scenarios/financial-plan", json=missing)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"

    invalid = _financial_request(scenario)
    invalid["currency"] = "SENTINEL_SECRET"
    response = client.post("/api/v1/scenarios/financial-plan", json=invalid)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_currency"
    assert "SENTINEL_SECRET" not in response.text

    scenario["untrusted_extra"] = "SENTINEL_SECRET"
    response = client.post("/api/v1/scenarios/financial-plan", json=_financial_request(scenario))
    assert response.status_code == 422
    assert "SENTINEL_SECRET" not in response.text


def test_financial_endpoint_distinguishes_infeasible_and_search_budget(client):
    impossible = _financial_tradeoff_scenario()
    impossible["orders"][0]["demand"] = 100
    response = client.post("/api/v1/scenarios/financial-plan", json=_financial_request(impossible))
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "infeasible"
    assert "plan" not in response.json()

    orders = [f"O{i}" for i in range(5)]
    locations = ["D", *orders]
    travel = {source: {target: (0 if source == target else 1) for target in locations}
              for source in locations}
    bounded = {"depot": "D", "vehicles": [{"id": f"V{i}", "capacity": 1} for i in range(8)],
               "orders": [{"id": oid, "demand": 1, "window": [0, 100], "service": 0, "penalty": 0}
                          for oid in orders], "travel": travel}
    response = client.post("/api/v1/scenarios/financial-plan", json=_financial_request(bounded))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "work_budget_exceeded"
    assert "plan" not in response.json()


def test_vqc_forwards_financial_terms_and_disables_proxy_fallback(client, scenario, monkeypatch):
    from backend import learning_inference

    small = _two_order_one_vehicle(scenario)
    financial_terms = {"currency": "USD", "operating_cost_per_travel_minute": 0.5,
                       "penalty_rate_unit": "currency_per_late_minute"}
    observed = {}

    def unavailable(*args, **kwargs):
        observed.update(kwargs)
        raise learning_inference.InferenceError("unsupported_capability", "runtime unavailable")

    monkeypatch.setattr(learning_inference, "predict_dispatch", unavailable)
    response = client.post("/api/v1/scenarios/vqc-plan", json={
        "forecast_scenario": small, "nominal_scenario": small,
        "theta": [0.31, -0.47], "model_version": "fixture-v1",
        "financial_terms": financial_terms})
    assert observed["financial_terms"] == financial_terms
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_capability"
    assert "fallback" not in response.json()


def test_vqc_financial_mode_returns_currency_estimate_and_independent_validation(client, scenario):
    small = _two_order_one_vehicle(scenario)
    response = client.post("/api/v1/scenarios/vqc-plan", json={
        "forecast_scenario": small, "nominal_scenario": small,
        "theta": [0.31, -0.47], "model_version": "fixture-v1",
        "financial_terms": {"currency": "USD", "operating_cost_per_travel_minute": 0.5,
                            "penalty_rate_unit": "currency_per_late_minute"}})
    assert response.status_code == 200
    result = response.json()
    assert result["validation"]["valid"]
    assert result["financial_validation"]["valid"]
    assert result["financial_estimate"]["currency"] == "USD"
    assert result["estimated_cost_currency"] == "USD"
    assert "not known to have been trained" in result["model_suitability_warning"]


def test_vqc_rejects_invalid_financial_terms_without_echo(client, scenario, monkeypatch):
    from backend import learning_inference

    small = _two_order_one_vehicle(scenario)

    def invalid_terms(*args, **kwargs):
        raise learning_inference.InferenceError("invalid_financial_terms", "financial terms rejected")

    monkeypatch.setattr(learning_inference, "predict_dispatch", invalid_terms)
    response = client.post("/api/v1/scenarios/vqc-plan", json={
        "forecast_scenario": small, "nominal_scenario": small,
        "theta": [0.31, -0.47], "model_version": "fixture-v1",
        "financial_terms": {"currency": "SECRET_SENTINEL"}})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_financial_terms"
    assert "SECRET_SENTINEL" not in response.text


def test_qaoa_route_exposes_checked_decision_and_explicit_execution_label(client, scenario):
    response = client.post("/api/v1/scenarios/qaoa", json={
        "scenario": _two_order_one_vehicle(scenario), "shots": 16, "angle_grid": 3})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    assert result["decision"]["validation"]["valid"]
    assert result["execution_type"] in {"local_simulator", "classical"}
    if result["fallback"]["used"]:
        assert result["execution_type"] == "classical"


def test_frozen_vqc_inference_endpoint_returns_checked_decision_without_retraining(client, scenario):
    small = _two_order_one_vehicle(scenario)
    response = client.post("/api/v1/scenarios/vqc-plan", json={
        "forecast_scenario": small, "nominal_scenario": small,
        "theta": [0.31, -0.47], "model_version": "fixture-v1"})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "completed"
    assert result["validation"]["valid"]
    assert result["execution_type"] == "local_cpu_exact_statevector"
    assert result["model"]["version"] == "fixture-v1"
    assert result["model"]["version_source"] == "caller-supplied identifier; not registry-verified"
    assert result["model"]["training_provenance"] == "unverified caller-supplied vector"


def test_vqc_inference_rejects_untrusted_extra_fields_without_echo(client, scenario):
    small = _two_order_one_vehicle(scenario)
    small["secret-token-value"] = "sentinel"
    response = client.post("/api/v1/scenarios/vqc-plan", json={
        "forecast_scenario": small, "nominal_scenario": small,
        "theta": [0.31, -0.47], "model_version": "fixture-v1"})
    assert response.status_code == 422
    assert "sentinel" not in response.text


def test_qpiai_local_backend_never_falls_back_when_capability_is_unavailable(client, scenario, monkeypatch):
    from backend import learning_inference

    small = _two_order_one_vehicle(scenario)

    def unavailable(*args, **kwargs):
        raise learning_inference.InferenceError("unsupported_capability", "requested runtime unavailable")

    monkeypatch.setattr(learning_inference, "predict_dispatch", unavailable)
    response = client.post("/api/v1/scenarios/vqc-plan", json={
        "forecast_scenario": small, "nominal_scenario": small,
        "theta": [0.31, -0.47], "model_version": "fixture-v1",
        "execution_backend": "qpiai_local"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_capability"
    assert "fallback" not in response.json()


def test_vqc_execution_backend_unknown_value_is_rejected_without_fallback(client, scenario):
    small = _two_order_one_vehicle(scenario)
    response = client.post("/api/v1/scenarios/vqc-plan", json={
        "forecast_scenario": small, "nominal_scenario": small,
        "theta": [0.31, -0.47], "model_version": "fixture-v1",
        "execution_backend": "DO_NOT_ECHO_SENTINEL"})
    assert response.status_code == 422
    assert "DO_NOT_ECHO_SENTINEL" not in response.text
    assert "fallback" not in response.json()


def test_vqc_classical_fallback_budget_exhaustion_is_not_hidden(client, scenario, monkeypatch):
    from backend import domain, learning_inference

    small = _two_order_one_vehicle(scenario)

    def unsupported(*args, **kwargs):
        raise learning_inference.InferenceError("unsupported_capability", "not supported")

    def exhausted(*args, **kwargs):
        raise domain.SolveBudgetExceeded()

    monkeypatch.setattr(learning_inference, "predict_dispatch", unsupported)
    monkeypatch.setattr(domain, "solve_exact", exhausted)
    response = client.post("/api/v1/scenarios/vqc-plan", json={
        "forecast_scenario": small, "nominal_scenario": small,
        "theta": [0.31, -0.47], "model_version": "fixture-v1"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "solver_budget_exceeded"
    assert "fallback" not in response.json()


def test_risk_endpoint_replays_and_independently_validates_each_scenario(client, scenario):
    small = _two_order_one_vehicle(scenario)
    plan = solve_exact(small)
    response = client.post("/api/v1/risk/evaluate", json={
        "forecast_scenario": small, "nominal_scenario": small, "plan": plan,
        "realized_scenarios": [{"scenario_id": "case-a", "travel": small["travel"]}],
        "cvar_alpha": 0.5})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "completed"
    assert result["estimator"]["scenario_count"] == 1
    assert result["scenarios"][0]["validation"]["valid"]
    assert result["cvar"]["sample_count"] == 1


def test_risk_endpoint_rejects_unknown_or_malformed_payload_without_echo(client, scenario):
    small = _two_order_one_vehicle(scenario)
    plan = solve_exact(small)
    payload = {"forecast_scenario": small, "nominal_scenario": small, "plan": plan,
               "realized_scenarios": [{"scenario_id": "ok", "travel": small["travel"]}],
               "unknown_private_field": "DO_NOT_ECHO_SENTINEL"}
    response = client.post("/api/v1/risk/evaluate", json=payload)
    assert response.status_code == 422
    assert "DO_NOT_ECHO_SENTINEL" not in response.text
    payload.pop("unknown_private_field")
    payload["realized_scenarios"] = [{"scenario_id": "bad", "travel": {"private": "DO_NOT_ECHO_SENTINEL"}}]
    response = client.post("/api/v1/risk/evaluate", json=payload)
    assert response.status_code == 422
    assert "DO_NOT_ECHO_SENTINEL" not in response.text


def test_vqc_inference_bound_uses_visible_validated_classical_fallback(client, scenario):
    large = json.loads(json.dumps(scenario))
    large["vehicles"] = [{"id": "V1", "capacity": 4}, {"id": "V2", "capacity": 4}]
    large["orders"].append({"id": "E", "demand": 1, "window": [0, 50], "service": 1, "penalty": 1})
    for origin in large["travel"].values():
        origin["E"] = 1
    large["travel"]["E"] = {key: 1 for key in ["D", "A", "B", "C", "E"]}
    large["travel"]["E"]["E"] = 0
    response = client.post("/api/v1/scenarios/vqc-plan", json={
        "forecast_scenario": large, "nominal_scenario": large,
        "theta": [0.31, -0.47], "model_version": "fixture-v1"})
    assert response.status_code == 200
    result = response.json()
    assert result["execution_type"] == "classical"
    assert result["fallback"] == {"used": True, "reason": "unsupported_capability"}
    assert result["validation"]["valid"]


def test_recovery_endpoint_preserves_locked_prefix_and_validates_new_plan(client, scenario):
    original_plan = solve_exact(scenario)
    prefixes = {route["vehicle_id"]: route["order_ids"][:1] for route in original_plan["routes"]}
    response = client.post("/api/v1/recovery/plan", json={
        "original_scenario": scenario, "original_plan": original_plan,
        "updated_scenario": scenario, "locked_prefixes": prefixes, "change_penalty": 1.0})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "recovered"
    assert result["validation"]["valid"]
    assert result["state_basis"].startswith("depot_replay")


def test_recovery_endpoint_returns_stable_conflict_for_missing_locked_vehicle(client, scenario):
    original_plan = solve_exact(scenario)
    locked_vehicle = original_plan["routes"][0]["vehicle_id"]
    prefix = original_plan["routes"][0]["order_ids"][:1]
    updated = json.loads(json.dumps(scenario))
    updated["vehicles"] = [row for row in updated["vehicles"] if row["id"] != locked_vehicle]
    response = client.post("/api/v1/recovery/plan", json={
        "original_scenario": scenario, "original_plan": original_plan,
        "updated_scenario": updated, "locked_prefixes": {locked_vehicle: prefix}})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "locked_work_vehicle_unavailable"


def test_invalid_scenario_has_stable_field_error_without_echoing_input(client, scenario):
    scenario["orders"][0]["window"] = [9, 1]
    scenario["api_key"] = "sentinel-secret-value"
    response = client.post("/api/v1/scenarios/validate", json=scenario)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_scenario"
    assert "window" in response.text
    assert "sentinel-secret-value" not in response.text


def test_malformed_container_types_return_field_errors(client):
    response = client.post("/api/v1/scenarios/validate", json={"vehicles": None, "orders": "bad"})
    assert response.status_code == 422
    fields = {item["field"] for item in response.json()["error"]["details"]}
    assert "vehicles" in fields and "orders" in fields


def test_plan_validation_recomputes_input_and_detects_forged_cost(client, scenario):
    plan_response = client.post("/api/v1/scenarios/plan", json=scenario).json()["plan"]
    plan_response["objective"] += 10
    response = client.post("/api/v1/plans/validate", json={"scenario": scenario, "plan": plan_response})
    assert response.status_code == 200
    assert not response.json()["valid"]
    assert any(error["field"] == "plan.objective" for error in response.json()["errors"])


@pytest.mark.parametrize("bad_vehicle,bad_order", [({}, "A"), ("V1", {"malformed": "identifier"})])
def test_plan_validator_rejects_unhashable_external_identifiers_without_crashing(
        client, scenario, bad_vehicle, bad_order):
    malformed = {"status": "feasible", "routes": [{"vehicle_id": bad_vehicle,
                  "order_ids": [bad_order]}], "travel_cost": 0, "lateness_exposure": 0, "objective": 0}
    response = client.post("/api/v1/plans/validate", json={"scenario": scenario, "plan": malformed})
    assert response.status_code == 200
    assert not response.json()["valid"]
    assert response.json()["errors"]
    assert "malformed" not in response.text


def test_route_enumeration_budget_rejects_high_factorial_pool(client):
    n = 8
    ids = [f"O{i}" for i in range(n)]
    locations = ["D", *ids]
    scenario = {"depot": "D", "vehicles": [{"id": "V1", "capacity": 99},
                                              {"id": "V2", "capacity": 99}],
                "orders": [{"id": oid, "demand": 1, "window": [0, 100],
                            "service": 1, "penalty": 1} for oid in ids],
                "travel": {origin: {dest: (0 if origin == dest else 1) for dest in locations}
                           for origin in locations}}
    response = client.post("/api/v1/scenarios/validate", json=scenario)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_scenario"
    assert any(item["field"] == "vehicles" for item in response.json()["error"]["details"])


def test_non_object_rows_are_rejected_instead_of_silently_dropped(client, scenario):
    scenario["vehicles"].insert(0, "sentinel-secret-value")
    response = client.post("/api/v1/scenarios/validate", json=scenario)
    assert response.status_code == 422
    assert any(item["field"] == "vehicles[0]" for item in response.json()["error"]["details"])
    assert "sentinel-secret-value" not in response.text


def test_unknown_field_names_and_malicious_order_ids_are_not_echoed(client, scenario):
    scenario["private-secret-value"] = "x"
    response = client.post("/api/v1/scenarios/validate", json=scenario)
    assert response.status_code == 422
    assert "private-secret-value" not in response.text
    bad_plan = {"status": "feasible", "routes": [{"vehicle_id": "V1",
                 "order_ids": ["credential-sentinel"]}], "travel_cost": 0,
                "lateness_exposure": 0, "objective": 0}
    response = client.post("/api/v1/plans/validate", json={"scenario": scenario | {"private-secret-value": None},
                                                           "plan": bad_plan})
    # Scenario boundary itself reports only generic, non-echoing field paths.
    assert "credential-sentinel" not in response.text


def test_api_registers_sanitized_framework_validation_handler(client):
    assert RequestValidationError in client.app.exception_handlers


def test_async_job_can_be_polled_to_success_and_keeps_execution_metadata(client, scenario):
    submitted = client.post("/api/v1/jobs", json=scenario)
    assert submitted.status_code == 202
    job_id = submitted.json()["id"]
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        current = client.get(f"/api/v1/jobs/{job_id}").json()
        if current["status"] in {"succeeded", "failed", "infeasible", "timed_out"}:
            break
        time.sleep(0.01)
    assert current["status"] == "succeeded"
    assert current["result"]["validation"]["valid"]
    assert current["metadata"]["execution_type"] == "local_classical"
    assert current["metadata"]["solver"] == "exact_enumeration"
    assert len(current["metadata"]["request_sha256"]) == 64
    assert current["metadata"]["runtime"]["python"]


def _qaoa_job_request(scenario):
    return {"scenario": _two_order_one_vehicle(scenario), "seed": 17,
            "shots": 24, "warm_start": True, "angle_grid": 3}


def _poll_job(client, job_id, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = client.get(f"/api/v1/jobs/{job_id}").json()
        if current["status"] in {"succeeded", "failed", "infeasible", "timed_out", "cancelled"}:
            return current
        time.sleep(0.005)
    raise AssertionError("job did not reach a terminal state")


def test_qaoa_job_runs_local_solver_and_persists_raw_decision_and_settings(client, scenario):
    response = client.post("/api/v1/jobs/qaoa", json=_qaoa_job_request(scenario))
    assert response.status_code == 202
    submitted = response.json()
    assert submitted["metadata"]["solver"] == "route_qaoa_with_exact_fallback"
    current = _poll_job(client, submitted["id"])
    assert current["status"] == "succeeded"
    result = current["result"]
    assert result["status"] == "success"
    assert result["decision"]["validation"]["valid"]
    assert result["execution_type"] in {"local_simulator", "classical"}
    assert result["fallback"]["used"] == (result["decision"]["origin"] == "classical_exact_fallback")
    assert result["optimization"]["execution_type"] in {"local_python_statevector_simulator", "not_run"}
    if result["optimization"].get("raw_result") is not None:
        assert isinstance(result["optimization"]["raw_result"]["raw_samples"], list)
    metadata = current["metadata"]
    assert metadata["settings"]["seed"] == 17
    assert metadata["settings"]["shots"] == 24
    assert metadata["settings"]["execution_backend"] == "local_python_statevector"
    assert len(metadata["request_sha256"]) == 64


def test_qaoa_job_validates_explicit_settings_and_redacts_worker_exception(client, scenario, monkeypatch):
    invalid = client.post("/api/v1/jobs/qaoa", json={"scenario": _two_order_one_vehicle(scenario),
        "seed": 1, "shots": 513, "warm_start": True, "angle_grid": 3})
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "invalid_solver_settings"

    import backend.jobs as jobs
    from threading import Event
    started, release = Event(), Event()

    def fail_with_secret(*args, **kwargs):
        started.set()
        assert release.wait(2)
        raise RuntimeError("QAOA SECRET_SENTINEL")

    monkeypatch.setattr(jobs, "solve_qaoa_dispatch", fail_with_secret)
    submitted = client.post("/api/v1/jobs/qaoa", json=_qaoa_job_request(scenario)).json()
    assert started.wait(2)
    release.set()
    terminal = _poll_job(client, submitted["id"])
    assert terminal["status"] == "failed"
    assert terminal["error"]["code"] == "solver_failure"
    assert "SECRET_SENTINEL" not in json.dumps(terminal)


def test_qaoa_fault_uses_explicit_validated_classical_fallback(client, scenario, monkeypatch):
    import backend.qaoa_dispatch as qaoa_dispatch

    def fail_backend(*args, **kwargs):
        raise RuntimeError("simulator fault")

    monkeypatch.setattr(qaoa_dispatch, "run_qaoa", fail_backend)
    submitted = client.post("/api/v1/jobs/qaoa", json=_qaoa_job_request(scenario)).json()
    terminal = _poll_job(client, submitted["id"])
    assert terminal["status"] == "succeeded"
    result = terminal["result"]
    assert result["execution_type"] == "classical"
    assert result["decision"]["origin"] == "classical_exact_fallback"
    assert result["decision"]["validation"]["valid"]
    assert result["fallback"]["used"]
    assert result["optimization"]["qaoa_failure"]["code"] == "QAOA_FAILED"


def test_qaoa_job_timeout_and_cancellation_suppress_late_results(tmp_path, scenario, monkeypatch):
    import backend.jobs as jobs
    from threading import Event

    started, release, finished = Event(), Event(), Event()
    original = jobs.solve_qaoa_dispatch

    def blocked(*args, **kwargs):
        started.set()
        try:
            if not release.wait(3):
                raise AssertionError("test did not release QAOA worker")
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(jobs, "solve_qaoa_dispatch", blocked)
    manager = JobManager(tmp_path / "qaoa-lifecycle.sqlite3", max_workers=1, timeout_seconds=5)
    try:
        request = _qaoa_job_request(scenario)
        timed = manager.submit_qaoa(request["scenario"], seed=17, shots=24, warm_start=True, angle_grid=3)
        assert started.wait(2)
        with manager._connect() as connection:
            connection.execute("UPDATE jobs SET deadline_epoch=? WHERE id=?", (time.time() - 1, timed["id"]))
        assert manager.get(timed["id"])["status"] == "timed_out"
        with pytest.raises(JobCapacityError):
            manager.submit_qaoa(request["scenario"], seed=17, shots=24, warm_start=True, angle_grid=3)
        release.set()
        assert finished.wait(3)
        assert manager.get(timed["id"])["result"] is None

        # Second worker is independently cancelled while blocked.
        started.clear(); release.clear(); finished.clear()
        cancelled = manager.submit_qaoa(request["scenario"], seed=17, shots=24, warm_start=True, angle_grid=3)
        assert started.wait(2)
        assert manager.cancel(cancelled["id"])["status"] == "cancelled"
        release.set()
        assert finished.wait(3)
        state = manager.get(cancelled["id"])
        assert state["status"] == "cancelled" and state["result"] is None
    finally:
        release.set()
        manager.close(wait=True)


def test_cancel_endpoint_transitions_job_to_cancelled(client, scenario, monkeypatch):
    import backend.jobs as jobs
    from threading import Event

    original = jobs.solve_exact
    worker_started = Event()
    release_worker = Event()

    def sync_solve(value):
        worker_started.set()
        release_worker.wait(timeout=3.0)
        return original(value)

    monkeypatch.setattr(jobs, "solve_exact", sync_solve)
    try:
        submitted = client.post("/api/v1/jobs", json=scenario).json()
        assert worker_started.wait(timeout=3.0)
        response = client.post(f"/api/v1/jobs/{submitted['id']}/cancel")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
    finally:
        release_worker.set()


def test_job_timeout_is_persisted_and_late_result_not_published(tmp_path, scenario, monkeypatch):
    import backend.jobs as jobs

    original = jobs.solve_exact
    def slow_solve(value):
        time.sleep(0.12)
        return original(value)
    monkeypatch.setattr(jobs, "solve_exact", slow_solve)
    manager = JobManager(tmp_path / "timeout.sqlite3", max_workers=1, timeout_seconds=0.03)
    try:
        submitted = manager.submit(scenario)
        time.sleep(0.05)
        current = manager.get(submitted["id"])
        assert current["status"] == "timed_out"
        assert current["result"] is None
    finally:
        manager.close(wait=True)


def test_restart_marks_inflight_job_as_interrupted(tmp_path, scenario):
    path = tmp_path / "restart.sqlite3"
    manager = JobManager(path, max_workers=1)
    manager.close(wait=True)
    with sqlite3.connect(path) as con:
        con.execute("""INSERT INTO jobs(id,status,scenario_json,result_json,error_json,submitted_at,started_at,
                    completed_at,deadline_epoch,execution_type,solver) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    ("interrupted", "running", json.dumps(scenario), None, None,
                     "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00", None,
                     time.time() + 60, "local_classical", "exact_enumeration"))
        con.execute("INSERT INTO runtime_lock VALUES(1,?,?,?)",
                    (4294967295, "stale-owner", "2020-01-01T00:00:00+00:00"))
    recovered = JobManager(path, max_workers=1)
    try:
        job = recovered.get("interrupted")
        assert job["status"] == "failed"
        assert job["error"]["code"] == "restart_interrupted"
    finally:
        recovered.close(wait=True)


def test_restart_recovers_qaoa_job_and_keeps_settings_and_hash(tmp_path, scenario):
    path = tmp_path / "qaoa-restart.sqlite3"
    manager = JobManager(path, max_workers=1)
    request = _qaoa_job_request(scenario)
    submitted = manager.submit_qaoa(request["scenario"], seed=request["seed"], shots=request["shots"],
                                    warm_start=request["warm_start"], angle_grid=request["angle_grid"])
    _poll_until = time.monotonic() + 5
    while time.monotonic() < _poll_until and manager.get(submitted["id"])["status"] not in {
            "succeeded", "failed", "infeasible", "timed_out"}:
        time.sleep(0.005)
    with manager._connect() as con:
        con.execute("UPDATE jobs SET status='running',result_json=NULL,error_json=NULL,completed_at=NULL WHERE id=?",
                    (submitted["id"],))
    manager.close(wait=True)
    with sqlite3.connect(path) as con:
        con.execute("INSERT INTO runtime_lock VALUES(1,?,?,?)",
                    (4294967295, "stale-qaoa-owner", "2020-01-01T00:00:00+00:00"))
    recovered = JobManager(path, max_workers=1)
    try:
        row = recovered.get(submitted["id"])
        assert row["status"] == "failed"
        assert row["error"]["code"] == "restart_interrupted"
        assert row["metadata"]["settings"]["seed"] == 17
        assert row["metadata"]["execution_type"] == "local_qaoa_job"
        assert len(row["metadata"]["request_sha256"]) == 64
    finally:
        recovered.close(wait=True)


def test_unknown_job_returns_stable_not_found(client):
    response = client.get("/api/v1/jobs/not-a-real-job")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "job_not_found"


def test_manager_bounds_active_jobs(tmp_path, scenario, monkeypatch):
    import backend.jobs as jobs
    original = jobs.solve_exact
    def slow_solve(value):
        time.sleep(0.2)
        return original(value)
    monkeypatch.setattr(jobs, "solve_exact", slow_solve)
    manager = JobManager(tmp_path / "capacity.sqlite3", max_workers=1, timeout_seconds=2)
    try:
        manager.submit(scenario)
        with pytest.raises(JobCapacityError):
            manager.submit(scenario)
    finally:
        manager.close(wait=True)


def test_sqlite_database_rejects_second_live_manager_and_reopens_after_close(tmp_path):
    path = tmp_path / "lease.sqlite3"
    first = JobManager(path, max_workers=1)
    try:
        with pytest.raises(RuntimeError, match="live runtime owner"):
            JobManager(path, max_workers=1)
    finally:
        first.close(wait=True)
    second = JobManager(path, max_workers=1)
    second.close(wait=True)


@pytest.mark.parametrize("terminal_action", ["timeout", "cancel"])
def test_terminal_state_does_not_release_slot_while_solver_thread_still_runs(
        tmp_path, scenario, monkeypatch, terminal_action):
    import backend.jobs as jobs
    from threading import Event
    original = jobs.solve_exact
    started, release, done = Event(), Event(), Event()
    def slow_solve(value):
        started.set()
        try:
            if not release.wait(2):
                raise AssertionError("test did not release exact solver")
            return original(value)
        finally:
            done.set()
    monkeypatch.setattr(jobs, "solve_exact", slow_solve)
    manager = JobManager(tmp_path / f"{terminal_action}.sqlite3", max_workers=1,
                         timeout_seconds=2)
    try:
        first = manager.submit(scenario)
        assert started.wait(2)
        if terminal_action == "timeout":
            with manager._connect() as connection:
                connection.execute("UPDATE jobs SET deadline_epoch=? WHERE id=?", (time.time() - 1, first["id"]))
            assert manager.get(first["id"])["status"] == "timed_out"
        else:
            assert manager.cancel(first["id"])["status"] == "cancelled"
        with pytest.raises(JobCapacityError):
            manager.submit(scenario)
        release.set()
        assert done.wait(2)
        deadline = time.monotonic() + 2
        while manager._active_count() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert manager._active_count() == 0
        second = manager.submit(scenario)
        assert second["id"]
    finally:
        release.set()
        manager.close(wait=True)


def test_timeout_during_budgeted_search_holds_slot_and_never_overwrites_timeout(
        tmp_path, monkeypatch):
    import backend.jobs as jobs
    from threading import Event
    original = jobs.solve_exact
    worker_started = Event()
    release_worker = Event()
    worker_done = Event()

    def delayed_budgeted_solve(value):
        worker_started.set()
        try:
            if not release_worker.wait(2):
                raise AssertionError("test did not release the solver worker")
            return original(value)
        finally:
            worker_done.set()

    monkeypatch.setattr(jobs, "solve_exact", delayed_budgeted_solve)
    manager = JobManager(tmp_path / "budget-timeout.sqlite3", max_workers=1, timeout_seconds=2)
    try:
        first = manager.submit(_high_branching_assignment_case())
        assert worker_started.wait(2)
        # Expire the persisted deadline only after the worker is confirmed
        # inside the solver, avoiding scheduler-dependent sleeps.
        with manager._connect() as connection:
            connection.execute("UPDATE jobs SET deadline_epoch=? WHERE id=?", (time.time() - 1, first["id"]))
        assert manager.get(first["id"])["status"] == "timed_out"
        with pytest.raises(JobCapacityError):
            manager.submit(_high_branching_assignment_case())
        release_worker.set()
        assert worker_done.wait(2)
        deadline = time.monotonic() + 1
        while manager._active_count() and time.monotonic() < deadline:
            time.sleep(0.001)
        expired = manager.get(first["id"])
        assert expired["status"] == "timed_out"
        assert expired["error"]["code"] == "timeout"
        assert expired["result"] is None
        # Once the worker has actually exited, its retained slot is reusable.
        second = manager.submit(_high_branching_assignment_case())
        assert second["id"]
    finally:
        release_worker.set()
        manager.close(wait=True)


def test_completed_success_is_not_overwritten_by_a_late_timeout_poll(tmp_path, scenario):
    manager = JobManager(tmp_path / "completion-race.sqlite3", max_workers=1, timeout_seconds=2)
    try:
        job = manager.submit(scenario)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and manager.get(job["id"])["status"] not in {
                "succeeded", "failed", "infeasible"}:
            time.sleep(0.005)
        with sqlite3.connect(manager.db_path) as con:
            con.execute("UPDATE jobs SET deadline_epoch=? WHERE id=?", (time.time() - 1, job["id"]))
        assert manager.get(job["id"])["status"] == "succeeded"
    finally:
        manager.close(wait=True)
