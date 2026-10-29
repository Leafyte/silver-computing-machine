"""Versioned FastAPI surface for local dispatch and persisted jobs."""

from __future__ import annotations

from typing import Any
from threading import BoundedSemaphore
from datetime import datetime, timezone

# pyrefly: ignore[missing-import]
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from math import isfinite

from .domain import SolveBudgetExceeded, generate_routes, solve_exact, validate_scenario
from .jobs import JobCapacityError, JobManager
from .recovery_store import RecoveryAuditStore
from .validator import validate_plan
from .auth import AuthConfig, TokenAuthMiddleware


def _canonical_scenario(raw: dict[str, Any]) -> dict[str, Any]:
    """Copy only recognized scenario fields into solver/storage boundaries."""
    vehicles = raw.get("vehicles", [])
    orders = raw.get("orders", [])
    return {
        "depot": raw.get("depot"),
        "vehicles": ([{"id": row.get("id"), "capacity": row.get("capacity")}
                      for row in vehicles if isinstance(row, dict)]
                     if isinstance(vehicles, list) else vehicles),
        "orders": ([{key: row.get(key) for key in ("id", "demand", "window", "service", "penalty")}
                     | ({"hard_window": row["hard_window"]} if "hard_window" in row else {})
                    for row in orders if isinstance(row, dict)]
                   if isinstance(orders, list) else orders),
        "travel": raw.get("travel"),
    }


def _field_errors(errors: list[str]) -> list[dict[str, str]]:
    # Return stable schema paths only; do not echo values embedded in IDs.
    output = []
    for message in errors:
        path = message.split(":", 1)[0]
        if path.startswith("coverage."):
            path = "coverage"
        output.append({"field": path, "message": "invalid or missing value"})
    return output


def _unknown_scenario_fields(raw: dict[str, Any]) -> list[dict[str, str]]:
    allowed = {"depot", "vehicles", "orders", "travel"}
    output = [{"field": "scenario", "message": "unsupported field"} for key in raw if key not in allowed]
    for collection, fields in (("vehicles", {"id", "capacity"}),
                               ("orders", {"id", "demand", "window", "service", "penalty", "hard_window"})):
        rows = raw.get(collection)
        if isinstance(rows, list):
            for index, row in enumerate(rows):
                if isinstance(row, dict):
                    output.extend({"field": f"{collection}[{index}]", "message": "unsupported field"}
                                  for key in row if key not in fields)
    return output


def _invalid_scenario(raw: Any) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    if not isinstance(raw, dict):
        return None, JSONResponse(status_code=422, content={"error": {
            "code": "invalid_scenario", "message": "scenario must be a JSON object",
            "details": [{"field": "scenario", "message": "invalid value"}]}})
    unknown = _unknown_scenario_fields(raw)
    checked = validate_scenario(raw)
    if unknown or not checked["valid"]:
        return None, JSONResponse(status_code=422, content={"error": {
            "code": "invalid_scenario", "message": "scenario validation failed",
            "details": unknown + _field_errors(checked["errors"])}})
    scenario = _canonical_scenario(raw)
    return scenario, None


def _canonical_validated_plan(raw: Any) -> dict[str, Any] | None:
    """Drop no plan semantics, but reject unknown keys before audit storage."""
    plan_fields = {"status", "routes", "travel_cost", "lateness_exposure", "objective"}
    route_fields = {"vehicle_id", "order_ids", "stops", "return_time", "travel_cost",
                    "lateness_exposure", "objective"}
    stop_fields = {"order_id", "arrival", "service_start", "departure", "lateness", "penalty_exposure"}
    if not isinstance(raw, dict) or set(raw) != plan_fields or not isinstance(raw.get("routes"), list):
        return None
    routes = []
    for route in raw["routes"]:
        if not isinstance(route, dict) or set(route) != route_fields or not isinstance(route.get("stops"), list):
            return None
        stops = []
        for stop in route["stops"]:
            if not isinstance(stop, dict) or set(stop) != stop_fields:
                return None
            stops.append({field: stop[field] for field in stop_fields})
        routes.append({field: route[field] for field in route_fields if field != "stops"} | {"stops": stops})
    return {field: raw[field] for field in plan_fields if field != "routes"} | {"routes": routes}


def _validated_locked_prefixes(raw: Any, original: dict[str, Any], plan: dict[str, Any]) -> dict[str, list[str]] | None:
    if not isinstance(raw, dict):
        return None
    vehicles = {row["id"] for row in original["vehicles"]}
    routes = {route["vehicle_id"]: route["order_ids"] for route in plan["routes"]}
    validated: dict[str, list[str]] = {}
    for vehicle_id, prefix in raw.items():
        if (not isinstance(vehicle_id, str) or vehicle_id not in vehicles or
                not isinstance(prefix, list) or any(not isinstance(order_id, str) for order_id in prefix)):
            return None
        route = routes.get(vehicle_id)
        if prefix and (route is None or route[:len(prefix)] != prefix):
            return None
        validated[vehicle_id] = list(prefix)
    return validated


def create_app(job_manager: JobManager | None = None, *, auth_config: AuthConfig | None = None,
               eta_db_path=None, eta_registry=None, eta_uncertainty_registry=None,
               eta_controller=None, eta_job_manager=None) -> FastAPI:
    manager = job_manager
    app = FastAPI(title="DebugLeaf Dispatch API", version="1.0.0")
    auth = auth_config or AuthConfig.from_environment()
    app.state.auth_mode = auth.mode if auth.ready else "misconfigured"
    from .india_eta.body_limit import EtaBodyLimit
    app.add_middleware(EtaBodyLimit)
    app.add_middleware(TokenAuthMiddleware, config=auth)
    app.state.job_manager = manager
    compute_slots = BoundedSemaphore(2)

    def jobs() -> JobManager:
        nonlocal manager
        if manager is None:
            manager = JobManager()
            app.state.job_manager = manager
        return manager

    recovery_store: RecoveryAuditStore | None = None

    def recovery_audits() -> RecoveryAuditStore:
        nonlocal recovery_store
        runtime_manager = jobs()  # claims the shared SQLite process lease first
        if recovery_store is None:
            recovery_store = RecoveryAuditStore(runtime_manager.db_path)
            app.state.recovery_audit_store = recovery_store
        return recovery_store

    def capacity_error() -> JSONResponse:
        return JSONResponse(status_code=429, content={"error": {
            "code": "capacity_reached", "message": "local compute capacity reached"}})

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "api_version": "v1"}

    @app.post("/api/v1/scenarios/validate")
    def scenario_validate(payload: Any = Body(...)) -> Any:
        scenario, error = _invalid_scenario(payload)
        if error:
            return error
        checked = validate_scenario(scenario)
        return {"valid": checked["valid"], "errors": [], "execution_type": "local_classical"}

    @app.post("/api/v1/scenarios/routes")
    def routes(payload: Any = Body(...)) -> Any:
        scenario, error = _invalid_scenario(payload)
        if error:
            return error
        if not compute_slots.acquire(blocking=False):
            return capacity_error()
        try:
            result = generate_routes(scenario)
        except Exception:
            return JSONResponse(status_code=500, content={"error": {
                "code": "solver_failure", "message": "route generation failed"}})
        finally:
            compute_slots.release()
        return {"routes": result, "count": len(result), "execution_type": "local_classical"}

    @app.post("/api/v1/scenarios/plan")
    def plan(payload: Any = Body(...)) -> Any:
        scenario, error = _invalid_scenario(payload)
        if error:
            return error
        if not compute_slots.acquire(blocking=False):
            return capacity_error()
        try:
            result = solve_exact(scenario)
            if result["status"] == "infeasible":
                return JSONResponse(status_code=409, content={"error": {
                    "code": "infeasible", "message": "no exact feasible dispatch plan exists"},
                    "plan": result, "execution_type": "local_classical"})
            validation = validate_plan(scenario, result)
            if not validation["valid"]:
                return JSONResponse(status_code=500, content={"error": {
                    "code": "validation_failure", "message": "independent plan validation failed"}})
            return {"plan": result, "validation": validation, "execution_type": "local_classical"}
        except SolveBudgetExceeded:
            return JSONResponse(status_code=503, content={"error": {
                "code": SolveBudgetExceeded.code, "message": "dispatch search budget exceeded; no plan was published"}})
        except Exception:
            return JSONResponse(status_code=500, content={"error": {
                "code": "solver_failure", "message": "dispatch solver failed"}})
        finally:
            compute_slots.release()

    @app.post("/api/v1/scenarios/financial-plan")
    def financial_plan(payload: Any = Body(...)) -> Any:
        required = {"scenario", "currency", "operating_cost_per_travel_minute", "penalty_rate_unit"}
        if not isinstance(payload, dict) or set(payload) != required:
            return JSONResponse(status_code=422, content={"error": {
                "code": "invalid_request", "message": "financial request requires scenario, currency, operating_cost_per_travel_minute, and penalty_rate_unit"}})
        scenario, error = _invalid_scenario(payload["scenario"])
        if error:
            return error
        if not compute_slots.acquire(blocking=False):
            return capacity_error()
        try:
            from .financial import FinancialError, solve_financial_dispatch, validate_financial_plan
            terms = {"currency": payload["currency"],
                     "operating_cost_per_travel_minute": payload["operating_cost_per_travel_minute"],
                     "penalty_rate_unit": payload["penalty_rate_unit"]}
            result = solve_financial_dispatch(scenario, **terms)
            independent = validate_financial_plan(
                scenario, result.get("plan"), result.get("financial_objective"), **terms)
            if not result.get("proxy_validation", {}).get("valid") or not independent["valid"]:
                return JSONResponse(status_code=500, content={"error": {
                    "code": "financial_validation_failure", "message": "financial plan failed independent validation"}})
            result["financial_validation"] = independent
            return result
        except FinancialError as exc:
            status = (409 if exc.code == "infeasible" else
                      503 if exc.code == "work_budget_exceeded" else
                      500 if exc.code.endswith("validation_failure") or exc.code == "validation_failure" else 422)
            return JSONResponse(status_code=status, content=exc.as_dict())
        except Exception:
            return JSONResponse(status_code=500, content={"error": {
                "code": "financial_solver_failure", "message": "financial dispatch solver failed"}})
        finally:
            compute_slots.release()

    @app.post("/api/v1/scenarios/qaoa")
    def qaoa_plan(payload: Any = Body(...)) -> Any:
        if not isinstance(payload, dict) or set(payload) - {"scenario", "seed", "shots", "warm_start", "angle_grid"}:
            return JSONResponse(status_code=422, content={"error": {
                "code": "invalid_request", "message": "unsupported QAOA request fields"}})
        scenario, error = _invalid_scenario(payload.get("scenario"))
        if error:
            return error
        seed, shots = payload.get("seed", 7), payload.get("shots", 128)
        warm_start, angle_grid = payload.get("warm_start", True), payload.get("angle_grid", 8)
        if (isinstance(seed, bool) or not isinstance(seed, int) or
                isinstance(shots, bool) or not isinstance(shots, int) or not 1 <= shots <= 512 or
                not isinstance(warm_start, bool) or isinstance(angle_grid, bool) or
                not isinstance(angle_grid, int) or not 2 <= angle_grid <= 10):
            return JSONResponse(status_code=422, content={"error": {
                "code": "invalid_solver_settings", "message": "QAOA settings exceed the local bounds"}})
        try:
            if not compute_slots.acquire(blocking=False):
                return capacity_error()
            try:
                from .domain import generate_routes
                if len(generate_routes(scenario)) > 12:
                    return JSONResponse(status_code=422, content={"error": {
                        "code": "unsupported_capability", "message": "local QAOA supports at most 12 route variables"}})
                from .qaoa_dispatch import solve_qaoa_dispatch
                result = solve_qaoa_dispatch(scenario, seed=seed, shots=shots,
                                             warm_start=warm_start, angle_grid=angle_grid)
            finally:
                compute_slots.release()
            if result.get("status") == "invalid_input":
                return JSONResponse(status_code=422, content={"error": {
                    "code": "invalid_scenario", "message": "scenario rejected by QAOA path"}})
            return result
        except SolveBudgetExceeded:
            return JSONResponse(status_code=503, content={"error": {
                "code": SolveBudgetExceeded.code, "message": "dispatch search budget exceeded; no plan was published"}})
        except Exception:
            return JSONResponse(status_code=500, content={"error": {
                "code": "solver_failure", "message": "QAOA dispatch failed"}})

    @app.post("/api/v1/scenarios/vqc-plan")
    def vqc_plan(payload: Any = Body(...)) -> Any:
        required = {"forecast_scenario", "nominal_scenario", "theta", "model_version"}
        optional = {"execution_backend", "financial_terms"}
        if (not isinstance(payload, dict) or not required <= set(payload) or
                set(payload) - required - optional):
            return JSONResponse(status_code=422, content={"error": {
                "code": "invalid_request", "message": "expected forecast_scenario, nominal_scenario, theta, model_version, and optional execution_backend or financial_terms"}})
        execution_backend = payload.get("execution_backend", "reference")
        if not compute_slots.acquire(blocking=False):
            return capacity_error()
        try:
            from .learning_inference import InferenceError, predict_dispatch
            inference_kwargs = {"model_version": payload["model_version"],
                                "execution_backend": execution_backend}
            if "financial_terms" in payload:
                inference_kwargs["financial_terms"] = payload["financial_terms"]
            result = predict_dispatch(payload["forecast_scenario"], payload["nominal_scenario"],
                                      payload["theta"], **inference_kwargs)
            return result
        except InferenceError as exc:
            if ("financial_terms" not in payload and execution_backend != "qpiai_local" and
                    exc.code in {"unsupported_capability", "validation_failure"}):
                scenario, validation_error = _invalid_scenario(payload["forecast_scenario"])
                if validation_error is None:
                    try:
                        from .domain import solve_exact
                        from .validator import validate_plan
                        plan = solve_exact(scenario)
                        report = validate_plan(scenario, plan) if plan.get("status") == "feasible" else None
                        if plan.get("status") == "infeasible" or (report and report["valid"]):
                            return {"status": plan["status"], "execution_type": "classical",
                                    "plan": plan, "validation": report,
                                    "fallback": {"used": True, "reason": exc.code},
                                    "vqc_error_code": exc.code}
                    except SolveBudgetExceeded:
                        return JSONResponse(status_code=503, content={"error": {
                            "code": SolveBudgetExceeded.code, "message": "dispatch search budget exceeded; fallback plan was not proven"}})
                    except Exception:
                        pass
            status = 409 if exc.code == "infeasible" else 422 if exc.code.startswith("invalid_") or exc.code == "unsupported_capability" else 500
            return JSONResponse(status_code=status, content=exc.as_dict())
        except SolveBudgetExceeded:
            return JSONResponse(status_code=503, content={"error": {
                "code": SolveBudgetExceeded.code, "message": "dispatch search budget exceeded; no fallback plan was published"}})
        except Exception:
            scenario, validation_error = _invalid_scenario(payload["forecast_scenario"])
            if ("financial_terms" not in payload and execution_backend != "qpiai_local" and
                    validation_error is None):
                try:
                    from .domain import solve_exact
                    from .validator import validate_plan
                    plan = solve_exact(scenario)
                    report = validate_plan(scenario, plan) if plan.get("status") == "feasible" else None
                    if plan.get("status") == "infeasible" or (report and report["valid"]):
                        return {"status": plan["status"], "execution_type": "classical",
                                "plan": plan, "validation": report,
                                "fallback": {"used": True, "reason": "inference_failure"},
                                "vqc_error_code": "inference_failure"}
                except SolveBudgetExceeded:
                    return JSONResponse(status_code=503, content={"error": {
                        "code": SolveBudgetExceeded.code, "message": "dispatch search budget exceeded; fallback plan was not proven"}})
                except Exception:
                    pass
            return JSONResponse(status_code=500, content={"error": {
                "code": "inference_failure", "message": "VQC dispatch inference failed"}})
        finally:
            compute_slots.release()

    @app.post("/api/v1/recovery/plan")
    def recover_plan(payload: Any = Body(...)) -> Any:
        fields = {"original_scenario", "original_plan", "updated_scenario", "locked_prefixes", "change_penalty"}
        if (not isinstance(payload, dict) or not {"original_scenario", "original_plan", "updated_scenario", "locked_prefixes"} <= set(payload)
                or set(payload) - fields):
            return JSONResponse(status_code=422, content={"error": {
                "code": "invalid_request", "message": "recovery request fields are incomplete or unsupported"}})
        original, error = _invalid_scenario(payload["original_scenario"])
        if error:
            return error
        updated, error = _invalid_scenario(payload["updated_scenario"])
        if error:
            return error
        original_plan = _canonical_validated_plan(payload["original_plan"])
        if original_plan is None:
            return JSONResponse(status_code=422, content={"error": {
                "code": "invalid_original_plan", "message": "original plan has an invalid or unsupported shape"}})
        original_report = validate_plan(original, original_plan)
        if not original_report["valid"]:
            return JSONResponse(status_code=422, content={"error": {
                "code": "original_plan_unvalidated", "message": "original plan must pass independent validation",
                "details": _field_errors(original_report["errors"])}})
        prefixes = _validated_locked_prefixes(payload["locked_prefixes"], original, original_plan)
        if prefixes is None:
            return JSONResponse(status_code=422, content={"error": {
                "code": "invalid_locked_prefix", "message": "locked prefixes must be proven ordered prefixes of the original plan"}})
        penalty = payload.get("change_penalty", 0.0)
        if (isinstance(penalty, bool) or not isinstance(penalty, (int, float)) or
                not isfinite(penalty) or penalty < 0):
            return JSONResponse(status_code=422, content={"error": {
                "code": "invalid_change_penalty", "message": "change_penalty must be finite and non-negative"}})
        if not compute_slots.acquire(blocking=False):
            return capacity_error()
        recovery_started_at = datetime.now(timezone.utc).isoformat()
        try:
            from .recovery import RecoveryError, recover_dispatch
            result = recover_dispatch(original, original_plan, updated, prefixes,
                                      change_penalty=penalty)
            report = validate_plan(updated, result.get("recovered_plan"))
            if not report["valid"]:
                raise RecoveryError("validation_failure", "recovered plan failed independent validation")
            result["validation"] = report
            try:
                audit = recovery_audits().append(status="recovered", original_scenario=original,
                    updated_scenario=updated, original_plan=original_plan, locked_prefixes=prefixes,
                    change_penalty=penalty, result=result, started_at=recovery_started_at)
            except Exception:
                return JSONResponse(status_code=503, content={"error": {
                    "code": "audit_persistence_failed", "message": "recovery result was not published because audit storage failed"}})
            return {**result, "audit": {"audit_id": audit["audit_id"], "scope_sha256": audit["scope_sha256"],
                                         "started_at": audit["started_at"], "completed_at": audit["completed_at"], "hashes": audit["hashes"]}}
        except RecoveryError as exc:
            status = 409 if exc.code in {"recovery_infeasible", "locked_prefix_infeasible",
                                         "locked_work_vehicle_unavailable"} else 422
            try:
                audit = recovery_audits().append(status="failed", original_scenario=original,
                    updated_scenario=updated, original_plan=original_plan, locked_prefixes=prefixes,
                    change_penalty=penalty, failure={"code": exc.code, "message": str(exc)},
                    started_at=recovery_started_at)
            except Exception:
                return JSONResponse(status_code=503, content={"error": {
                    "code": "audit_persistence_failed", "message": "failed recovery could not be recorded"}})
            return JSONResponse(status_code=status, content={"error": {
                "code": exc.code, "message": str(exc),
                "audit": {"audit_id": audit["audit_id"], "scope_sha256": audit["scope_sha256"], "started_at": audit["started_at"], "completed_at": audit["completed_at"],
                          "hashes": audit["hashes"]}}})
        except Exception:
            try:
                audit = recovery_audits().append(status="failed", original_scenario=original,
                    updated_scenario=updated, original_plan=original_plan, locked_prefixes=prefixes,
                    change_penalty=penalty,
                    failure={"code": "recovery_failure", "message": "dispatch recovery failed"},
                    started_at=recovery_started_at)
            except Exception:
                return JSONResponse(status_code=503, content={"error": {
                    "code": "audit_persistence_failed", "message": "failed recovery could not be recorded"}})
            return JSONResponse(status_code=500, content={"error": {
                "code": "recovery_failure", "message": "dispatch recovery failed",
                "audit": {"audit_id": audit["audit_id"], "scope_sha256": audit["scope_sha256"], "started_at": audit["started_at"], "completed_at": audit["completed_at"],
                          "hashes": audit["hashes"]}}})
        finally:
            compute_slots.release()

    @app.get("/api/v1/recovery/audits/latest")
    def latest_recovery_audits(scope_sha256: str | None = None) -> Any:
        try:
            if scope_sha256 is not None and (len(scope_sha256) != 64 or any(c not in "0123456789abcdef" for c in scope_sha256)):
                return JSONResponse(status_code=422, content={"error": {"code": "invalid_scope", "message": "scope_sha256 must be a lowercase SHA-256 digest"}})
            return recovery_audits().latest(scope_sha256)
        except Exception:
            return JSONResponse(status_code=503, content={"error": {
                "code": "audit_unavailable", "message": "recovery audit store unavailable"}})

    @app.get("/api/v1/recovery/audits/{audit_id}")
    def recovery_audit(audit_id: str) -> Any:
        try:
            record = recovery_audits().get(audit_id)
        except Exception:
            return JSONResponse(status_code=503, content={"error": {
                "code": "audit_unavailable", "message": "recovery audit store unavailable"}})
        if record is None:
            return JSONResponse(status_code=404, content={"error": {
                "code": "audit_not_found", "message": "recovery audit record not found"}})
        return record

    @app.post("/api/v1/risk/evaluate")
    def evaluate_risk(payload: Any = Body(...)) -> Any:
        fields = {"forecast_scenario", "nominal_scenario", "plan", "realized_scenarios", "cvar_alpha"}
        required = {"forecast_scenario", "nominal_scenario", "plan", "realized_scenarios"}
        if not isinstance(payload, dict) or not required <= set(payload) or set(payload) - fields:
            return JSONResponse(status_code=422, content={"error": {
                "code": "invalid_request", "message": "risk request fields are incomplete or unsupported"}})
        forecast, error = _invalid_scenario(payload["forecast_scenario"])
        if error:
            return error
        nominal, error = _invalid_scenario(payload["nominal_scenario"])
        if error:
            return error
        plan_value = _canonical_validated_plan(payload["plan"])
        if plan_value is None:
            return JSONResponse(status_code=422, content={"error": {
                "code": "invalid_plan", "message": "plan has an invalid or unsupported shape"}})
        if not compute_slots.acquire(blocking=False):
            return capacity_error()
        try:
            from .risk import RiskError, evaluate_plan_risk
            return evaluate_plan_risk(forecast, nominal, plan_value,
                                      payload["realized_scenarios"],
                                      cvar_alpha=payload.get("cvar_alpha"))
        except RiskError as exc:
            status = 422 if exc.code != "validation_failure" else 500
            return JSONResponse(status_code=status, content=exc.as_dict())
        except Exception:
            return JSONResponse(status_code=500, content={"error": {
                "code": "risk_evaluation_failure", "message": "dispatch risk evaluation failed"}})
        finally:
            compute_slots.release()

    @app.post("/api/v1/plans/validate")
    def plan_validate(payload: Any = Body(...)) -> Any:
        if not isinstance(payload, dict):
            return JSONResponse(status_code=422, content={"error": {
                "code": "invalid_request", "message": "request must be an object"}})
        scenario, error = _invalid_scenario(payload.get("scenario"))
        if error:
            return error
        report = validate_plan(scenario, payload.get("plan"))
        return {"valid": report["valid"], "feasible": report["feasible"],
                "errors": _field_errors(report["errors"]), "recomputed": report["recomputed"]}

    @app.post("/api/v1/scenarios/submit", status_code=202)
    @app.post("/api/v1/jobs", status_code=202)
    def submit_job(payload: Any = Body(...)) -> Any:
        scenario, error = _invalid_scenario(payload)
        if error:
            return error
        try:
            job = jobs().submit(scenario)
        except JobCapacityError:
            return JSONResponse(status_code=429, content={"error": {
                "code": "capacity_reached", "message": "dispatch job capacity reached"}})
        except Exception:
            return JSONResponse(status_code=503, content={"error": {
                "code": "job_submission_failed", "message": "dispatch job could not be submitted"}})
        return job

    @app.post("/api/v1/jobs/qaoa", status_code=202)
    def submit_qaoa_job(payload: Any = Body(...)) -> Any:
        fields = {"scenario", "seed", "shots", "warm_start", "angle_grid"}
        if not isinstance(payload, dict) or set(payload) != fields:
            return JSONResponse(status_code=422, content={"error": {
                "code": "invalid_request", "message": "QAOA job requires scenario and explicit bounded settings"}})
        seed, shots, warm_start, angle_grid = (payload["seed"], payload["shots"],
                                                payload["warm_start"], payload["angle_grid"])
        if (isinstance(seed, bool) or not isinstance(seed, int) or not -(2**31) <= seed < 2**31 or
                isinstance(shots, bool) or not isinstance(shots, int) or not 1 <= shots <= 512 or
                not isinstance(warm_start, bool) or isinstance(angle_grid, bool) or
                not isinstance(angle_grid, int) or not 2 <= angle_grid <= 10):
            return JSONResponse(status_code=422, content={"error": {
                "code": "invalid_solver_settings", "message": "QAOA job settings exceed local bounds"}})
        scenario, error = _invalid_scenario(payload["scenario"])
        if error:
            return error
        if not compute_slots.acquire(blocking=False):
            return capacity_error()
        try:
            if len(generate_routes(scenario)) > 12:
                return JSONResponse(status_code=422, content={"error": {
                    "code": "unsupported_capability", "message": "local QAOA jobs support at most 12 route variables"}})
        except Exception:
            return JSONResponse(status_code=422, content={"error": {
                "code": "invalid_scenario", "message": "QAOA candidates could not be generated"}})
        finally:
            compute_slots.release()
        try:
            job = jobs().submit_qaoa(scenario, seed=seed, shots=shots,
                                     warm_start=warm_start, angle_grid=angle_grid)
        except JobCapacityError:
            return JSONResponse(status_code=429, content={"error": {
                "code": "capacity_reached", "message": "dispatch job capacity reached"}})
        except ValueError:
            return JSONResponse(status_code=422, content={"error": {
                "code": "invalid_solver_settings", "message": "QAOA job settings exceed local bounds"}})
        except Exception:
            return JSONResponse(status_code=503, content={"error": {
                "code": "job_submission_failed", "message": "QAOA job could not be submitted"}})
        return job

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str) -> Any:
        job = jobs().get(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": {
                "code": "job_not_found", "message": "dispatch job not found"}})
        return job

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> Any:
        job = jobs().cancel(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": {
                "code": "job_not_found", "message": "dispatch job not found"}})
        return job

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": {
            "code": "invalid_request", "message": "request body is invalid"}})

    from .india_eta.api import create_eta_router
    app.include_router(create_eta_router(app, auth, db_path=eta_db_path,
                                        registry=eta_registry, uncertainty_registry=eta_uncertainty_registry,
                                        controller=eta_controller,
                                        job_manager=eta_job_manager))
    return app


app = create_app()
