"""Persistent, approval-gated ETA acting workflow.

The controller never activates an optimized candidate automatically. A new
candidate is only a proposal until an authenticated dispatcher/admin approves
it under an optimistic version/context guard.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import re
import uuid
from typing import Any, Callable, Mapping

from backend.validator import validate_plan
from .errors import EtaError
from .optimization import evaluate_plan, optimize_dispatch
from .projection import project_remaining
from .store import EtaStore, dumps

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ACTOR_KEYS = {"subject", "role"}
_PLAN_KEYS = {"plan_id", "version", "model_version", "as_of", "scenario", "weighted_scenarios", "plan",
              "currency", "operating_rate", "switching_cost_threshold", "switching_cost", "cooldown_seconds", "expiry_seconds", "mode",
              "hysteresis_cost_threshold",
              "validation", "expected_cost", "eta_model_identity"}
_EVENT_KEYS = {"event_id", "plan_id", "plan_version", "model_version", "as_of", "observed_state",
               "scenario", "weighted_scenarios", "event_type", "currency", "operating_rate", "mode"}
_APPROVE_KEYS = {"base_version", "event_id", "model_version", "context_hash"}
_EVENT_TYPES = {"eta_update", "delay", "vehicle_status", "route_change"}


def _now_default() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise EtaError("invalid_request", f"{field} must be an ISO-8601 timestamp")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EtaError("invalid_request", f"{field} must be an ISO-8601 timestamp") from exc
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise EtaError("invalid_request", f"{field} must include a timezone")
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _hash(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def _fail(code: str, message: str, status: int = 422) -> None:
    raise EtaError(code, message, status)


def _object(value: Any, fields: set[str], required: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - fields or required - set(value):
        _fail("invalid_request", f"{label} schema is invalid")
    return value


def _actor(value: Any) -> tuple[str, str]:
    if isinstance(value, str):
        # A role-only value is allowed for trusted server adapters; payloads
        # have no corresponding field and cannot self-assign this value.
        if value in {"dispatcher", "admin"}:
            return value, value
        _fail("forbidden", "Dispatcher or admin authorization is required", 403)
    if not isinstance(value, Mapping) or set(value) != _ACTOR_KEYS:
        _fail("forbidden", "Dispatcher or admin authorization is required", 403)
    subject, role = value.get("subject"), value.get("role")
    if not isinstance(subject, str) or not subject.strip() or len(subject) > 160 or role not in {"dispatcher", "admin"}:
        _fail("forbidden", "Dispatcher or admin authorization is required", 403)
    return subject, role


def _finite(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        _fail("invalid_request", f"{label} must be finite and at least {minimum}")
    return float(value)


class EtaController:
    def __init__(self, db_path: str, *, clock: Callable[[], datetime] | None = None,
                 optimizer: Callable[..., dict[str, Any]] | None = None,
                 evaluator: Callable[..., dict[str, Any]] | None = None,
                 projector: Callable[..., dict[str, Any]] | None = None) -> None:
        self.store = EtaStore(db_path)
        self._clock = clock or _now_default
        self._optimizer = optimizer or optimize_dispatch
        self._evaluator = evaluator or evaluate_plan
        self._projector = projector or project_remaining

    def _now(self) -> str:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise RuntimeError("controller clock must return an aware datetime")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def register_plan(self, payload: dict[str, Any], actor: Any) -> dict[str, Any]:
        subject, role = _actor(actor)
        data = _object(payload, _PLAN_KEYS, {"plan_id", "version", "model_version", "as_of", "scenario",
                   "weighted_scenarios", "plan"}, "plan")
        plan_id = _valid_id(data["plan_id"], "plan_id")
        model_version = _valid_id(data["model_version"], "model_version")
        eta_model_identity = _model_identity(data.get("eta_model_identity"), model_version)
        if isinstance(data["version"], bool) or data["version"] != 1:
            _fail("invalid_plan", "Initial plan version must be 1")
        as_of = _iso(data["as_of"], "as_of")
        if data.get("mode", "synthetic_scenario") != "synthetic_scenario":
            _fail("unsupported_capability", "Live activation requires a verified live-state adapter", 501)
        scenario = deepcopy(data["scenario"])
        weighted = deepcopy(data["weighted_scenarios"])
        plan = deepcopy(data["plan"])
        validation = validate_plan(scenario, plan)
        if not validation.get("valid") or not validation.get("feasible"):
            _fail("validation_failure", "Initial plan failed independent validation")
        currency = data.get("currency", "XTS")
        rate = _finite(data.get("operating_rate", 1.0), "operating_rate")
        cost_check = self._evaluator(scenario, plan, weighted, currency=currency, operating_rate=rate)
        if not cost_check.get("valid") or cost_check.get("expected_cost") is None:
            _fail("validation_failure", "Initial plan failed independent scenario cost validation")
        if "validation" in data and _hash(data["validation"]) != _hash(validation):
            _fail("validation_failure", "Supplied validation does not match independent validation")
        if "expected_cost" in data and _hash(data["expected_cost"]) != _hash(cost_check["expected_cost"]):
            _fail("validation_failure", "Supplied expected cost does not match independent recomputation")
        cost = cost_check["expected_cost"]
        min_benefit = _finite(data.get("switching_cost_threshold", 0), "switching_cost_threshold")
        switching_cost = _finite(data.get("switching_cost", 0), "switching_cost")
        hysteresis = _finite(data.get("hysteresis_cost_threshold", 5), "hysteresis_cost_threshold")
        cooldown = _finite(data.get("cooldown_seconds", 120), "cooldown_seconds")
        expiry = _finite(data.get("expiry_seconds", 900), "expiry_seconds", minimum=1)
        with self.store.transaction() as conn:
            exists = conn.execute("SELECT 1 FROM eta_plans WHERE plan_id=?", (plan_id,)).fetchone()
            if exists:
                _fail("conflict", "Plan identifier is already registered", 409)
            vehicle_ids = [r.get("vehicle_id") for r in plan.get("routes", []) if isinstance(r, dict)]
            if len(set(vehicle_ids)) != len(vehicle_ids):
                _fail("invalid_plan", "Initial plan assigns a vehicle more than once")
            active_routes = conn.execute("SELECT plan_id,plan_json FROM eta_plans WHERE active=1").fetchall()
            active_vehicles = {route.get("vehicle_id") for active_row in active_routes
                               for route in json.loads(active_row["plan_json"]).get("routes", [])}
            for vehicle_id in vehicle_ids:
                if vehicle_id in active_vehicles:
                    _fail("vehicle_already_active", "Vehicle already has an active ETA manifest", 409)
            conn.execute("""INSERT INTO eta_plans
              (plan_id,version,model_version,as_of,scenario_json,weighted_json,plan_json,cost_json,currency,
               operating_rate,min_benefit,cooldown_seconds,expiry_seconds,switching_cost,hysteresis_cost_threshold,
               eta_model_json,mode,active)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
              (plan_id, 1, model_version, as_of, dumps(scenario), dumps(weighted), dumps(plan), dumps(cost),
               currency, rate, min_benefit, cooldown, expiry, switching_cost, hysteresis,
               dumps(eta_model_identity) if eta_model_identity is not None else None, "synthetic_scenario"))
            _write_manifests(conn, plan_id, 1, plan, validation, {}, self._now())
            _audit(conn, plan_id, "plan_registered", subject, plan_id, self._now(),
                   {"role": role, "model_version": model_version, "mode": "synthetic_scenario"})
        return self.get_active(plan_id)

    def process_event(self, payload: dict[str, Any], *, cancel_check: Callable[[], bool] | None = None) -> dict[str, Any]:
        data = _object(payload, _EVENT_KEYS, {"event_id", "plan_id", "plan_version", "model_version", "as_of",
                   "observed_state", "scenario", "weighted_scenarios", "event_type"}, "event")
        event_id, plan_id = _valid_id(data["event_id"], "event_id"), _valid_id(data["plan_id"], "plan_id")
        if isinstance(data["plan_version"], bool) or not isinstance(data["plan_version"], int) or data["plan_version"] < 1:
            _fail("invalid_request", "plan_version must be a positive integer")
        model_version = _valid_id(data["model_version"], "model_version")
        as_of = _iso(data["as_of"], "as_of")
        if data["event_type"] not in _EVENT_TYPES:
            _fail("invalid_request", "event_type is unsupported")
        observed_state = _validate_observed_state(data["observed_state"])
        if any(row["position_type"] != "simulated" for row in observed_state):
            _fail("unsupported_capability", "Live event processing requires a verified live-state adapter", 501)
        if data.get("mode", "synthetic_scenario") != "synthetic_scenario":
            _fail("unsupported_capability", "Live event processing requires a verified live-state adapter", 501)
        scenario, weighted = deepcopy(data["scenario"]), deepcopy(data["weighted_scenarios"])
        event_body = deepcopy(data)
        body_hash = _hash(event_body)
        context_hash = _hash({"event": event_body, "observed_state": observed_state})
        now = self._now()

        # Reserve sequence/id before expensive optimization. A later event or
        # cancellation invalidates this worker's publication guard.
        with self.store.transaction() as conn:
            prior = conn.execute("SELECT * FROM eta_events WHERE event_id=?", (event_id,)).fetchone()
            if prior:
                if prior["body_hash"] != body_hash:
                    _fail("idempotency_conflict", "Event identifier was already used with different content", 409)
                if prior["status"] == "processing":
                    return {"event_id": event_id, "plan_id": plan_id, "status": "processing"}
                return json.loads(prior["result_json"])
            active = conn.execute("SELECT * FROM eta_plans WHERE plan_id=? AND active=1", (plan_id,)).fetchone()
            if active is None:
                _fail("not_found", "Active plan was not found", 404)
            if int(active["version"]) != data["plan_version"] or active["model_version"] != model_version:
                _fail("stale_context", "Event plan or model version is stale", 409)
            previous_as_of = active["latest_event_at"] or active["as_of"]
            if _dt(as_of) <= _dt(previous_as_of):
                _fail("out_of_order_event", "Event timestamp must be newer than the active context", 409)
            latest_state = conn.execute("SELECT observed_state_json FROM eta_events WHERE plan_id=? ORDER BY as_of DESC LIMIT 1",
                                        (plan_id,)).fetchone()
            if latest_state:
                _assert_prefix_monotone(json.loads(latest_state["observed_state_json"]), observed_state)
            conn.execute("UPDATE eta_plans SET latest_event_at=?,latest_event_hash=? WHERE plan_id=? AND version=? AND active=1",
                         (as_of, context_hash, plan_id, data["plan_version"]))
            conn.execute("INSERT INTO eta_events(event_id,plan_id,body_hash,as_of,status,result_json,created_at,observed_state_json) VALUES(?,?,?,?,?,?,?,?)",
                         (event_id, plan_id, body_hash, as_of, "processing", "{}", now, dumps(observed_state)))
            conn.execute("UPDATE eta_proposals SET status='superseded' WHERE plan_id=? AND status='pending'", (plan_id,))
            plan_snapshot = json.loads(active["plan_json"])
            old_scenario = json.loads(active["scenario_json"])
            old_weighted = json.loads(active["weighted_json"])
            current_version = int(active["version"])
            min_benefit = float(active["min_benefit"])
            cooldown = float(active["cooldown_seconds"])
            expires_in = float(active["expiry_seconds"])
            switching_cost = float(active["switching_cost"])
            hysteresis = float(active["hysteresis_cost_threshold"])
            currency = data.get("currency", active["currency"])
            rate = _finite(data.get("operating_rate", active["operating_rate"]), "operating_rate")
            last_proposal_at = active["last_proposal_at"]

        try:
            result = self._evaluate_event(plan_id, event_id, model_version, context_hash, as_of,
                    scenario, weighted, plan_snapshot, old_scenario, old_weighted, observed_state,
                    currency, rate, min_benefit, cooldown, expires_in, last_proposal_at, current_version, switching_cost, hysteresis)
            if cancel_check is not None and cancel_check():
                raise EtaError("cancelled", "ETA event was cancelled", 409)
            with self.store.transaction() as conn:
                guard = conn.execute("SELECT latest_event_hash,version,active FROM eta_plans WHERE plan_id=? AND version=?",
                                     (plan_id, current_version)).fetchone()
                event = conn.execute("SELECT status FROM eta_events WHERE event_id=?", (event_id,)).fetchone()
                if cancel_check is not None and cancel_check():
                    raise EtaError("cancelled", "ETA event was cancelled", 409)
                if not guard or not guard["active"] or guard["latest_event_hash"] != context_hash or not event or event["status"] == "cancelled":
                    raise EtaError("stale_context", "Event context changed before proposal publication", 409)
                proposal = result.pop("_proposal", None)
                if proposal:
                    conn.execute("INSERT INTO eta_proposals(proposal_id,event_id,plan_id,base_version,model_version,context_hash,expires_at,status,proposal_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                                 (proposal["proposal_id"], event_id, plan_id, current_version, model_version,
                                  context_hash, proposal["expires_at"], "pending", dumps(proposal), now))
                    conn.execute("UPDATE eta_plans SET last_proposal_at=? WHERE plan_id=? AND version=?",
                                 (now, plan_id, current_version))
                    _audit(conn, plan_id, "proposal_created", "system", proposal["proposal_id"], now,
                           {"event_id": event_id, "net_benefit": proposal["net_benefit"]})
                    result["proposal_id"] = proposal["proposal_id"]
                    result["proposal"] = proposal
                conn.execute("UPDATE eta_events SET status='completed',result_json=? WHERE event_id=?",
                             (dumps(result), event_id))
            return result
        except EtaError as exc:
            with self.store.transaction() as conn:
                current = conn.execute("SELECT status FROM eta_events WHERE event_id=?", (event_id,)).fetchone()
                if current and current["status"] == "processing":
                    conn.execute("UPDATE eta_events SET status=?,result_json=? WHERE event_id=?",
                                 ("cancelled" if exc.code == "cancelled" else "failed",
                                  dumps({"event_id": event_id, "status": "cancelled" if exc.code == "cancelled" else "failed",
                                         "error": {"code": exc.code, "message": exc.message}}), event_id))
            raise
        except Exception as exc:
            with self.store.transaction() as conn:
                conn.execute("UPDATE eta_events SET status='failed',result_json=? WHERE event_id=? AND status='processing'",
                             (dumps({"event_id": event_id, "status": "failed", "error": {"code": "optimization_failed", "message": "Bounded dispatch optimization failed"}}), event_id))
            raise EtaError("optimization_failed", "Bounded dispatch optimization failed", 500) from exc

    def _evaluate_event(self, plan_id: str, event_id: str, model_version: str, context_hash: str, as_of: str,
                        scenario: dict[str, Any], weighted: list[dict[str, Any]], active_plan: dict[str, Any],
                        active_scenario: dict[str, Any], active_weighted: list[dict[str, Any]], observed_state: list[dict[str, Any]],
                        currency: str, rate: float, min_benefit: float, cooldown: float, expires_in: float,
                        last_proposal_at: str | None, current_version: int, switching_cost: float,
                        hysteresis: float) -> dict[str, Any]:
        current = self._projector(scenario, active_plan, observed_state, weighted,
                                  currency=currency, operating_rate=rate)
        if current.get("status") != "projected":
            return {"event_id": event_id, "plan_id": plan_id, "status": "ALERT", "reason": "remaining_state_invalid",
                    "current_projection": current, "context_hash": context_hash}
        keep_eval = current.get("evaluation")
        if keep_eval is None:
            _fail("validation_failure", "Current remaining plan could not be independently evaluated")
        keep_cost = _cost(keep_eval)
        prior_projection = self._projector(active_scenario, active_plan, observed_state, active_weighted,
                                           currency=currency, operating_rate=rate)
        prior_cost = _cost(prior_projection.get("evaluation")) if prior_projection.get("status") == "projected" else None
        material_delta = (keep_cost - prior_cost) if prior_cost is not None else 0.0
        if abs(material_delta) <= min_benefit:
            risk = _projection_has_material_risk(current)
            decision = "ALERT" if risk else "KEEP"
            return {"event_id": event_id, "plan_id": plan_id, "status": decision,
                    "reason": "material_risk_without_material_cost_change" if decision == "ALERT" else "no_material_change",
                    "expected_cost_before": prior_cost, "expected_cost_after": keep_cost,
                    "material_cost_delta": material_delta, "current_projection": current, "context_hash": context_hash}
        optimum = self._optimizer(scenario, weighted, currency=currency, operating_rate=rate,
                                  locked_prefixes=_prefix_map(observed_state), vehicle_states=observed_state, quantum=True)
        candidate = optimum.get("plan")
        validation = optimum.get("validation")
        if not isinstance(validation, dict) or not validation.get("valid") or not validation.get("feasible"):
            return {"event_id": event_id, "plan_id": plan_id, "status": "ALERT", "reason": "candidate_validation_failed",
                    "candidate": _safe_optimum(optimum), "current_projection": current, "context_hash": context_hash}
        if current.get("projection_mode") != "synthetic_scenario":
            _fail("unsupported_capability", "Only declared simulated-state activation is currently supported", 501)
        projected_candidate = self._projector(scenario, candidate, observed_state, weighted,
                                              currency=currency, operating_rate=rate)
        candidate_cost = _cost(projected_candidate.get("evaluation")) if projected_candidate.get("valid", True) else None
        if candidate_cost is None:
            return {"event_id": event_id, "plan_id": plan_id, "status": "ALERT", "reason": "candidate_remaining_state_invalid",
                    "candidate": _safe_optimum(optimum), "current_projection": current, "context_hash": context_hash}
        improvement = keep_cost - candidate_cost
        unchanged = _same_remaining_routes(current.get("plan", active_plan), projected_candidate.get("plan", candidate), observed_state)
        if unchanged:
            decision = "ALERT" if _projection_has_material_risk(current) else "KEEP"
            return {"event_id": event_id, "plan_id": plan_id, "status": decision,
                    "reason": "material_risk_without_beneficial_replan" if decision == "ALERT" else "no_material_route_change",
                    "expected_cost_before": keep_cost, "expected_cost_after": candidate_cost, "net_benefit": improvement,
                    "current_projection": current, "candidate": _safe_optimum(optimum), "context_hash": context_hash}
        net_benefit = improvement - switching_cost
        required_benefit = _required_benefit(self.store, plan_id, current_version, active_plan,
                                             candidate, min_benefit, hysteresis)
        if net_benefit <= required_benefit:
            reverse = required_benefit > min_benefit
            return {"event_id": event_id, "plan_id": plan_id, "status": "ALERT",
                    "reason": "reverse_plan_hysteresis_active" if reverse else "net_benefit_below_switching_threshold",
                    "expected_cost_before": keep_cost, "expected_cost_after": candidate_cost,
                    "switching_cost": switching_cost, "net_benefit": net_benefit,
                    "required_net_benefit": required_benefit,
                    "current_projection": current, "candidate": _safe_optimum(optimum), "context_hash": context_hash}
        if last_proposal_at and (_dt(self._now()) - _dt(last_proposal_at)).total_seconds() < cooldown:
            return {"event_id": event_id, "plan_id": plan_id, "status": "ALERT", "reason": "proposal_cooldown_active",
                    "expected_cost_before": keep_cost, "expected_cost_after": candidate_cost,
                    "switching_cost": switching_cost, "net_benefit": net_benefit,
                    "required_net_benefit": required_benefit,
                    "current_projection": current, "candidate": _safe_optimum(optimum), "context_hash": context_hash}
        proposal_id = "eta-prop-" + uuid.uuid4().hex
        expires_at = (self._clock().astimezone(timezone.utc) + timedelta(seconds=expires_in)).isoformat().replace("+00:00", "Z")
        proposal = {"proposal_id": proposal_id, "event_id": event_id, "plan_id": plan_id,
                    "base_version": current_version, "model_version": model_version, "context_hash": context_hash,
                    "created_at": self._now(), "expires_at": expires_at,
                    "plan": candidate, "projected_plan": projected_candidate.get("plan"),
                    "validation": validation, "projected_validation": projected_candidate.get("validation"),
                    "expected_cost_before": keep_cost, "expected_cost_after": candidate_cost,
                    "switching_cost": switching_cost, "net_benefit": net_benefit,
                    "currency": currency, "mode": "synthetic_scenario",
                    "scenario": scenario, "weighted_scenarios": weighted,
                    "optimizer": _safe_optimum(optimum)}
        return {"event_id": event_id, "plan_id": plan_id, "status": "PROPOSE_REPLAN",
                "reason": "validated_net_benefit_exceeds_switching_threshold", "context_hash": context_hash,
                "expected_cost_before": keep_cost, "expected_cost_after": candidate_cost,
                "switching_cost": switching_cost, "net_benefit": net_benefit,
                "_proposal": proposal}

    def approve_proposal(self, proposal_id: str, payload: dict[str, Any], actor: Any) -> dict[str, Any]:
        subject, role = _actor(actor)
        proposal_id = _valid_id(proposal_id, "proposal_id")
        data = _object(payload, _APPROVE_KEYS, _APPROVE_KEYS, "approval")
        if isinstance(data["base_version"], bool) or not isinstance(data["base_version"], int) or data["base_version"] < 1:
            _fail("invalid_request", "base_version must be a positive integer")
        event_id = _valid_id(data["event_id"], "event_id")
        model_version = _valid_id(data["model_version"], "model_version")
        context_hash = data["context_hash"]
        if not isinstance(context_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", context_hash):
            _fail("invalid_request", "context_hash is invalid")
        now = self._now()
        found = self.store.read("SELECT status,expires_at FROM eta_proposals WHERE proposal_id=?", (proposal_id,))
        if found and found[0]["status"] == "pending" and _dt(found[0]["expires_at"]) <= _dt(now):
            with self.store.transaction() as conn:
                conn.execute("UPDATE eta_proposals SET status='expired' WHERE proposal_id=? AND status='pending'", (proposal_id,))
            _fail("proposal_expired", "Proposal has expired", 409)
        with self.store.transaction() as conn:
            row = conn.execute("SELECT * FROM eta_proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
            if not row:
                _fail("not_found", "Proposal was not found", 404)
            if row["status"] != "pending":
                _fail("proposal_not_pending", "Proposal is not pending approval", 409)
            if _dt(row["expires_at"]) <= _dt(now):
                conn.execute("UPDATE eta_proposals SET status='expired' WHERE proposal_id=?", (proposal_id,))
                _fail("proposal_expired", "Proposal has expired", 409)
            if (int(row["base_version"]) != data["base_version"] or row["event_id"] != event_id or
                    row["model_version"] != model_version or row["context_hash"] != context_hash):
                _fail("stale_context", "Approval context does not match the proposal", 409)
            planrow = conn.execute("SELECT * FROM eta_plans WHERE plan_id=? AND version=? AND active=1",
                                   (row["plan_id"], row["base_version"])).fetchone()
            if not planrow or planrow["model_version"] != model_version or planrow["latest_event_hash"] != context_hash:
                _fail("stale_context", "Active plan or event context has changed", 409)
            ev = conn.execute("SELECT status,result_json,observed_state_json,as_of FROM eta_events WHERE event_id=?", (event_id,)).fetchone()
            if not ev or ev["status"] != "completed":
                _fail("event_cancelled", "Source event is not active", 409)
            async_table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='eta_async_jobs'").fetchone()
            if async_table:
                job = conn.execute("SELECT status FROM eta_async_jobs WHERE event_id=?", (event_id,)).fetchone()
                if job and job["status"] != "succeeded":
                    _fail("event_job_not_succeeded", "ETA event job is not in the succeeded state", 409)
            event_result = json.loads(ev["result_json"])
            if event_result.get("context_hash") != context_hash:
                _fail("stale_context", "Event context hash no longer matches", 409)
            proposal = json.loads(row["proposal_json"])
            candidate = proposal.get("plan")
            scenario = proposal.get("scenario", json.loads(planrow["scenario_json"]))
            weighted = proposal.get("weighted_scenarios", json.loads(planrow["weighted_json"]))
            observed_state = json.loads(ev["observed_state_json"])
            validation = validate_plan(scenario, candidate)
            if not validation.get("valid") or not validation.get("feasible"):
                _fail("validation_failure", "Proposal failed independent revalidation")
            active_plan = json.loads(planrow["plan_json"])
            projection_kwargs = {"currency": planrow["currency"], "operating_rate": float(planrow["operating_rate"])}
            try:
                keep_projection = self._projector(scenario, active_plan, observed_state, weighted, **projection_kwargs)
                candidate_projection = self._projector(scenario, candidate, observed_state, weighted, **projection_kwargs)
            except (TypeError, ValueError, KeyError):
                # State/route mismatches must become a stable rejection, never
                # leak a projector exception or permit partially applied state.
                _fail("validation_failure", "Remaining-state projection failed independent revalidation")
            if keep_projection.get("status") != "projected" or candidate_projection.get("status") != "projected":
                _fail("validation_failure", "Remaining-state projection failed independent revalidation")
            if candidate_projection.get("evaluation", {}).get("hard_window_violation_frequency", 1.0) > 0:
                _fail("validation_failure", "Candidate violates a hard window in the frozen scenarios")
            keep_cost = _cost(keep_projection.get("evaluation"))
            candidate_cost = _cost(candidate_projection.get("evaluation"))
            if keep_cost is None or candidate_cost is None:
                _fail("validation_failure", "Remaining-state cost failed independent revalidation")
            net_benefit = keep_cost - candidate_cost - float(planrow["switching_cost"])
            required_benefit = _required_benefit_conn(conn, row["plan_id"], int(row["base_version"]), active_plan,
                                                      candidate, float(planrow["min_benefit"]),
                                                      float(planrow["hysteresis_cost_threshold"]))
            if net_benefit <= required_benefit:
                _fail("validation_failure", "Recomputed net benefit no longer meets the switching/hysteresis threshold")
            candidate_vehicles = {r.get("vehicle_id") for r in candidate.get("routes", []) if r.get("vehicle_id")}
            other_active = conn.execute("SELECT plan_id,plan_json FROM eta_plans WHERE active=1 AND plan_id<>?",
                                        (row["plan_id"],)).fetchall()
            other_vehicles = {route.get("vehicle_id") for active_row in other_active
                              for route in json.loads(active_row["plan_json"]).get("routes", [])}
            if candidate_vehicles & other_vehicles:
                _fail("vehicle_already_active", "Candidate vehicle assignment conflicts with another active plan", 409)
            # Preserve old manifests, atomically advance the active version,
            # and write every new vehicle manifest in the same transaction.
            new_version = int(row["base_version"]) + 1
            conn.execute("UPDATE eta_plans SET active=0 WHERE plan_id=? AND version=? AND active=1",
                         (row["plan_id"], row["base_version"]))
            conn.execute("""INSERT INTO eta_plans
              (plan_id,version,model_version,as_of,scenario_json,weighted_json,plan_json,cost_json,currency,
               operating_rate,min_benefit,cooldown_seconds,expiry_seconds,switching_cost,hysteresis_cost_threshold,
               eta_model_json,mode,active,latest_event_at,latest_event_hash,last_proposal_at)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              (row["plan_id"], new_version, model_version, ev["as_of"], dumps(scenario), dumps(weighted),
               dumps(candidate), dumps(candidate_projection["evaluation"]["expected_cost"]), planrow["currency"], planrow["operating_rate"],
               planrow["min_benefit"], planrow["cooldown_seconds"], planrow["expiry_seconds"], planrow["switching_cost"],
               planrow["hysteresis_cost_threshold"],
               planrow["eta_model_json"], "synthetic_scenario", 1, planrow["latest_event_at"], planrow["latest_event_hash"], now))
            conn.execute("UPDATE eta_proposals SET status='approved',decided_at=?,decided_by=? WHERE proposal_id=? AND status='pending'",
                         (now, subject, proposal_id))
            _write_manifests(conn, row["plan_id"], new_version, candidate, validation,
                             {"proposal_id": proposal_id, "event_id": event_id, "approved_by": subject,
                              "role": role, "approved_at": now, "net_benefit": net_benefit}, now,
                             candidate_projection.get("plan"))
            new_vehicles = {r.get("vehicle_id") for r in candidate.get("routes", []) if r.get("vehicle_id")}
            old_vehicles = {r.get("vehicle_id") for r in active_plan.get("routes", []) if r.get("vehicle_id")}
            for vehicle_id in sorted(old_vehicles - new_vehicles):
                manifest_version = _next_manifest_version(conn, vehicle_id)
                revoked = {"vehicle_id": vehicle_id, "plan_id": row["plan_id"], "plan_version": new_version,
                           "manifest_version": manifest_version,
                           "plan_hash": _hash(candidate), "ordered_stops": [], "completed_stop_references": [],
                           "approval": {"proposal_id": proposal_id, "event_id": event_id, "approved_by": subject,
                                        "role": role, "approved_at": now}, "mode": "synthetic_scenario",
                           "status": "revoked", "created_at": now}
                conn.execute("INSERT INTO eta_manifests(vehicle_id,version,plan_id,manifest_json,created_at) VALUES(?,?,?,?,?)",
                             (vehicle_id, manifest_version, row["plan_id"], dumps(revoked), now))
            _audit(conn, row["plan_id"], "proposal_approved", subject, proposal_id, now,
                   {"role": role, "event_id": event_id, "version": new_version})
            return {"plan_id": row["plan_id"], "version": new_version, "proposal_id": proposal_id,
                    "status": "approved", "plan": candidate, "validation": validation,
                    "expected_cost": candidate_projection["evaluation"]["expected_cost"],
                    "net_benefit": net_benefit, "manifests_written": len(new_vehicles | old_vehicles),
                    "mode": "synthetic_scenario"}

    def get_proposal(self, proposal_id: str) -> dict[str, Any]:
        rows = self.store.read("SELECT * FROM eta_proposals WHERE proposal_id=?", (_valid_id(proposal_id, "proposal_id"),))
        if not rows:
            _fail("not_found", "Proposal was not found", 404)
        row = rows[0]
        return {"proposal_id": row["proposal_id"], "event_id": row["event_id"], "plan_id": row["plan_id"],
                "base_version": row["base_version"], "status": row["status"], "expires_at": row["expires_at"],
                "proposal": json.loads(row["proposal_json"])}

    def get_manifest(self, vehicle_id: str, version: int | None = None) -> dict[str, Any]:
        vehicle_id = _valid_id(vehicle_id, "vehicle_id")
        if version is None:
            rows = self.store.read("SELECT * FROM eta_manifests WHERE vehicle_id=? ORDER BY version DESC,created_at DESC LIMIT 1", (vehicle_id,))
        else:
            if isinstance(version, bool) or not isinstance(version, int) or version < 1:
                _fail("invalid_request", "manifest version must be a positive integer")
            rows = self.store.read("SELECT * FROM eta_manifests WHERE vehicle_id=? AND version=?", (vehicle_id, version))
        if not rows:
            _fail("not_found", "Manifest was not found", 404)
        return json.loads(rows[0]["manifest_json"])

    def get_active(self, plan_id: str) -> dict[str, Any]:
        plan_id = _valid_id(plan_id, "plan_id")
        rows = self.store.read("SELECT * FROM eta_plans WHERE plan_id=? AND active=1", (plan_id,))
        if not rows:
            _fail("not_found", "Active plan was not found", 404)
        row = rows[0]
        proposals = self.store.read("SELECT proposal_id,status FROM eta_proposals WHERE plan_id=? ORDER BY created_at DESC LIMIT 10", (plan_id,))
        return {"plan_id": row["plan_id"], "version": row["version"], "model_version": row["model_version"],
                "eta_model_identity": json.loads(row["eta_model_json"]) if row["eta_model_json"] else None,
                "as_of": row["as_of"], "scenario": json.loads(row["scenario_json"]),
                "weighted_scenarios": json.loads(row["weighted_json"]), "plan": json.loads(row["plan_json"]),
                "expected_cost": json.loads(row["cost_json"]), "currency": row["currency"],
                "mode": row["mode"], "latest_event_at": row["latest_event_at"],
                "pending_proposals": [{"proposal_id": p["proposal_id"], "status": p["status"]} for p in proposals]}

    def cancel_event(self, event_id: str) -> dict[str, Any]:
        event_id = _valid_id(event_id, "event_id")
        now = self._now()
        with self.store.transaction() as conn:
            row = conn.execute("SELECT * FROM eta_events WHERE event_id=?", (event_id,)).fetchone()
            if not row:
                _fail("not_found", "Event was not found", 404)
            if row["status"] == "completed":
                result = json.loads(row["result_json"])
                proposal_id = result.get("proposal_id")
                if proposal_id:
                    conn.execute("UPDATE eta_proposals SET status='cancelled' WHERE proposal_id=? AND status='pending'", (proposal_id,))
                elif result.get("status") != "KEEP":
                    _fail("event_not_cancellable", "Completed event has no pending proposal", 409)
            elif row["status"] not in {"processing", "cancelled"}:
                _fail("event_not_cancellable", "Event is not cancellable", 409)
            conn.execute("UPDATE eta_events SET status='cancelled',result_json=? WHERE event_id=?",
                         (dumps({"event_id": event_id, "status": "cancelled"}), event_id))
            async_table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='eta_async_jobs'").fetchone()
            if async_table:
                conn.execute("UPDATE eta_async_jobs SET status='cancelled' WHERE event_id=? AND status IN ('queued','running','succeeded')",
                             (event_id,))
            _audit(conn, row["plan_id"], "event_cancelled", "system", event_id, now, {})
            return {"event_id": event_id, "status": "cancelled"}


def _valid_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        _fail("invalid_request", f"{field} is invalid")
    return value


def _model_identity(value: Any, model_version: str) -> dict[str, str] | None:
    if value is None:
        return None
    identity = _object(value, {"model_id", "model_version", "model_sha256"},
                        {"model_id", "model_version", "model_sha256"}, "eta_model_identity")
    model_id = _valid_id(identity["model_id"], "eta_model_identity.model_id")
    identity_version = _valid_id(identity["model_version"], "eta_model_identity.model_version")
    digest = identity["model_sha256"]
    if identity_version != model_version:
        _fail("invalid_request", "eta_model_identity.model_version must match model_version")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        _fail("invalid_request", "eta_model_identity.model_sha256 must be 64 lowercase hexadecimal characters")
    return {"model_id": model_id, "model_version": identity_version, "model_sha256": digest}


def _validate_observed_state(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 16:
        _fail("invalid_request", "observed_state must contain 1 to 16 vehicle rows")
    seen: set[str] = set()
    normalized = []
    fields = {"vehicle_id", "position", "position_type", "elapsed_minutes", "available_at", "completed_prefix"}
    for row in value:
        _object(row, fields, fields, "observed_state row")
        vid = _valid_id(row["vehicle_id"], "vehicle_id")
        position = _valid_id(row["position"], "position")
        if vid in seen or row["position_type"] not in {"actual", "simulated"}:
            _fail("invalid_request", "observed_state vehicle IDs must be unique and position_type supported")
        seen.add(vid)
        elapsed = _finite(row["elapsed_minutes"], "elapsed_minutes")
        available = _finite(row["available_at"], "available_at")
        prefix = row["completed_prefix"]
        if not isinstance(prefix, list):
            _fail("invalid_request", "completed_prefix must be a list")
        norm_prefix = []
        previous = -math.inf
        for item in prefix:
            _object(item, {"order_id", "completed_at"}, {"order_id", "completed_at"}, "completed stop")
            oid = _valid_id(item["order_id"], "order_id")
            when = _finite(item["completed_at"], "completed_at")
            if when < previous or when > max(elapsed, available):
                _fail("invalid_request", "completed stop timestamps must be ordered and not in the future")
            previous = when
            norm_prefix.append({"order_id": oid, "completed_at": when})
        normalized.append({"vehicle_id": vid, "position": position, "position_type": row["position_type"],
                           "elapsed_minutes": elapsed, "available_at": available,
                           "completed_prefix": norm_prefix})
    return normalized


def _assert_prefix_monotone(previous: list[dict[str, Any]], current: list[dict[str, Any]]) -> None:
    current_by_vehicle = {row["vehicle_id"]: row for row in current}
    for prior in previous:
        old_rows = prior["completed_prefix"]
        new_row = current_by_vehicle.get(prior["vehicle_id"])
        new_rows = new_row["completed_prefix"] if new_row else []
        if [x["order_id"] for x in new_rows[:len(old_rows)]] != [x["order_id"] for x in old_rows]:
            _fail("state_regression", "A newer event cannot remove or reassign a completed stop", 409)
        for old, new in zip(old_rows, new_rows):
            if not math.isclose(old["completed_at"], new["completed_at"], rel_tol=0.0, abs_tol=1e-9):
                _fail("state_regression", "A completed stop timestamp is immutable", 409)
        if new_row and (new_row["elapsed_minutes"] < prior["elapsed_minutes"] or
                        new_row["available_at"] < prior["available_at"]):
            _fail("state_regression", "Vehicle position clocks cannot move backwards", 409)


def _cost(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    if value.get("valid") is False:
        return None
    expected = value.get("expected_cost")
    if isinstance(expected, dict):
        amount = expected.get("amount", expected.get("total"))
    else:
        amount = value.get("expected_remaining_cost", {}).get("total_cost_proxy")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount):
        return None
    return float(amount)


def _projection_has_material_risk(projection: dict[str, Any]) -> bool:
    costs = projection.get("expected_remaining_cost")
    if isinstance(costs, dict):
        for field in ("late_penalty_cost", "hard_window_violation_frequency"):
            value = costs.get(field, 0)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
                return True
    evaluation = projection.get("evaluation")
    if isinstance(evaluation, dict):
        value = evaluation.get("hard_window_violation_frequency", 0)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
            return True
    return False


def _prefix_map(state: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {row["vehicle_id"]: [stop["order_id"] for stop in row["completed_prefix"]] for row in state}


def _same_remaining_routes(left: dict[str, Any], right: dict[str, Any], state: list[dict[str, Any]]) -> bool:
    locked = {k: v for k, v in _prefix_map(state).items()}
    def suffixes(plan: dict[str, Any]) -> dict[str, list[str]]:
        out = {}
        for route in plan.get("routes", []):
            vid, ids = route.get("vehicle_id"), list(route.get("order_ids", []))
            prefix = locked.get(vid, [])
            if ids[:len(prefix)] != prefix:
                return {"__invalid__": []}
            out[vid] = ids[len(prefix):]
        return out
    return suffixes(left) == suffixes(right)


def _route_signature(plan: dict[str, Any]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(sorted((str(route.get("vehicle_id")), tuple(route.get("order_ids", [])))
                        for route in plan.get("routes", [])))


def _required_benefit(store: EtaStore, plan_id: str, base_version: int, active_plan: dict[str, Any],
                      candidate: dict[str, Any], minimum: float, hysteresis: float) -> float:
    with store.connection() as conn:
        return _required_benefit_conn(conn, plan_id, base_version, active_plan, candidate, minimum, hysteresis)


def _required_benefit_conn(conn: Any, plan_id: str, base_version: int, active_plan: dict[str, Any],
                           candidate: dict[str, Any], minimum: float, hysteresis: float) -> float:
    if base_version <= 1:
        return minimum
    previous = conn.execute("SELECT plan_json FROM eta_plans WHERE plan_id=? AND version=?",
                            (plan_id, base_version - 1)).fetchone()
    if previous:
        prior_plan = json.loads(previous["plan_json"])
        if (_route_signature(candidate) == _route_signature(prior_plan) and
                _route_signature(active_plan) != _route_signature(prior_plan)):
            return minimum + hysteresis
    return minimum


def _safe_optimum(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result.get(key) for key in ("schema_version", "plan", "validation", "expected_cost", "quantum",
            "classical_reference", "fallback", "execution_type", "coefficients")}


def _write_manifests(conn: Any, plan_id: str, version: int, plan: dict[str, Any], validation: dict[str, Any],
                     approval: dict[str, Any], created_at: str, projected: dict[str, Any] | None = None) -> None:
    routes = plan.get("routes", [])
    metrics_by_vehicle = {}
    recomputed = validation.get("recomputed") or {}
    for route in recomputed.get("routes", []):
        metrics_by_vehicle[route.get("vehicle_id")] = route.get("stops", [])
    for route in routes:
        vid = route.get("vehicle_id")
        if not vid:
            continue
        manifest_version = _next_manifest_version(conn, vid)
        projected_route = next((r for r in (projected or {}).get("routes", []) if r.get("vehicle_id") == vid), None)
        stops = []
        if projected_route:
            for metric in projected_route.get("remaining_stops", []):
                stops.append({"order_id": metric["order_id"], "arrival": metric.get("arrival"),
                              "service_start": metric.get("service_start"), "departure": metric.get("departure"),
                              "status": "completed", "immutable": True,
                              "completed_at": metric.get("completed_at")})
            future = projected_route.get("future_stops", [])
            by_order: dict[str, list[dict[str, Any]]] = {}
            for metric in future:
                by_order.setdefault(metric["order_id"], []).append({"scenario_id": metric.get("scenario_id"),
                    "arrival": metric.get("arrival"), "service_start": metric.get("service_start"),
                    "departure": metric.get("departure"), "lateness": metric.get("lateness")})
            completed_ids = {x["order_id"] for x in stops}
            for oid in route.get("order_ids", []):
                if oid in completed_ids:
                    continue
                scenarios = by_order.get(oid, [])
                stops.append({"order_id": oid, "status": "planned", "immutable": False,
                              "scenario_etas": scenarios})
        else:
            metrics = metrics_by_vehicle.get(vid, [])
            for i, oid in enumerate(route.get("order_ids", [])):
                metric = metrics[i] if i < len(metrics) else {}
                stops.append({"order_id": oid, "arrival": metric.get("arrival"),
                              "service_start": metric.get("service_start"), "departure": metric.get("departure"),
                              "status": "planned", "immutable": False})
        manifest = {"vehicle_id": vid, "plan_id": plan_id, "plan_version": version,
                    "manifest_version": manifest_version,
                    "plan_hash": _hash(plan), "ordered_stops": stops,
                    "completed_stop_references": [x["order_id"] for x in stops if x["immutable"]], "approval": approval or None,
                    "mode": "synthetic_scenario", "created_at": created_at}
        conn.execute("INSERT INTO eta_manifests(vehicle_id,version,plan_id,manifest_json,created_at) VALUES(?,?,?,?,?)",
                     (vid, manifest_version, plan_id, dumps(manifest), created_at))


def _next_manifest_version(conn: Any, vehicle_id: str) -> int:
    row = conn.execute("SELECT COALESCE(MAX(version),0)+1 AS next_version FROM eta_manifests WHERE vehicle_id=?",
                       (vehicle_id,)).fetchone()
    return int(row["next_version"])


def _audit(conn: Any, plan_id: str, action: str, actor: str, entity_id: str | None, at: str, detail: dict[str, Any]) -> None:
    conn.execute("INSERT INTO eta_audit(plan_id,action,actor,entity_id,at,detail_json) VALUES(?,?,?,?,?,?)",
                 (plan_id, action, actor, entity_id, at, dumps(detail)))
