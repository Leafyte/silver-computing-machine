"""Additive ETA API. Approval identity comes exclusively from server auth config."""
from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
import os
from pathlib import Path
import threading
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .errors import EtaError


def create_eta_router(app, auth, *, db_path=None, registry=None, uncertainty_registry=None,
                      controller=None, job_manager=None):
    lock = threading.RLock()
    state = {"controller": controller, "jobs": job_manager}
    configured_path = Path(db_path or os.environ.get("DEBUGLEAF_ETA_DB_PATH", ".runtime/india_eta.sqlite3"))

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            if state["jobs"] is not None:
                state["jobs"].close(wait=True)

    router = APIRouter(prefix="/api/v1/eta", tags=["India ETA"], lifespan=lifespan)

    def service():
        with lock:
            if state["controller"] is None:
                from .controller import EtaController
                state["controller"] = EtaController(configured_path)
                app.state.eta_controller = state["controller"]
            return state["controller"]

    def jobs():
        with lock:
            if state["jobs"] is None:
                from .jobs import EtaJobs
                state["jobs"] = EtaJobs(service(), configured_path)
                app.state.eta_jobs = state["jobs"]
            return state["jobs"]

    def principal(request):
        if auth.mode != "token" or not auth.ready:
            raise EtaError("configured_auth_required", "assignment changes require configured token-role authentication", status=403)
        value = request.headers.get("authorization", "")
        raw = value[7:].encode() if value.startswith("Bearer ") else b""
        for role in ("dispatcher", "admin"):
            if hmac.compare_digest(raw, auth.tokens[role].encode()):
                return {"subject": role, "role": role}
        raise EtaError("authorization_denied", "dispatcher or admin approval is required", status=403)

    def call(function, *args, **kwargs):
        try:
            result = function(*args, **kwargs)
            if result is None:
                return JSONResponse(status_code=404, content={"error": {"code": "not_found", "message": "ETA resource not found"}})
            return result
        except EtaError as exc:
            # Messages originate in allowlisted application code; never echo input or arbitrary exceptions.
            return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}})
        except Exception:
            return JSONResponse(status_code=503, content={"error": {"code": "eta_failure", "message": "ETA operation could not complete"}})

    @router.post("/predictions")
    def predict(payload: dict[str, Any]):
        from .prediction import predict_eta
        return call(predict_eta, payload, registry=registry)

    @router.post("/sla")
    def sla(payload: dict[str, Any]):
        from .uncertainty import evaluate_sla
        return call(evaluate_sla, payload, registry=uncertainty_registry)

    @router.post("/compatibility")
    def compatibility(payload: dict[str, Any]):
        from .compatibility import check_compatibility
        return call(check_compatibility, payload)

    @router.post("/plans")
    def register(payload: dict[str, Any], request: Request):
        def operation():
            actor = principal(request)
            from .source_adapter import model_identity_for_version
            if "eta_model_identity" in payload:
                raise EtaError("invalid_request", "Model identity is resolved by the server", status=422)
            registered = dict(payload)
            selected_id = registered.pop("eta_model_id", None)
            identity = model_identity_for_version(registered.get("model_version"), registry=registry, model_id=selected_id)
            if identity is not None:
                registered["eta_model_identity"] = identity
            return service().register_plan(registered, actor)
        return call(operation)

    @router.get("/plans/{plan_id}")
    def active(plan_id: str):
        return call(lambda: service().get_active(plan_id))

    @router.post("/events", status_code=202)
    def event(payload: dict[str, Any]):
        def operation():
            from .jobs import EtaJobs
            from .source_adapter import resolve_eta_update
            base = {key: value for key, value in payload.items() if key != "eta_update"}
            EtaJobs._validate(base)
            identity = None
            if "eta_update" in payload:
                identity = service().get_active(payload["plan_id"]).get("eta_model_identity")
            resolved = resolve_eta_update(payload, registry=registry, model_identity=identity)
            EtaJobs._validate(resolved)
            return jobs().submit(resolved)
        return call(operation)

    @router.get("/events/{event_id}")
    def event_status(event_id: str):
        return call(lambda: jobs().get(event_id))

    @router.post("/events/{event_id}/cancel")
    def cancel(event_id: str):
        return call(lambda: jobs().cancel(event_id))

    @router.get("/proposals/{proposal_id}")
    def proposal(proposal_id: str):
        return call(lambda: service().get_proposal(proposal_id))

    @router.post("/proposals/{proposal_id}/approve")
    def approve(proposal_id: str, payload: dict[str, Any], request: Request):
        def operation():
            actor = principal(request)
            return service().approve_proposal(proposal_id, payload, actor)
        return call(operation)

    @router.get("/manifests/{vehicle_id}")
    def manifest(vehicle_id: str, version: int | None = None):
        return call(lambda: service().get_manifest(vehicle_id, version=version))

    return router
