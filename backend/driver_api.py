"""Authenticated, vehicle-scoped driver API for manifest reads and reports."""
from __future__ import annotations

import hmac
import os
from pathlib import Path
import threading
import time
from typing import Any

from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse

from .auth import AuthConfig
from .driver_sessions import DriverError, DriverSessionStore


def create_driver_router(app, auth: AuthConfig) -> APIRouter:
    router = APIRouter(prefix="/api/v1/driver", tags=["Driver"])
    store = DriverSessionStore()
    app.state.driver_session_store = store
    attempts: dict[str, list[float]] = {}
    attempts_lock = threading.Lock()

    def bearer(request: Request) -> str:
        value = request.headers.get("authorization", "")
        return value[7:] if value.startswith("Bearer ") else ""

    def actor(request: Request) -> dict[str, str]:
        return store.authenticate(bearer(request))

    def eta_controller():
        controller = getattr(app.state, "eta_controller", None)
        if controller is None:
            from .india_eta.controller import EtaController
            path = os.environ.get("DEBUGLEAF_ETA_DB_PATH", ".runtime/india_eta.sqlite3")
            controller = EtaController(Path(path))
            app.state.eta_controller = controller
        return controller

    def call(function):
        try:
            return function()
        except DriverError as exc:
            return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}})
        except Exception:
            return JSONResponse(status_code=503, content={"error": {"code": "driver_service_unavailable", "message": "Driver operation could not complete"}})

    @router.post("/sessions")
    def pair(request: Request, payload: Any = Body(...)):
        def operation():
            if not isinstance(payload, dict) or set(payload) != {"pairing_code"} or not isinstance(payload["pairing_code"], str):
                raise DriverError("invalid_pairing", "Pairing request is invalid")
            source = request.client.host if request.client else "unknown"
            now = time.time()
            with attempts_lock:
                recent = [stamp for stamp in attempts.get(source, []) if now - stamp < 60]
                if len(recent) >= 5:
                    raise DriverError("pairing_rate_limited", "Too many pairing attempts; try again later", 429)
                recent.append(now)
                attempts[source] = recent
            token, identity = store.redeem(payload["pairing_code"])
            return {"access_token": token, **identity, "token_type": "bearer"}
        return call(operation)

    @router.get("/me")
    def me(request: Request):
        return call(lambda: actor(request))

    @router.delete("/sessions/current")
    def logout(request: Request):
        def operation():
            identity = actor(request)
            store.revoke(bearer(request))
            return {"revoked": True, "driver_id": identity["driver_id"]}
        return call(operation)

    @router.get("/state")
    def state(request: Request):
        def operation():
            identity = actor(request)
            manifest = eta_controller().get_manifest(identity["vehicle_id"])
            reports = store.reports(identity["vehicle_id"], manifest["plan_id"])
            stop_status = {row["order_id"]: "completed" if row.get("immutable") else "pending"
                           for row in manifest.get("ordered_stops", [])}
            for report in reports:
                if report["stop_id"] and report["kind"] in {"stop_arrived", "stop_completed"}:
                    stop_status[report["stop_id"]] = "arrived" if report["kind"] == "stop_arrived" else "completed"
            return {"driver_id": identity["driver_id"], "vehicle_id": identity["vehicle_id"],
                    "manifest": manifest, "stop_status": stop_status, "reports": reports,
                    "capabilities": ["manifest_read", "stop_arrival_report", "stop_completion_report", "delay_report"],
                    "state_source": "backend manifest plus driver-reported events"}
        return call(operation)

    @router.post("/reports", status_code=202)
    def report(request: Request, payload: Any = Body(...)):
        def operation():
            identity = actor(request)
            manifest = eta_controller().get_manifest(identity["vehicle_id"])
            return store.submit_report(identity, manifest, payload)
        return call(operation)

    @router.get("/reports")
    def reports(request: Request, limit: int = 100):
        def operation():
            if not 1 <= limit <= 500:
                raise DriverError("invalid_limit", "limit must be between 1 and 500")
            token = bearer(request).encode()
            permitted = auth.mode == "local" and auth.ready
            if auth.mode == "token" and auth.ready:
                permitted = any(hmac.compare_digest(token, auth.tokens[role].encode())
                                for role in ("dispatcher", "admin"))
            if not permitted:
                raise DriverError("authorization_denied", "Dispatcher access is required", 403)
            return store.list_reports(limit)
        return call(operation)

    return router
