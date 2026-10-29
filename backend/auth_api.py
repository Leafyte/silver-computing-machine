"""Login endpoints for configured local demo accounts."""
from __future__ import annotations

import threading
import time
from typing import Any

from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse

from .demo_auth import DemoAuthError, DemoAuthService


def create_auth_router(app, auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])
    service = DemoAuthService(auth, app.state.driver_session_store)
    app.state.demo_auth_service = service
    attempts: dict[str, list[float]] = {}
    attempts_lock = threading.Lock()

    @router.post("/login")
    def login(request: Request, payload: Any = Body(...)):
        source = request.client.host if request.client else "unknown"
        now = time.time()
        with attempts_lock:
            recent = [stamp for stamp in attempts.get(source, []) if now - stamp < 60]
            if len(recent) >= 8:
                return JSONResponse(status_code=429, content={"error": {"code": "login_rate_limited", "message": "Too many login attempts; try again later"}})
            recent.append(now)
            attempts[source] = recent
        if not isinstance(payload, dict) or set(payload) != {"username", "password"}:
            return JSONResponse(status_code=422, content={"error": {"code": "invalid_login", "message": "Username or password is incorrect"}})
        try:
            return service.login(payload.get("username"), payload.get("password"))
        except DemoAuthError as exc:
            return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}})

    return router
