"""Small opt-in bearer-token authorization hook for localhost development.

This module is not production authentication. Tokens are injected from named
environment variables by default and are never included in errors or logs.
"""
from __future__ import annotations

import hmac
import json
import os
from dataclasses import dataclass
from typing import Mapping


ROLES = ("dispatcher", "researcher", "admin")
TOKEN_ENV = {
    "dispatcher": "DEBUGLEAF_DISPATCHER_TOKEN",
    "researcher": "DEBUGLEAF_RESEARCHER_TOKEN",
    "admin": "DEBUGLEAF_ADMIN_TOKEN",
}


class AuthConfigurationError(ValueError):
    """Invalid auth mode configuration; never includes credential values."""


@dataclass(frozen=True)
class AuthConfig:
    mode: str
    tokens: Mapping[str, str]
    ready: bool

    @classmethod
    def from_mapping(cls, mode: str, tokens: Mapping[str, str] | None = None) -> "AuthConfig":
        values = dict(tokens or {})
        if mode == "local":
            return cls(mode="local", tokens={}, ready=True)
        if mode != "token":
            return cls(mode=mode, tokens={}, ready=False)
        ready = (set(values) == set(ROLES) and
                 all(isinstance(values.get(role), str) and values[role] for role in ROLES) and
                 len(set(values.values())) == len(ROLES))
        return cls(mode="token", tokens=values if ready else {}, ready=ready)

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "AuthConfig":
        env = os.environ if environ is None else environ
        mode = env.get("DEBUGLEAF_AUTH_MODE", "local").strip().lower()
        if mode != "token":
            return cls.from_mapping(mode)
        tokens = {role: env.get(variable, "") for role, variable in TOKEN_ENV.items()}
        return cls.from_mapping(mode, tokens)


# Route policy is intentionally based on implemented API routes. Driver routes
# use separate opaque, vehicle-scoped session tokens in backend.driver_api.
_COMMON_RESEARCHER = {
    ("GET", "/api/v1/health"),
    ("POST", "/api/v1/scenarios/validate"), ("POST", "/api/v1/scenarios/routes"),
    ("POST", "/api/v1/scenarios/plan"), ("POST", "/api/v1/scenarios/qaoa"),
    ("POST", "/api/v1/scenarios/vqc-plan"), ("POST", "/api/v1/risk/evaluate"),
    ("POST", "/api/v1/plans/validate"), ("POST", "/api/v1/jobs"),
    ("GET", "/api/v1/jobs"),
    ("POST", "/api/v1/scenarios/submit"), ("POST", "/api/v1/jobs/qaoa"),
}

_DISPATCHER_ONLY = {
    ("POST", "/api/v1/scenarios/financial-plan"),
    ("POST", "/api/v1/recovery/plan"),
}

_ADMIN_ONLY = {
    ("POST", "/api/v1/scenarios/financial-plan"),
    ("POST", "/api/v1/recovery/plan"),
    ("GET", "/api/v1/recovery/audits/latest"),
}


def _dynamic_route_allowed(method: str, path: str) -> bool:
    parts = path.split("/")
    if len(parts) == 5 and parts[:3] == ["", "api", "v1"]:
        if parts[3] == "jobs" and parts[4]:
            return method == "GET"
        if parts[3] == "recovery" and parts[4] == "audits":
            return False
    if len(parts) == 6 and parts[:4] == ["", "api", "v1", "jobs"] and parts[4] and parts[5] == "cancel":
        return method == "POST"
    if len(parts) == 6 and parts[:5] == ["", "api", "v1", "recovery", "audits"] and parts[5]:
        return method == "GET"
    return False


def _route_role_allowed(role: str, method: str, path: str) -> bool:
    method = method.upper()
    if path.startswith("/api/v1/eta/"):
        return _eta_route_allowed(role, method, path)
    if role == "admin":
        return ((method, path) in (_COMMON_RESEARCHER | _DISPATCHER_ONLY | _ADMIN_ONLY) or
                _dynamic_route_allowed(method, path))
    if role == "researcher":
        if (method, path) in _COMMON_RESEARCHER:
            return True
        if _dynamic_route_allowed(method, path) and method == "GET":
            return True
        if _dynamic_route_allowed(method, path) and method == "POST" and path.endswith("/cancel"):
            return True
        return False
    if role == "dispatcher":
        if (method, path) in _COMMON_RESEARCHER:
            return True
        if (method, path) in _DISPATCHER_ONLY:
            return True
        if _dynamic_route_allowed(method, path) and method == "GET":
            return True
        if _dynamic_route_allowed(method, path) and method == "POST" and path.endswith("/cancel"):
            return True
    return False


def _eta_route_allowed(role: str, method: str, path: str) -> bool:
    parts = path.strip("/").split("/")
    if parts[:3] != ["api", "v1", "eta"] or role not in ROLES:
        return False
    resource = parts[3:]
    if method == "POST" and resource in (["predictions"], ["sla"], ["compatibility"]):
        return True
    if method == "GET" and len(resource) == 2 and resource[0] in ("plans", "events", "proposals", "manifests"):
        return bool(resource[1])
    if role not in ("dispatcher", "admin"):
        return False
    if method == "POST" and resource in (["plans"], ["events"]):
        return True
    return (method == "POST" and len(resource) == 3 and bool(resource[1])
            and (resource[0], resource[2]) in (("events", "cancel"), ("proposals", "approve")))


def authorize_request(config: AuthConfig, method: str, path: str,
                     authorization: bytes) -> tuple[int, str, str, bool] | None:
    """Return a safe denial tuple, or None when the request may proceed."""
    # Driver routes validate their separate, vehicle-scoped session token in
    # backend.driver_api. The public pairing endpoint is one-time and bounded.
    if path.startswith("/api/v1/driver/"):
        return None
    if method.upper() == "POST" and path == "/api/v1/auth/login":
        return None
    if config.mode == "local" and config.ready:
        return None
    if not config.ready:
        return 503, "auth_misconfigured", "token authorization is not configured", False
    method = method.upper()
    if method == "GET" and path == "/api/v1/health":
        return None
    token = authorization[len(b"Bearer "):] if authorization.startswith(b"Bearer ") else b""
    if not token:
        return 401, "authentication_required", "bearer token required", True
    matched_roles: list[str] = []
    for role in ROLES:
        expected = config.tokens[role].encode("utf-8")
        if hmac.compare_digest(token, expected):
            matched_roles.append(role)
    if not matched_roles:
        return 401, "authentication_failed", "bearer token is invalid", True
    if not any(_route_role_allowed(role, method, path) for role in matched_roles):
        return 403, "authorization_denied", "role is not allowed for this endpoint", False
    return None


class TokenAuthMiddleware:
    """Pure ASGI middleware, with a credential-free default local mode."""

    def __init__(self, app, *, config: AuthConfig):
        self.app = app
        self.config = config

    @staticmethod
    async def _respond(send, status: int, code: str, message: str, *, challenge: bool = False) -> None:
        body = json.dumps({"error": {"code": code, "message": message}}, separators=(",", ":")).encode()
        headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
        if challenge:
            headers.append((b"www-authenticate", b"Bearer"))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        method = scope.get("method", "GET").upper()
        raw_headers = dict(scope.get("headers", []))
        denial = authorize_request(self.config, method, path, raw_headers.get(b"authorization", b""))
        if denial:
            status, code, message, challenge = denial
            await self._respond(send, status, code, message, challenge=challenge)
            return
        await self.app(scope, receive, send)
