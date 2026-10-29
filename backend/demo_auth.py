"""Local demo login for the dispatcher web client and driver Expo client.

The credentials are for a localhost demo only. Production identity and session
management are outside this prototype's scope.
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import time
from typing import Any

from .auth import AuthConfig
from .driver_sessions import DriverSessionStore


_DEFAULT_USERS = [
    {"username": "dispatcher", "password": "dispatch-demo-2026", "role": "dispatcher"},
    {"username": "driver", "password": "driver-demo-2026", "role": "driver",
     "driver_id": "demo-driver", "vehicle_id": "V1"},
]


class DemoAuthError(ValueError):
    def __init__(self, code: str, message: str, status: int = 401):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


class DemoAuthService:
    def __init__(self, auth: AuthConfig, drivers: DriverSessionStore):
        raw = os.environ.get("DEBUGLEAF_DEMO_USERS")
        try:
            users = json.loads(raw) if raw else _DEFAULT_USERS
        except json.JSONDecodeError:
            raise ValueError("demo login configuration is invalid") from None
        if not isinstance(users, list) or len(users) > 16:
            raise ValueError("demo login configuration is invalid")
        self.users: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in users:
            if (not isinstance(row, dict) or row.get("role") not in {"dispatcher", "admin", "driver"} or
                    not isinstance(row.get("username"), str) or not row["username"] or
                    not isinstance(row.get("password"), str) or len(row["password"]) < 8):
                raise ValueError("demo login configuration is invalid")
            required = {"username", "password", "role"} | ({"driver_id", "vehicle_id"} if row["role"] == "driver" else set())
            if set(row) != required or row["username"] in seen:
                raise ValueError("demo login configuration is invalid")
            if row["role"] == "driver" and not all(isinstance(row[key], str) and row[key] for key in ("driver_id", "vehicle_id")):
                raise ValueError("demo login configuration is invalid")
            seen.add(row["username"])
            self.users.append({key: value for key, value in row.items()})
        self.auth = auth
        self.drivers = drivers
        self.web_sessions: dict[str, tuple[str, float]] = {}

    def login(self, username: Any, password: Any) -> dict[str, Any]:
        if not isinstance(username, str) or not isinstance(password, str):
            raise DemoAuthError("invalid_login", "Username or password is incorrect")
        user = next((row for row in self.users if hmac.compare_digest(username.encode(), row["username"].encode())), None)
        expected = user["password"] if user else "not-a-valid-demo-password"
        if not hmac.compare_digest(password.encode(), expected.encode()):
            raise DemoAuthError("invalid_login", "Username or password is incorrect")
        role = user["role"]
        if role == "driver":
            token, identity = self.drivers.create_demo_session(user["driver_id"], user["vehicle_id"])
            return {"access_token": token, "token_type": "bearer", "role": role, **identity}
        if self.auth.mode == "token":
            if not self.auth.ready:
                raise DemoAuthError("auth_misconfigured", "Dispatcher token authorization is not configured", 503)
            token = self.auth.tokens[role]
        else:
            token = secrets.token_urlsafe(32)
            self.web_sessions[token] = (role, time.time() + 8 * 60 * 60)
        return {"access_token": token, "token_type": "bearer", "role": role,
                "expires_in": 8 * 60 * 60 if self.auth.mode != "token" else None}

    def role_for_local_token(self, token: str) -> str | None:
        record = self.web_sessions.get(token)
        if not record:
            return None
        role, expires_at = record
        if expires_at <= time.time():
            self.web_sessions.pop(token, None)
            return None
        return role
