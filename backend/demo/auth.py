"""Authentication and scoped session management for hackathon demo.

Enforces:
  - Driver-scoped bearer tokens (opaque random hex)
  - One-time pairing code redemption
  - Expiry and immediate revocation
  - Ownership enforcement (prevents Phone A from reading or modifying Phone B)
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from .store import DemoStore


class DemoAuthManager:
    """Handles pairing and session verification for demo mobile clients."""

    def __init__(self, store: DemoStore, session_ttl_minutes: int = 240) -> None:
        self.store = store
        self.session_ttl = timedelta(minutes=session_ttl_minutes)

    def redeem_code(self, code: str) -> dict[str, Any] | None:
        """Atomically redeem a one-time pairing code for an opaque session token."""
        clean_code = code.strip().upper()
        token = "demo_sess_" + secrets.token_hex(20)
        expires_at = (datetime.now(timezone.utc) + self.session_ttl).isoformat()

        driver_info = self.store.redeem_code(clean_code, token, expires_at)
        if not driver_info:
            return None

        return {
            "token": token,
            "expires_at": expires_at,
            "driver_id": driver_info["driver_id"],
            "vehicle_id": driver_info["vehicle_id"],
            "driver_name": driver_info["driver_name"],
        }

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify token validity, expiration, and non-revocation."""
        session = self.store.get_session(token)
        if not session:
            return None

        # Check expiration
        try:
            exp = datetime.fromisoformat(session["expires_at"])
            if datetime.now(timezone.utc) > exp:
                return None
        except Exception:
            return None

        return {
            "driver_id": session["driver_id"],
            "vehicle_id": session["vehicle_id"],
            "driver_name": session["driver_name"],
        }

    def revoke_token(self, token: str) -> bool:
        """Revoke current session token."""
        return self.store.revoke_session(token)
