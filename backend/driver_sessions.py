"""Scoped driver pairing sessions and durable driver-reported events.

This is a local prototype identity boundary, not a replacement for an IdP.
Pairing codes are server-configured, one-time, and never persisted in cleartext.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any


class DriverError(ValueError):
    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class DriverSessionStore:
    def __init__(self, db_path: str | Path | None = None, *, pairings: list[dict[str, str]] | None = None,
                 session_seconds: int = 8 * 60 * 60):
        raw_pairings = os.environ.get("DEBUGLEAF_DRIVER_PAIRING_CODES", "[]")
        try:
            configured = json.loads(raw_pairings) if pairings is None else pairings
        except json.JSONDecodeError:
            raise ValueError("driver pairing configuration is invalid") from None
        if not isinstance(configured, list) or len(configured) > 64:
            raise ValueError("driver pairing configuration is invalid")
        self.pairings: list[tuple[str, str, str]] = []
        for item in configured:
            if (not isinstance(item, dict) or set(item) != {"code", "driver_id", "vehicle_id"} or
                    not all(isinstance(item[key], str) and item[key] for key in item) or
                    len(item["code"]) < 12 or len(item["code"]) > 128):
                raise ValueError("driver pairing configuration is invalid")
            self.pairings.append((_digest(item["code"]), item["driver_id"], item["vehicle_id"]))
        self.session_seconds = max(60, min(int(session_seconds), 24 * 60 * 60))
        self.db_path = Path(db_path or os.environ.get("DEBUGLEAF_DRIVER_DB_PATH", ".runtime/drivers.sqlite3"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS redeemed_pairings(code_sha256 TEXT PRIMARY KEY, redeemed_at REAL NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS driver_sessions(
                token_sha256 TEXT PRIMARY KEY, driver_id TEXT NOT NULL, vehicle_id TEXT NOT NULL,
                issued_at REAL NOT NULL, expires_at REAL NOT NULL, revoked INTEGER NOT NULL DEFAULT 0)""")
            db.execute("""CREATE TABLE IF NOT EXISTS driver_reports(
                event_id TEXT PRIMARY KEY, body_sha256 TEXT NOT NULL, vehicle_id TEXT NOT NULL,
                driver_id TEXT NOT NULL, plan_id TEXT NOT NULL, manifest_version INTEGER NOT NULL,
                kind TEXT NOT NULL, stop_id TEXT, severity TEXT, note TEXT NOT NULL, created_at TEXT NOT NULL)""")

    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def redeem(self, code: str) -> tuple[str, dict[str, str]]:
        candidate = _digest(code)
        matched = next((row for row in self.pairings if secrets.compare_digest(candidate, row[0])), None)
        if matched is None:
            raise DriverError("pairing_failed", "Pairing code is invalid or unavailable", 401)
        _, driver_id, vehicle_id = matched
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            exists = db.execute("SELECT 1 FROM redeemed_pairings WHERE code_sha256=?", (candidate,)).fetchone()
            if exists:
                db.rollback()
                raise DriverError("pairing_used", "Pairing code has already been redeemed", 409)
            db.execute("INSERT INTO redeemed_pairings VALUES(?,?)", (candidate, now))
            db.execute("INSERT INTO driver_sessions VALUES(?,?,?,?,?,0)",
                       (_digest(token), driver_id, vehicle_id, now, now + self.session_seconds))
        return token, {"driver_id": driver_id, "vehicle_id": vehicle_id,
                       "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + self.session_seconds))}

    def create_demo_session(self, driver_id: str, vehicle_id: str) -> tuple[str, dict[str, str]]:
        """Issue a short-lived session for a configured local demo account."""
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self._connect() as db:
            db.execute("INSERT INTO driver_sessions VALUES(?,?,?,?,?,0)",
                       (_digest(token), driver_id, vehicle_id, now, now + self.session_seconds))
        return token, {"driver_id": driver_id, "vehicle_id": vehicle_id,
                       "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + self.session_seconds))}

    def authenticate(self, token: str) -> dict[str, str]:
        if not token or len(token) > 256:
            raise DriverError("authentication_required", "Valid driver session required", 401)
        with self._connect() as db:
            row = db.execute("SELECT driver_id,vehicle_id,expires_at,revoked FROM driver_sessions WHERE token_sha256=?",
                             (_digest(token),)).fetchone()
        if not row or row["revoked"] or row["expires_at"] <= time.time():
            raise DriverError("authentication_required", "Valid driver session required", 401)
        return {
            "driver_id": row["driver_id"],
            "vehicle_id": row["vehicle_id"],
            "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(row["expires_at"])),
        }

    def revoke(self, token: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE driver_sessions SET revoked=1 WHERE token_sha256=?", (_digest(token),))

    def reports(self, vehicle_id: str, plan_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM driver_reports WHERE vehicle_id=? AND plan_id=? ORDER BY created_at,event_id",
                              (vehicle_id, plan_id)).fetchall()
        return [{key: row[key] for key in ("event_id", "driver_id", "vehicle_id", "plan_id", "manifest_version",
                                            "kind", "stop_id", "severity", "note", "created_at")} for row in rows]

    def list_reports(self, limit: int) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT event_id,driver_id,vehicle_id,plan_id,manifest_version,kind,stop_id,severity,note,created_at FROM driver_reports ORDER BY created_at DESC,event_id LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def submit_report(self, actor: dict[str, str], manifest: dict[str, Any], payload: Any) -> dict[str, Any]:
        required = {"event_id", "manifest_version", "kind"}
        allowed = required | {"stop_id", "severity", "note"}
        if (not isinstance(payload, dict) or not required <= set(payload) or set(payload) - allowed or
                not isinstance(payload.get("event_id"), str) or not 1 <= len(payload["event_id"]) <= 64 or
                isinstance(payload.get("manifest_version"), bool) or not isinstance(payload.get("manifest_version"), int) or
                payload["manifest_version"] != manifest.get("manifest_version")):
            raise DriverError("invalid_report", "Driver report does not match the active manifest")
        kind = payload["kind"]
        if kind not in {"stop_arrived", "stop_completed", "delay_report", "driver_message"}:
            raise DriverError("invalid_report", "Driver report kind is unsupported")
        stop_id = payload.get("stop_id")
        stops = manifest.get("ordered_stops", [])
        stop_ids = [row.get("order_id") for row in stops if isinstance(row, dict)]
        if kind in {"stop_arrived", "stop_completed"} or stop_id is not None:
            if not isinstance(stop_id, str) or stop_id not in stop_ids:
                raise DriverError("invalid_report", "Stop is not in this driver's active manifest")
        severity = payload.get("severity")
        if kind == "delay_report" and severity not in {"light", "moderate", "heavy"}:
            raise DriverError("invalid_report", "Delay severity must be light, moderate or heavy")
        note = payload.get("note", "")
        if not isinstance(note, str) or len(note) > 280:
            raise DriverError("invalid_report", "Driver note exceeds the 280 character limit")
        if kind == "driver_message" and not note.strip():
            raise DriverError("invalid_report", "Driver message cannot be empty")
        event = {"event_id": payload["event_id"], "manifest_version": payload["manifest_version"],
                 "kind": kind, "stop_id": stop_id, "severity": severity, "note": note}
        body_hash = _digest(json.dumps(event, sort_keys=True, separators=(",", ":")))
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT body_sha256 FROM driver_reports WHERE event_id=?", (event["event_id"],)).fetchone()
            if prior:
                if not secrets.compare_digest(prior["body_sha256"], body_hash):
                    db.rollback()
                    raise DriverError("duplicate_event_conflict", "Event ID was already used for different report data", 409)
                row = db.execute("SELECT * FROM driver_reports WHERE event_id=?", (event["event_id"],)).fetchone()
                return {key: row[key] for key in ("event_id", "kind", "stop_id", "severity", "created_at")}
            if kind in {"stop_arrived", "stop_completed"}:
                rows = db.execute("SELECT stop_id,kind FROM driver_reports WHERE vehicle_id=? AND plan_id=? AND kind IN ('stop_arrived','stop_completed') ORDER BY rowid",
                                  (actor["vehicle_id"], manifest["plan_id"])).fetchall()
                completed = {row["stop_id"] for row in rows if row["kind"] == "stop_completed"}
                arrived = {row["stop_id"] for row in rows if row["kind"] == "stop_arrived"}
                initial_completed = {row.get("order_id") for row in stops if row.get("immutable")}
                completed |= initial_completed
                next_stop = next((oid for oid in stop_ids if oid not in completed), None)
                if stop_id != next_stop:
                    db.rollback()
                    raise DriverError("stop_order_conflict", "Only the next uncompleted manifest stop can be updated", 409)
                if kind == "stop_arrived" and stop_id in arrived:
                    db.rollback()
                    raise DriverError("stop_already_arrived", "Arrival for this stop was already reported", 409)
            db.execute("""INSERT INTO driver_reports(event_id,body_sha256,vehicle_id,driver_id,plan_id,manifest_version,
                kind,stop_id,severity,note,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (event["event_id"], body_hash, actor["vehicle_id"], actor["driver_id"], manifest["plan_id"],
                 manifest["manifest_version"], kind, stop_id, severity, note, now))
        return {**event, "vehicle_id": actor["vehicle_id"], "plan_id": manifest["plan_id"], "created_at": now,
                "status": "recorded"}
