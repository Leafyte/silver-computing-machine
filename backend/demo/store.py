"""Thread-safe SQLite storage for hackathon phone demo.

Manages:
  - Pairing codes & driver sessions
  - Driver route states & active manifest versions
  - Congestion reports & active delay overlays
  - Durable planning jobs
  - Client acknowledgements & decision audit log
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DemoStore:
    """SQLite-backed store for demo state isolation."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_dir = Path(".runtime")
            db_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(db_dir / "demo_store.sqlite3")
        else:
            self.db_path = str(db_path)
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._init_db()

    from contextlib import contextmanager

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS pairing_codes (
                    code TEXT PRIMARY KEY,
                    driver_id TEXT NOT NULL,
                    vehicle_id TEXT NOT NULL,
                    driver_name TEXT NOT NULL,
                    is_used INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    driver_id TEXT NOT NULL,
                    vehicle_id TEXT NOT NULL,
                    driver_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    is_revoked INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS driver_state (
                    driver_id TEXT PRIMARY KEY,
                    vehicle_id TEXT NOT NULL,
                    route_id TEXT NOT NULL,
                    corridor_id TEXT NOT NULL,
                    route_version INTEGER NOT NULL DEFAULT 1,
                    manifest_revision INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'en_route',
                    last_ack_version INTEGER DEFAULT 0,
                    last_ack_at TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS congestion_reports (
                    event_id TEXT PRIMARY KEY,
                    driver_id TEXT NOT NULL,
                    vehicle_id TEXT NOT NULL,
                    segment_id TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    delay_seconds REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    note TEXT
                );

                CREATE TABLE IF NOT EXISTS planning_jobs (
                    job_id TEXT PRIMARY KEY,
                    driver_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    outcome TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    principal TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)

    # -------------------------------------------------------------------------
    # Pairing & Sessions
    # -------------------------------------------------------------------------

    def seed_pairing_codes(self, codes: list[dict[str, str]]) -> None:
        with self._lock, self._get_connection() as conn:
            for item in codes:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO pairing_codes (code, driver_id, vehicle_id, driver_name, is_used, created_at)
                    VALUES (?, ?, ?, ?, 0, ?)
                    """,
                    (item["code"], item["driver_id"], item["vehicle_id"], item["driver_name"], _utcnow_iso()),
                )

    def redeem_code(self, code: str, session_token: str, expires_at: str) -> dict[str, Any] | None:
        with self._lock, self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM pairing_codes WHERE code = ? AND is_used = 0", (code,))
            row = cur.fetchone()
            if not row:
                return None
            driver_id = row["driver_id"]
            vehicle_id = row["vehicle_id"]
            driver_name = row["driver_name"]

            # Mark code as used
            cur.execute("UPDATE pairing_codes SET is_used = 1 WHERE code = ?", (code,))
            # Insert session
            now = _utcnow_iso()
            cur.execute(
                """
                INSERT INTO sessions (token, driver_id, vehicle_id, driver_name, created_at, expires_at, is_revoked)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                (session_token, driver_id, vehicle_id, driver_name, now, expires_at),
            )
            return {
                "driver_id": driver_id,
                "vehicle_id": vehicle_id,
                "driver_name": driver_name,
            }

    def get_session(self, token: str) -> dict[str, Any] | None:
        with self._lock, self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM sessions WHERE token = ? AND is_revoked = 0", (token,))
            row = cur.fetchone()
            if not row:
                return None
            return dict(row)

    def revoke_session(self, token: str) -> bool:
        with self._lock, self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE sessions SET is_revoked = 1 WHERE token = ?", (token,))
            return cur.rowcount > 0

    # -------------------------------------------------------------------------
    # Driver State
    # -------------------------------------------------------------------------

    def init_driver_state_if_missing(
        self, driver_id: str, vehicle_id: str, route_id: str, corridor_id: str
    ) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO driver_state (driver_id, vehicle_id, route_id, corridor_id, route_version, manifest_revision, status, updated_at)
                VALUES (?, ?, ?, ?, 1, 1, 'en_route', ?)
                """,
                (driver_id, vehicle_id, route_id, corridor_id, _utcnow_iso()),
            )

    def get_driver_state(self, driver_id: str) -> dict[str, Any] | None:
        with self._lock, self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM driver_state WHERE driver_id = ?", (driver_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def update_driver_route(
        self, driver_id: str, route_id: str, corridor_id: str
    ) -> dict[str, Any]:
        with self._lock, self._get_connection() as conn:
            cur = conn.cursor()
            now = _utcnow_iso()
            cur.execute(
                """
                UPDATE driver_state
                SET route_id = ?, corridor_id = ?, route_version = route_version + 1, manifest_revision = manifest_revision + 1, updated_at = ?
                WHERE driver_id = ?
                """,
                (route_id, corridor_id, now, driver_id),
            )
            cur.execute("SELECT * FROM driver_state WHERE driver_id = ?", (driver_id,))
            return dict(cur.fetchone())

    def record_acknowledgement(self, driver_id: str, version: int) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                UPDATE driver_state
                SET last_ack_version = ?, last_ack_at = ?
                WHERE driver_id = ?
                """,
                (version, _utcnow_iso(), driver_id),
            )

    # -------------------------------------------------------------------------
    # Congestion Reports & Overlays
    # -------------------------------------------------------------------------

    def save_congestion_report(
        self,
        event_id: str,
        driver_id: str,
        vehicle_id: str,
        segment_id: str,
        severity: str,
        delay_seconds: float,
        expires_at: str,
        note: str = "",
    ) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO congestion_reports (event_id, driver_id, vehicle_id, segment_id, severity, delay_seconds, created_at, expires_at, note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, driver_id, vehicle_id, segment_id, severity, delay_seconds, _utcnow_iso(), expires_at, note),
            )

    def get_active_overlays(self) -> dict[str, float]:
        """Return {segment_id: delay_seconds} for non-expired reports."""
        with self._lock, self._get_connection() as conn:
            cur = conn.cursor()
            now = _utcnow_iso()
            cur.execute("SELECT segment_id, delay_seconds FROM congestion_reports WHERE expires_at > ?", (now,))
            rows = cur.fetchall()
            overlays: dict[str, float] = {}
            for r in rows:
                seg = r["segment_id"]
                # Keep maximum reported delay for a segment
                overlays[seg] = max(overlays.get(seg, 0.0), float(r["delay_seconds"]))
            return overlays

    # -------------------------------------------------------------------------
    # Planning Jobs & Audit
    # -------------------------------------------------------------------------

    def create_or_get_job(self, job_id: str, driver_id: str, request_id: str) -> dict[str, Any]:
        with self._lock, self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM planning_jobs WHERE request_id = ?", (request_id,))
            row = cur.fetchone()
            if row:
                return dict(row)
            now = _utcnow_iso()
            cur.execute(
                """
                INSERT INTO planning_jobs (job_id, driver_id, request_id, status, outcome, result_json, created_at, updated_at)
                VALUES (?, ?, ?, 'running', NULL, NULL, ?, ?)
                """,
                (job_id, driver_id, request_id, now, now),
            )
            return {
                "job_id": job_id,
                "driver_id": driver_id,
                "request_id": request_id,
                "status": "running",
            }

    def complete_job(self, job_id: str, outcome: str, result: dict[str, Any]) -> None:
        with self._lock, self._get_connection() as conn:
            now = _utcnow_iso()
            conn.execute(
                """
                UPDATE planning_jobs
                SET status = 'completed', outcome = ?, result_json = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (outcome, json.dumps(result), now, job_id),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM planning_jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            res = dict(row)
            if res.get("result_json"):
                try:
                    res["result"] = json.loads(res["result_json"])
                except Exception:
                    res["result"] = None
            return res

    def log_audit(self, event_type: str, principal: str, payload: Mapping[str, Any]) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_events (event_type, principal, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (event_type, principal, json.dumps(dict(payload)), _utcnow_iso()),
            )
