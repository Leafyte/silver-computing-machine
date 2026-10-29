"""SQLite persistence for India ETA proposals, approvals, and manifests.

All ETA tables use their own prefix so the existing job and recovery stores are
untouched. Each write uses BEGIN IMMEDIATE; approval is consequently a single
serialized compare-and-swap transaction across plan, proposal, manifest, and
audit records.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class EtaStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self.connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS eta_plans (
              plan_id TEXT NOT NULL, version INTEGER NOT NULL, model_version TEXT NOT NULL,
              as_of TEXT NOT NULL, scenario_json TEXT NOT NULL, weighted_json TEXT NOT NULL,
              plan_json TEXT NOT NULL, cost_json TEXT NOT NULL, currency TEXT NOT NULL,
              operating_rate REAL NOT NULL, min_benefit REAL NOT NULL, cooldown_seconds REAL NOT NULL,
              expiry_seconds REAL NOT NULL, switching_cost REAL NOT NULL DEFAULT 0,
              hysteresis_cost_threshold REAL NOT NULL DEFAULT 5, eta_model_json TEXT,
              mode TEXT NOT NULL, active INTEGER NOT NULL,
              latest_event_at TEXT, latest_event_hash TEXT, last_proposal_at TEXT,
              PRIMARY KEY(plan_id, version)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS eta_one_active_plan ON eta_plans(plan_id) WHERE active=1;
            CREATE TABLE IF NOT EXISTS eta_events (
              event_id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, body_hash TEXT NOT NULL,
              as_of TEXT NOT NULL, status TEXT NOT NULL, result_json TEXT NOT NULL,
              created_at TEXT NOT NULL, observed_state_json TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS eta_proposals (
              proposal_id TEXT PRIMARY KEY, event_id TEXT NOT NULL UNIQUE, plan_id TEXT NOT NULL,
              base_version INTEGER NOT NULL, model_version TEXT NOT NULL, context_hash TEXT NOT NULL,
              expires_at TEXT NOT NULL, status TEXT NOT NULL, proposal_json TEXT NOT NULL,
              created_at TEXT NOT NULL, decided_at TEXT, decided_by TEXT
            );
            CREATE INDEX IF NOT EXISTS eta_pending_by_plan ON eta_proposals(plan_id,status);
            CREATE TABLE IF NOT EXISTS eta_manifests (
              vehicle_id TEXT NOT NULL, version INTEGER NOT NULL, plan_id TEXT NOT NULL,
              manifest_json TEXT NOT NULL, created_at TEXT NOT NULL,
              PRIMARY KEY(vehicle_id,version)
            );
            CREATE TABLE IF NOT EXISTS eta_audit (
              audit_id INTEGER PRIMARY KEY AUTOINCREMENT, plan_id TEXT NOT NULL,
              action TEXT NOT NULL, actor TEXT NOT NULL, entity_id TEXT,
              at TEXT NOT NULL, detail_json TEXT NOT NULL
            );
            """)
            # Additive migrations for local databases created by early ETA
            # controller iterations; never touch non-ETA tables.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(eta_plans)")}
            if "switching_cost" not in columns:
                conn.execute("ALTER TABLE eta_plans ADD COLUMN switching_cost REAL NOT NULL DEFAULT 0")
            if "hysteresis_cost_threshold" not in columns:
                conn.execute("ALTER TABLE eta_plans ADD COLUMN hysteresis_cost_threshold REAL NOT NULL DEFAULT 5")
            if "eta_model_json" not in columns:
                conn.execute("ALTER TABLE eta_plans ADD COLUMN eta_model_json TEXT")
            event_columns = {row[1] for row in conn.execute("PRAGMA table_info(eta_events)")}
            if "observed_state_json" not in event_columns:
                conn.execute("ALTER TABLE eta_events ADD COLUMN observed_state_json TEXT NOT NULL DEFAULT '[]'")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def read(self, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return list(conn.execute(query, params).fetchall())
