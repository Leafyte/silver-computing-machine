from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from backend.india_eta.store import EtaStore


def test_eta_tables_are_namespaced_and_transaction_rolls_back(tmp_path: Path) -> None:
    db = tmp_path / "jobs.sqlite3"
    store = EtaStore(db)
    with store.transaction() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS existing_jobs(job_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO existing_jobs(job_id) VALUES('job-1')")
    with pytest.raises(RuntimeError):
        with store.transaction() as conn:
            conn.execute("INSERT INTO eta_audit(plan_id,action,actor,entity_id,at,detail_json) VALUES('p','test','x',NULL,'now','{}')")
            raise RuntimeError("force transaction rollback")
    tables = {row[0] for row in store.read("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"eta_plans", "eta_events", "eta_proposals", "eta_manifests", "eta_audit", "existing_jobs"} <= tables
    assert store.read("SELECT job_id FROM existing_jobs")[0][0] == "job-1"
    assert store.read("SELECT * FROM eta_audit") == []


def test_concurrent_begin_immediate_transactions_serialize(tmp_path: Path) -> None:
    store = EtaStore(tmp_path / "concurrent.sqlite3")
    with store.transaction() as conn:
        conn.execute("INSERT INTO eta_events(event_id,plan_id,body_hash,as_of,status,result_json,created_at) VALUES('e','p','h','2026-01-01T00:00:00Z','processing','{}','now')")
    with store.transaction() as conn:
        conn.execute("UPDATE eta_events SET status='completed' WHERE event_id='e'")
    assert store.read("SELECT status FROM eta_events WHERE event_id='e'")[0][0] == "completed"


def test_model_identity_column_is_added_to_legacy_eta_database(tmp_path: Path) -> None:
    db = tmp_path / "legacy-eta.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE eta_plans (
          plan_id TEXT NOT NULL, version INTEGER NOT NULL, model_version TEXT NOT NULL,
          as_of TEXT NOT NULL, scenario_json TEXT NOT NULL, weighted_json TEXT NOT NULL,
          plan_json TEXT NOT NULL, cost_json TEXT NOT NULL, currency TEXT NOT NULL,
          operating_rate REAL NOT NULL, min_benefit REAL NOT NULL, cooldown_seconds REAL NOT NULL,
          expiry_seconds REAL NOT NULL, switching_cost REAL NOT NULL DEFAULT 0,
          hysteresis_cost_threshold REAL NOT NULL DEFAULT 5, mode TEXT NOT NULL, active INTEGER NOT NULL,
          latest_event_at TEXT, latest_event_hash TEXT, last_proposal_at TEXT,
          PRIMARY KEY(plan_id, version))""")
        conn.execute("""INSERT INTO eta_plans
          (plan_id,version,model_version,as_of,scenario_json,weighted_json,plan_json,cost_json,currency,
           operating_rate,min_benefit,cooldown_seconds,expiry_seconds,mode,active)
          VALUES('legacy',1,'v1','2026-01-01T00:00:00Z','{}','[]','{}','{}','XTS',1,0,120,900,'synthetic_scenario',1)""")
    store = EtaStore(db)
    columns = {row[1] for row in store.read("PRAGMA table_info(eta_plans)")}
    assert "eta_model_json" in columns
    legacy = store.read("SELECT eta_model_json,model_version FROM eta_plans WHERE plan_id='legacy'")[0]
    assert legacy["eta_model_json"] is None
    assert legacy["model_version"] == "v1"
