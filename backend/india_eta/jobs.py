"""Bounded durable event jobs. Cancellation suppresses late proposals and output."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sqlite3
import threading
import time
import uuid
from importlib.metadata import version, PackageNotFoundError

from .errors import EtaError


def _now():
    return datetime.now(timezone.utc).isoformat()


class EtaJobs:
    def __init__(self, controller, db_path, *, max_workers=2, timeout_seconds=10.0,
                 max_retained_jobs=256, worker=None):
        if type(max_workers) is not int or not 1 <= max_workers <= 2:
            raise ValueError("ETA workers must be 1 or 2")
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("invalid ETA deadline")
        if type(max_retained_jobs) is not int or not 1 <= max_retained_jobs <= 1000:
            raise ValueError("invalid ETA retention budget")
        self.controller, self.db_path = controller, Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.max_workers = max_workers
        self.max_retained_jobs = max_retained_jobs
        self._worker = worker or controller.process_event
        self._lock = threading.RLock()
        self._futures = {}
        self._closed = False
        self._owner = uuid.uuid4().hex
        self._versions = {}
        for dependency in ("fastapi", "pydantic"):
            try:
                self._versions[dependency] = version(dependency)
            except PackageNotFoundError:
                self._versions[dependency] = "unavailable"
        code = hashlib.sha256()
        for module in sorted(Path(__file__).parent.glob("*.py")):
            code.update(module.name.encode()); code.update(module.read_bytes())
        self._code_sha256 = code.hexdigest()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="eta-event")
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS eta_async_runtime (id INTEGER PRIMARY KEY, pid INTEGER, owner TEXT)")
            db.execute("""CREATE TABLE IF NOT EXISTS eta_async_jobs (
                event_id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, request_json TEXT NOT NULL,
                status TEXT NOT NULL, result_json TEXT, error_code TEXT, submitted_at TEXT,
                completed_at TEXT, deadline REAL NOT NULL, metadata_json TEXT NOT NULL)""")
            db.execute("BEGIN IMMEDIATE")
            lease = db.execute("SELECT pid FROM eta_async_runtime WHERE id=1").fetchone()
            if lease:
                from backend.jobs import JobManager
                if JobManager._pid_alive(lease[0]):
                    self._executor.shutdown(wait=False)
                    raise EtaError("capacity_reached", "ETA event runtime is already owned", status=503)
            stale = [r[0] for r in db.execute("SELECT event_id FROM eta_async_jobs WHERE status IN ('queued','running')")]
            db.execute("UPDATE eta_async_jobs SET status='failed',error_code='restart_interrupted',completed_at=? WHERE status IN ('queued','running')", (_now(),))
            db.execute("INSERT OR REPLACE INTO eta_async_runtime VALUES (1,?,?)", (os.getpid(), self._owner))
        for event_id in stale:
            self._invalidate(event_id)

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)

    def _invalidate(self, event_id):
        try:
            self.controller.cancel_event(event_id)
        except EtaError as exc:
            # Boundary rejection can occur before the controller reserves an event.
            if exc.code != "not_found":
                raise

    @staticmethod
    def _validate(payload):
        required = {"event_id", "plan_id", "plan_version", "model_version", "as_of", "observed_state", "scenario", "weighted_scenarios", "event_type"}
        allowed = required | {"currency", "operating_rate", "mode"}
        if not isinstance(payload, dict) or not required <= payload.keys() or payload.keys() - allowed:
            raise EtaError("invalid_request", "invalid ETA event fields", status=422)
        if not isinstance(payload["event_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", payload["event_id"]):
            raise EtaError("invalid_request", "invalid event identifier", status=422)
        try:
            body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (ValueError, TypeError):
            raise EtaError("invalid_request", "ETA event must contain finite JSON values", status=422) from None
        if len(body.encode()) > 256_000:
            raise EtaError("invalid_request", "ETA event exceeds local input budget", status=422)
        return body

    def submit(self, payload):
        body = self._validate(payload)
        digest = hashlib.sha256(body.encode()).hexdigest()
        event_id = payload["event_id"]
        with self._lock:
            if self._closed:
                raise EtaError("capacity_reached", "ETA runtime is closed", status=503)
            existing = self.get(event_id)
            if existing:
                if existing["request_sha256"] != digest:
                    raise EtaError("duplicate_event_conflict", "event identifier already has different inputs", status=409)
                return existing
            self._futures = {key: future for key, future in self._futures.items() if not future.done()}
            if sum(not f.done() for f in self._futures.values()) >= self.max_workers:
                raise EtaError("capacity_reached", "ETA event capacity reached", status=429)
            with self._connect() as db:
                count = db.execute("SELECT COUNT(*) FROM eta_async_jobs").fetchone()[0]
            if count >= self.max_retained_jobs:
                raise EtaError("retention_capacity_reached", "ETA history budget reached; preserve history before starting a fresh local database", status=503)
            metadata = {"execution_type": "local_cpu", "python": platform.python_version(), "backend": "eta_controller", "seed": 7, "max_workers": self.max_workers, "max_retained_jobs": self.max_retained_jobs, "timeout_seconds": self.timeout_seconds,
                        "dependency_versions": self._versions, "implementation_sha256": self._code_sha256,
                        "data_provenance": "synthetic_joint_scenarios; full source inputs persisted",
                        "qpu_job_id": "not_applicable", "quantum_sdk": "not used"}
            with self._connect() as db:
                db.execute("INSERT INTO eta_async_jobs VALUES (?,?,?,?,?,?,?,?,?,?)", (event_id, digest, body, "queued", None, None, _now(), None, time.time() + self.timeout_seconds, json.dumps(metadata)))
            self._futures[event_id] = self._executor.submit(self._execute, event_id, json.loads(body))
        return self.get(event_id)

    def _stopped(self, event_id):
        with self._connect() as db:
            row = db.execute("SELECT status,deadline FROM eta_async_jobs WHERE event_id=?", (event_id,)).fetchone()
        return not row or row[0] not in ("queued", "running") or time.time() >= row[1]

    def _execute(self, event_id, payload):
        with self._connect() as db:
            db.execute("UPDATE eta_async_jobs SET status='running' WHERE event_id=? AND status='queued'", (event_id,))
        result, code, encoded_result = None, None, None
        try:
            if not self._stopped(event_id):
                result = self._worker(payload, cancel_check=lambda: self._stopped(event_id))
                encoded_result = json.dumps(result, allow_nan=False)
                if len(encoded_result.encode()) > 1_048_576:
                    raise EtaError("result_budget_exceeded", "ETA result exceeds local storage budget", status=503)
        except EtaError as exc:
            code = exc.code
        except Exception:
            code = "event_failure"
        with self._lock:
            self.get(event_id)
            if self._stopped(event_id):
                self._invalidate(event_id)
                return
            if code:
                self._invalidate(event_id)
            with self._connect() as db:
                db.execute("UPDATE eta_async_jobs SET status=?,result_json=?,error_code=?,completed_at=? WHERE event_id=? AND status='running'", ("failed" if code else "succeeded", encoded_result if not code else None, code, _now(), event_id))

    def get(self, event_id):
        with self._lock:
            with self._connect() as db:
                row = db.execute("SELECT request_hash,status,result_json,error_code,submitted_at,completed_at,deadline,metadata_json FROM eta_async_jobs WHERE event_id=?", (event_id,)).fetchone()
                if not row:
                    return None
                digest, status, result, code, submitted, completed, deadline, metadata = row
                expired = status in ("queued", "running") and time.time() >= deadline
                if expired:
                    status, code, result, completed = "timed_out", "timeout", None, _now()
                    db.execute("UPDATE eta_async_jobs SET status=?,error_code=?,result_json=NULL,completed_at=? WHERE event_id=?", (status, code, completed, event_id))
            if expired:
                self._invalidate(event_id)
            return {"event_id": event_id, "status": status, "request_sha256": digest, "result": json.loads(result) if result else None, "error": {"code": code, "message": "ETA event did not complete"} if code else None, "submitted_at": submitted, "completed_at": completed, "deadline_epoch": deadline, "metadata": json.loads(metadata)}

    def cancel(self, event_id):
        with self._lock:
            if not self.get(event_id):
                return None
            with self._connect() as db:
                db.execute("UPDATE eta_async_jobs SET status='cancelled',result_json=NULL,error_code='cancelled',completed_at=? WHERE event_id=? AND status IN ('queued','running','succeeded')", (_now(), event_id))
            self._invalidate(event_id)
            return self.get(event_id)

    def close(self, *, wait=True):
        with self._lock:
            self._closed = True
            for event_id, future in list(self._futures.items()):
                if not future.done():
                    self.cancel(event_id)
        self._executor.shutdown(wait=wait, cancel_futures=True)
        with self._connect() as db:
            db.execute("DELETE FROM eta_async_runtime WHERE id=1 AND owner=?", (self._owner,))
