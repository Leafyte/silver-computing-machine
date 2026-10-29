"""Deterministic job cancellation, race, duplicate-cancel, restart, and live HTTP tests."""

import json
import logging
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from backend.api import create_app
from backend.domain import solve_exact
from backend.jobs import JobCapacityError, JobManager

ROOT = Path(__file__).parents[1]


def _scenario() -> dict[str, Any]:
    return json.loads((ROOT / "fixtures" / "dispatch_tiny.json").read_text())


class LogCaptureHandler(logging.Handler):
    """Capture logs without logging secret or scenario payloads."""
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def capture_jobs_logger():
    logger = logging.getLogger("backend.jobs")
    handler = LogCaptureHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


def test_live_http_cancellation_lifecycle(tmp_path, capture_jobs_logger):
    """Real localhost HTTP cancellation lifecycle test on Windows loopback with ephemeral port.

    Validates:
    - Worker starts only after durable 'running' state in SQLite.
    - Synchronizes on worker start.
    - Asserts 'running' and null result.
    - Concurrent GET /health while worker is blocked.
    - POST /cancel with bounded response time.
    - Worker observes cancellation signal and exits.
    - Worker attempts non-null late return, which is discarded.
    - Persisted state remains 'cancelled' with null result.
    - Future.result(timeout=...) completes and executor slot is cleaned up.
    - Logs contain state transitions and late discard without scenario data.
    """
    db_file = tmp_path / "live_http_cancel.sqlite3"
    scenario = _scenario()

    worker_started = threading.Event()
    worker_exited = threading.Event()
    cancel_signal_observed = threading.Event()

    def test_worker_hook(job_id: str, sc: dict[str, Any]):
        # Worker must start only after durable running in SQLite
        with sqlite3.connect(db_file, timeout=5) as con:
            con.row_factory = sqlite3.Row
            row = con.execute("SELECT status, result_json FROM jobs WHERE id=?", (job_id,)).fetchone()
            assert row is not None
            assert row["status"] == "running"
            assert row["result_json"] is None
        worker_started.set()

        # Wait on per-job cancel event with a bounded timeout
        event = manager.cancel_event(job_id)
        assert event is not None
        if event.wait(timeout=3.0):
            cancel_signal_observed.set()
        try:
            # Attempt a non-null late return
            return solve_exact(sc)
        finally:
            worker_exited.set()

    manager = JobManager(db_file, max_workers=2, timeout_seconds=10.0, _solver_hook=test_worker_hook)
    app = create_app(manager)

    # Bind ephemeral socket synchronously to eliminate loopback port conflicts
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_level="warning",
        timeout_graceful_shutdown=0.1,
        timeout_keep_alive=0,
    )
    sock = config.bind_socket()
    port = sock.getsockname()[1]
    server = uvicorn.Server(config)
    server_thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [sock]},
        daemon=True,
        name="live-uvicorn",
    )
    server_thread.start()

    try:
        # Bounded server-readiness coordination
        startup_deadline = time.monotonic() + 5.0
        while not server.started and time.monotonic() < startup_deadline:
            time.sleep(0.005)
        assert server.started, "Uvicorn failed to start within 5s"

        base_url = f"http://127.0.0.1:{port}"

        # 1. Submit job through real HTTP
        submit_req = urllib.request.Request(
            f"{base_url}/api/v1/jobs",
            data=json.dumps(scenario).encode("utf-8"),
            headers={"Content-Type": "application/json", "Connection": "close"},
            method="POST",
        )
        with urllib.request.urlopen(submit_req, timeout=3.0) as resp:
            assert resp.status == 202
            submit_data = json.loads(resp.read().decode())
        job_id = submit_data["id"]
        assert submit_data["status"] in {"queued", "running"}

        fut = manager.future(job_id)
        assert fut is not None

        # 2. Wait for worker to start and durably record 'running'
        assert worker_started.wait(timeout=3.0), "Worker did not signal start within bound"

        # 3. GET /jobs/{id} - assert running and null result
        get_req = urllib.request.Request(f"{base_url}/api/v1/jobs/{job_id}", headers={"Connection": "close"})
        with urllib.request.urlopen(get_req, timeout=3.0) as resp:
            assert resp.status == 200
            get_data = json.loads(resp.read().decode())
        assert get_data["status"] == "running"
        assert get_data["result"] is None

        # 4. Request health while worker is blocked
        health_req = urllib.request.Request(f"{base_url}/api/v1/health", headers={"Connection": "close"})
        with urllib.request.urlopen(health_req, timeout=3.0) as resp:
            assert resp.status == 200
            health_data = json.loads(resp.read().decode())
        assert health_data == {"status": "ok", "api_version": "v1"}

        # 5. POST /jobs/{id}/cancel - assert bounded response
        cancel_t0 = time.perf_counter()
        cancel_req = urllib.request.Request(
            f"{base_url}/api/v1/jobs/{job_id}/cancel",
            data=b"{}",
            headers={"Content-Type": "application/json", "Connection": "close"},
            method="POST",
        )
        with urllib.request.urlopen(cancel_req, timeout=3.0) as resp:
            assert resp.status == 200
            cancel_data = json.loads(resp.read().decode())
        cancel_duration = time.perf_counter() - cancel_t0
        assert cancel_duration < 1.0, f"Cancel response took {cancel_duration}s, exceeded 1.0s bound"
        assert cancel_data["status"] == "cancelled"
        assert cancel_data["result"] is None

        # 6. Verify cancellation signal observed by worker
        assert cancel_signal_observed.wait(timeout=3.0), "Worker did not observe cancel event"

        # 7. Verify worker exited
        assert worker_exited.wait(timeout=3.0), "Worker did not exit within bound"

        # 8. Wait on Future completion directly without sleep polling
        try:
            fut.result(timeout=3.0)
        except Exception:
            pass

        # 9. GET /jobs/{id} after worker exit - persisted cancelled, result is null (no late publish)
        with urllib.request.urlopen(get_req, timeout=3.0) as resp:
            assert resp.status == 200
            final_data = json.loads(resp.read().decode())
        assert final_data["status"] == "cancelled"
        assert final_data["result"] is None

        # 10. Verify slot / future cleanup directly
        assert manager._active_count() == 0, "Worker slot was not freed after worker exit"

        # 11. Verify logging assertions
        log_messages = [r.getMessage() for r in capture_jobs_logger.records]
        assert any(f"job state transition: job_id={job_id} status=running" in m for m in log_messages)
        assert any(f"job state transition: job_id={job_id} status=cancelled" in m for m in log_messages)
        assert any(f"late result discarded for cancelled job: job_id={job_id}" in m for m in log_messages)

        # Verify no scenario or secret data in captured log messages
        for msg in log_messages:
            assert "depot" not in msg
            assert "vehicles" not in msg
            assert "orders" not in msg
            assert "DO_NOT_ECHO" not in msg

    finally:
        server.should_exit = True
        server_thread.join(timeout=3.0)
        assert not server_thread.is_alive(), "Uvicorn server thread did not terminate within bound"
        sock.close()
        manager.close(wait=True)


def test_deterministic_cancel_wins_race(tmp_path):
    """Deterministic cancel-wins race regression: cancel takes lock before worker finishes late result."""
    db_file = tmp_path / "cancel_wins.sqlite3"
    scenario = _scenario()

    worker_at_finish_line = threading.Event()
    worker_may_return = threading.Event()
    worker_done = threading.Event()

    def race_worker(job_id: str, sc: dict[str, Any]):
        try:
            worker_at_finish_line.set()
            assert worker_may_return.wait(timeout=3.0)
            return solve_exact(sc)
        finally:
            worker_done.set()

    manager = JobManager(db_file, max_workers=1, timeout_seconds=5.0, _solver_hook=race_worker)
    try:
        sub = manager.submit(scenario)
        job_id = sub["id"]
        fut = manager.future(job_id)
        assert fut is not None

        assert worker_at_finish_line.wait(timeout=3.0)

        # Cancel arrives while worker is about to return
        cancel_resp = manager.cancel(job_id)
        assert cancel_resp["status"] == "cancelled"
        assert cancel_resp["result"] is None

        # Release worker to attempt return
        worker_may_return.set()
        assert worker_done.wait(timeout=3.0)

        # Wait on Future completion directly without sleep polling
        try:
            fut.result(timeout=3.0)
        except Exception:
            pass

        # Verify cancel won: persisted state is cancelled, result is None
        current = manager.get(job_id)
        assert current["status"] == "cancelled"
        assert current["result"] is None

        # Slot is released immediately
        assert manager._active_count() == 0
    finally:
        worker_may_return.set()
        manager.close(wait=True)


def test_deterministic_completion_wins_race(tmp_path):
    """Deterministic completion-wins race regression: worker completes before cancel is processed."""
    db_file = tmp_path / "completion_wins.sqlite3"
    scenario = _scenario()

    worker_completed = threading.Event()

    def immediate_worker(job_id: str, sc: dict[str, Any]):
        res = solve_exact(sc)
        worker_completed.set()
        return res

    manager = JobManager(db_file, max_workers=1, timeout_seconds=5.0, _solver_hook=immediate_worker)
    try:
        sub = manager.submit(scenario)
        job_id = sub["id"]
        fut = manager.future(job_id)
        assert fut is not None

        assert worker_completed.wait(timeout=3.0)

        # Wait for worker Future to complete without sleep polling
        fut.result(timeout=3.0)

        terminal_state = manager.get(job_id)
        assert terminal_state["status"] == "succeeded"
        assert terminal_state["result"] is not None

        # Attempt cancel after completion
        cancel_resp = manager.cancel(job_id)
        # Cancel must preserve the terminal state and not transition to cancelled
        assert cancel_resp["status"] == "succeeded"
        assert cancel_resp["result"] == terminal_state["result"]
    finally:
        manager.close(wait=True)


def test_duplicate_cancel_idempotence(tmp_path):
    """Duplicate-cancel regression: repeated cancel calls return existing record idempotently."""
    db_file = tmp_path / "duplicate_cancel.sqlite3"
    scenario = _scenario()

    worker_started = threading.Event()
    worker_release = threading.Event()

    def blocked_worker(job_id: str, sc: dict[str, Any]):
        worker_started.set()
        worker_release.wait(timeout=3.0)
        return solve_exact(sc)

    manager = JobManager(db_file, max_workers=1, timeout_seconds=5.0, _solver_hook=blocked_worker)
    try:
        sub = manager.submit(scenario)
        job_id = sub["id"]

        assert worker_started.wait(timeout=3.0)

        # First cancel
        c1 = manager.cancel(job_id)
        assert c1["status"] == "cancelled"
        assert c1["result"] is None

        # Duplicate cancel
        c2 = manager.cancel(job_id)
        assert c2["status"] == "cancelled"
        assert c2["result"] is None
        assert c2["id"] == c1["id"]

        # Triplicate cancel
        c3 = manager.cancel(job_id)
        assert c3["status"] == "cancelled"

        worker_release.set()
    finally:
        worker_release.set()
        manager.close(wait=True)


def test_restart_recovery_preserves_cancelled_and_fails_in_flight(tmp_path, capture_jobs_logger):
    """Restart recovery regression: in-flight jobs fail on restart while cancelled jobs are preserved."""
    db_file = tmp_path / "restart_recovery.sqlite3"

    # Pre-seed SQLite database with various states simulating a crashed backend
    with sqlite3.connect(db_file) as con:
        con.execute("""CREATE TABLE jobs (
            id TEXT PRIMARY KEY, status TEXT NOT NULL, scenario_json TEXT NOT NULL,
            result_json TEXT, error_json TEXT, submitted_at TEXT NOT NULL,
            started_at TEXT, completed_at TEXT, deadline_epoch REAL NOT NULL,
            execution_type TEXT NOT NULL, solver TEXT NOT NULL,
            settings_json TEXT, request_sha256 TEXT, runtime_json TEXT
        )""")
        con.execute("""CREATE TABLE runtime_lock (
            singleton INTEGER PRIMARY KEY CHECK(singleton=1), pid INTEGER NOT NULL,
            owner_token TEXT NOT NULL, started_at TEXT NOT NULL
        )""")
        now = "2026-10-29T00:00:00+00:00"
        dummy_sc = json.dumps({"depot": "D"})
        dummy_err = json.dumps({"code": "cancelled", "message": "job cancelled"})
        # 1. Already cancelled
        con.execute("INSERT INTO jobs VALUES ('j-cancelled', 'cancelled', ?, NULL, ?, ?, ?, ?, 9999999999.0, 'local_classical', 'exact_enumeration', NULL, 'req1', NULL)",
                    (dummy_sc, dummy_err, now, now, now))
        # 2. In-flight running
        con.execute("INSERT INTO jobs VALUES ('j-running', 'running', ?, NULL, NULL, ?, ?, NULL, 9999999999.0, 'local_classical', 'exact_enumeration', NULL, 'req2', NULL)",
                    (dummy_sc, now, now))
        # 3. In-flight queued
        con.execute("INSERT INTO jobs VALUES ('j-queued', 'queued', ?, NULL, NULL, ?, NULL, NULL, 9999999999.0, 'local_classical', 'exact_enumeration', NULL, 'req3', NULL)",
                    (dummy_sc, now))
        # 4. Already succeeded
        con.execute("INSERT INTO jobs VALUES ('j-succeeded', 'succeeded', ?, '{\"plan\":{}}', NULL, ?, ?, ?, 9999999999.0, 'local_classical', 'exact_enumeration', NULL, 'req4', NULL)",
                    (dummy_sc, now, now, now))

    # Instantiate new JobManager (triggers _recover_interrupted)
    manager = JobManager(db_file, max_workers=2)
    try:
        # Check preserved states
        j_cancelled = manager.get("j-cancelled")
        assert j_cancelled["status"] == "cancelled"
        assert j_cancelled["error"]["code"] == "cancelled"

        j_succeeded = manager.get("j-succeeded")
        assert j_succeeded["status"] == "succeeded"
        assert j_succeeded["result"] == {"plan": {}}

        # Check recovered states
        j_running = manager.get("j-running")
        assert j_running["status"] == "failed"
        assert j_running["error"]["code"] == "restart_interrupted"

        j_queued = manager.get("j-queued")
        assert j_queued["status"] == "failed"
        assert j_queued["error"]["code"] == "restart_interrupted"

        # Verify log recorded recovery count = 2
        log_messages = [r.getMessage() for r in capture_jobs_logger.records]
        assert any("recovered interrupted jobs on startup: count=2" in m for m in log_messages)
    finally:
        manager.close(wait=True)


@pytest.mark.parametrize("repetition", range(5))
def test_live_http_cancellation_repetition_for_flakiness(tmp_path, repetition):
    """Repeat the live HTTP cancellation test 5 times consecutively to verify stability."""
    db_file = tmp_path / f"repeat_{repetition}.sqlite3"
    scenario = _scenario()

    worker_started = threading.Event()
    worker_exited = threading.Event()
    cancel_event_observed = threading.Event()

    def worker_hook(job_id: str, sc: dict[str, Any]):
        worker_started.set()
        ev = manager.cancel_event(job_id)
        if ev and ev.wait(timeout=2.0):
            cancel_event_observed.set()
        try:
            return solve_exact(sc)
        finally:
            worker_exited.set()

    manager = JobManager(db_file, max_workers=2, timeout_seconds=5.0, _solver_hook=worker_hook)
    app = create_app(manager)

    # Bind ephemeral socket synchronously to eliminate loopback port conflicts
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_level="error",
        timeout_graceful_shutdown=0.1,
        timeout_keep_alive=0,
    )
    sock = config.bind_socket()
    port = sock.getsockname()[1]
    server = uvicorn.Server(config)
    server_thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [sock]},
        daemon=True,
    )
    server_thread.start()

    try:
        # Bounded server-readiness coordination
        startup_deadline = time.monotonic() + 5.0
        while not server.started and time.monotonic() < startup_deadline:
            time.sleep(0.005)
        assert server.started, "Uvicorn server failed to start within 5s"

        base_url = f"http://127.0.0.1:{port}"

        submit_req = urllib.request.Request(
            f"{base_url}/api/v1/jobs",
            data=json.dumps(scenario).encode("utf-8"),
            headers={"Content-Type": "application/json", "Connection": "close"},
            method="POST",
        )
        with urllib.request.urlopen(submit_req, timeout=3.0) as resp:
            job_id = json.loads(resp.read().decode())["id"]

        fut = manager.future(job_id)
        assert fut is not None

        assert worker_started.wait(timeout=2.0)

        cancel_req = urllib.request.Request(
            f"{base_url}/api/v1/jobs/{job_id}/cancel",
            data=b"{}",
            headers={"Content-Type": "application/json", "Connection": "close"},
            method="POST",
        )
        with urllib.request.urlopen(cancel_req, timeout=3.0) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode())
        assert res["status"] == "cancelled"

        assert cancel_event_observed.wait(timeout=2.0)
        assert worker_exited.wait(timeout=2.0)

        # Wait on Future completion directly without sleep polling
        try:
            fut.result(timeout=2.0)
        except Exception:
            pass

        get_req = urllib.request.Request(f"{base_url}/api/v1/jobs/{job_id}", headers={"Connection": "close"})
        with urllib.request.urlopen(get_req, timeout=3.0) as resp:
            final_job = json.loads(resp.read().decode())
        assert final_job["status"] == "cancelled"
        assert final_job["result"] is None

        # Verify active slots freed directly
        assert manager._active_count() == 0

    finally:
        server.should_exit = True
        server_thread.join(timeout=3.0)
        assert not server_thread.is_alive(), "Uvicorn server thread did not terminate within bound"
        sock.close()
        manager.close(wait=True)
