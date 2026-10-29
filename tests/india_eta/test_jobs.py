import json
import sqlite3
import threading
import time

import pytest

from backend.india_eta.errors import EtaError
from backend.india_eta.jobs import EtaJobs


def event(event_id="event1"):
    return {"event_id": event_id, "plan_id": "plan1", "plan_version": 1,
            "model_version": "fixture-v1", "as_of": "2026-10-29T10:01:00Z",
            "observed_state": {}, "scenario": {}, "weighted_scenarios": [], "event_type": "eta_update"}


class Controller:
    def __init__(self):
        self.cancelled = set()
        self.calls = 0

    def process_event(self, payload, *, cancel_check=None):
        self.calls += 1
        return {"status": "KEEP", "event_id": payload["event_id"]}

    def cancel_event(self, event_id):
        self.cancelled.add(event_id)


def terminal(jobs, event_id="event1"):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        result = jobs.get(event_id)
        if result["status"] not in ("queued", "running"):
            return result
        time.sleep(.005)
    raise AssertionError("bounded ETA job did not terminate")


def test_durable_event_success_and_changed_duplicate_rejected(tmp_path):
    controller = Controller()
    jobs = EtaJobs(controller, tmp_path / "eta.sqlite")
    try:
        jobs.submit(event())
        result = terminal(jobs)
        assert result["status"] == "succeeded"
        assert result["result"]["status"] == "KEEP"
        assert jobs.submit(event()) == result
        assert controller.calls == 1
        with pytest.raises(EtaError) as exc:
            jobs.submit({**event(), "plan_version": 2})
        assert exc.value.code == "duplicate_event_conflict"
        assert "request_json" not in result
        assert result["metadata"]["dependency_versions"]["fastapi"] != "unavailable"
        assert len(result["metadata"]["implementation_sha256"]) == 64
    finally:
        jobs.close()
    reopened = EtaJobs(controller, tmp_path / "eta.sqlite")
    try:
        assert reopened.get("event1") == result
    finally:
        reopened.close()


@pytest.mark.parametrize("operation", ["cancel", "timeout"])
def test_cancel_timeout_hold_capacity_and_suppress_late_output(tmp_path, operation):
    entered, release = threading.Event(), threading.Event()
    controller = Controller()
    def worker(payload, *, cancel_check=None):
        entered.set()
        assert release.wait(2)
        return {"status": "PROPOSE_REPLAN", "proposal_id": "late"}
    jobs = EtaJobs(controller, tmp_path / "eta.sqlite", max_workers=1,
                   timeout_seconds=.05 if operation == "timeout" else 2, worker=worker)
    try:
        jobs.submit(event())
        assert entered.wait(1)
        if operation == "cancel":
            result = jobs.cancel("event1")
        else:
            time.sleep(.06)
            result = jobs.get("event1")
        assert result["status"] in ("cancelled", "timed_out")
        with pytest.raises(EtaError) as exc:
            jobs.submit(event("event2"))
        assert exc.value.code == "capacity_reached"
        release.set()
    finally:
        release.set()
        jobs.close()
    assert jobs.get("event1")["result"] is None
    assert "event1" in controller.cancelled


def test_restart_recovers_unfinished_job_and_invalidates_proposal(tmp_path):
    controller = Controller()
    path = tmp_path / "eta.sqlite"
    jobs = EtaJobs(controller, path)
    jobs.close()
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO eta_async_jobs VALUES (?,?,?,?,?,?,?,?,?,?)", ("crashed", "hash", "{}", "running", None, None, "start", None, time.time()+100, "{}"))
    jobs = EtaJobs(controller, path)
    try:
        assert jobs.get("crashed")["error"]["code"] == "restart_interrupted"
        assert jobs.get("crashed")["status"] == "failed"
        assert "crashed" in controller.cancelled
    finally:
        jobs.close()


def test_worker_errors_are_stable_and_unknown_fields_not_persisted(tmp_path):
    sentinel = "TEST_NON_SECRET_SENTINEL"
    def worker(payload, *, cancel_check=None):
        raise RuntimeError(sentinel)
    jobs = EtaJobs(Controller(), tmp_path / "eta.sqlite", worker=worker)
    try:
        with pytest.raises(EtaError):
            jobs.submit({**event(), "unknown": sentinel})
        assert jobs.get("event1") is None
        jobs.submit(event())
        result = terminal(jobs)
        assert result["error"]["code"] == "event_failure"
        assert sentinel not in json.dumps(result)
    finally:
        jobs.close()


def test_second_live_runtime_rejected(tmp_path):
    jobs = EtaJobs(Controller(), tmp_path / "eta.sqlite")
    try:
        with pytest.raises(EtaError) as exc:
            EtaJobs(Controller(), tmp_path / "eta.sqlite")
        assert exc.value.code == "capacity_reached"
    finally:
        jobs.close()


def test_history_budget_rejects_new_work_without_losing_idempotency(tmp_path):
    jobs = EtaJobs(Controller(), tmp_path / "eta.sqlite", max_retained_jobs=1)
    try:
        jobs.submit(event())
        result = terminal(jobs)
        assert jobs.submit(event()) == result
        with pytest.raises(EtaError) as exc:
            jobs.submit(event("event2"))
        assert exc.value.code == "retention_capacity_reached"
        assert jobs.get("event2") is None
    finally:
        jobs.close()


def test_nonfinite_worker_result_is_failed_without_publication(tmp_path):
    controller = Controller()
    jobs = EtaJobs(controller, tmp_path / "eta.sqlite", worker=lambda *a, **kw: {"cost": float("nan")})
    try:
        jobs.submit(event())
        result = terminal(jobs)
        assert result["status"] == "failed" and result["result"] is None
        assert "event1" in controller.cancelled
    finally:
        jobs.close()
