"""Bounded, durable SQLite-backed exact-plan job manager.

This local V1 runner uses worker threads around the bounded exact solver. A
cancel or timeout prevents publication of a result; Python threads cannot be
forcibly stopped, so an already-running exact solve retains its bounded worker
slot until it returns. In-flight jobs are failed explicitly during recovery.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
import ctypes
import hashlib
import platform
import sys
from importlib.metadata import PackageNotFoundError, version as package_version
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .domain import SolveBudgetExceeded, solve_exact
from .qaoa_dispatch import solve_qaoa_dispatch
from .validator import validate_plan


class JobCapacityError(RuntimeError):
    """The bounded in-memory worker queue is full."""


logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    """Manage local dispatch jobs with a durable audit record.

    ``max_workers`` also bounds active computations. At most ``max_workers``
    total queued/running jobs are admitted, so no unbounded executor queue is
    created. The database path defaults to ``DEBUGLEAF_DB_PATH`` or a local
    ``.runtime/debugleaf.sqlite3`` file.
    """

    def __init__(self, db_path: str | Path | None = None, *, max_workers: int = 2,
                 timeout_seconds: float = 30.0, _solver_hook: Any = None):
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        configured = db_path or os.environ.get("DEBUGLEAF_DB_PATH", ".runtime/debugleaf.sqlite3")
        self.db_path = Path(configured)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self._solver_hook = _solver_hook
        self._owner_token = str(uuid.uuid4())
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="dispatch-job")
        self._futures: dict[str, Future[Any]] = {}
        self._cancelled: set[str] = set()
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._init_db()
        self._claim_runtime()
        self._recover_interrupted()

    def cancel_event(self, job_id: str) -> threading.Event | None:
        with self._lock:
            return self._cancel_events.get(job_id)

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled

    def future(self, job_id: str) -> Future[Any] | None:
        with self._lock:
            return self._futures.get(job_id)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, scenario_json TEXT NOT NULL,
                result_json TEXT, error_json TEXT, submitted_at TEXT NOT NULL,
                started_at TEXT, completed_at TEXT, deadline_epoch REAL NOT NULL,
                execution_type TEXT NOT NULL, solver TEXT NOT NULL,
                settings_json TEXT, request_sha256 TEXT, runtime_json TEXT
            )""")
            existing = {row["name"] for row in con.execute("PRAGMA table_info(jobs)")}
            for name, sql_type in (("settings_json", "TEXT"), ("request_sha256", "TEXT"),
                                   ("runtime_json", "TEXT")):
                if name not in existing:
                    con.execute(f"ALTER TABLE jobs ADD COLUMN {name} {sql_type}")
            con.execute("""CREATE TABLE IF NOT EXISTS runtime_lock (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1), pid INTEGER NOT NULL,
                owner_token TEXT NOT NULL, started_at TEXT NOT NULL
            )""")

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if os.name == "nt":
            # `os.kill(pid, 0)` requests rights Windows often denies even for a
            # dead PID. Query-limited handles distinguish no such process from
            # an existing process whose details cannot be read (fail closed).
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
            kernel.GetExitCodeProcess.restype = ctypes.c_int
            kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
            handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                try:
                    exit_code = ctypes.c_ulong()
                    if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                        return True
                    return exit_code.value == 259  # STILL_ACTIVE
                finally:
                    kernel.CloseHandle(ctypes.c_void_p(handle))
            return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER => PID absent
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except (OSError, OverflowError, ValueError):
            return False

    def _claim_runtime(self) -> None:
        """Claim one writer/worker process before recovery mutates job rows."""
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            current = con.execute("SELECT pid FROM runtime_lock WHERE singleton=1").fetchone()
            if current and self._pid_alive(int(current["pid"])):
                con.rollback()
                self._executor.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError("job database already has a live runtime owner")
            con.execute("INSERT OR REPLACE INTO runtime_lock(singleton,pid,owner_token,started_at) VALUES(1,?,?,?)",
                        (os.getpid(), self._owner_token, _now()))

    def _release_runtime(self) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM runtime_lock WHERE singleton=1 AND owner_token=?", (self._owner_token,))

    def _recover_interrupted(self) -> None:
        # A process restart cannot safely resume an in-memory solver call.
        with self._connect() as con:
            cur = con.execute("""UPDATE jobs SET status='failed',
                error_json=?, completed_at=? WHERE status IN ('queued','running','cancel_requested')""",
                (json.dumps({"code": "restart_interrupted", "message": "job interrupted by backend restart"}), _now()))
            if cur.rowcount > 0:
                logger.info("recovered interrupted jobs on startup: count=%d", cur.rowcount)

    def _active_count(self) -> int:
        # Timed-out/cancelled workers may still consume CPU. Keep them inside
        # the bound until the actual Future ends, even though their result is
        # already terminal from the API's perspective.
        return sum(not future.done() for future in self._futures.values())

    def capacity(self) -> dict[str, int]:
        """Return bounded worker capacity without exposing runtime internals."""
        with self._lock:
            active = self._active_count()
        return {"max_slots": self.max_workers, "active_slots": active,
                "available_slots": max(0, self.max_workers - active)}

    def submit(self, scenario: dict[str, Any]) -> dict[str, Any]:
        return self._submit(scenario, settings={"timeout_seconds": self.timeout_seconds,
                                               "max_workers": self.max_workers,
                                               "seed": None,
                                               "seed_status": "not_applicable_deterministic_exact_solver"},
                            solver="exact_enumeration", execution_type="local_classical",
                            runner=self._run)

    def submit_qaoa(self, scenario: dict[str, Any], *, seed: int, shots: int,
                    warm_start: bool, angle_grid: int) -> dict[str, Any]:
        """Submit a bounded local statevector QAOA attempt with exact fallback."""
        if (isinstance(seed, bool) or not isinstance(seed, int) or not -(2**31) <= seed < 2**31 or
                isinstance(shots, bool) or not isinstance(shots, int) or not 1 <= shots <= 512 or
                not isinstance(warm_start, bool) or isinstance(angle_grid, bool) or
                not isinstance(angle_grid, int) or not 2 <= angle_grid <= 10):
            raise ValueError("invalid QAOA settings")
        settings = {"timeout_seconds": self.timeout_seconds, "max_workers": self.max_workers,
                    "seed": seed, "shots": shots, "warm_start": warm_start,
                    "angle_grid": angle_grid, "execution_backend": "local_python_statevector",
                    "seed_status": "explicit_caller_supplied"}
        return self._submit(scenario, settings=settings,
                            solver="route_qaoa_with_exact_fallback", execution_type="local_qaoa_job",
                            runner=self._run_qaoa)

    def _submit(self, scenario: dict[str, Any], *, settings: dict[str, Any],
                solver: str, execution_type: str, runner: Any) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("job manager is shut down")
            if self._active_count() >= self.max_workers:
                raise JobCapacityError("job capacity reached")
            job_id = str(uuid.uuid4())
            submitted = _now()
            deadline = time.time() + self.timeout_seconds
            scenario_json = json.dumps(scenario, sort_keys=True, separators=(",", ":"))
            runtime = {"python": platform.python_version(), "platform": platform.platform()}
            try:
                runtime["fastapi"] = package_version("fastapi")
            except PackageNotFoundError:
                runtime["fastapi"] = "unavailable"
            request_material = json.dumps({"scenario": scenario, "settings": settings},
                                          sort_keys=True, separators=(",", ":"))
            request_sha256 = hashlib.sha256(request_material.encode()).hexdigest()
            with self._connect() as con:
                con.execute("""INSERT INTO jobs
                    (id,status,scenario_json,submitted_at,deadline_epoch,execution_type,solver,
                     settings_json,request_sha256,runtime_json)
                    VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (job_id, "queued", scenario_json, submitted, deadline, execution_type,
                     solver, json.dumps(settings), request_sha256, json.dumps(runtime)))
            future = self._executor.submit(runner, job_id, scenario, settings)
            self._futures[job_id] = future
            self._cancel_events[job_id] = threading.Event()
            future.add_done_callback(lambda done, key=job_id: self._future_completed(key, done))
            return self.get(job_id)  # queued or already running/completed

    def _future_completed(self, job_id: str, future: Future[Any]) -> None:
        # A future cancelled before _run starts has no worker finally block.
        with self._lock:
            if future.cancelled() and self._futures.get(job_id) is future:
                self._futures.pop(job_id, None)
                self._cancelled.discard(job_id)
                self._cancel_events.pop(job_id, None)
            if self._closed and not self._futures:
                self._release_runtime()

    def _set_state(self, job_id: str, status: str, *, result: Any = None,
                   error: dict[str, str] | None = None, started_at: str | None = None,
                   completed: bool = False) -> None:
        err_code = error.get("code") if error else None
        logger.info("job state transition: job_id=%s status=%s completed=%s error_code=%s",
                    job_id, status, completed, err_code)
        updates = ["status=?", "result_json=?", "error_json=?"]
        values: list[Any] = [status, json.dumps(result) if result is not None else None,
                             json.dumps(error) if error is not None else None]
        if started_at is not None:
            updates.append("started_at=?")
            values.append(started_at)
        if completed:
            updates.append("completed_at=?")
            values.append(_now())
        values.append(job_id)
        with self._connect() as con:
            con.execute(f"UPDATE jobs SET {', '.join(updates)} WHERE id=?", values)

    def _run(self, job_id: str, scenario: dict[str, Any], _settings: dict[str, Any] | None = None) -> None:
        solve_fn = (lambda: self._solver_hook(job_id, scenario)) if self._solver_hook is not None else (lambda: solve_exact(scenario))
        self._run_solver(job_id, scenario, solve_fn, qaoa=False)

    def _run_qaoa(self, job_id: str, scenario: dict[str, Any], settings: dict[str, Any]) -> None:
        self._run_solver(job_id, scenario,
                         lambda: solve_qaoa_dispatch(
                             scenario, seed=settings["seed"], shots=settings["shots"],
                             warm_start=settings["warm_start"], angle_grid=settings["angle_grid"]),
                         qaoa=True)

    def _run_solver(self, job_id: str, scenario: dict[str, Any], solve: Any, *, qaoa: bool) -> None:
        with self._lock:
            if job_id in self._cancelled:
                self._set_state(job_id, "cancelled", error={"code": "cancelled", "message": "job cancelled"}, completed=True)
                self._futures.pop(job_id, None)
                self._cancelled.discard(job_id)
                self._cancel_events.pop(job_id, None)
                return
            with self._connect() as con:
                row = con.execute("SELECT deadline_epoch,status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None or row["status"] != "queued":
                self._futures.pop(job_id, None)
                self._cancel_events.pop(job_id, None)
                return
            if time.time() >= row["deadline_epoch"]:
                self._set_state(job_id, "timed_out", error={"code": "timeout", "message": "job exceeded its time limit"}, completed=True)
                self._futures.pop(job_id, None)
                self._cancel_events.pop(job_id, None)
                return
            self._set_state(job_id, "running", started_at=_now())
        try:
            output = solve()
            if qaoa:
                if output.get("status") == "infeasible":
                    plan = output.get("decision", {}).get("plan")
                    report = None
                elif output.get("status") == "success":
                    decision = output.get("decision") or {}
                    plan = decision.get("plan")
                    report = validate_plan(scenario, plan) if isinstance(plan, dict) else None
                    if report and report.get("valid"):
                        decision["validation"] = report
                        output["decision"] = decision
                else:
                    plan, report = None, None
            else:
                plan = output
                report = validate_plan(scenario, plan) if plan.get("status") == "feasible" else None
            with self._lock:
                if job_id in self._cancelled:
                    logger.info("late result discarded for cancelled job: job_id=%s", job_id)
                    self._set_state(job_id, "cancelled", error={"code": "cancelled", "message": "job cancelled; result discarded"}, completed=True)
                else:
                    with self._connect() as con:
                        row = con.execute("SELECT deadline_epoch,status FROM jobs WHERE id=?", (job_id,)).fetchone()
                    if row is None or row["status"] in ("timed_out", "cancelled", "failed"):
                        return
                    if time.time() >= row["deadline_epoch"]:
                        logger.info("late result discarded for timed out job: job_id=%s", job_id)
                        self._set_state(job_id, "timed_out", error={"code": "timeout", "message": "job exceeded its time limit; result discarded"}, completed=True)
                    elif (qaoa and output.get("status") == "infeasible" and
                          isinstance(plan, dict) and plan.get("status") == "infeasible"):
                        self._set_state(job_id, "infeasible", result=output, completed=True)
                    elif not qaoa and isinstance(plan, dict) and plan.get("status") == "infeasible":
                        self._set_state(job_id, "infeasible", result=plan, completed=True)
                    elif not report or not report["valid"]:
                        self._set_state(job_id, "failed", error={"code": "validation_failure", "message": "independent plan validation failed"}, completed=True)
                    else:
                        result = output if qaoa else {"plan": plan, "validation": report}
                        self._set_state(job_id, "succeeded", result=result, completed=True)
        except SolveBudgetExceeded:
            # Exhaustion means neither optimum nor infeasibility was proven.
            # Keep the stable cause, and never publish an incumbent.
            with self._lock:
                current = self.get(job_id)
                if current and current["status"] not in {"cancelled", "timed_out", "failed", "infeasible", "succeeded"}:
                    self._set_state(job_id, "failed", error={"code": SolveBudgetExceeded.code,
                                                              "message": "dispatch search budget exceeded"}, completed=True)
        except Exception:
            # Do not persist exception text: it may contain input or environment details.
            with self._lock:
                current = self.get(job_id)
                if current and current["status"] not in {"cancelled", "timed_out", "failed", "infeasible", "succeeded"}:
                    self._set_state(job_id, "failed", error={"code": "solver_failure", "message": "dispatch solver failed"}, completed=True)
        finally:
            with self._lock:
                self._futures.pop(job_id, None)
                self._cancelled.discard(job_id)
                self._cancel_events.pop(job_id, None)
                if self._closed and not self._futures:
                    self._release_runtime()

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        # Timeout is reflected by polling even if solve_exact has not returned;
        # a late worker result is discarded in _run.
        if row["status"] in ("queued", "running") and time.time() >= row["deadline_epoch"]:
            with self._lock:
                with self._connect() as con:
                    con.execute("""UPDATE jobs SET status='timed_out',error_json=?,completed_at=?
                        WHERE id=? AND status IN ('queued','running') AND deadline_epoch<=?""",
                        (json.dumps({"code": "timeout", "message": "job exceeded its time limit"}),
                         _now(), job_id, time.time()))
                    row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return {"id": row["id"], "status": row["status"],
                "scenario": json.loads(row["scenario_json"]),
                "result": json.loads(row["result_json"]) if row["result_json"] else None,
                "error": json.loads(row["error_json"]) if row["error_json"] else None,
                "submitted_at": row["submitted_at"], "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "metadata": {"execution_type": row["execution_type"], "solver": row["solver"],
                             "settings": json.loads(row["settings_json"]) if row["settings_json"] else None,
                             "request_sha256": row["request_sha256"],
                             "runtime": json.loads(row["runtime_json"]) if row["runtime_json"] else None}}

    def list(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Return the newest durable jobs with their bounded request/result records."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        with self._connect() as con:
            rows = con.execute("SELECT id FROM jobs ORDER BY submitted_at DESC, id DESC LIMIT ?",
                               (limit,)).fetchall()
        return [job for row in rows if (job := self.get(row["id"])) is not None]

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            current = self.get(job_id)
            if current is None:
                return None
            if current["status"] in {"succeeded", "failed", "infeasible", "cancelled", "timed_out"}:
                return current
            self._cancelled.add(job_id)
            event = self._cancel_events.get(job_id)
            if event is not None:
                event.set()
            future = self._futures.get(job_id)
            cancelled_before_start = future.cancel() if future is not None else False
            self._set_state(job_id, "cancelled", error={"code": "cancelled", "message": "job cancelled; result will not be published"}, completed=True)
            if cancelled_before_start:
                self._futures.pop(job_id, None)
                self._cancelled.discard(job_id)
                self._cancel_events.pop(job_id, None)
            return self.get(job_id)

    def close(self, *, wait: bool = False) -> None:
        with self._lock:
            self._closed = True
            for job_id, future in list(self._futures.items()):
                event = self._cancel_events.get(job_id)
                if event is not None:
                    event.set()
                if future.cancel():
                    self._set_state(job_id, "cancelled", error={"code": "cancelled", "message": "backend shutting down"}, completed=True)
                    self._futures.pop(job_id, None)
                    self._cancelled.discard(job_id)
                    self._cancel_events.pop(job_id, None)
        self._executor.shutdown(wait=wait, cancel_futures=True)
        with self._lock:
            if not self._futures:
                self._release_runtime()
