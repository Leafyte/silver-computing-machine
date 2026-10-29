"""Bounded real-loopback synthetic demonstration; no real credentials or vehicles."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TOKEN = "PUBLIC_NON_SECRET_ETA_SMOKE_DISPATCHER"
TOKENS = {"dispatcher": TOKEN, "researcher": "PUBLIC_NON_SECRET_ETA_SMOKE_RESEARCHER", "admin": "PUBLIC_NON_SECRET_ETA_SMOKE_ADMIN"}


def requests_for(name, *, switching_cost=0, changed=True, eta_request=None):
    from backend.domain import generate_routes
    vehicle = "V_" + name
    scenario = {"depot": "D", "vehicles": [{"id": vehicle, "capacity": 2}],
                "orders": [{"id": "A", "demand": 1, "window": [0, 12], "service": 0, "penalty": 10},
                           {"id": "B", "demand": 1, "window": [0, 100], "service": 0, "penalty": 0}],
                "travel": {"D": {"D": 0, "A": 10, "B": 1}, "A": {"D": 1, "A": 0, "B": 1}, "B": {"D": 1, "A": 1, "B": 0}}}
    route = next(r for r in generate_routes(scenario) if r["order_ids"] == ["A", "B"])
    plan = {"status": "feasible", "routes": [route], **{key: route[key] for key in ("objective", "travel_cost", "lateness_exposure")}}
    def weighted(s):
        return [{"id": "joint1", "weight": 1., "travel": deepcopy(s["travel"]), "source_label": "synthetic_overlay", "prediction_target": "travel_time"}]
    register = {"plan_id": name, "version": 1, "model_version": "fixture-v1", "as_of": "2026-10-29T10:00:00Z", "scenario": scenario,
                "weighted_scenarios": weighted(scenario), "plan": plan, "currency": "XTS", "operating_rate": 1., "switching_cost": switching_cost,
                "cooldown_seconds": 30, "mode": "synthetic_scenario"}
    updated = deepcopy(scenario)
    if changed:
        updated["travel"]["D"]["A"] = 20
    event = {"event_id": name + "_event", "plan_id": name, "plan_version": 1, "model_version": "fixture-v1", "as_of": "2026-10-29T10:01:00Z", "event_type": "delay",
             "scenario": updated, "weighted_scenarios": weighted(updated), "mode": "synthetic_scenario",
             "observed_state": [{"vehicle_id": vehicle, "position": "D", "position_type": "simulated", "elapsed_minutes": 1., "available_at": 1., "completed_prefix": []}]}
    if eta_request is not None:
        prediction_request = deepcopy(eta_request)
        prediction_request.update(origin={"id": "D"}, destination={"id": "A"}, prediction_time=event["as_of"])
        prediction_request["features"]["observed_at"] = {name: event["as_of"] for name in ("distance_km", "traffic_index")}
        register["model_version"] = event["model_version"] = prediction_request["model_version"]
        event["eta_update"] = {"schema_version": "india_eta.synthetic_update.v1", "arcs": [
            {"origin": "D", "destination": "A", "prediction_request": prediction_request,
             "residual_minutes_by_scenario": {"joint1": 0}}]}
        # The server computes this arc. The client supplies only a preserved baseline.
        event["scenario"] = deepcopy(scenario)
        event["weighted_scenarios"] = weighted(scenario)
    return register, event


def serve(db, port):
    from backend.api import create_app
    from backend.auth import AuthConfig
    import uvicorn
    app = create_app(auth_config=AuthConfig.from_mapping("token", TOKENS), eta_db_path=db)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        serve(sys.argv[2], int(sys.argv[3]))
        return
    runtime = ROOT / ".runtime" / ("india-http-" + str(time.time_ns()))
    runtime.mkdir(parents=True)
    db = runtime / "eta.sqlite3"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    report = {"schema_version": "india_eta.http_evidence.v1", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "execution_type": "real_localhost_http", "source": "synthetic_fixture", "live_activation": False, "qpu": False,
              "worker_budget": 2, "request_timeout_seconds": 5, "responses": {}, "request_count": 0}
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    request_lock = threading.Lock()

    def request(method, path, payload=None):
        with request_lock:
            report["request_count"] += 1
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + TOKEN})
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def start_server():
        log = (runtime / "server.log").open("ab")
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--serve", str(db), str(port)], cwd=ROOT,
                                  stdout=log, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        log.close()
        for _ in range(80):
            if child.poll() is not None:
                raise RuntimeError("loopback server exited; inspect saved local server.log")
            try:
                if request("GET", "/api/v1/health")[0] == 200:
                    return child
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(.1)
        child.terminate(); child.wait(timeout=5)
        raise RuntimeError("loopback server did not become ready within 8 seconds")

    def wait_event(event_id):
        for _ in range(100):
            code, job = request("GET", "/api/v1/eta/events/" + event_id)
            if code != 200:
                raise AssertionError(job)
            if job["status"] not in ("queued", "running"):
                return job
            time.sleep(.02)
        raise AssertionError("event exceeded bounded polling window")

    child = start_server()
    try:
        prediction_request = json.loads((ROOT / "fixtures/india_eta/prediction_request.json").read_text())
        prediction_request["features"]["distance_km"] = 1.875
        code, prediction = request("POST", "/api/v1/eta/predictions", prediction_request)
        assert code == 200 and prediction["predicted_duration_minutes"] == 20, prediction
        report["responses"]["prediction"] = prediction
        sla = {"schema_version": "india_eta.sla.request.v1", "prediction": prediction, "deadline": "2026-01-01T10:25:00+05:30",
               "late_penalty_per_minute": 6, "currency": "XTS", "scenario_source": "synthetic_fixture",
               "scenarios": [{"residual_minutes": value, "weight": 1/3} for value in (-5, 0, 10)]}
        code, risk = request("POST", "/api/v1/eta/sla", sla)
        assert code == 200 and abs(risk["expected_penalty"] - 10) < 1e-9, risk
        report["responses"]["sla_oracle"] = risk
        for name, expected, switching, changed in (("keep", "KEEP", 0, False), ("alert", "ALERT", 1000, True), ("propose", "PROPOSE_REPLAN", 0, True)):
            register, event = requests_for(name, switching_cost=switching, changed=changed,
                                          eta_request=prediction_request if changed else None)
            code, registered = request("POST", "/api/v1/eta/plans", register)
            assert code == 200, registered
            code, submitted = request("POST", "/api/v1/eta/events", event)
            assert code == 202, submitted
            job = wait_event(event["event_id"])
            assert job["status"] == "succeeded" and job["result"]["status"] == expected, job
            report["responses"][name] = job
            (ROOT / "fixtures/india_eta" / ("http_" + name + ".json")).write_text(json.dumps({"schema_version": "india_eta.demo.v1", "source": "synthetic_fixture", "register": register, "event": event, "observed_result": job}, indent=2) + "\n")
            if expected == "PROPOSE_REPLAN":
                proposed_event_request = deepcopy(event)
                proposal = job["result"]["proposal"]
                source = proposal["weighted_scenarios"][0]["eta_source_provenance"][0]
                assert source["predicted_duration_minutes"] == 20
                assert source["compatibility"]["decision"] == "scenario_only"
                assert source["model_sha256"] == prediction["model_sha256"]
                report["responses"]["eta_to_dispatch_provenance"] = source
                approval = {"base_version": 1, "event_id": event["event_id"], "model_version": event["model_version"], "context_hash": proposal["context_hash"]}
                path = "/api/v1/eta/proposals/" + job["result"]["proposal_id"] + "/approve"
                code, approved = request("POST", path, approval)
                assert code == 200 and approved["version"] == 2, approved
                report["responses"]["approval"] = approved
                code, stale = request("POST", path, approval)
                assert code == 409, stale
                report["responses"]["stale_approval"] = stale
        with ThreadPoolExecutor(max_workers=2) as pool:
            invalid = list(pool.map(lambda _: request("POST", "/api/v1/eta/predictions", {"unknown": "NON_SECRET_SENTINEL"})[0], range(20)))
        assert invalid == [422] * 20
        report["invalid_request_stress"] = {"requests": 20, "concurrency": 2, "statuses": invalid}
        # Ten independent tiny jobs, submitted at concurrency two; admission rejection is explicit.
        inputs = []
        for index in range(10):
            registration, event = requests_for("stress" + str(index), changed=False)
            assert request("POST", "/api/v1/eta/plans", registration)[0] == 200
            inputs.append(event)
        with ThreadPoolExecutor(max_workers=2) as pool:
            admissions = list(pool.map(lambda event: (event["event_id"], request("POST", "/api/v1/eta/events", event)[0]), inputs))
        admitted = [event_id for event_id, status in admissions if status == 202]
        outcomes = {event_id: wait_event(event_id)["status"] for event_id in admitted}
        assert all(status in (202, 429) for _, status in admissions)
        assert all(status == "succeeded" for status in outcomes.values())
        report["job_stress"] = {"submissions": 10, "concurrency": 2, "admission_statuses": admissions, "terminal_states": outcomes}
        child.terminate(); child.wait(timeout=5)
        child = start_server()
        code, manifest = request("GET", "/api/v1/eta/manifests/V_propose")
        assert code == 200 and manifest["plan_version"] == 2, manifest
        code, duplicate = request("POST", "/api/v1/eta/events", proposed_event_request)
        assert code == 202 and duplicate["status"] == "succeeded", duplicate
        report["responses"]["restart_manifest"] = manifest
        report["responses"]["duplicate_after_restart"] = {"event_id": duplicate["event_id"], "status": duplicate["status"], "request_sha256": duplicate["request_sha256"]}
        report["status"] = "passed"
    finally:
        if child.poll() is None:
            child.terminate(); child.wait(timeout=5)
        report["wall_seconds"] = time.perf_counter() - started_wall
        report["driver_process_cpu_seconds"] = time.process_time() - started_cpu
        report["server_cpu_seconds"] = "unavailable"
        try:
            import psutil
            report["driver_peak_memory_note"] = "rss_at_completion_not_peak"
            report["driver_rss_bytes"] = psutil.Process().memory_info().rss
        except ImportError:
            report["driver_rss_bytes"] = "unavailable"
        report["runtime_directory"] = str(runtime)
        output = ROOT / "evidence/india_eta/http-smoke.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "wall_seconds": report["wall_seconds"], "evidence": str(output), "server_stopped": child.poll() is not None}))


if __name__ == "__main__":
    main()
