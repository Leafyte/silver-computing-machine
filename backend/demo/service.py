"""Core application service and route publication coordinator for the phone demo.

Orchestrates:
  - Current driver state & GeoJSON route retrieval
  - Congestion admission & segment delay overlays
  - Candidate scoring & deterministic tie-break selection
  - Automatic publication policy (same vehicle, remaining stops covered)
  - Full audit logging of all decisions
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from .routes import RouteCatalog, RouteCandidate
from .store import DemoStore


class DemoCoordinatorService:
    """Coordinates driver actions, incident admissions, and route publication."""

    def __init__(self, store: DemoStore, catalog: RouteCatalog | None = None) -> None:
        self.store = store
        self.catalog = catalog or RouteCatalog()
        self._ensure_default_state()

    def _ensure_default_state(self) -> None:
        """Seed default routes for demo drivers."""
        # Vehicle 1 -> Driver 1
        d1 = self.catalog.get_default_route("vehicle_1")
        if d1:
            self.store.init_driver_state_if_missing("driver_1", "vehicle_1", d1.route_id, d1.corridor_id)
        # Vehicle 2 -> Driver 2
        d2 = self.catalog.get_default_route("vehicle_2")
        if d2:
            self.store.init_driver_state_if_missing("driver_2", "vehicle_2", d2.route_id, d2.corridor_id)

    # -------------------------------------------------------------------------
    # Driver State & Navigation
    # -------------------------------------------------------------------------

    def get_driver_state(self, driver_id: str) -> dict[str, Any] | None:
        state = self.store.get_driver_state(driver_id)
        if not state:
            return None

        route_id = state["route_id"]
        candidate = self.catalog.get_candidate(route_id)
        overlays = self.store.get_active_overlays()
        current_duration = self.catalog.score_route(candidate, overlays) if candidate else 0.0

        return {
            "driver_id": driver_id,
            "vehicle_id": state["vehicle_id"],
            "route_id": route_id,
            "corridor_id": state["corridor_id"],
            "route_version": state["route_version"],
            "manifest_revision": state["manifest_revision"],
            "status": state["status"],
            "estimated_duration_seconds": current_duration,
            "estimated_distance_metres": candidate.total_distance_metres if candidate else 0.0,
            "active_overlays": overlays,
            "geojson": candidate.to_geojson() if candidate else None,
            "last_ack_version": state.get("last_ack_version", 0),
            "last_ack_at": state.get("last_ack_at"),
            "updated_at": state["updated_at"],
        }

    # -------------------------------------------------------------------------
    # Planning Route Requests
    # -------------------------------------------------------------------------

    def request_plan_route(
        self, driver_id: str, request_id: str, base_route_version: int
    ) -> dict[str, Any]:
        """Driver taps 'Plan route'. Runs deterministic candidate evaluation."""
        state = self.store.get_driver_state(driver_id)
        if not state:
            raise ValueError(f"Unknown driver {driver_id}")

        job_id = f"job_plan_{uuid.uuid4().hex[:12]}"
        job = self.store.create_or_get_job(job_id, driver_id, request_id)
        if job["status"] == "completed":
            return job

        # Evaluate alternatives
        result = self._evaluate_and_decide(driver_id, state)
        self.store.complete_job(job_id, result["decision"], result)
        self.store.log_audit(
            event_type="plan_route_evaluated",
            principal=driver_id,
            payload={"job_id": job_id, "request_id": request_id, "result": result},
        )
        return {"job_id": job_id, "status": "completed", "outcome": result["decision"], "result": result}

    # -------------------------------------------------------------------------
    # Congestion Reports
    # -------------------------------------------------------------------------

    def report_congestion(
        self,
        driver_id: str,
        event_id: str,
        base_route_version: int,
        severity: str = "heavy",
        note: str = "",
    ) -> dict[str, Any]:
        """Driver taps 'Report Congestion'. Admits report and evaluates auto-publication."""
        state = self.store.get_driver_state(driver_id)
        if not state:
            raise ValueError(f"Unknown driver {driver_id}")

        # Map severity to synthetic delay: moderate = +3 min (180s), heavy = +8 min (480s)
        delay_seconds = 480.0 if severity.lower() == "heavy" else 180.0
        # TTL: 5 minutes (300 seconds)
        now_dt = datetime.now(timezone.utc)
        expires_at = (now_dt + timedelta(seconds=300)).isoformat()

        # Identify driver's active road segment
        current_route = self.catalog.get_candidate(state["route_id"])
        target_segment = current_route.segments[0].segment_id if current_route and current_route.segments else "seg_unknown"

        # Persist report
        self.store.save_congestion_report(
            event_id=event_id,
            driver_id=driver_id,
            vehicle_id=state["vehicle_id"],
            segment_id=target_segment,
            severity=severity,
            delay_seconds=delay_seconds,
            expires_at=expires_at,
            note=note,
        )

        job_id = f"job_cong_{uuid.uuid4().hex[:12]}"
        job = self.store.create_or_get_job(job_id, driver_id, event_id)

        # Run decision evaluation under updated congestion
        decision_result = self._evaluate_and_decide(driver_id, state)
        self.store.complete_job(job_id, decision_result["decision"], decision_result)

        self.store.log_audit(
            event_type="congestion_reported_and_evaluated",
            principal=driver_id,
            payload={
                "event_id": event_id,
                "job_id": job_id,
                "severity": severity,
                "delay_seconds": delay_seconds,
                "decision": decision_result,
            },
        )

        return {
            "status": "accepted",
            "event_id": event_id,
            "job_id": job_id,
            "delay_applied_seconds": delay_seconds,
            "segment_id": target_segment,
            "decision": decision_result["decision"],
            "published_new_route": decision_result.get("published", False),
            "current_route_version": decision_result["route_version"],
        }

    # -------------------------------------------------------------------------
    # Evaluation & Auto-Publication Engine
    # -------------------------------------------------------------------------

    def _evaluate_and_decide(self, driver_id: str, state: Mapping[str, Any]) -> dict[str, Any]:
        """Compare current route against alternatives under active congestion."""
        vehicle_id = state["vehicle_id"]
        current_candidate = self.catalog.get_candidate(state["route_id"])
        if not current_candidate:
            return {"decision": "ALERT", "reason": "No active route found", "route_version": state["route_version"]}

        overlays = self.store.get_active_overlays()
        current_cost = self.catalog.score_route(current_candidate, overlays)

        # Retrieve alternatives for same vehicle
        alternatives = self.catalog.get_alternatives(vehicle_id)
        if len(alternatives) <= 1:
            return {
                "decision": "KEEP",
                "reason": "Single road corridor configured",
                "cost_seconds": current_cost,
                "route_version": state["route_version"],
            }

        # Score all alternatives
        scored: list[tuple[float, RouteCandidate]] = []
        for alt in alternatives:
            cost = self.catalog.score_route(alt, overlays)
            scored.append((cost, alt))

        # Sort by cost (seconds), deterministic tie-break by route_id
        scored.sort(key=lambda item: (item[0], item[1].route_id))
        best_cost, best_candidate = scored[0]

        # Check switching rule:
        # Must reduce travel time by at least 120s (2 min) AND 10%
        time_saved = current_cost - best_cost
        relative_improvement = time_saved / max(current_cost, 1.0)

        if best_candidate.route_id != current_candidate.route_id and time_saved >= 120.0 and relative_improvement >= 0.10:
            # AUTO-PUBLISH: Publish new route version
            new_state = self.store.update_driver_route(
                driver_id=driver_id,
                route_id=best_candidate.route_id,
                corridor_id=best_candidate.corridor_id,
            )
            return {
                "decision": "PUBLISHED_NEW_ROUTE",
                "published": True,
                "previous_route": current_candidate.name,
                "previous_cost_seconds": current_cost,
                "new_route": best_candidate.name,
                "new_cost_seconds": best_cost,
                "time_saved_seconds": time_saved,
                "route_version": new_state["route_version"],
                "manifest_revision": new_state["manifest_revision"],
            }
        else:
            return {
                "decision": "KEEP",
                "published": False,
                "reason": f"Current route retained (cost={current_cost:.0f}s vs alternate={best_cost:.0f}s)",
                "route_version": state["route_version"],
                "current_cost_seconds": current_cost,
            }

    def acknowledge(self, driver_id: str, version: int) -> None:
        self.store.record_acknowledgement(driver_id, version)
        self.store.log_audit(
            event_type="driver_acknowledgement",
            principal=driver_id,
            payload={"version": version},
        )
