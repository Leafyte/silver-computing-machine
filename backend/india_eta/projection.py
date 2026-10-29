"""Live-state or explicitly simulated projection of remaining route ETAs.

Completed work is retained as an immutable prefix. Remaining times begin from
the supplied vehicle position and availability clock; this module never
replays a live route from the depot.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from backend.domain import validate_scenario
from backend.validator import validate_plan

from .optimization import SCHEMA_VERSION, _validate_terms


def project_remaining(
    scenario: dict[str, Any], plan: dict[str, Any], vehicle_states: Sequence[Mapping[str, Any]],
    weighted_scenarios: Sequence[Mapping[str, Any]], *, currency: str = "XTS", operating_rate: float = 1.0,
) -> dict[str, Any]:
    """Recompute future stop timing from explicit state for every assigned vehicle.

    ``vehicle_states`` entries use ``vehicle_id, position, position_type,
    elapsed_minutes, available_at, completed_prefix``. Both time fields are
    absolute minutes on the source plan clock. The larger clock is used as
    the next available time.
    """
    _validate_terms(scenario, weighted_scenarios, currency, operating_rate)
    refreshed_plan = _recompute_plan_from_ordered_routes(scenario, plan)
    static_validation = validate_plan(scenario, refreshed_plan) if refreshed_plan is not None else {
        "valid": False, "feasible": False, "errors": ["plan routes could not be independently recomputed"],
        "recomputed": None}
    if not static_validation["valid"]:
        return {"schema_version": SCHEMA_VERSION, "status": "invalid_plan", "valid": False,
                "errors": static_validation["errors"], "plan_validation": static_validation}
    if not isinstance(vehicle_states, Sequence) or isinstance(vehicle_states, (str, bytes)):
        raise ValueError("vehicle_states must be a list of explicit state objects")
    state_by_vehicle: dict[str, Mapping[str, Any]] = {}
    for index, state in enumerate(vehicle_states):
        if not isinstance(state, Mapping):
            raise ValueError(f"vehicle_states[{index}] must be an object")
        vid = state.get("vehicle_id")
        if not isinstance(vid, str) or vid in state_by_vehicle:
            raise ValueError(f"vehicle_states[{index}].vehicle_id must be unique")
        state_by_vehicle[vid] = state
    route_by_vehicle = {route["vehicle_id"]: route for route in refreshed_plan["routes"]}
    saved_route_by_vehicle = {route["vehicle_id"]: route for route in plan["routes"]}
    missing = sorted(set(route_by_vehicle) - set(state_by_vehicle))
    if missing:
        return {"schema_version": SCHEMA_VERSION, "status": "unsupported_missing_vehicle_state", "valid": False,
                "missing_vehicle_ids": missing, "plan_validation": static_validation,
                "message": "live projection requires explicit actual or simulated state for each assigned vehicle"}
    known_locations = {scenario["depot"], *(o["id"] for o in scenario["orders"])}
    orders = {o["id"]: o for o in scenario["orders"]}
    completed_global: set[str] = set()
    normalized = {}
    for vid, route in route_by_vehicle.items():
        state = state_by_vehicle[vid]
        position, mode = state.get("position"), state.get("position_type")
        if not isinstance(position, str) or position not in known_locations:
            raise ValueError(f"vehicle_states.{vid}.position must be a known location or depot")
        if mode not in ("actual", "simulated"):
            raise ValueError(f"vehicle_states.{vid}.position_type must be 'actual' or 'simulated'")
        elapsed = _finite_time(state.get("elapsed_minutes"), f"vehicle_states.{vid}.elapsed_minutes")
        available = _finite_time(state.get("available_at"), f"vehicle_states.{vid}.available_at")
        clock = max(elapsed, available)
        prefix_rows = state.get("completed_prefix")
        if not isinstance(prefix_rows, Sequence) or isinstance(prefix_rows, (str, bytes)):
            raise ValueError(f"vehicle_states.{vid}.completed_prefix must be a list")
        prefix_ids, timestamps = [], []
        for pidx, row in enumerate(prefix_rows):
            if not isinstance(row, Mapping) or not isinstance(row.get("order_id"), str):
                raise ValueError(f"vehicle_states.{vid}.completed_prefix[{pidx}] requires order_id and completed_at")
            oid = row["order_id"]
            completed_at = _finite_time(row.get("completed_at"), f"vehicle_states.{vid}.completed_prefix[{pidx}].completed_at")
            if completed_at > clock + 1e-9:
                raise ValueError(f"vehicle_states.{vid}.completed_prefix[{pidx}].completed_at is after vehicle availability")
            if timestamps and completed_at < timestamps[-1]:
                raise ValueError(f"vehicle_states.{vid}.completed_prefix timestamps must be nondecreasing")
            timestamps.append(completed_at); prefix_ids.append(oid)
        if tuple(prefix_ids) != tuple(route["order_ids"][:len(prefix_ids)]):
            raise ValueError(f"vehicle_states.{vid}.completed_prefix must exactly match the immutable planned route prefix")
        if completed_global.intersection(prefix_ids):
            raise ValueError("a completed stop cannot appear in multiple vehicle prefixes")
        completed_global.update(prefix_ids)
        suffix = route["order_ids"][len(prefix_ids):]
        if position in orders and position not in prefix_ids and position not in suffix:
            # A current position may represent a non-order location only when
            # it is the depot; order locations must belong to this route's
            # completed/remaining state to avoid a fabricated arc.
            raise ValueError(f"vehicle_states.{vid}.position order location is not part of this route")
        if len(prefix_ids) and position == scenario["depot"]:
            raise ValueError(f"vehicle_states.{vid}.position cannot jump back to depot after completed stops")
        if position in route["order_ids"]:
            route_position = route["order_ids"].index(position)
            allowed = ({len(prefix_ids) - 1, len(prefix_ids)} if prefix_ids else {0})
            if route_position not in allowed:
                raise ValueError(f"vehicle_states.{vid}.position skips an uncompleted route stop")
        normalized[vid] = {"position": position, "position_type": mode, "elapsed_minutes": elapsed,
                           "available_at": available, "clock": clock, "prefix_ids": prefix_ids,
                           "prefix_rows": prefix_rows, "suffix": suffix}

    projected_routes = []
    for vid, route in route_by_vehicle.items():
        state = normalized[vid]
        saved_route = saved_route_by_vehicle[vid]
        existing = {row["order_id"]: row for row in saved_route.get("stops", []) if isinstance(row, Mapping)}
        completed_stops = []
        for entry in state["prefix_rows"]:
            oid = entry["order_id"]
            carried = dict(existing.get(oid, next(row for row in route["stops"] if row["order_id"] == oid)))
            carried.update({"completed": True, "immutable": True, "completed_at": float(entry["completed_at"])})
            completed_stops.append(carried)
        projected_routes.append({"vehicle_id": vid, "order_ids": list(route["order_ids"]),
                                 "position": state["position"], "position_type": state["position_type"],
                                 "elapsed_minutes": state["elapsed_minutes"],
                                 "available_at": state["available_at"],
                                 "remaining_stops": completed_stops, "future_stops": []})

    scenario_costs = []
    for row in weighted_scenarios:
        total_minutes = total_transport = total_penalty = 0.0
        scenario_hard_violation = False
        for projected in projected_routes:
            state = normalized[projected["vehicle_id"]]
            location, clock = state["position"], state["clock"]
            route_minutes = route_penalty = 0.0
            for oid in state["suffix"]:
                order = orders[oid]
                leg = float(row["travel"][location][oid]); route_minutes += leg
                arrival = clock + leg
                start = max(arrival, float(order["window"][0]))
                lateness = max(0.0, start - float(order["window"][1]))
                departure = start + float(order["service"])
                hard_violation = bool(order.get("hard_window", False) and lateness > 0)
                scenario_hard_violation |= hard_violation
                projected["future_stops"].append({"order_id": oid, "scenario_id": row["id"],
                    "arrival": arrival, "service_start": start, "departure": departure,
                    "waiting": start - arrival, "service_minutes": float(order["service"]),
                    "lateness": lateness, "penalty_exposure": lateness * float(order["penalty"]),
                    "hard_window_violation": hard_violation})
                route_penalty += lateness * float(order["penalty"])
                clock, location = departure, oid
            route_minutes += float(row["travel"][location][scenario["depot"]])
            total_minutes += route_minutes; total_penalty += route_penalty
        # Apply caller-declared units; this does not establish calibrated India costs.
        total_transport = float(operating_rate) * total_minutes
        scenario_costs.append({"id": row["id"], "weight": float(row["weight"]),
                               "remaining_travel_minutes": total_minutes,
                               "transport_cost_proxy": total_transport,
                               "currency": currency,
                               "late_penalty_cost": total_penalty,
                               "total_cost_proxy": total_transport + total_penalty,
                               "hard_window_violation": scenario_hard_violation})
    # Preserve scenario-specific future ETA rows separately; don't collapse
    # joint outcomes into a falsely precise point prediction.
    expected = {"remaining_travel_minutes": sum(x["weight"] * x["remaining_travel_minutes"] for x in scenario_costs),
                "currency": currency,
                "transport_cost_proxy": sum(x["weight"] * x["transport_cost_proxy"] for x in scenario_costs),
                "late_penalty_cost": sum(x["weight"] * x["late_penalty_cost"] for x in scenario_costs),
                "total_cost_proxy": sum(x["weight"] * x["total_cost_proxy"] for x in scenario_costs),
                "hard_window_violation_frequency": sum(x["weight"] for x in scenario_costs if x["hard_window_violation"])}
    evaluation = {"valid": True,
                  "expected_cost": {"amount": expected["transport_cost_proxy"] + expected["late_penalty_cost"],
                                    "currency": currency,
                                    "transport_cost": expected["transport_cost_proxy"],
                                    "late_penalty_cost": expected["late_penalty_cost"]},
                  "scenario_costs": scenario_costs,
                  "hard_window_violation_frequency": expected["hard_window_violation_frequency"]}
    projection_mode = ("synthetic_scenario" if all(value["position_type"] == "simulated"
                                                   for value in normalized.values()) else "actual_state")
    return {"schema_version": SCHEMA_VERSION, "status": "projected", "valid": True,
            "projection_mode": projection_mode, "evaluation": evaluation,
            "state_mode": {vid: value["position_type"] for vid, value in normalized.items()},
            "plan": {"routes": projected_routes},
            "prefix_validation": {"valid": True, "completed_stops": sorted(completed_global),
                                  "immutable_prefixes": {vid: list(value["prefix_ids"]) for vid, value in normalized.items()}},
            "expected_remaining_cost": expected, "scenario_costs": scenario_costs,
            "scenario_sources": [{"id": row["id"], "source_label": row["source_label"],
                                  "prediction_target": row["prediction_target"]} for row in weighted_scenarios],
            "source_note": "caller-labeled travel-time matrices; no India full-delivery model is used as a travel arc",
            "scenario_specific_future_stops": True,
            "validator_scope": "state-aware prefix and route projection; backend.validator.validate_plan is depot-static and intentionally not applied to live state"}


def _finite_time(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be a finite non-negative plan-clock minute value")
    return float(value)


def _recompute_plan_from_ordered_routes(scenario: dict[str, Any], plan: Any) -> dict[str, Any] | None:
    """Refresh nominal route metrics from source and route order, ignoring stale cached ETAs."""
    scenario_check = validate_scenario(scenario)
    if not scenario_check["valid"] or not isinstance(plan, Mapping) or plan.get("status") != "feasible":
        return None
    source_routes = plan.get("routes")
    if not isinstance(source_routes, list):
        return None
    orders = {row["id"]: row for row in scenario["orders"]}
    vehicles = {row["id"]: row for row in scenario["vehicles"]}
    rebuilt = []
    for source in source_routes:
        if not isinstance(source, Mapping):
            return None
        vid, ids = source.get("vehicle_id"), source.get("order_ids")
        if (not isinstance(vid, str) or vid not in vehicles or not isinstance(ids, list) or not ids or
                any(not isinstance(oid, str) or oid not in orders for oid in ids)):
            return None
        location, clock, travel_minutes, penalty = scenario["depot"], 0.0, 0.0, 0.0
        stops = []
        for oid in ids:
            order = orders[oid]
            leg = float(scenario["travel"][location][oid]); travel_minutes += leg
            arrival = clock + leg
            start = max(arrival, float(order["window"][0]))
            lateness = max(0.0, start - float(order["window"][1]))
            if order.get("hard_window", False) and lateness > 0:
                # The validator below returns a precise, independent hard-window error.
                pass
            exposure = lateness * float(order["penalty"]); penalty += exposure
            departure = start + float(order["service"])
            stops.append({"order_id": oid, "arrival": arrival, "service_start": start,
                          "departure": departure, "lateness": lateness,
                          "penalty_exposure": exposure})
            location, clock = oid, departure
        back = float(scenario["travel"][location][scenario["depot"]]); travel_minutes += back
        rebuilt.append({"vehicle_id": vid, "order_ids": list(ids), "stops": stops,
                        "return_time": clock + back, "travel_cost": travel_minutes,
                        "lateness_exposure": penalty, "objective": travel_minutes + penalty})
    return {"status": "feasible", "routes": rebuilt,
            "objective": sum(row["objective"] for row in rebuilt),
            "travel_cost": sum(row["travel_cost"] for row in rebuilt),
            "lateness_exposure": sum(row["lateness_exposure"] for row in rebuilt)}
