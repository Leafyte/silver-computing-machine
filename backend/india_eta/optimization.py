"""Versioned India ETA coefficient adapter over the unchanged route QUBO.

The adapter consumes complete, joint travel-time matrices. It never trains or
changes the ETA model and recomputes stochastic costs from source matrices.
"""
from __future__ import annotations

from itertools import product
import math
from typing import Any, Mapping, Sequence

from backend.domain import generate_routes, validate_scenario
from backend.qaoa import run_qaoa
from backend.qubo import build_route_qubo, enumerate_qubo, ising_energy, qubo_to_ising
from backend.qaoa_dispatch import _plan_for_bits
from backend.validator import validate_plan
from experiments_v1_1.quantum_optimization import lp_augmented_relaxation


SCHEMA_VERSION = "debugleaf.india_eta.optimization.v1"
MAX_JOINT_SCENARIOS = 100
MAX_QUANTUM_VARIABLES = 8


def evaluate_plan(
    scenario: dict[str, Any], plan: dict[str, Any], weighted_scenarios: Sequence[Mapping[str, Any]], *,
    currency: str = "XTS", operating_rate: float = 1.0,
) -> dict[str, Any]:
    """Recompute deterministic validity and weighted transport/SLA cost.

    Hard-window violation frequency is reported separately and is not treated
    as a probabilistic feasibility guarantee.
    """
    _validate_terms(scenario, weighted_scenarios, currency, operating_rate)
    deterministic = validate_plan(scenario, plan)
    if not deterministic["valid"]:
        return {"schema_version": SCHEMA_VERSION, "valid": False,
                "errors": deterministic["errors"], "deterministic_validation": deterministic,
                "expected_cost": None}
    expected, by_scenario = _independent_plan_cost(scenario, plan, weighted_scenarios,
                                                   currency, float(operating_rate))
    return {"schema_version": SCHEMA_VERSION, "valid": True,
            "deterministic_validation": deterministic, "expected_cost": expected,
            "scenario_costs": by_scenario,
            "hard_window_violation_frequency": _weighted_hard_violation(scenario, plan, weighted_scenarios),
            "provenance": {"cost_source": "independent_recomputation_from_weighted_scenario_matrices",
                           "currency": currency, "scenario_count": len(weighted_scenarios),
                           "scenario_sources": [{"id": row["id"],
                               "source_label": _scenario_source(row)} for row in weighted_scenarios],
                           "india_calibration_inferred": False}}


def optimize_dispatch(
    scenario: dict[str, Any], weighted_scenarios: Sequence[Mapping[str, Any]], *,
    currency: str = "XTS", operating_rate: float = 1.0,
    locked_prefixes: Mapping[str, Sequence[str]] | None = None,
    vehicle_states: Sequence[Mapping[str, Any]] | None = None,
    quantum: bool = True, seed: int = 7,
) -> dict[str, Any]:
    """Select a dispatch from all feasible route candidates using common costs.

    Quantum sampling is attempted only for complete candidate pools up to
    eight variables. Larger route pools go directly to a bounded classical
    selector; candidates are never truncated to fit a circuit.
    """
    _validate_terms(scenario, weighted_scenarios, currency, operating_rate)
    if not isinstance(quantum, bool):
        raise ValueError("quantum must be a boolean")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    state_by_vehicle = _validate_vehicle_states(scenario, vehicle_states)
    state_prefixes = {vid: tuple(row["prefix_ids"]) for vid, row in state_by_vehicle.items() if row["prefix_ids"]}
    prefixes = _validate_prefixes(scenario, locked_prefixes if locked_prefixes is not None else state_prefixes)
    for vid, state in state_by_vehicle.items():
        if prefixes.get(vid, ()) != tuple(state["prefix_ids"]):
            raise ValueError(f"locked_prefixes for {vid} must match vehicle_states.completed_prefix")
    all_routes = generate_routes(scenario)
    routes = _apply_prefixes(scenario, all_routes, prefixes, state_by_vehicle or None)
    # Coverage remains over every source order. Completed prefixes stay in a
    # vehicle-locked route candidate so the exact-cover QUBO cannot discard
    # historical work when another vehicle handles the remaining orders.
    required = [o["id"] for o in scenario["orders"]]

    # Route candidate costs are independently calculated from every supplied
    # joint scenario, rather than reusing cached nominal route objective fields.
    candidate_rows = []
    route_by_id: dict[str, dict[str, Any]] = {}
    scenario_route_costs: dict[str, dict[str, float]] = {}
    for index, route in enumerate(routes):
        rid = f"{route['vehicle_id']}:{index}:" + ",".join(route["order_ids"])
        state = state_by_vehicle.get(route["vehicle_id"])
        scenario_costs = [_route_cost(scenario, route, row["travel"], float(operating_rate), state)
                           for row in weighted_scenarios]
        expected_cost = sum(float(row["weight"]) * cost for row, cost in zip(weighted_scenarios, scenario_costs))
        route_by_id[rid] = route
        scenario_route_costs[rid] = {str(row["id"]): cost for row, cost in zip(weighted_scenarios, scenario_costs)}
        candidate_rows.append({"id": rid, "vehicle": route["vehicle_id"],
                               "stops": list(route["order_ids"]),
                               "cost": expected_cost})

    if not required:
        plan = {"status": "feasible", "routes": [], "objective": 0.0,
                "travel_cost": 0.0, "lateness_exposure": 0.0}
        validation = validate_plan(scenario, plan)
        return _result(plan, validation, evaluate_plan(scenario, plan, weighted_scenarios,
                        currency=currency, operating_rate=operating_rate), None, None,
                        {"used": False, "reason": None}, "classical_trivial")
    if not candidate_rows:
        raise ValueError("infeasible: no route candidates cover the remaining orders")

    # Exact selector is the oracle/fallback over precisely the QUBO candidate
    # pool, with one route per vehicle and exact remaining-order coverage.
    classical = _exact_candidate_selection(candidate_rows, required, route_by_id)
    qubo = build_route_qubo(candidate_rows, required)
    exhaustive = _audit_qubo(qubo) if qubo.n <= 12 else None
    lp_steps = 5000
    lp_vector = lp_augmented_relaxation(qubo, steps=lp_steps) if qubo.n <= MAX_QUANTUM_VARIABLES else None
    quantum_result: dict[str, Any] = {"status": "not_run", "execution_type": None,
                                     "raw_samples": [], "repaired_samples": [],
                                     "selected_raw_plan": None, "selected_raw_validation": None,
                                     "lp_warm_start": None}
    fallback = {"used": False, "reason": None}
    selected_plan = None
    execution_type = "local_classical_exact_candidate_selection"
    if quantum and qubo.n <= MAX_QUANTUM_VARIABLES:
        try:
            quantum_run = run_qaoa(qubo, seed=seed, shots=256, warm_start=True,
                                   relaxation=lp_vector, angle_grid=8)
            raw_samples = []
            repaired_samples = []
            for row in quantum_run.raw_samples:
                bits = row["bits"]
                raw_plan = _plan_for_bits([route_by_id[c["id"]] for c in candidate_rows], bits)
                raw_validation = validate_plan(scenario, raw_plan)
                raw_state_validation = _evaluate_remaining_plan(scenario, raw_plan, weighted_scenarios,
                                                                 state_by_vehicle, currency, float(operating_rate))
                raw_samples.append({"bits": bits, "energy": row["energy"],
                                    "qubo_validation": row["validation"],
                                    "plan": raw_plan, "independent_validation": raw_validation,
                                    "state_validation": raw_state_validation})
                repair = row.get("repair")
                if repair is not None:
                    repaired_plan = _plan_for_bits([route_by_id[c["id"]] for c in candidate_rows], repair["bits"])
                    repaired_samples.append({"source_bits": bits, "bits": repair["bits"],
                                             "edits": repair["edits"], "plan": repaired_plan,
                                             "qubo_validation": repair["validation"],
                                             "independent_validation": validate_plan(scenario, repaired_plan),
                                             "state_validation": _evaluate_remaining_plan(scenario, repaired_plan,
                                                 weighted_scenarios, state_by_vehicle, currency, float(operating_rate))})
            eligible = [r for r in raw_samples if r["qubo_validation"]["feasible"] and
                            r["independent_validation"]["valid"] and r["state_validation"]["valid"]]
            chosen = min(eligible, key=lambda r: (sum(float(candidate_rows[i]["cost"])
                        for i, bit in enumerate(r["bits"]) if bit == "1"), r["bits"])) if eligible else None
            quantum_result = {"status": "success" if chosen else "no_valid_raw_sample",
                              "execution_type": quantum_run.backend,
                              "mode": quantum_run.mode, "angles": quantum_run.angles,
                              "expected_qubo_energy": quantum_run.expected_qubo_energy,
                              "raw_counts": quantum_run.raw_counts, "raw_samples": raw_samples,
                              "repaired_samples": repaired_samples,
                              "selected_raw_plan": chosen["plan"] if chosen else None,
                              "selected_raw_validation": chosen["independent_validation"] if chosen else None,
                              "lp_warm_start": {"method": "v1.1_augmented_lagrangian_set_partitioning_relaxation",
                                  "steps": lp_steps, "vector": [float(x) for x in lp_vector],
                                  "mapping": "candidate_rows order equals QUBO variable order",
                                  "clamp_applied_by_qaoa": [0.05, 0.95]}}
            if chosen:
                selected_plan = chosen["plan"]
                execution_type = "local_python_statevector_simulator"
            else:
                fallback = {"used": True, "reason": "no raw QAOA sample passed QUBO and independent plan validation"}
        except Exception as exc:
            # Exception messages may contain source values; keep response stable and non-secret.
            quantum_result = {**quantum_result, "status": "failed", "failure_type": type(exc).__name__}
            fallback = {"used": True, "reason": "local QAOA execution or validation failed"}
    elif quantum:
        fallback = {"used": True, "reason": f"complete candidate pool has {qubo.n} variables; local limit is {MAX_QUANTUM_VARIABLES}; no candidates were truncated"}
        quantum_result["status"] = "unsupported_candidate_pool_size"
    if selected_plan is None:
        if classical is None:
            raise ValueError("infeasible: no feasible exact cover exists in bounded route candidate pool")
        selected_plan = classical["plan"]
        if quantum and not fallback["used"]:
            fallback = {"used": True, "reason": "quantum path not requested; selected matched classical reference"}
        execution_type = "local_classical_exact_candidate_selection"

    validation = validate_plan(scenario, selected_plan)
    stochastic = evaluate_plan(scenario, selected_plan, weighted_scenarios,
                               currency=currency, operating_rate=operating_rate)
    state_validation = _evaluate_remaining_plan(scenario, selected_plan, weighted_scenarios,
                                                 state_by_vehicle, currency, float(operating_rate))
    if not validation["valid"] or not stochastic["valid"] or not state_validation["valid"]:
        raise ValueError("validation_failure: selected plan failed independent validation")
    coeff = {"schema_version": "debugleaf.india_eta.coefficients.v1", "currency": currency,
             "operating_rate_per_travel_minute": float(operating_rate),
             "candidate_costs": [{"id": row["id"], "vehicle": row["vehicle"],
                  "stops": list(row["stops"]), "expected_cost": row["cost"],
                  "scenario_costs": scenario_route_costs[row["id"]]} for row in candidate_rows],
             "qubo": {"variable_order": list(qubo.variable_ids), "linear": list(qubo.linear),
                      "quadratic": {f"{i},{j}": v for (i, j), v in qubo.quadratic.items()},
                      "offset": qubo.offset, "coverage_penalty": qubo.coverage_penalty,
                      "vehicle_penalty": qubo.vehicle_penalty,
                      "exhaustive_audit": exhaustive},
             "locked_prefixes": {k: list(v) for k, v in prefixes.items()},
             "scenario_ids_weights": [{"id": row["id"], "weight": float(row["weight"]),
                 "source_label": _scenario_source(row)} for row in weighted_scenarios],
             "source_note": "scenario labels are caller-supplied; no India provenance or calibration is inferred"}
    result = _result(selected_plan, validation, stochastic, coeff, quantum_result,
                   fallback, execution_type, classical)
    if state_by_vehicle:
        result["expected_cost"] = state_validation["expected_cost"]
        result["remaining_cost_validation"] = state_validation
        result["state_validation"] = {"valid": True,
            "checked_vehicle_ids": sorted(state_by_vehicle),
            "position_clock_rule": "start from each explicit position at max(elapsed_minutes, available_at)",
            "historical_prefix_excluded_from_expected_cost": True}
    return result


def _result(plan: dict[str, Any], validation: dict[str, Any], stochastic: dict[str, Any],
            coefficients: dict[str, Any] | None, quantum: dict[str, Any] | None,
            fallback: dict[str, Any], execution_type: str,
            classical: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "plan": plan, "validation": validation,
            "expected_cost": stochastic.get("expected_cost"), "stochastic_validation": stochastic,
            "coefficients": coefficients,
            "quantum": quantum or {"status": "not_applicable", "raw_samples": [], "repaired_samples": []},
            "classical_reference": ({"plan": classical["plan"], "expected_cost": classical["cost"],
                 "candidate_pool_exact": True} if classical else None),
            "fallback": fallback, "execution_type": execution_type}


def _validate_terms(scenario: Any, scenarios: Sequence[Mapping[str, Any]], currency: Any, rate: Any) -> None:
    checked = validate_scenario(scenario)
    if not checked["valid"]:
        raise ValueError("invalid scenario: " + "; ".join(checked["errors"]))
    if (not isinstance(currency, str) or len(currency) != 3 or not currency.isascii() or
            currency.upper() != currency or not currency.isalpha()):
        raise ValueError("currency must be an uppercase three-letter code")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate < 0:
        raise ValueError("operating_rate must be finite and non-negative")
    if not isinstance(scenarios, Sequence) or isinstance(scenarios, (str, bytes)) or not 1 <= len(scenarios) <= MAX_JOINT_SCENARIOS:
        raise ValueError(f"weighted_scenarios must contain 1..{MAX_JOINT_SCENARIOS} complete scenarios")
    seen, total = set(), 0.0
    locations = {scenario["depot"], *(o["id"] for o in scenario["orders"])}
    for index, row in enumerate(scenarios):
        if not isinstance(row, Mapping) or set(("id", "weight", "travel")) - set(row):
            raise ValueError(f"weighted_scenarios[{index}] requires id, weight, and travel")
        ident, weight, travel = row["id"], row["weight"], row["travel"]
        label = row.get("source_label", row.get("source"))
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"weighted_scenarios[{index}].source_label is required (for example 'synthetic_overlay')")
        target = row.get("prediction_target")
        lowered = label.lower().replace("-", "_").replace(" ", "_")
        if target != "travel_time" or any(token in lowered for token in ("full_delivery", "delivery_duration", "whole_delivery")):
            raise ValueError(f"weighted_scenarios[{index}] cannot use a whole-delivery prediction as a travel arc")
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError(f"weighted_scenarios[{index}].id must be unique and non-empty")
        seen.add(ident)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"weighted_scenarios[{index}].weight must be finite and positive")
        if not isinstance(travel, Mapping):
            raise ValueError(f"weighted_scenarios[{index}].travel must be a complete duration matrix")
        for origin in locations:
            if not isinstance(travel.get(origin), Mapping):
                raise ValueError(f"weighted_scenarios[{index}].travel.{origin} missing")
            for dest in locations:
                value = travel[origin].get(dest)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"weighted_scenarios[{index}].travel.{origin}.{dest} must be finite non-negative minutes")
        total += float(weight)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("weighted_scenarios weights must sum to 1")


def _scenario_source(row: Mapping[str, Any]) -> str:
    label = row.get("source_label", row.get("source"))
    return label.strip() if isinstance(label, str) and label.strip() else "caller_not_labeled"


def _validate_vehicle_states(scenario: dict[str, Any], states: Sequence[Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    if states is None:
        return {}
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        raise ValueError("vehicle_states must be a list")
    vehicles = {v["id"] for v in scenario["vehicles"]}
    orders = {o["id"] for o in scenario["orders"]}
    locations = orders | {scenario["depot"]}
    output: dict[str, dict[str, Any]] = {}
    completed_all: set[str] = set()
    for index, row in enumerate(states):
        if not isinstance(row, Mapping):
            raise ValueError(f"vehicle_states[{index}] must be an object")
        vid = row.get("vehicle_id")
        if not isinstance(vid, str) or vid not in vehicles or vid in output:
            raise ValueError(f"vehicle_states[{index}].vehicle_id must be unique and known")
        position, mode = row.get("position"), row.get("position_type")
        if not isinstance(position, str) or position not in locations or mode not in ("actual", "simulated"):
            raise ValueError(f"vehicle_states[{vid}] requires a known position and actual/simulated position_type")
        elapsed, available = row.get("elapsed_minutes"), row.get("available_at")
        for field, value in (("elapsed_minutes", elapsed), ("available_at", available)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"vehicle_states[{vid}].{field} must be finite non-negative minutes")
        clock = max(float(elapsed), float(available))
        raw_prefix = row.get("completed_prefix")
        if not isinstance(raw_prefix, Sequence) or isinstance(raw_prefix, (str, bytes)):
            raise ValueError(f"vehicle_states[{vid}].completed_prefix must be a list")
        prefix, rows, previous = [], [], -math.inf
        for pindex, item in enumerate(raw_prefix):
            if (not isinstance(item, Mapping) or not isinstance(item.get("order_id"), str) or
                    item.get("order_id") not in orders):
                raise ValueError(f"vehicle_states[{vid}].completed_prefix[{pindex}] has an unknown order")
            oid, when = item["order_id"], item.get("completed_at")
            if oid in prefix or oid in completed_all:
                raise ValueError("completed prefix stops must be unique across the fleet")
            if isinstance(when, bool) or not isinstance(when, (int, float)) or not math.isfinite(when) or when < previous or when > clock:
                raise ValueError(f"vehicle_states[{vid}].completed_prefix timestamps must be ordered and no later than available_at")
            prefix.append(oid); rows.append({"order_id": oid, "completed_at": float(when)})
            previous = float(when)
        completed_all.update(prefix)
        output[vid] = {"position": position, "position_type": mode,
                       "elapsed_minutes": float(elapsed), "available_at": float(available),
                       "clock": clock, "prefix_ids": tuple(prefix), "prefix_rows": rows}
    return output


def _route_cost(scenario: dict[str, Any], route: dict[str, Any], travel: Mapping[str, Any], rate: float,
                state: Mapping[str, Any] | None = None) -> float:
    orders = {o["id"]: o for o in scenario["orders"]}
    if state:
        location, clock = state["position"], state["clock"]
        future_ids = route["order_ids"][len(state["prefix_ids"]):]
    else:
        location, clock, future_ids = scenario["depot"], 0.0, route["order_ids"]
    travel_minutes, penalty_cost = 0.0, 0.0
    for oid in future_ids:
        order = orders[oid]
        leg = float(travel[location][oid]); travel_minutes += leg
        start = max(clock + leg, float(order["window"][0]))
        penalty_cost += max(0.0, start - float(order["window"][1])) * float(order["penalty"])
        clock, location = start + float(order["service"]), oid
    travel_minutes += float(travel[location][scenario["depot"]])
    return rate * travel_minutes + penalty_cost


def _independent_plan_cost(scenario: dict[str, Any], plan: dict[str, Any], scenarios: Sequence[Mapping[str, Any]],
                           currency: str, rate: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    orders = {o["id"]: o for o in scenario["orders"]}
    costs = []
    for row in scenarios:
        travel_minutes = penalty = 0.0
        for route in plan["routes"]:
            loc, clock = scenario["depot"], 0.0
            for oid in route["order_ids"]:
                order = orders[oid]; leg = float(row["travel"][loc][oid])
                travel_minutes += leg
                start = max(clock + leg, float(order["window"][0]))
                penalty += max(0.0, start - float(order["window"][1])) * float(order["penalty"])
                clock, loc = start + float(order["service"]), oid
            travel_minutes += float(row["travel"][loc][scenario["depot"]])
        transport = rate * travel_minutes
        costs.append({"id": row["id"], "weight": float(row["weight"]), "currency": currency,
                      "travel_minutes": travel_minutes, "transport_cost": transport,
                      "late_penalty_cost": penalty, "total_cost": transport + penalty})
    result = {"currency": currency, "expected_travel_minutes": sum(x["weight"] * x["travel_minutes"] for x in costs),
              "expected_transport_cost": sum(x["weight"] * x["transport_cost"] for x in costs),
              "expected_late_penalty_cost": sum(x["weight"] * x["late_penalty_cost"] for x in costs),
              "total": sum(x["weight"] * x["total_cost"] for x in costs),
              "cost_recomputed_from_source_matrices": True}
    return result, costs


def _evaluate_remaining_plan(scenario: dict[str, Any], plan: dict[str, Any],
                             scenarios: Sequence[Mapping[str, Any]], states: Mapping[str, Mapping[str, Any]],
                             currency: str, rate: float) -> dict[str, Any]:
    """Independent state-aware repricing from raw joint matrices and vehicle state."""
    if not states:
        expected, rows = _independent_plan_cost(scenario, plan, scenarios, currency, rate)
        amount = expected["total"]
        return {"valid": True, "expected_cost": {"amount": amount, "currency": currency,
                "transport_cost": expected["expected_transport_cost"],
                "late_penalty_cost": expected["expected_late_penalty_cost"]},
                "scenario_costs": rows, "hard_window_violation_frequency":
                _weighted_hard_violation(scenario, plan, scenarios),
                "historical_prefix_excluded": False}
    orders = {o["id"]: o for o in scenario["orders"]}
    seen_vehicles = set()
    for route in plan["routes"]:
        vid = route["vehicle_id"]
        state = states.get(vid)
        if state is None:
            return {"valid": False, "errors": [f"missing current state for vehicle {vid}"]}
        if vid in seen_vehicles or tuple(route["order_ids"][:len(state["prefix_ids"])]) != tuple(state["prefix_ids"]):
            return {"valid": False, "errors": [f"selected route violates locked prefix for vehicle {vid}"]}
        if state["position"] != scenario["depot"] and state["position"] not in route["order_ids"]:
            return {"valid": False, "errors": [f"selected route does not contain current position for vehicle {vid}"]}
        seen_vehicles.add(vid)
    scenario_rows = []
    expected_minutes = expected_transport = expected_penalty = 0.0
    hard_frequency = 0.0
    for row in scenarios:
        world_minutes = world_penalty = 0.0
        world_hard = False
        for route in plan["routes"]:
            state = states[route["vehicle_id"]]
            location, clock = state["position"], state["clock"]
            remaining = route["order_ids"][len(state["prefix_ids"]):]
            for oid in remaining:
                order = orders[oid]
                leg = float(row["travel"][location][oid]); world_minutes += leg
                start = max(clock + leg, float(order["window"][0]))
                lateness = max(0.0, start - float(order["window"][1]))
                world_penalty += lateness * float(order["penalty"])
                world_hard |= bool(order.get("hard_window", False) and lateness > 0)
                clock, location = start + float(order["service"]), oid
            world_minutes += float(row["travel"][location][scenario["depot"]])
        transport = rate * world_minutes
        total = transport + world_penalty
        scenario_rows.append({"id": row["id"], "weight": float(row["weight"]),
                              "currency": currency, "remaining_travel_minutes": world_minutes,
                              "transport_cost": transport, "late_penalty_cost": world_penalty,
                              "total_cost": total, "hard_window_violation": world_hard})
        weight = float(row["weight"])
        expected_minutes += weight * world_minutes
        expected_transport += weight * transport
        expected_penalty += weight * world_penalty
        hard_frequency += weight * int(world_hard)
    return {"valid": True, "expected_cost": {"amount": expected_transport + expected_penalty,
            "currency": currency, "transport_cost": expected_transport,
            "late_penalty_cost": expected_penalty, "remaining_travel_minutes": expected_minutes},
            "scenario_costs": scenario_rows, "hard_window_violation_frequency": hard_frequency,
            "historical_prefix_excluded": True,
            "cost_source": "independently recomputed from explicit vehicle state and weighted source matrices"}


def _weighted_hard_violation(scenario: dict[str, Any], plan: dict[str, Any], scenarios: Sequence[Mapping[str, Any]]) -> float:
    orders = {o["id"]: o for o in scenario["orders"]}
    frequency = 0.0
    for row in scenarios:
        violated = False
        for route in plan["routes"]:
            loc, clock = scenario["depot"], 0.0
            for oid in route["order_ids"]:
                order = orders[oid]; start = max(clock + float(row["travel"][loc][oid]), float(order["window"][0]))
                violated |= bool(order.get("hard_window", False) and start > float(order["window"][1]))
                clock, loc = start + float(order["service"]), oid
        frequency += float(row["weight"]) * int(violated)
    return frequency


def _validate_prefixes(scenario: dict[str, Any], prefixes: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    if not isinstance(prefixes, Mapping):
        raise ValueError("locked_prefixes must map vehicle ids to completed stop prefixes")
    vehicles = {v["id"] for v in scenario["vehicles"]}; orders = {o["id"] for o in scenario["orders"]}
    output, seen = {}, set()
    for vid, prefix in prefixes.items():
        if not isinstance(vid, str) or vid not in vehicles or not isinstance(prefix, Sequence) or isinstance(prefix, (str, bytes)):
            raise ValueError("locked_prefixes contains an unknown vehicle or invalid prefix")
        values = tuple(prefix)
        if (any(not isinstance(oid, str) or oid not in orders for oid in values) or
                len(set(values)) != len(values) or seen.intersection(values)):
            raise ValueError("locked prefixes must contain distinct known stops, without cross-vehicle duplicates")
        seen.update(values); output[vid] = values
    return output


def _apply_prefixes(scenario: dict[str, Any], routes: Sequence[dict[str, Any]],
                    prefixes: Mapping[str, tuple[str, ...]],
                    states: Mapping[str, Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    completed = {oid for p in prefixes.values() for oid in p}
    result = []
    for route in routes:
        vid, stops = route["vehicle_id"], tuple(route["order_ids"])
        if states is not None and vid not in states:
            continue
        prefix = prefixes.get(vid, ())
        if prefix and stops[:len(prefix)] != prefix:
            continue
        if not prefix and completed.intersection(stops):
            continue
        if prefix and len(stops) < len(prefix):
            continue
        state = states.get(vid) if states else None
        if state and state["position"] != scenario["depot"]:
            if state["position"] not in stops:
                continue
            position_index = stops.index(state["position"])
            allowed_indices = ({len(prefix) - 1, len(prefix)} if prefix else {0})
            if position_index not in allowed_indices:
                continue
        result.append(route)
    return result


def _exact_candidate_selection(candidates: Sequence[Mapping[str, Any]], required: Sequence[str],
                               route_by_id: Mapping[str, dict[str, Any]]) -> dict[str, Any] | None:
    by_stop = {s: [] for s in required}
    for row in candidates:
        for stop in row["stops"]:
            if stop in by_stop:
                by_stop[stop].append(row)
    if any(not values for values in by_stop.values()):
        return None
    best, work, limit = None, 0, 100_000
    def visit(covered: frozenset[str], vehicles: frozenset[str], chosen: tuple[Mapping[str, Any], ...], cost: float) -> None:
        nonlocal best, work
        work += 1
        if work > limit:
            raise ValueError("solver_budget_exceeded: exact candidate selection reached 100000 nodes")
        if len(covered) == len(required):
            if best is None or cost < best[0]: best = (cost, chosen)
            return
        if best is not None and cost >= best[0]: return
        remaining = [s for s in required if s not in covered]
        target = min(remaining, key=lambda s: sum(c["vehicle"] not in vehicles and not (set(c["stops"]) & covered) for c in by_stop[s]))
        for row in by_stop[target]:
            subset = set(row["stops"])
            if row["vehicle"] not in vehicles and not (subset & covered):
                visit(covered | subset, vehicles | {row["vehicle"]}, chosen + (row,), cost + float(row["cost"]))
    visit(frozenset(), frozenset(), (), 0.0)
    if best is None: return None
    routes = [route_by_id[row["id"]] for row in best[1]]
    plan = {"status": "feasible", "routes": routes,
            "objective": sum(float(r["objective"]) for r in routes),
            "travel_cost": sum(float(r["travel_cost"]) for r in routes),
            "lateness_exposure": sum(float(r["lateness_exposure"]) for r in routes)}
    return {"plan": plan, "cost": best[0], "search_nodes": work}


def _audit_qubo(qubo: Any) -> dict[str, Any]:
    rows = enumerate_qubo(qubo)
    offset, h, j = qubo_to_ising(qubo)
    for bits in product((0, 1), repeat=qubo.n):
        converted = offset + sum(h[i] * (1 - 2 * bits[i]) for i in range(qubo.n)) + sum(
            val * (1 - 2 * bits[i]) * (1 - 2 * bits[k]) for (i, k), val in j.items())
        if not math.isclose(qubo.energy(bits), converted, rel_tol=1e-9, abs_tol=1e-9) or not math.isclose(qubo.energy(bits), ising_energy(qubo, bits), rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError("QUBO/Ising exhaustive equivalence failure")
    return {"assignment_count": len(rows), "all_energy_checks_passed": True,
            "qubo_to_ising_offset": offset, "qubo_to_ising_linear": list(h),
            "qubo_to_ising_quadratic": {f"{i},{j}": v for (i, j), v in j.items()},
            "feasible_assignment_count": sum(bool(row["feasible"]) for row in rows),
            "minimum_feasible_business_cost": min((row["business_cost"] for row in rows if row["feasible"]), default=None)}
