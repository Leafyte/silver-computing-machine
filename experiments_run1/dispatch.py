"""Bounded risk-conditioned QUBO/QAOA dispatch for RUN 1 evaluations.

This module keeps predictive feature scaling and the native benchmark adapter
outside the optimizer. Route candidates must already have been evaluated by
the source-specific generator; selected routes are always rechecked by the
caller-provided native evaluator. No QAOA call occurs in model training.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from itertools import permutations, product
import json
import math
import random
import time
from typing import Any, Callable, Mapping, Sequence

from backend.qubo import build_route_qubo, decode_bitstring, validate_selection
from experiments_v1_1.artifact_contract import (
    CONTRACT_SCHEMA_VERSION,
    HybridDispatchArtifact,
    validate_hybrid_artifact,
)
from experiments_v1_1.quantum_optimization import (
    execute_qaoa_benchmark,
    fast_qaoa_state,
    lp_augmented_relaxation,
    precompute_hamiltonian_spectrum,
    solve_classical_exact,
)


MAX_ROUTE_VARIABLES = 8
DEFAULT_SOFT_CUTOFF_FRACTION = 0.90
SYNTHETIC_TRANSPORT_RATE_SCU = 0.50
SYNTHETIC_DUTY_RATE_SCU = 0.25
DEFAULT_SYNTHETIC_PENALTY_RATE_SCU = 1.0


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"native evaluator returned unsupported value type {type(value).__name__}")


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    out = float(value)
    if not math.isfinite(out) or out < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return out


def _route_id(route: Sequence[int]) -> str:
    return "route:" + ">".join(str(int(value)) for value in route)


def _native_route_candidate(
    instance: Any, route: Sequence[int], vehicle: str, expected_breach_severity: float,
    *, penalty_rate: float = DEFAULT_SYNTHETIC_PENALTY_RATE_SCU,
) -> dict[str, Any]:
    """Convert one native-feasible route into a leakage-safe dispatch row."""
    result = instance.evaluate_route(route)
    if not result.feasible:
        raise ValueError(f"route {_route_id(route)} is not native-feasible: {result.status}")
    nodes = tuple(int(node) for node in route)
    windows = instance.time_windows
    horizon = max(float(due) - float(ready) for ready, due in windows)
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("native source has no positive time-window horizon")
    soft_slacks = [float(due) * DEFAULT_SOFT_CUTOFF_FRACTION +
                   float(ready) * (1.0 - DEFAULT_SOFT_CUTOFF_FRACTION) - float(arrival)
                   for (ready, due), arrival in zip((windows[node] for node in nodes), result.arrivals)]
    travel = _finite_nonnegative(result.travel_duration, "native travel_duration")
    service = _finite_nonnegative(result.service_duration, "native service_duration")
    waiting = _finite_nonnegative(result.waiting_duration, "native waiting_duration")
    duty = travel + service + waiting
    return {
        "id": _route_id(nodes), "vehicle": vehicle, "stops": nodes, "route": nodes,
        "features": (min(soft_slacks) / horizon, float(result.duration) / horizon),
        "forecast_cost": SYNTHETIC_TRANSPORT_RATE_SCU * travel + SYNTHETIC_DUTY_RATE_SCU * duty,
        "expected_breach_severity": _finite_nonnegative(expected_breach_severity, "train_only_expected_breach_severity"),
        "penalty_rate": float(penalty_rate), "severity_estimate_split": "train",
        "transport_cost_scu": SYNTHETIC_TRANSPORT_RATE_SCU * travel,
        "labor_cost_scu": SYNTHETIC_DUTY_RATE_SCU * duty,
        "native_forecast_features": {
            "duration": float(result.duration),
            "travel_duration": travel,
            "service_duration": service,
            "waiting_duration": waiting,
            "optimized_departure_time": result.optimized_departure_time,
        },
    }


def _route_order_proposals(nodes: Sequence[int]) -> list[tuple[int, ...]]:
    """Small deterministic native-evaluated ordering set; never reads BKS."""
    route = tuple(nodes)
    proposed = [route, tuple(reversed(route))]
    if len(route) > 2:
        proposed.extend(route[offset:] + route[:offset] for offset in range(1, len(route)))
        proposed.extend(route[:index] + tuple(reversed(route[index:stop])) + route[stop:]
                        for index in range(len(route) - 1) for stop in range(index + 2, len(route) + 1))
    return list(dict.fromkeys(proposed))


def _best_native_order(instance: Any, nodes: Sequence[int], vehicle: str,
                       severity: float, penalty_rate: float) -> dict[str, Any] | None:
    feasible = []
    for route in _route_order_proposals(nodes):
        try:
            feasible.append(_native_route_candidate(instance, route, vehicle, severity,
                                                   penalty_rate=penalty_rate))
        except ValueError:
            continue
    if not feasible:
        return None
    return min(feasible, key=lambda row: (float(row["forecast_cost"]), tuple(row["route"])))


def _native_order_variants(instance: Any, nodes: Sequence[int], vehicle: str,
                           severity: float, penalty_rate: float, *, max_variants: int = 2
                           ) -> list[dict[str, Any]]:
    """Find bounded feasible order alternatives using native source evaluation."""
    if len(nodes) <= 7:
        proposals = permutations(nodes)
    else:
        proposals = iter(_route_order_proposals(nodes))
    feasible: dict[str, dict[str, Any]] = {}
    for route in proposals:
        try:
            candidate = _native_route_candidate(instance, route, vehicle, severity,
                                                penalty_rate=penalty_rate)
        except ValueError:
            continue
        feasible[candidate["id"]] = candidate
    ordered = sorted(feasible.values(), key=lambda row: (float(row["forecast_cost"]), tuple(row["route"])))
    return ordered[:max_variants]


def _greedy_native_partition(instance: Any, severity: float, penalty_rate: float,
                             *, max_stops_per_route: int) -> list[list[int]] | None:
    """Build source-native feasible routes by bounded earliest-deadline insertion."""
    windows = instance.time_windows
    remaining = sorted(instance.customers,
                       key=lambda node: (float(windows[node][1]), float(windows[node][0]), int(node)))
    routes: list[list[int]] = []
    while remaining:
        vehicle = f"vehicle-{len(routes) + 1:02d}"
        seed = remaining.pop(0)
        try:
            base = _native_route_candidate(instance, (seed,), vehicle, severity,
                                           penalty_rate=penalty_rate)
        except ValueError:
            return None
        route = list(base["route"])
        current = base
        while len(route) < max_stops_per_route and remaining:
            best = None
            # Examine only the next twelve earliest-deadline unassigned
            # customers. Each proposed route uses the native source checker.
            for node in remaining[:12]:
                for position in range(len(route) + 1):
                    proposal = tuple(route[:position] + [node] + route[position:])
                    try:
                        candidate = _native_route_candidate(instance, proposal, vehicle, severity,
                                                            penalty_rate=penalty_rate)
                    except ValueError:
                        continue
                    increase = float(candidate["forecast_cost"]) - float(current["forecast_cost"])
                    rank = (increase, int(node), tuple(proposal))
                    if best is None or rank < best[0]:
                        best = (rank, int(node), candidate)
            if best is None:
                break
            _, node, current = best
            route = list(current["route"])
            remaining.remove(node)
        routes.append(route)
        if len(routes) > int(instance.num_vehicles):
            return None
    return routes


def _ortools_solomon_partition(instance: Any, *, time_limit_seconds: int = 5) -> tuple[list[list[int]], dict[str, Any]] | None:
    """Bounded Solomon-only VRPTW constructive solver; final validator stays native."""
    if str(instance.source).lower() != "solomon":
        return None
    try:
        from importlib.metadata import version
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError:
        return None
    scale = 1000
    coordinates = instance.coordinates
    demands = instance.demands
    service = instance.service_times
    windows = instance.time_windows
    horizon = max(int(round(float(due) * scale)) for _, due in windows)
    vehicles = int(instance.num_vehicles)
    manager = pywrapcp.RoutingIndexManager(instance.num_customers + 1, vehicles, instance.depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance(a: int, b: int) -> int:
        dx = float(coordinates[a][0]) - float(coordinates[b][0])
        dy = float(coordinates[a][1]) - float(coordinates[b][1])
        return int(round(math.hypot(dx, dy) * scale))

    def distance_callback(from_index: int, to_index: int) -> int:
        return distance(manager.IndexToNode(from_index), manager.IndexToNode(to_index))

    transit = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    def time_callback(from_index: int, to_index: int) -> int:
        origin = manager.IndexToNode(from_index)
        destination = manager.IndexToNode(to_index)
        return distance(origin, destination) + int(round(float(service[origin]) * scale))

    time_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(time_index, horizon, horizon, False, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")
    for node in range(instance.num_customers + 1):
        cumul = time_dimension.CumulVar(manager.NodeToIndex(node))
        ready, due = windows[node]
        cumul.SetRange(int(round(float(ready) * scale)), int(round(float(due) * scale)))

    def demand_callback(from_index: int) -> int:
        node = manager.IndexToNode(from_index)
        return int(round(float(demands[node]) * scale)) if node != instance.depot else 0

    demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
    capacity = int(round(float(instance.capacity) * scale))
    routing.AddDimensionWithVehicleCapacity(demand_index, 0, [capacity] * vehicles, True, "Capacity")
    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parameters.time_limit.seconds = int(time_limit_seconds)
    parameters.log_search = False
    started = time.perf_counter()
    solution = routing.SolveWithParameters(parameters)
    elapsed = time.perf_counter() - started
    if solution is None:
        return None
    routes = []
    for vehicle in range(vehicles):
        index = routing.Start(vehicle)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != instance.depot:
                route.append(int(node))
            index = solution.Value(routing.NextVar(index))
        if route:
            routes.append(route)
    if not routes:
        return None
    return routes, {"solver": "OR-Tools RoutingModel Solomon VRPTW",
                    "solver_version": version("ortools"),
                    "time_limit_seconds": time_limit_seconds,
                    "wall_seconds": elapsed,
                    "distance_scale": scale,
                    "final_validator": "NativeInstance.evaluate_solution"}


def build_native_candidate_pool(
    instance: Any,
    *,
    severity_by_source: Mapping[str, float],
    penalty_rate: float = DEFAULT_SYNTHETIC_PENALTY_RATE_SCU,
    quantum_group_limit: int = MAX_ROUTE_VARIABLES // 2,
) -> dict[str, Any]:
    """Build a complete, deterministic route-order pool without consulting BKS.

    Customers are partitioned by native time-window order and round-robin
    assignment. Every ordering is checked by the canonical source evaluator;
    Dabia feasibility and travel use native ATFs through `evaluate_route`.
    At most four customer groups receive two QUBO variables each; remaining
    complete routes are fixed outside the quantum block and all routes are
    passed together to the native whole-instance checker.
    """
    if not hasattr(instance, "evaluate_route") or not hasattr(instance, "evaluate_solution"):
        raise TypeError("instance must expose native route and full-solution evaluators")
    source = str(instance.source).lower()
    if source not in severity_by_source:
        raise ValueError(f"missing train-only severity for source {source}")
    if quantum_group_limit < 1 or quantum_group_limit * 2 > MAX_ROUTE_VARIABLES:
        raise ValueError("quantum_group_limit must be between 1 and 4")
    if len(instance.customers) != instance.num_customers:
        raise ValueError("native customer IDs are not the expected one-based complete range")
    windows = instance.time_windows
    ordered = sorted(instance.customers,
                     key=lambda node: (float(windows[node][1]), float(windows[node][0]), int(node)))
    n_groups = min(int(instance.num_vehicles), max(1, math.ceil(instance.num_customers / 6)))
    attempts = min(max(1, len(ordered)), 8)
    chosen_groups = None
    chosen_options: list[list[dict[str, Any]]] | None = None
    construction_detail: dict[str, Any] = {
        "method": "earliest-window round-robin partition; deterministic route order proposals",
        "used_bks": False,
    }
    for offset in range(attempts):
        rotated = ordered[offset:] + ordered[:offset]
        groups = [[] for _ in range(n_groups)]
        for index, node in enumerate(rotated):
            groups[index % n_groups].append(node)
        options: list[list[dict[str, Any]]] = []
        valid_partition = True
        for group_index, group in enumerate(groups):
            # Unique route orders are ranked by only source-native forecast
            # cost; no stochastic replay/test outcome or BKS is consulted.
            first = _best_native_order(instance, group, f"vehicle-{group_index + 1:02d}",
                                       float(severity_by_source[source]), penalty_rate)
            if first is None:
                valid_partition = False
                break
            options.append([first])
        if not valid_partition:
            continue
        baseline_routes = [variants[0] for variants in options]
        baseline = instance.evaluate_solution([row["route"] for row in baseline_routes])
        if baseline.valid:
            chosen_groups, chosen_options = groups, options
            break
    if chosen_groups is None or chosen_options is None:
        max_stops = max(1, math.ceil(instance.num_customers /
                                     min(int(instance.num_vehicles), max(1, math.ceil(instance.num_customers / 6)))))
        greedy = _greedy_native_partition(instance, float(severity_by_source[source]), penalty_rate,
                                           max_stops_per_route=max_stops)
        if greedy:
            options = []
            for index, group in enumerate(greedy):
                route = _best_native_order(instance, group, f"vehicle-{index + 1:02d}",
                                           float(severity_by_source[source]), penalty_rate)
                if route is None:
                    options = []
                    break
                options.append([route])
            if options and instance.evaluate_solution([row[0]["route"] for row in options]).valid:
                chosen_groups, chosen_options = greedy, options
                construction_detail = {"method": "bounded earliest-deadline native insertion",
                                       "used_bks": False, "max_stops_per_route": max_stops}
    if chosen_groups is None or chosen_options is None:
        ortools_result = _ortools_solomon_partition(instance)
        if ortools_result is not None:
            routes, solver_detail = ortools_result
            options = []
            for index, route in enumerate(routes):
                candidate = _best_native_order(instance, route, f"vehicle-{index + 1:02d}",
                                                float(severity_by_source[source]), penalty_rate)
                if candidate is None:
                    options = []
                    break
                options.append([candidate])
            if options and instance.evaluate_solution([row[0]["route"] for row in options]).valid:
                chosen_groups = [list(row[0]["stops"]) for row in options]
                chosen_options = options
                construction_detail = {"method": "bounded OR-Tools Solomon VRPTW; native validation applied",
                                       "used_bks": False, **solver_detail}
    if chosen_groups is None or chosen_options is None:
        raise ValueError("no deterministic time-window-feasible complete route partition found without BKS")

    quantum_group_count = min(len(chosen_groups), quantum_group_limit)
    candidate_routes: list[dict[str, Any]] = []
    fixed_routes: list[dict[str, Any]] = []
    for index, variants in enumerate(chosen_options):
        if index < quantum_group_count:
            candidate_routes.extend(variants)
        else:
            fixed_routes.append(variants[0])

    # First add an alternate route ordering for a fixed customer set where
    # native feasibility permits. If route-time constraints leave no alternate
    # order, try one-customer exchanges between paired vehicles; both sides of
    # an exchange are added together, so the QUBO's exact-cover constraints
    # force a coherent plan rather than a partial move.
    for group_index in range(quantum_group_count):
        if len(candidate_routes) >= MAX_ROUTE_VARIABLES:
            break
        base = chosen_options[group_index][0]
        variants = _native_order_variants(
            instance, base["stops"], base["vehicle"], float(severity_by_source[source]),
            penalty_rate, max_variants=2)
        for alternate in variants:
            if alternate["id"] != base["id"] and len(candidate_routes) < MAX_ROUTE_VARIABLES:
                candidate_routes.append(alternate)

    exchange_groups: list[list[int]] = []
    for first in range(0, quantum_group_count - 1, 2):
        exchange_groups.append([first, first + 1])
    completed_exchanges = []
    for first, second in exchange_groups:
        if len(candidate_routes) + 2 > MAX_ROUTE_VARIABLES:
            break
        left, right = chosen_options[first][0], chosen_options[second][0]
        found = None
        for left_customer in left["stops"]:
            for right_customer in right["stops"]:
                left_nodes = tuple(node for node in left["stops"] if node != left_customer) + (right_customer,)
                right_nodes = tuple(node for node in right["stops"] if node != right_customer) + (left_customer,)
                left_alt = _best_native_order(instance, left_nodes, left["vehicle"],
                                               float(severity_by_source[source]), penalty_rate)
                if left_alt is None:
                    continue
                right_alt = _best_native_order(instance, right_nodes, right["vehicle"],
                                                float(severity_by_source[source]), penalty_rate)
                if right_alt is not None:
                    found = (left_alt, right_alt, int(left_customer), int(right_customer))
                    break
            if found is not None:
                break
        if found is not None:
            left_alt, right_alt, left_customer, right_customer = found
            candidate_routes.extend((left_alt, right_alt))
            completed_exchanges.append({"vehicles": [left["vehicle"], right["vehicle"]],
                                        "customer_exchange": [left_customer, right_customer],
                                        "alternative_route_ids": [left_alt["id"], right_alt["id"]]})
    required_stops = [int(node) for group in chosen_groups[:quantum_group_count] for node in group]
    fixed_stops = [int(node) for group in chosen_groups[quantum_group_count:] for node in group]
    evaluator_result = instance.evaluate_solution([row["route"] for row in fixed_routes] +
                                                    [row["route"] for row in (variants[0] for variants in chosen_options[:quantum_group_count])])
    if not evaluator_result.valid:
        raise RuntimeError(f"constructed native plan failed whole-instance validation: {evaluator_result.status}")
    return {
        "candidate_routes": candidate_routes,
        "required_stops": required_stops,
        "fixed_routes": fixed_routes,
        "fixed_stops": fixed_stops,
        "all_source_customer_ids": list(instance.customers),
        "native_evaluator": instance.evaluate_solution,
        "construction": {
            **construction_detail,
            "route_count": len(chosen_groups),
            "quantum_group_count": quantum_group_count,
            "candidate_variable_count": len(candidate_routes),
            "fixed_route_count": len(fixed_routes),
            "paired_customer_exchanges": completed_exchanges,
            "baseline_native_validation": _jsonable(evaluator_result),
            "native_objective": instance.native_objective,
            "source": source,
            "instance_id": instance.instance_id,
        },
    }


def _predict(model: Any, features: Sequence[float]) -> float:
    predictor = getattr(model, "predict_proba", None)
    if not callable(predictor):
        raise TypeError("model must expose predict_proba(features); scaler ownership stays with the model")
    raw = predictor(features)
    # Accept either a scalar or a two-class [non-event, event] result.
    if isinstance(raw, (tuple, list)):
        if len(raw) != 2:
            raise ValueError("predict_proba sequence must contain [non-event, event] probabilities")
        raw = raw[1]
    p = _finite_nonnegative(raw, "predicted probability")
    if p > 1:
        raise ValueError("predicted probability must lie in [0, 1]")
    return p


def score_candidates(
    candidate_routes: Sequence[Mapping[str, Any]], model: Any, *, enforce_quantum_bound: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Attach model probabilities and homogeneous risk-conditioned cost.

    Required candidate fields: id, vehicle, stops, route, features,
    forecast_cost, expected_breach_severity, penalty_rate, and
    severity_estimate_split='train'. The last three
    values must all use the declared decision objective unit (e.g. USD).
    `route` is passed intact to the native validator and is not reconstructed.
    """
    if not candidate_routes:
        raise ValueError("candidate_routes must be non-empty")
    if enforce_quantum_bound and len(candidate_routes) > MAX_ROUTE_VARIABLES:
        raise ValueError(f"candidate route count must be <= {MAX_ROUTE_VARIABLES}")
    scored: list[dict[str, Any]] = []
    probabilities: dict[str, float] = {}
    ids: set[str] = set()
    for raw in candidate_routes:
        required = {"id", "vehicle", "stops", "route", "features", "forecast_cost",
                    "expected_breach_severity", "penalty_rate", "severity_estimate_split"}
        missing = required - set(raw)
        if missing:
            raise ValueError(f"candidate missing fields: {sorted(missing)}")
        if raw["severity_estimate_split"] != "train":
            raise ValueError(f"candidate {raw.get('id')} severity estimate must come from the training split")
        forbidden = {"label", "train_label", "lateness_severity", "realized_cost", "realized_financial_cost",
                     "actual_cost", "test_outcome"}
        if forbidden & set(raw):
            raise ValueError(f"candidate contains outcome fields: {sorted(forbidden & set(raw))}")
        cid = str(raw["id"])
        if not cid or cid in ids:
            raise ValueError("candidate ids must be non-empty and unique")
        ids.add(cid)
        features = raw["features"]
        if not isinstance(features, (tuple, list)) or not features:
            raise ValueError(f"candidate {cid} features must be a non-empty sequence")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in features):
            raise ValueError(f"candidate {cid} features must be finite numbers")
        p = _predict(model, features)
        base = _finite_nonnegative(raw["forecast_cost"], f"{cid}.forecast_cost")
        severity = _finite_nonnegative(raw["expected_breach_severity"], f"{cid}.expected_breach_severity")
        rate = _finite_nonnegative(raw["penalty_rate"], f"{cid}.penalty_rate")
        risk_cost = p * severity * rate
        enriched = dict(raw)
        enriched.update({"id": cid, "cost": base + risk_cost,
                         "risk_probability": p, "expected_risk_cost": risk_cost})
        scored.append(enriched)
        probabilities[cid] = p
    return scored, probabilities


def _native_eval(evaluator: Callable[[Sequence[Any]], Any], selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    routes = [row["route"] for row in selected]
    try:
        result = _jsonable(evaluator(routes))
    except Exception as exc:
        return {"valid": False, "feasible": False, "status": "native_validation_error",
                "error_type": type(exc).__name__, "error": str(exc)}
    if not isinstance(result, dict):
        raise TypeError("native evaluator must return a mapping or dataclass mapping")
    # Do not infer validity from a numeric objective or solver flag.
    return result


def _native_feasible(result: Mapping[str, Any]) -> bool:
    return result.get("valid") is True and result.get("feasible", True) is True


def _plan_evaluation(
    qubo: Any, bits: Sequence[int], candidates: Sequence[Mapping[str, Any]],
    native_evaluator: Callable[[Sequence[Any]], Any],
    fixed_routes: Sequence[Mapping[str, Any]] = (), fixed_cost: float = 0.0,
) -> dict[str, Any]:
    validation = validate_selection(qubo, bits)
    if not validation["feasible"]:
        raise ValueError("optimizer selection is not exactly covering and vehicle-feasible")
    selected = [row for row, bit in zip(candidates, bits) if int(bit)]
    native = _native_eval(native_evaluator, [*fixed_routes, *selected])
    if not _native_feasible(native):
        raise ValueError(f"native validation rejected selected plan: {native.get('status', native.get('errors'))}")
    return {"selected_route_ids": [row["id"] for row in fixed_routes] + validation["selected_route_ids"],
            "quantum_subproblem_route_ids": validation["selected_route_ids"],
            "fixed_route_ids": [row["id"] for row in fixed_routes],
            "expected_objective": validation["business_cost"] + fixed_cost,
            "native_validation": native}


def _enumerate_feasible(qubo: Any) -> list[tuple[list[int], dict[str, Any]]]:
    rows = []
    for bits in product((0, 1), repeat=qubo.n):
        check = validate_selection(qubo, bits)
        if check["feasible"]:
            rows.append(([int(bit) for bit in bits], check))
    return rows


def _replay_raw_samples(qubo: Any, qaoa: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct a diagnostic sampler replay from the V1.1 summary.

    The V1.1 API only reports angles rounded to 1e-6 and omits original raw
    draws. This replay cannot establish that these were the exact original
    samples, so its provenance and comparison status travel with the rows.
    """
    raw_relaxation = qaoa.get("relaxation_vector")
    if not isinstance(raw_relaxation, list) or len(raw_relaxation) != qubo.n:
        raise ValueError("QAOA result is missing the LP warm-start vector")
    probabilities = [min(0.95, max(0.05, float(value))) for value in raw_relaxation]
    energies, _ = precompute_hamiltonian_spectrum(qubo)
    state = fast_qaoa_state(qubo.n, energies, qaoa["optimized_gammas"],
                            qaoa["optimized_betas"], probabilities)
    rng = random.Random(int(qaoa["seed"]))
    cumulative: list[float] = []
    total = 0.0
    for amplitude in state:
        total += amplitude.real * amplitude.real + amplitude.imag * amplitude.imag
        cumulative.append(total)
    rows = []
    for shot in range(1, int(qaoa["shots"]) + 1):
        draw = rng.random()
        low, high = 0, len(cumulative) - 1
        while low < high:
            middle = (low + high) // 2
            if draw < cumulative[middle]:
                high = middle
            else:
                low = middle + 1
        bitstring = format(low, f"0{qubo.n}b")
        bits = [int(bit) for bit in bitstring]
        rows.append({"shot": shot, "bitstring": bitstring,
                     "qubo_energy": qubo.energy(bits),
                     "constraint_validation": validate_selection(qubo, bits)})
    observed_feasible = sum(row["constraint_validation"]["feasible"] for row in rows)
    return {"rows": rows,
            "provenance": "diagnostic replay from V1.1 angles rounded to 1e-6; original raw samples unavailable",
            "benchmark_feasible_shots": int(qaoa["feasible_shots"]),
            "replayed_feasible_shots": observed_feasible,
            "feasible_count_matches": observed_feasible == int(qaoa["feasible_shots"]),
            "is_original_raw_output": False}


def _run_qpiai_local(qubo: Any, relaxation: Sequence[float], *, seed: int, shots: int,
                     gamma: float, beta: float) -> dict[str, Any]:
    """Attempt the genuine QpiAI Opt + Quantum local statevector adapter."""
    from backend.qpiai_adapter import run_qpiai_warm_start

    result = run_qpiai_warm_start(qubo, relaxation, gamma=gamma, beta=beta,
                                  shots=shots, seed=seed)
    decoded = []
    best = None
    counts = result.get("raw_counts", {})
    if result.get("status") == "success":
        for bitstring, count in sorted(counts.items()):
            row = decode_bitstring(qubo, str(bitstring))
            row["count"] = int(count)
            decoded.append(row)
            validation = row["validation"]
            if validation["feasible"]:
                key = (float(validation["business_cost"]), str(bitstring))
                if best is None or key < best[0]:
                    best = (key, validation["selected_route_ids"], str(bitstring))
        if sum(int(value) for value in counts.values()) != shots:
            return {**result, "status": "failed", "error": {"code": "SHOT_COUNT_MISMATCH"},
                    "decoded_raw_counts": decoded, "best_feasible_routes": None}
        if result.get("relaxed_solution") != [float(value) for value in relaxation]:
            return {**result, "status": "failed", "error": {"code": "LP_VECTOR_MISMATCH"},
                    "decoded_raw_counts": decoded, "best_feasible_routes": None}
    result["decoded_raw_counts"] = decoded
    result["best_feasible_routes"] = best[1] if best is not None else None
    result["best_feasible_bitstring"] = best[2] if best is not None else None
    result["raw_feasible_count"] = sum(int(row["count"]) for row in decoded
                                      if row["validation"]["feasible"])
    result["independent_decode_validation"] = "route QUBO variable order; constraints recomputed locally"
    return result


def _make_artifact(
    *, candidates: Sequence[Mapping[str, Any]], probabilities: Mapping[str, float],
    qubo: Any, metadata: Mapping[str, Any], qaoa: Mapping[str, Any],
    fixed_routes: Sequence[Mapping[str, Any]] = (), fixed_probabilities: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Emit v1.1-compatible contract plus explicit unit/provenance extensions."""
    if metadata.get("objective_unit") in (None, "", "native_duration"):
        raise ValueError("dispatch QUBO requires a declared non-native decision objective unit; Duration is separate")
    fixed_probabilities = fixed_probabilities or {}
    candidate_ids = [str(row["id"]) for row in candidates]
    coefficients = {str(row["id"]): float(row["cost"]) for row in candidates}
    is_usd = str(metadata["objective_unit"]).upper() in {"USD", "USD_ASSUMPTION"}
    payload = HybridDispatchArtifact(
        schema_version=CONTRACT_SCHEMA_VERSION,
        model_and_dataset_version={
            "model_id": str(metadata["model_id"]),
            "model_version": str(metadata["model_version"]),
            "dataset_id": str(metadata["dataset_id"]),
            "dataset_sha256": str(metadata["dataset_sha256"]),
            "is_reference_fixture": bool(metadata.get("is_reference_fixture", False)),
            "model_sha256": metadata.get("model_sha256"),
            "model_artifact_path": metadata.get("model_artifact_path"),
            "scaler_sha256": metadata.get("scaler_sha256"),
            "scaler_artifact_path": metadata.get("scaler_artifact_path"),
            "target_specification": metadata.get("target_specification"),
            "circuit_specification": metadata.get("circuit_specification"),
            "actual_execution_provenance": qaoa.get("execution_provenance",
                "custom local exact statevector QAOA simulator; no QCloud/QPU"),
        },
        split_identifiers={
            "replicate_id": str(metadata["replicate_id"]),
            "train_split_sha256": str(metadata["train_split_sha256"]),
            "validation_split_sha256": str(metadata["validation_split_sha256"]),
            "test_split_sha256": str(metadata["test_split_sha256"]),
            "evaluation_split": str(metadata["evaluation_split"]),
        },
        scenario_and_seed_Provenance={
            "scenario_id": str(metadata["scenario_id"]),
            "family_seed": metadata.get("family_seed"),
            "split_seed": metadata.get("split_seed"),
            "qaoa_seed": int(qaoa["seed"]),
            "source_revision": metadata.get("source_revision"),
            "source_files_sha256": metadata.get("source_files_sha256", {}),
        },
        risk_predictions_and_calibration={
            "route_probabilities": dict(probabilities),
            "readout_type": str(metadata.get("readout_type", "model.predict_proba")),
            "brier_score_validation": metadata.get("brier_score_validation"),
            "ece_validation": metadata.get("ece_validation"),
            "feature_schema": metadata.get("feature_schema"),
            "scaler_fit_split": "train_only",
        },
        candidate_routes_and_coefficients={
            "candidate_route_ids": candidate_ids,
            "transport_cost_usd": {str(row["id"]): row.get("transport_cost_usd") for row in candidates},
            "labor_cost_usd": {str(row["id"]): row.get("labor_cost_usd") for row in candidates},
            "expected_sla_penalty_usd": {
                str(row["id"]): float(row["expected_risk_cost"]) if is_usd else None for row in candidates},
            "total_route_coefficients_usd": coefficients if is_usd else {rid: None for rid in candidate_ids},
            "objective_unit": str(metadata["objective_unit"]),
            "route_cost_components": {
                str(row["id"]): {
                    "vehicle": str(row["vehicle"]),
                    "stops": list(row["stops"]),
                    "route_payload": list(row["route"]),
                    "raw_features": list(row["features"]),
                    "forecast_cost": float(row["forecast_cost"]),
                    "expected_risk_cost": float(row["expected_risk_cost"]),
                    "transport_cost_scu": row.get("transport_cost_scu"),
                    "labor_cost_scu": row.get("labor_cost_scu"),
                } for row in candidates
            },
            "fixed_routes_outside_quantum_subproblem": [
                {"id": str(row["id"]), "vehicle": str(row["vehicle"]),
                 "stops": list(row["stops"]), "route_payload": list(row["route"]),
                 "raw_features": list(row["features"]),
                 "risk_probability": fixed_probabilities[str(row["id"])],
                 "forecast_cost": float(row["forecast_cost"]),
                 "expected_risk_cost": float(row["expected_risk_cost"]),
                 "risk_conditioned_cost": float(row["cost"])}
                for row in fixed_routes
            ],
        },
        qubo_and_constraints={
            "n_variables": qubo.n,
            "variable_order": list(qubo.variable_ids),
            "linear": list(qubo.linear),
            "quadratic": {f"{i},{j}": value for (i, j), value in qubo.quadratic.items()},
            "offset": qubo.offset,
            "coverage_penalty": qubo.coverage_penalty,
            "vehicle_penalty": qubo.vehicle_penalty,
            "required_stops": list(qubo.required_stops),
            "fixed_routes_outside_quantum_subproblem": [str(row["id"]) for row in fixed_routes],
        },
        execution_backend_and_config={
            "backend_name": qaoa["backend"],
            "execution_mode": "lp_augmented_warm_start_qaoa_p1",
            "is_simulated": True,
            "qpu_authorized": False,
            "layers": 1,
            "shots": int(qaoa["shots"]),
            "qaoa_seed": int(qaoa["seed"]),
            "relaxation_vector": qaoa.get("relaxation_vector"),
            "sdk_versions": qaoa.get("sdk_versions"),
            "compiled_qubo_validation": qaoa.get("compiled_qubo_validation"),
            "local_simulator_jobs": qaoa.get("local_simulator_jobs", 0),
            "qpiai_local_attempt": qaoa.get("qpiai_local_attempt"),
            "cloud_jobs": 0,
            "qpu_jobs": 0,
        },
    ).to_dict()
    contract_check = validate_hybrid_artifact(payload)
    # The v1.1 validator checks structure/hash only. Run 1 adds semantic checks.
    semantic_errors = []
    if payload["candidate_routes_and_coefficients"]["objective_unit"] != metadata["objective_unit"]:
        semantic_errors.append("objective unit mismatch")
    if payload["risk_predictions_and_calibration"]["scaler_fit_split"] != "train_only":
        semantic_errors.append("scaler was not declared train-only")
    if not is_usd and any(
        value is not None
        for field in ("transport_cost_usd", "labor_cost_usd", "expected_sla_penalty_usd", "total_route_coefficients_usd")
        for value in payload["candidate_routes_and_coefficients"][field].values()
    ):
        semantic_errors.append("synthetic non-USD objective was placed in a legacy USD field")
    return {"adapter_version": "debugleaf.run1_dispatch.v1",
            "embedded_contract": payload,
            "compatibility_mapping": {
                "legacy_contract_schema": CONTRACT_SCHEMA_VERSION,
                "run1_objective_unit": str(metadata["objective_unit"]),
                "legacy_usd_fields": "present for structural compatibility; null when the RUN 1 objective is SCU",
                "scu_fields": "candidate_routes_and_coefficients.route_cost_components.*_scu",
                "currency_conversion_performed": False,
                "financial_currency_claim": bool(is_usd and metadata.get("financial_unit")),
                "rates": {"transport_scu_per_native_travel_unit": SYNTHETIC_TRANSPORT_RATE_SCU,
                          "labor_scu_per_native_duty_unit": SYNTHETIC_DUTY_RATE_SCU,
                          "sla_scu_per_late_time_unit": DEFAULT_SYNTHETIC_PENALTY_RATE_SCU},
            },
            "contract_validation": contract_check,
            "semantic_validation": {"valid": not semantic_errors, "errors": semantic_errors,
                                    "note": "v1.1 contract hash and key checks do not establish semantic correctness"}}


def run_dispatch(
    candidate_routes: Sequence[Mapping[str, Any]],
    required_stops: Sequence[str],
    model: Any,
    native_evaluator: Callable[[Sequence[Any]], Any],
    *,
    metadata: Mapping[str, Any],
    fixed_routes: Sequence[Mapping[str, Any]] = (),
    fixed_stops: Sequence[str] = (),
    seed: int = 7,
    shots: int = 256,
    grid_size: int = 4,
    max_qaoa_seconds: float = 30.0,
) -> dict[str, Any]:
    """Run risk-priced QUBO, LP warm-start p=1 QAOA, and native revalidation.

    Candidate pool is capped at eight variables. Exhaustive enumeration gives
    a certified optimum under the same *predicted* risk-conditioned objective.
    If QAOA returns no feasible raw sample, that exact candidate-pool solution
    is returned with an explicit fallback label. Every selected plan is passed
    to the independent native evaluator before publication.
    """
    required_metadata = {
        "model_id", "model_version", "dataset_id", "dataset_sha256", "replicate_id",
        "train_split_sha256", "validation_split_sha256", "test_split_sha256",
        "evaluation_split", "scenario_id", "objective_unit", "feature_schema",
        "severity_fit_split_sha256",
    }
    if required_metadata - set(metadata):
        raise ValueError(f"metadata missing: {sorted(required_metadata - set(metadata))}")
    if metadata["evaluation_split"] != "test":
        raise ValueError("RUN 1 dispatch artifact may only evaluate the frozen test split")
    if metadata["severity_fit_split_sha256"] != metadata["train_split_sha256"]:
        raise ValueError("severity estimator provenance must match the frozen training split hash")
    scored, probabilities = score_candidates(candidate_routes, model)
    scored_fixed, fixed_probabilities = score_candidates(fixed_routes, model, enforce_quantum_bound=False) if fixed_routes else ([], {})
    if set(row["id"] for row in scored) & set(row["id"] for row in scored_fixed):
        raise ValueError("fixed and QUBO candidate route ids must be disjoint")
    if set(stop for row in scored for stop in row["stops"]) & set(fixed_stops):
        raise ValueError("fixed route stops overlap the QUBO subproblem customer set")
    if set(stop for row in scored_fixed for stop in row["stops"]) != set(fixed_stops):
        raise ValueError("fixed_stops must exactly match the customers covered by fixed_routes")
    if metadata.get("all_source_customer_ids") is not None:
        if set(metadata["all_source_customer_ids"]) != set(fixed_stops) | set(required_stops):
            raise ValueError("fixed and QUBO stops do not cover the complete declared source instance")
    fixed_cost = sum(float(row["cost"]) for row in scored_fixed)
    qubo_candidates = [{"id": row["id"], "vehicle": row["vehicle"], "stops": row["stops"], "cost": row["cost"]}
                       for row in scored]
    qubo = build_route_qubo(qubo_candidates, required_stops)
    lp = lp_augmented_relaxation(qubo)
    if len(lp) != qubo.n or any(not math.isfinite(float(p)) or not 0 <= p <= 1 for p in lp):
        raise ValueError("LP relaxation returned an invalid warm-start vector")
    exact = solve_classical_exact(qubo)
    if not exact["feasible"]:
        raise ValueError("candidate pool has no exact-cover plan")
    qaoa = execute_qaoa_benchmark(
        qubo, layers=1, warm_start=True, relaxation_mode="lp_augmented",
        optimizer_method="grid", grid_size=grid_size, shots=shots, seed=seed,
        wall_deadline_seconds=max_qaoa_seconds,
        certified_optimum=float(exact["best_objective"]),
    )
    qpiai_qaoa = _run_qpiai_local(
        qubo, lp, seed=seed, shots=shots,
        gamma=float(qaoa["optimized_gammas"][0]), beta=float(qaoa["optimized_betas"][0]),
    )
    by_id = {row["id"]: row for row in scored}
    quantum_options = []
    if qpiai_qaoa.get("status") == "success" and qpiai_qaoa.get("best_feasible_routes"):
        quantum_options.append(("qpiai_local_simulator", qpiai_qaoa["best_feasible_routes"]))
    if qaoa.get("best_feasible_routes"):
        quantum_options.append(("custom_local_exact_statevector", qaoa["best_feasible_routes"]))
    selection_attempts = []
    selection_backend = "classical_exact_candidate_pool"
    fallback_reason = None
    selected_ids = None
    selected_eval = None
    for backend_name, route_ids in quantum_options:
        bits_try = [int(cid in set(route_ids)) for cid in qubo.variable_ids]
        try:
            eval_try = _plan_evaluation(qubo, bits_try, scored, native_evaluator, scored_fixed, fixed_cost)
        except ValueError as exc:
            selection_attempts.append({"backend": backend_name, "status": "native_validation_failed",
                                       "reason": str(exc)})
            continue
        selection_backend, selected_ids, selected_eval = backend_name, list(route_ids), eval_try
        selection_attempts.append({"backend": backend_name, "status": "selected"})
        break
    fallback_used = selected_eval is None
    if fallback_used:
        selected_ids = list(exact["selected_route_ids"])
        bits_exact = [int(cid in set(selected_ids)) for cid in qubo.variable_ids]
        selected_eval = _plan_evaluation(qubo, bits_exact, scored, native_evaluator, scored_fixed, fixed_cost)
        if not quantum_options:
            fallback_reason = "no feasible raw sample from QpiAI/custom QAOA; exact candidate-pool optimum used"
        else:
            fallback_reason = "all sampled quantum selections failed independent native validation; exact candidate-pool optimum used"
        selection_attempts.append({"backend": selection_backend, "status": "classical_exact_fallback"})

    # Same-scenario clairvoyant reference over the exact feasible candidate pool.
    # Only use comparable realized financial objective outputs; never compare
    # contractual/native Duration against synthetic USD or proxy units.
    feasible_plans = _enumerate_feasible(qubo)
    native_oracles = []
    for plan_bits, check in feasible_plans:
        plan_rows = [row for row, bit in zip(scored, plan_bits) if bit]
        value = _native_eval(native_evaluator, [*scored_fixed, *plan_rows])
        realized = value.get("realized_decision_cost")
        realized_financial = value.get("realized_financial_cost")
        native_oracles.append({"route_ids": check["selected_route_ids"],
                               "valid": _native_feasible(value),
                               "realized_decision_cost": realized,
                               "decision_objective_unit": value.get("decision_objective_unit"),
                               "realized_financial_cost": realized_financial,
                               "native_cost": value.get("native_cost"),
                               "native_validation": value})
    realized_values = [row["realized_decision_cost"] for row in native_oracles if row["valid"]
                       if isinstance(row["realized_decision_cost"], (int, float)) and
                       math.isfinite(float(row["realized_decision_cost"]))]
    # A clairvoyant reference is valid only when every entry uses the same
    # objective unit and all feasible candidate plans were evaluated.
    units = {row["decision_objective_unit"] for row in native_oracles if row["valid"] and
             row["realized_decision_cost"] is not None}
    all_plan_evaluations_completed = len(native_oracles) == len(feasible_plans)
    no_native_evaluator_errors = all(
        row["native_validation"].get("status") != "native_validation_error" for row in native_oracles)
    realized_decision = selected_eval["native_validation"].get("realized_decision_cost")
    clairvoyant = min(realized_values) if (realized_values and units == {metadata["objective_unit"]}
                                            and all_plan_evaluations_completed and no_native_evaluator_errors) else None
    decision_regret = (float(realized_decision) - float(clairvoyant)
                       if clairvoyant is not None and isinstance(realized_decision, (int, float)) else None)
    realized = selected_eval["native_validation"].get("realized_financial_cost")
    financial_values = [row["realized_financial_cost"] for row in native_oracles if row["valid"]
                        if isinstance(row["realized_financial_cost"], (int, float)) and
                        math.isfinite(float(row["realized_financial_cost"]))]
    financial_units = {row["native_validation"].get("financial_unit") for row in native_oracles
                       if row["valid"] and row["realized_financial_cost"] is not None}
    financial_unit = metadata.get("financial_unit")
    financial_clairvoyant = min(financial_values) if (financial_values and financial_unit and
                                                       financial_units == {financial_unit} and
                                                       all_plan_evaluations_completed and no_native_evaluator_errors) else None
    financial_regret = (float(realized) - float(financial_clairvoyant)
                        if financial_clairvoyant is not None and isinstance(realized, (int, float)) else None)
    native_values = [row["native_cost"] for row in native_oracles if row["valid"]
                     if isinstance(row["native_cost"], (int, float)) and math.isfinite(float(row["native_cost"]))]
    native_bks = metadata.get("native_bks_duration")
    native_gap = (float(selected_eval["native_validation"]["native_cost"]) - float(native_bks)
                  if isinstance(native_bks, (int, float)) and
                  isinstance(selected_eval["native_validation"].get("native_cost"), (int, float)) else None)
    qaoa_meta = {"backend": qpiai_qaoa["backend"] if selection_backend == "qpiai_local_simulator" else qaoa["backend"],
                 "seed": qaoa["seed"], "shots": qaoa["shots"],
                 "relaxation_vector": lp,
                 "execution_provenance": (
                     "QpiAI Opt QAOA compiled route QUBO; QpiAI Quantum local StatevectorSimulator; no cloud/QPU"
                     if selection_backend == "qpiai_local_simulator" else
                     "custom local exact statevector QAOA simulator; no QCloud/QPU"),
                 "sdk_versions": qpiai_qaoa.get("sdk_versions"),
                 "compiled_qubo_validation": qpiai_qaoa.get("compiled_qubo_validation"),
                 "qpiai_local_attempt": {
                     "status": qpiai_qaoa.get("status"),
                     "execution_type": qpiai_qaoa.get("execution_type", "not_run"),
                     "sdk_versions": qpiai_qaoa.get("sdk_versions"),
                     "compiled_qubo_validation": qpiai_qaoa.get("compiled_qubo_validation"),
                     "raw_counts": qpiai_qaoa.get("raw_counts"),
                     "decoded_raw_counts": qpiai_qaoa.get("decoded_raw_counts"),
                     "error": qpiai_qaoa.get("error"),
                 },
                 "local_simulator_jobs": int(selection_backend == "qpiai_local_simulator")}
    sample_replay = _replay_raw_samples(qubo, qaoa)
    artifact = _make_artifact(candidates=scored, probabilities=probabilities, qubo=qubo,
                              metadata=metadata, qaoa=qaoa_meta, fixed_routes=scored_fixed,
                              fixed_probabilities=fixed_probabilities)
    if not artifact["contract_validation"]["valid"] or not artifact["semantic_validation"]["valid"]:
        raise ValueError(f"dispatch artifact failed validation: {artifact}")
    return {
        "status": "validated_fallback" if fallback_used else "validated_qaoa",
        "fallback": {"used": fallback_used,
                     "reason": fallback_reason,
                     "method": "classical_exact_candidate_pool" if fallback_used else None},
        "scenario_id": metadata["scenario_id"],
        "evaluation_split": metadata["evaluation_split"],
        "route_risk_probabilities": probabilities,
        "fixed_route_risk_probabilities": fixed_probabilities,
        "fixed_routes": [{"id": row["id"], "vehicle": row["vehicle"],
                          "stops": list(row["stops"]), "route_payload": list(row["route"]),
                          "raw_features": list(row["features"]), "expected_cost": row["cost"]}
                         for row in scored_fixed],
        "fixed_route_expected_cost": fixed_cost,
        "candidate_routes_and_expected_costs": [
            {"id": row["id"], "vehicle": row["vehicle"],
             "stops": list(row["stops"]), "route_payload": list(row["route"]),
             "raw_features": list(row["features"]), "forecast_cost": row["forecast_cost"],
             "expected_risk_cost": row["expected_risk_cost"], "risk_probability": row["risk_probability"],
             "risk_conditioned_cost": row["cost"]} for row in scored
        ],
        "selected_plan": selected_eval,
        "expected_objective_regret_vs_exact_candidate_reference":
            float(selected_eval["expected_objective"]) - (float(exact["best_objective"]) + fixed_cost),
        "realized_decision_cost": realized_decision,
        "decision_objective_unit": selected_eval["native_validation"].get("decision_objective_unit"),
        "clairvoyant_same_candidate_set_decision_cost": clairvoyant,
        "realized_decision_regret": decision_regret,
        "realized_financial_cost": realized,
        "clairvoyant_same_candidate_set_financial_cost": financial_clairvoyant,
        "realized_financial_decision_regret": financial_regret,
        "native_duration_cost": selected_eval["native_validation"].get("native_cost"),
        "native_duration_bks": native_bks,
        "native_duration_gap_to_bks": native_gap,
        "native_candidate_plan_count": len(native_oracles),
        "native_feasible_candidate_plan_count": sum(bool(row["valid"]) for row in native_oracles),
        "native_evaluator_error_count": sum(row["native_validation"].get("status") == "native_validation_error"
                                           for row in native_oracles),
        "native_candidate_evaluations": native_oracles,
        "candidate_pool_construction": metadata.get("candidate_pool_construction", "caller-provided candidate pool"),
        "risk_conditioned_exact_reference": exact,
        "all_source_customers": list(fixed_stops) + list(required_stops),
        "quantum_subproblem_customer_ids": list(required_stops),
        "lp_warm_start_vector": lp,
        "qaoa": qaoa,
        "qpiai_qaoa": qpiai_qaoa,
        "selection_execution_backend": selection_backend,
        "selection_attempts": selection_attempts,
        "qaoa_sample_replay": sample_replay,
        "artifact": artifact,
    }


def compare_models_on_scenario(
    candidate_routes: Sequence[Mapping[str, Any]],
    required_stops: Sequence[str],
    models: Mapping[str, Any],
    native_evaluator: Callable[[Sequence[Any]], Any],
    *,
    metadata_by_model: Mapping[str, Mapping[str, Any]],
    fixed_routes: Sequence[Mapping[str, Any]] = (),
    fixed_stops: Sequence[str] = (),
    seed: int = 7,
    shots: int = 256,
    grid_size: int = 4,
    max_qaoa_seconds: float = 30.0,
) -> dict[str, Any]:
    """Evaluate predictive, classical, and decision-focused models on one
    frozen candidate pool, native scenario replay, seed, and QAOA budget.

    The caller must pass separately fitted models whose split selection was
    frozen before test evaluation. This function does not tune or fit models.
    """
    if not models or set(models) != set(metadata_by_model):
        raise ValueError("models and metadata_by_model must have the same non-empty keys")
    outputs = {}
    for name in sorted(models):
        outputs[name] = run_dispatch(
            candidate_routes, required_stops, models[name], native_evaluator,
            metadata=metadata_by_model[name], seed=seed, shots=shots,
            fixed_routes=fixed_routes, fixed_stops=fixed_stops,
            grid_size=grid_size, max_qaoa_seconds=max_qaoa_seconds,
        )
    candidate_ids = [str(row["id"]) for row in candidate_routes]
    fixed_ids = [str(row["id"]) for row in fixed_routes]
    if any([row["id"] for row in output["candidate_routes_and_expected_costs"]] != candidate_ids
           for output in outputs.values()):
        raise RuntimeError("model comparisons did not use the identical ordered candidate pool")
    if any([row["id"] for row in output["fixed_routes"]] != fixed_ids for output in outputs.values()):
        raise RuntimeError("model comparisons did not use the identical fixed route pool")
    return {
        "scenario_id": next(iter(outputs.values()))["scenario_id"],
        "evaluation_split": "test",
        "comparison_protocol": {
            "common_candidate_pool_ids": candidate_ids,
            "common_native_evaluation_callable": True,
            "common_qaoa_seed": seed,
            "common_qaoa_shots": shots,
            "common_grid_size": grid_size,
            "tuning_performed": False,
        },
        "methods": {
            name: {
                "status": output["status"],
                "selected_route_ids": output["selected_plan"]["selected_route_ids"],
                "expected_cost": output["selected_plan"]["expected_objective"],
                "expected_regret_vs_exact_candidate_reference": output[
                    "expected_objective_regret_vs_exact_candidate_reference"],
                "realized_decision_cost": output["realized_decision_cost"],
                "decision_objective_unit": output["decision_objective_unit"],
                "clairvoyant_same_candidate_set_decision_cost": output[
                    "clairvoyant_same_candidate_set_decision_cost"],
                "realized_decision_regret": output["realized_decision_regret"],
                "realized_financial_cost": output["realized_financial_cost"],
                "clairvoyant_same_candidate_set_financial_cost": output[
                    "clairvoyant_same_candidate_set_financial_cost"],
                "realized_financial_regret": output["realized_financial_decision_regret"],
                "native_duration_cost": output["native_duration_cost"],
                "native_duration_bks": output["native_duration_bks"],
                "native_duration_gap_to_bks": output["native_duration_gap_to_bks"],
                "fallback": output["fallback"],
                "full_dispatch_artifact": output["artifact"],
            } for name, output in outputs.items()
        },
        "limitations": [
            "Regret is only defined when native_evaluator returns a same-replay realized_financial_cost.",
            "Native benchmark Duration/BKS values are reported separately from decision-objective units.",
            "This function assumes caller-frozen train/validation/test provenance and performs no tuning.",
        ],
    }
