"""Independent Plan Validator and Financial Objective Evaluator.

This validator recomputes order coverage, vehicle load, arrival times,
hard windows, soft SLA lateness, transport, labor, and contractual penalties
from first principles without relying on solver flags or model outputs.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from benchmark_harness.interfaces import CandidateRoute, PlanValidationResult


DEFAULT_TRANSPORT_RATE = 0.50  # USD per travel minute
DEFAULT_LABOR_RATE = 0.25      # USD per duty minute


class IndependentPlanValidator:
    """Independent auditor for route-pool dispatch plans."""

    def __init__(
        self,
        transport_rate_per_min: float = DEFAULT_TRANSPORT_RATE,
        labor_rate_per_min: float = DEFAULT_LABOR_RATE,
        hard_window_mode: bool = False,
    ) -> None:
        if transport_rate_per_min < 0 or labor_rate_per_min < 0:
            raise ValueError("cost rates must be non-negative")
        self.transport_rate = float(transport_rate_per_min)
        self.labor_rate = float(labor_rate_per_min)
        self.hard_window_mode = bool(hard_window_mode)

    def validate_plan(
        self,
        selected_routes: Sequence[CandidateRoute],
        required_orders: Sequence[str],
        vehicles: Mapping[str, float],  # vehicle_id -> capacity
        order_demands: Mapping[str, float],  # order_id -> demand
        order_windows: Mapping[str, Tuple[float, float]],  # order_id -> (ready_min, due_min)
        order_services: Mapping[str, float],  # order_id -> service_duration_min
        order_penalties: Mapping[str, float],  # order_id -> penalty_rate_per_min
        travel_time_matrix: Mapping[str, Mapping[str, float]],  # node -> node -> travel_time_min
        depot_id: str = "D",
        depot_window: Optional[Tuple[float, float]] = None,
        realized_travel_matrix: Optional[Mapping[str, Mapping[str, float]]] = None,
        candidate_pool: Optional[Sequence[CandidateRoute]] = None,
    ) -> PlanValidationResult:
        """Independently validate a dispatch plan and recompute all business costs."""
        required_set = set(str(o) for o in required_orders)
        errors: List[str] = []

        # 1. Order coverage check (Exact Partition)
        covered_orders: List[str] = []
        duplicate_orders: List[str] = []
        seen_orders: Set[str] = set()

        for route in selected_routes:
            for oid in route.order_ids:
                oid_str = str(oid)
                if oid_str in seen_orders:
                    duplicate_orders.append(oid_str)
                else:
                    seen_orders.add(oid_str)
                covered_orders.append(oid_str)

        missing_orders = sorted(list(required_set - seen_orders))
        coverage_valid = (len(missing_orders) == 0 and len(duplicate_orders) == 0 and seen_orders == required_set)
        if missing_orders:
            errors.append(f"missing orders: {missing_orders}")
        if duplicate_orders:
            errors.append(f"duplicate orders: {duplicate_orders}")

        # 2. Vehicle usage and capacity check
        vehicle_loads: Dict[str, float] = {}
        vehicle_capacities: Dict[str, float] = dict(vehicles)
        seen_vehicles: Set[str] = set()
        capacity_valid = True

        for route in selected_routes:
            vid = str(route.vehicle_id)
            if vid in seen_vehicles:
                capacity_valid = False
                errors.append(f"vehicle {vid} used more than once in plan")
            seen_vehicles.add(vid)

            if vid not in vehicles:
                capacity_valid = False
                errors.append(f"unknown vehicle {vid}")
                cap = 0.0
            else:
                cap = vehicles[vid]

            route_demand = sum(order_demands.get(oid, 0.0) for oid in route.order_ids)
            vehicle_loads[vid] = route_demand
            if route_demand > cap + 1e-9:
                capacity_valid = False
                errors.append(f"vehicle {vid} load {route_demand} exceeds capacity {cap}")

        # 3. Timing, stop ETAs, windows and lateness
        matrix_to_use = realized_travel_matrix if realized_travel_matrix is not None else travel_time_matrix
        nominal_matrix = travel_time_matrix

        stop_nominal_etas: Dict[str, float] = {}
        stop_realized_etas: Dict[str, float] = {}
        stop_service_starts: Dict[str, float] = {}
        stop_lateness_min: Dict[str, float] = {}
        hard_violations = 0
        soft_breaches = 0
        total_travel_min = 0.0
        total_duty_min = 0.0

        for route in selected_routes:
            # Nominal trace
            t_nom = 0.0
            prev_node = depot_id
            for oid in route.stops:
                oid_str = str(oid)
                leg_time = nominal_matrix.get(prev_node, {}).get(oid_str, 0.0)
                t_nom += leg_time
                stop_nominal_etas[oid_str] = round(t_nom, 4)
                ready, _ = order_windows.get(oid_str, (0.0, float("inf")))
                serv = order_services.get(oid_str, 0.0)
                t_nom = max(t_nom, ready) + serv
                prev_node = oid_str

            # Realized trace
            t_real = 0.0
            prev_node = depot_id
            route_departure = 0.0
            for oid in route.stops:
                oid_str = str(oid)
                leg_time = matrix_to_use.get(prev_node, {}).get(oid_str, 0.0)
                total_travel_min += leg_time
                t_real += leg_time
                stop_realized_etas[oid_str] = round(t_real, 4)

                ready, due = order_windows.get(oid_str, (0.0, float("inf")))
                service_start = max(t_real, ready)
                stop_service_starts[oid_str] = round(service_start, 4)
                serv = order_services.get(oid_str, 0.0)

                # SLA lateness relative to due time
                lateness = max(0.0, service_start - due)
                stop_lateness_min[oid_str] = round(lateness, 4)
                if lateness > 1e-6:
                    soft_breaches += 1
                    if self.hard_window_mode or (t_real > due + 1e-6):
                        hard_violations += 1
                        if self.hard_window_mode:
                            errors.append(f"stop {oid_str} hard window violated: arrival {t_real:.2f} > due {due:.2f}")

                t_real = service_start + serv
                prev_node = oid_str

            # Return to depot leg
            return_leg = matrix_to_use.get(prev_node, {}).get(depot_id, 0.0)
            total_travel_min += return_leg
            t_real += return_leg
            route_duty = t_real - route_departure
            total_duty_min += route_duty

            if depot_window is not None:
                depot_due = depot_window[1]
                if t_real > depot_due + 1e-6:
                    hard_violations += 1
                    errors.append(f"vehicle {route.vehicle_id} returned to depot late: {t_real:.2f} > {depot_due:.2f}")

        time_windows_valid = (hard_violations == 0 if self.hard_window_mode else True)

        # 4. Independent Financial Cost Recomputation
        transport_cost = round(total_travel_min * self.transport_rate, 4)
        labor_cost = round(total_duty_min * self.labor_rate, 4)
        sla_penalty = round(
            sum(order_penalties.get(oid, 0.0) * stop_lateness_min.get(oid, 0.0) for oid in covered_orders),
            4,
        )
        total_financial = round(transport_cost + labor_cost + sla_penalty, 4)

        # 5. Feasibility and status
        is_feasible = (coverage_valid and capacity_valid and time_windows_valid)
        valid = is_feasible and len(errors) == 0
        status = "valid" if valid else (errors[0] if errors else "infeasible")

        # 6. Oracle Regret Recomputation (if candidate pool provided)
        oracle_best: Optional[float] = None
        regret: Optional[float] = None
        if candidate_pool is not None and is_feasible:
            oracle_best = self.find_clairvoyant_pool_optimum(
                candidate_pool=candidate_pool,
                required_orders=required_orders,
                vehicles=vehicles,
                order_demands=order_demands,
                order_windows=order_windows,
                order_services=order_services,
                order_penalties=order_penalties,
                travel_matrix=matrix_to_use,
                depot_id=depot_id,
            )
            if oracle_best is not None:
                regret = round(max(0.0, total_financial - oracle_best), 4)

        return PlanValidationResult(
            valid=valid,
            is_feasible=is_feasible,
            status=status,
            error_message="; ".join(errors) if errors else "",
            order_coverage_valid=coverage_valid,
            capacity_valid=capacity_valid,
            time_windows_valid=time_windows_valid,
            covered_orders=tuple(covered_orders),
            missing_orders=tuple(missing_orders),
            duplicate_orders=tuple(duplicate_orders),
            vehicles_used=len(seen_vehicles),
            vehicle_loads=vehicle_loads,
            vehicle_capacities=vehicle_capacities,
            stop_nominal_etas=stop_nominal_etas,
            stop_realized_etas=stop_realized_etas,
            stop_service_starts=stop_service_starts,
            stop_lateness_min=stop_lateness_min,
            hard_window_violations=hard_violations,
            soft_sla_breaches=soft_breaches,
            transport_cost_usd=transport_cost,
            labor_cost_usd=labor_cost,
            realized_sla_penalty_usd=sla_penalty,
            total_financial_cost_usd=total_financial,
            oracle_best_financial_cost_usd=oracle_best,
            decision_regret_usd=regret,
        )

    def find_clairvoyant_pool_optimum(
        self,
        candidate_pool: Sequence[CandidateRoute],
        required_orders: Sequence[str],
        vehicles: Mapping[str, float],
        order_demands: Mapping[str, float],
        order_windows: Mapping[str, Tuple[float, float]],
        order_services: Mapping[str, float],
        order_penalties: Mapping[str, float],
        travel_matrix: Mapping[str, Mapping[str, float]],
        depot_id: str = "D",
    ) -> Optional[float]:
        """Exhaustively enumerate all feasible combinations in the candidate pool to find true clairvoyant minimum cost."""
        n = len(candidate_pool)
        if n > 16:
            # Route pools for QUBO are capped at 8-12 variables. If larger, raise or prune.
            return None

        required_set = set(str(o) for o in required_orders)
        best_cost = float("inf")

        for mask in range(1, 1 << n):
            subset = [candidate_pool[i] for i in range(n) if (mask & (1 << i))]
            # Quick check: partition of required orders
            covered: List[str] = []
            duplicate = False
            for r in subset:
                for oid in r.order_ids:
                    if oid in covered:
                        duplicate = True
                        break
                    covered.append(oid)
                if duplicate:
                    break
            if duplicate or set(covered) != required_set:
                continue

            # Check distinct vehicles
            veh_ids = [r.vehicle_id for r in subset]
            if len(veh_ids) != len(set(veh_ids)):
                continue

            # Validate capacity
            cap_ok = True
            for r in subset:
                load = sum(order_demands.get(oid, 0.0) for oid in r.order_ids)
                if load > vehicles.get(r.vehicle_id, 0.0):
                    cap_ok = False
                    break
            if not cap_ok:
                continue

            # Compute realized cost
            val = self.validate_plan(
                selected_routes=subset,
                required_orders=required_orders,
                vehicles=vehicles,
                order_demands=order_demands,
                order_windows=order_windows,
                order_services=order_services,
                order_penalties=order_penalties,
                travel_time_matrix=travel_matrix,
                depot_id=depot_id,
                realized_travel_matrix=travel_matrix,
                candidate_pool=None,
            )
            if val.valid and val.total_financial_cost_usd < best_cost:
                best_cost = val.total_financial_cost_usd

        return round(best_cost, 4) if best_cost < float("inf") else None
