"""Independent mathematical verification of QUBO, Ising mapping, and Hamiltonian properties.

Verifies:
1. Binary decision-variable mapping and order.
2. Complete customer coverage constraint formulation.
3. At-most-one-route-per-vehicle conflict formulation.
4. Financial objective coefficient scaling.
5. Exact QUBO-to-Ising mapping (constant offset, h, J).
6. Energy equivalence between direct operational calculations, QUBO polynomial, and Ising Hamiltonian.
7. Constraint penalty magnitude bounds (proving ground-state feasibility vs spectral conditioning).
8. Exhaustive certificate of global optimality across 4-var (16), 8-var (256), and 12-var (4,096) instances.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from backend.domain import generate_routes
from backend.learning_dispatch import _feasible_plans, _route_id
from backend.qubo import (
    Qubo,
    build_route_qubo,
    qubo_to_ising,
    validate_selection,
)


def verify_qubo_isomorphism(qubo: Qubo) -> Dict[str, Any]:
    """Exhaustively verify direct penalties == QUBO polynomial == Ising Hamiltonian energy."""
    n = qubo.n
    total_assignments = 1 << n
    c_ising, h_ising, j_ising = qubo_to_ising(qubo)

    max_poly_direct_error = 0.0
    max_ising_direct_error = 0.0
    feasible_count = 0
    infeasible_count = 0
    min_feasible_energy = float("inf")
    min_infeasible_energy = float("inf")
    best_feasible_bits = None
    best_feasible_cost = float("inf")

    for int_val in range(total_assignments):
        bits = [(int_val >> (n - 1 - i)) & 1 for i in range(n)]
        bitstr = "".join(str(b) for b in bits)

        # 1. Direct business validation
        val = validate_selection(qubo, bits)
        business_cost = val["business_cost"]

        coverage_penalty = qubo.coverage_penalty * sum(
            (count - 1) ** 2 for count in val["coverage_counts"].values()
        )
        vehicle_penalty = 0.0
        for vehicle in set(c["vehicle"] for c in qubo.candidates):
            v_count = sum(int(b) for c, b in zip(qubo.candidates, bits) if c["vehicle"] == vehicle)
            vehicle_penalty += qubo.vehicle_penalty * v_count * (v_count - 1) / 2.0
        direct_energy = business_cost + coverage_penalty + vehicle_penalty

        # 2. QUBO polynomial energy
        poly_energy = qubo.energy(bits)
        poly_err = abs(poly_energy - direct_energy)
        if poly_err > max_poly_direct_error:
            max_poly_direct_error = poly_err

        # 3. Ising Hamiltonian energy
        z = [1 - 2 * b for b in bits]
        ising_val = (
            c_ising
            + sum(h * zi for h, zi in zip(h_ising, z))
            + sum(v * z[i] * z[k] for (i, k), v in j_ising.items())
        )
        ising_err = abs(ising_val - direct_energy)
        if ising_err > max_ising_direct_error:
            max_ising_direct_error = ising_err

        if val["feasible"]:
            feasible_count += 1
            if poly_energy < min_feasible_energy:
                min_feasible_energy = poly_energy
            if business_cost < best_feasible_cost:
                best_feasible_cost = business_cost
                best_feasible_bits = bitstr
        else:
            infeasible_count += 1
            if poly_energy < min_infeasible_energy:
                min_infeasible_energy = poly_energy

    penalty_gap = min_infeasible_energy - min_feasible_energy
    ground_state_is_feasible = penalty_gap > 0
    max_abs_j = max((abs(v) for v in j_ising.values()), default=0.0)
    max_abs_h = max((abs(v) for v in h_ising), default=0.0)

    return {
        "n_variables": n,
        "total_assignments_checked": total_assignments,
        "feasible_assignments": feasible_count,
        "infeasible_assignments": infeasible_count,
        "max_qubo_direct_discrepancy": max_poly_direct_error,
        "max_ising_direct_discrepancy": max_ising_direct_error,
        "min_feasible_energy": round(min_feasible_energy, 6),
        "min_infeasible_energy": round(min_infeasible_energy, 6),
        "penalty_margin_gap": round(penalty_gap, 6),
        "ground_state_is_feasible": ground_state_is_feasible,
        "best_feasible_bitstring": best_feasible_bits,
        "best_feasible_cost": round(best_feasible_cost, 6),
        "coverage_penalty": round(qubo.coverage_penalty, 6),
        "vehicle_penalty": round(qubo.vehicle_penalty, 6),
        "max_abs_ising_h": round(max_abs_h, 6),
        "max_abs_ising_J": round(max_abs_j, 6),
    }


def run_independent_mathematical_audit() -> Dict[str, Any]:
    """Audit mathematical properties across 4-var, 8-var, and 12-var dispatch QUBOs."""
    results: Dict[str, Any] = {}

    # Instance 1: Tiny 4-route benchmark instance (16 bitstrings)
    c4 = [
        {"id": "r0", "vehicle": "A", "stops": ["a"], "cost": 2.0},
        {"id": "r1", "vehicle": "A", "stops": ["a", "b"], "cost": 3.0},
        {"id": "r2", "vehicle": "B", "stops": ["b"], "cost": 2.0},
        {"id": "r3", "vehicle": "C", "stops": ["a", "b"], "cost": 6.0},
    ]
    q4 = build_route_qubo(c4, ["a", "b"], coverage_penalty=20.0, vehicle_penalty=15.0)
    results["tiny_4var"] = verify_qubo_isomorphism(q4)

    # Instance 2: 8-route dispatch fixture (256 bitstrings)
    fix_path = Path(__file__).resolve().parents[1] / "experiments_qaoa" / "dispatch_fixture.json"
    scenario = json.loads(fix_path.read_text(encoding="utf-8"))
    routes = generate_routes(scenario)
    c8 = [
        {
            "id": f"{r['vehicle_id']}:{'>'.join(r['order_ids'])}",
            "vehicle": r["vehicle_id"],
            "stops": r["order_ids"],
            "cost": float(r["travel_cost"]),
        }
        for r in routes
    ]
    q8 = build_route_qubo(c8, [o["id"] for o in scenario["orders"]])
    results["dispatch_8var"] = verify_qubo_isomorphism(q8)

    # Instance 3: 12-route 3-order prospective dispatch QUBO (4,096 bitstrings)
    sc12 = {
        "depot": "D",
        "vehicles": [{"id": "R1", "capacity": 3}, {"id": "R2", "capacity": 2}],
        "orders": [
            {"id": "P", "demand": 1, "window": [0, 5.5], "service": 0.7, "penalty": 4.0},
            {"id": "Q", "demand": 1, "window": [0, 7.5], "service": 0.8, "penalty": 1.5},
            {"id": "R", "demand": 2, "window": [1, 10.0], "service": 0.6, "penalty": 3.0},
        ],
        "travel": {
            "D": {"D": 0.0, "P": 2.1, "Q": 1.5, "R": 2.8},
            "P": {"D": 2.0, "P": 0.0, "Q": 1.4, "R": 2.2},
            "Q": {"D": 1.7, "P": 1.2, "Q": 0.0, "R": 1.9},
            "R": {"D": 2.6, "P": 2.1, "Q": 2.0, "R": 0.0},
        },
    }
    r12_all = generate_routes(sc12)
    active12 = {_route_id(r) for p in _feasible_plans(sc12, r12_all) for r in p["routes"]}
    c12 = [
        {
            "id": _route_id(r),
            "vehicle": r["vehicle_id"],
            "stops": r["order_ids"],
            "cost": round(float(r["travel_cost"]) * 0.50 + float(r["travel_cost"]) * 0.25, 4),
        }
        for r in r12_all
        if _route_id(r) in active12
    ]
    q12 = build_route_qubo(c12, [o["id"] for o in sc12["orders"]], coverage_penalty=18.0, vehicle_penalty=18.0)
    results["prospective_12var"] = verify_qubo_isomorphism(q12)

    # Penalty threshold verification on 8-var QUBO
    penalty_sweep = {}
    for pen in (1.5, 4.0, 8.0, 29.0, 100.0):
        qp = build_route_qubo(c8, [o["id"] for o in scenario["orders"]], coverage_penalty=pen, vehicle_penalty=pen)
        audit_p = verify_qubo_isomorphism(qp)
        penalty_sweep[f"penalty_{pen}"] = {
            "penalty": pen,
            "ground_state_is_feasible": audit_p["ground_state_is_feasible"],
            "penalty_margin_gap": audit_p["penalty_margin_gap"],
            "max_abs_ising_J": audit_p["max_abs_ising_J"],
        }
    results["penalty_threshold_study"] = penalty_sweep

    return results


if __name__ == "__main__":
    print(json.dumps(run_independent_mathematical_audit(), indent=2))
