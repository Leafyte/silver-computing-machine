"""Independent Mathematical Verifier for QUBO, Ising, and Constraint Penalties.

Exhaustively enumerates basis states on small QUBOs to audit:
1. Exact equality between direct business penalties and QUBO matrix energy.
2. Exact Ising Hamiltonian isomorphism under z_i = 1 - 2x_i (including constant offset).
3. Penalty margin gap (ensuring no infeasible state is lower in energy than any feasible state).
4. Ground state feasibility and optimal assignment identification.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class IndependentQUBOVerifier:
    """Rigorous mathematical verifier for route-pool QUBO formulations."""

    @staticmethod
    def direct_business_objective(
        route_costs: Sequence[float],
        route_stops: Sequence[Sequence[str]],
        route_vehicles: Sequence[str],
        required_stops: Sequence[str],
        bitstring: Sequence[int],
        coverage_penalty: float,
        vehicle_penalty: float,
    ) -> Tuple[float, bool]:
        """Direct computation of business cost + constraint penalties.

        E(x) = sum_i cost_i * x_i
             + P_cov * sum_{s in stops} (sum_{i covers s} x_i - 1)^2
             + P_veh * sum_{v} sum_{i < j on v} x_i * x_j
        """
        n = len(bitstring)
        # Direct route cost
        direct_cost = sum(route_costs[i] * bitstring[i] for i in range(n))

        # Coverage penalty
        cov_pen = 0.0
        cov_counts: Dict[str, int] = {s: 0 for s in required_stops}
        for i in range(n):
            if bitstring[i]:
                for s in route_stops[i]:
                    if s in cov_counts:
                        cov_counts[s] += 1
                    else:
                        cov_counts[s] = 1

        for s in required_stops:
            count = cov_counts.get(s, 0)
            cov_pen += coverage_penalty * ((count - 1) ** 2)

        # Vehicle penalty (at most one route per vehicle)
        veh_pen = 0.0
        veh_routes: Dict[str, List[int]] = {}
        for i in range(n):
            if bitstring[i]:
                veh_routes.setdefault(route_vehicles[i], []).append(i)

        for v, selected in veh_routes.items():
            if len(selected) > 1:
                # count pairs
                pairs = len(selected) * (len(selected) - 1) // 2
                veh_pen += vehicle_penalty * pairs

        total_energy = direct_cost + cov_pen + veh_pen
        is_feasible = (cov_pen == 0.0 and veh_pen == 0.0)
        return total_energy, is_feasible

    @staticmethod
    def qubo_energy(
        linear: Sequence[float],
        quadratic: Mapping[Tuple[int, int], float],
        offset: float,
        bitstring: Sequence[int],
    ) -> float:
        """Evaluate QUBO energy: offset + sum_i lin_i * x_i + sum_{i<j} quad_{ij} * x_i * x_j."""
        energy = float(offset)
        n = len(bitstring)
        for i in range(n):
            if bitstring[i]:
                energy += linear[i]
        for (i, j), val in quadratic.items():
            if bitstring[i] and bitstring[j]:
                energy += val
        return energy

    @staticmethod
    def qubo_to_ising(
        linear: Sequence[float],
        quadratic: Mapping[Tuple[int, int], float],
        offset: float,
    ) -> Tuple[float, List[float], Dict[Tuple[int, int], float]]:
        """Ising Hamiltonian under z_i = 1 - 2*x_i (x_i = (1 - z_i)/2).

        Returns:
            (constant_offset, magnetic_fields_h, coupling_matrix_J)
        """
        n = len(linear)
        c = float(offset)
        h = [0.0] * n
        J: Dict[Tuple[int, int], float] = {}

        # Linear contribution: lin_i * (1 - z_i)/2 = lin_i/2 - (lin_i/2)*z_i
        for i in range(n):
            c += linear[i] / 2.0
            h[i] -= linear[i] / 2.0

        # Quadratic contribution: quad_ij * (1 - z_i)/2 * (1 - z_j)/2
        # = quad_ij/4 * (1 - z_i - z_j + z_i*z_j)
        for (i, j), val in quadratic.items():
            c += val / 4.0
            h[i] -= val / 4.0
            h[j] -= val / 4.0
            J[(i, j)] = val / 4.0

        return c, h, J

    @staticmethod
    def ising_energy(
        c: float,
        h: Sequence[float],
        J: Mapping[Tuple[int, int], float],
        spins: Sequence[int],
    ) -> float:
        """Evaluate Ising energy: c + sum_i h_i * z_i + sum_{i<j} J_ij * z_i * z_j."""
        energy = float(c)
        n = len(spins)
        for i in range(n):
            energy += h[i] * spins[i]
        for (i, j), val in J.items():
            energy += val * spins[i] * spins[j]
        return energy

    def run_exhaustive_audit(
        self,
        qubo: Any,
        route_costs: Sequence[float],
        route_stops: Sequence[Sequence[str]],
        route_vehicles: Sequence[str],
        required_stops: Sequence[str],
    ) -> Dict[str, Any]:
        """Exhaustively verify all 2^n assignments."""
        n = qubo.n
        total_states = 1 << n
        linear = list(qubo.linear)
        quadratic = dict(qubo.quadratic)
        offset = float(qubo.offset)
        cov_penalty = float(qubo.coverage_penalty)
        veh_penalty = float(qubo.vehicle_penalty)

        c_ising, h_ising, J_ising = self.qubo_to_ising(linear, quadratic, offset)

        max_direct_qubo_diff = 0.0
        max_qubo_ising_diff = 0.0
        feasible_count = 0
        min_feasible_energy = float("inf")
        max_feasible_energy = float("-inf")
        min_infeasible_energy = float("inf")
        ground_state_bits: Optional[Tuple[int, ...]] = None
        min_overall_energy = float("inf")

        for int_rep in range(total_states):
            bits = tuple(int(b) for b in format(int_rep, f"0{n}b"))
            spins = tuple(1 - 2 * b for b in bits)

            direct_e, is_feas = self.direct_business_objective(
                route_costs, route_stops, route_vehicles, required_stops,
                bits, cov_penalty, veh_penalty,
            )
            q_e = self.qubo_energy(linear, quadratic, offset, bits)
            ising_e = self.ising_energy(c_ising, h_ising, J_ising, spins)

            diff1 = abs(direct_e - q_e)
            diff2 = abs(q_e - ising_e)
            if diff1 > max_direct_qubo_diff:
                max_direct_qubo_diff = diff1
            if diff2 > max_qubo_ising_diff:
                max_qubo_ising_diff = diff2

            if q_e < min_overall_energy:
                min_overall_energy = q_e
                ground_state_bits = bits

            if is_feas:
                feasible_count += 1
                if q_e < min_feasible_energy:
                    min_feasible_energy = q_e
                if q_e > max_feasible_energy:
                    max_feasible_energy = q_e
            else:
                if q_e < min_infeasible_energy:
                    min_infeasible_energy = q_e

        # Penalty margin gap: E_min(infeasible) - E_max(feasible)
        # For a properly penalized QUBO, every feasible state should ideally be cheaper than infeasible states
        penalty_margin_gap = min_infeasible_energy - max_feasible_energy if feasible_count > 0 else None
        ground_state_is_feasible = (min_overall_energy == min_feasible_energy)

        return {
            "n_variables": n,
            "basis_states_checked": total_states,
            "feasible_states_count": feasible_count,
            "feasible_fraction": feasible_count / total_states,
            "max_direct_vs_qubo_discrepancy": round(max_direct_qubo_diff, 14),
            "max_qubo_vs_ising_discrepancy": round(max_qubo_ising_diff, 14),
            "min_feasible_energy": round(min_feasible_energy, 4) if feasible_count else None,
            "max_feasible_energy": round(max_feasible_energy, 4) if feasible_count else None,
            "min_infeasible_energy": round(min_infeasible_energy, 4),
            "penalty_margin_gap": round(penalty_margin_gap, 4) if penalty_margin_gap is not None else None,
            "ground_state_bits": "".join(str(b) for b in ground_state_bits) if ground_state_bits else None,
            "ground_state_feasible": ground_state_is_feasible,
            "ground_state_energy": round(min_overall_energy, 4),
            "mathematically_isomorphic": (max_direct_qubo_diff < 1e-9 and max_qubo_ising_diff < 1e-9),
        }
