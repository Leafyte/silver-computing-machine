"""Independent Benchmark Harness Interfaces and Data Models.

Defines reusable data structures for scenarios, candidate routes, predictions,
plans, validation outcomes, quantum solver metrics, and 5-stage resource accounting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CandidateRoute:
    """A candidate route proposed by the route generation phase."""
    id: str
    vehicle_id: str
    order_ids: Tuple[str, ...]
    stops: Tuple[str, ...]  # Sequence of customer stops visited
    forecast_duration_min: float
    forecast_transport_cost: float
    forecast_labor_cost: float
    expected_breach_severity_min: float
    penalty_rate_per_min: float
    features: Tuple[float, float]
    raw_route: Optional[Tuple[int, ...]] = None  # Native node indices if available

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("route id must be non-empty")
        if not self.vehicle_id:
            raise ValueError("vehicle_id must be non-empty")
        if not self.stops:
            raise ValueError("route must visit at least one customer stop")
        if not math.isfinite(self.forecast_duration_min) or self.forecast_duration_min < 0:
            raise ValueError("forecast_duration_min must be finite and non-negative")


@dataclass(frozen=True)
class PlanValidationResult:
    """Complete independent validation result for a selected dispatch plan."""
    valid: bool
    is_feasible: bool
    status: str
    error_message: str
    # Constraint checks
    order_coverage_valid: bool
    capacity_valid: bool
    time_windows_valid: bool
    covered_orders: Tuple[str, ...]
    missing_orders: Tuple[str, ...]
    duplicate_orders: Tuple[str, ...]
    vehicles_used: int
    vehicle_loads: Mapping[str, float]
    vehicle_capacities: Mapping[str, float]
    # Detailed timing & ETAs
    stop_nominal_etas: Mapping[str, float]
    stop_realized_etas: Mapping[str, float]
    stop_service_starts: Mapping[str, float]
    stop_lateness_min: Mapping[str, float]
    hard_window_violations: int
    soft_sla_breaches: int
    # Recomputed financial objective components
    transport_cost_usd: float
    labor_cost_usd: float
    realized_sla_penalty_usd: float
    total_financial_cost_usd: float
    # Regret vs Oracle
    oracle_best_financial_cost_usd: Optional[float] = None
    decision_regret_usd: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "is_feasible": self.is_feasible,
            "status": self.status,
            "error_message": self.error_message,
            "order_coverage_valid": self.order_coverage_valid,
            "capacity_valid": self.capacity_valid,
            "time_windows_valid": self.time_windows_valid,
            "covered_orders": list(self.covered_orders),
            "missing_orders": list(self.missing_orders),
            "duplicate_orders": list(self.duplicate_orders),
            "vehicles_used": self.vehicles_used,
            "vehicle_loads": dict(self.vehicle_loads),
            "vehicle_capacities": dict(self.vehicle_capacities),
            "stop_nominal_etas": dict(self.stop_nominal_etas),
            "stop_realized_etas": dict(self.stop_realized_etas),
            "stop_service_starts": dict(self.stop_service_starts),
            "stop_lateness_min": dict(self.stop_lateness_min),
            "hard_window_violations": self.hard_window_violations,
            "soft_sla_breaches": self.soft_sla_breaches,
            "transport_cost_usd": self.transport_cost_usd,
            "labor_cost_usd": self.labor_cost_usd,
            "realized_sla_penalty_usd": self.realized_sla_penalty_usd,
            "total_financial_cost_usd": self.total_financial_cost_usd,
            "oracle_best_financial_cost_usd": self.oracle_best_financial_cost_usd,
            "decision_regret_usd": self.decision_regret_usd,
        }


@dataclass(frozen=True)
class QuantumSolverMetrics:
    """Metrics for quantum optimization execution on a route pool QUBO."""
    backend_name: str
    execution_type: str  # local_cpu_statevector, qcloud, qpu
    qubits: int
    circuit_depth: int
    single_qubit_gates: int
    two_qubit_gates: int
    layers: int
    shots: int
    circuit_evaluations: int
    feasible_sample_probability: float
    probability_best_known_subproblem: float
    best_feasible_qubo_energy: Optional[float]
    best_feasible_objective_cost: Optional[float]
    exact_subproblem_optimum: float
    subproblem_optimality_gap: Optional[float]
    fallback_used: bool
    fallback_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "execution_type": self.execution_type,
            "qubits": self.qubits,
            "circuit_depth": self.circuit_depth,
            "single_qubit_gates": self.single_qubit_gates,
            "two_qubit_gates": self.two_qubit_gates,
            "layers": self.layers,
            "shots": self.shots,
            "circuit_evaluations": self.circuit_evaluations,
            "feasible_sample_probability": self.feasible_sample_probability,
            "probability_best_known_subproblem": self.probability_best_known_subproblem,
            "best_feasible_qubo_energy": self.best_feasible_qubo_energy,
            "best_feasible_objective_cost": self.best_feasible_objective_cost,
            "exact_subproblem_optimum": self.exact_subproblem_optimum,
            "subproblem_optimality_gap": self.subproblem_optimality_gap,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class StageResourceBreakdown:
    """Precise 5-stage CPU and wall-clock breakdown for end-to-end solve."""
    preprocessing_wall_seconds: float
    preprocessing_cpu_seconds: float
    lp_relaxation_wall_seconds: float
    lp_relaxation_cpu_seconds: float
    angle_optimization_wall_seconds: float
    angle_optimization_cpu_seconds: float
    circuit_execution_wall_seconds: float
    circuit_execution_cpu_seconds: float
    validation_wall_seconds: float
    validation_cpu_seconds: float
    total_wall_seconds: float
    total_cpu_seconds: float
    peak_memory_bytes: int
    circuit_evaluations: int
    deadline_budget_seconds: Optional[float] = None
    budget_exceeded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preprocessing_wall_seconds": self.preprocessing_wall_seconds,
            "preprocessing_cpu_seconds": self.preprocessing_cpu_seconds,
            "lp_relaxation_wall_seconds": self.lp_relaxation_wall_seconds,
            "lp_relaxation_cpu_seconds": self.lp_relaxation_cpu_seconds,
            "angle_optimization_wall_seconds": self.angle_optimization_wall_seconds,
            "angle_optimization_cpu_seconds": self.angle_optimization_cpu_seconds,
            "circuit_execution_wall_seconds": self.circuit_execution_wall_seconds,
            "circuit_execution_cpu_seconds": self.circuit_execution_cpu_seconds,
            "validation_wall_seconds": self.validation_wall_seconds,
            "validation_cpu_seconds": self.validation_cpu_seconds,
            "total_wall_seconds": self.total_wall_seconds,
            "total_cpu_seconds": self.total_cpu_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "circuit_evaluations": self.circuit_evaluations,
            "deadline_budget_seconds": self.deadline_budget_seconds,
            "budget_exceeded": self.budget_exceeded,
        }


@dataclass(frozen=True)
class PredictionMetrics:
    """Predictive accuracy and probability calibration metrics."""
    sample_count: int
    positive_count: int
    positive_rate: float
    brier_score: float
    expected_calibration_error: float
    log_loss: Optional[float]
    brier_95ci_halfwidth: float
    calibration_bins: List[Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "positive_rate": self.positive_rate,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "log_loss": self.log_loss,
            "brier_95ci_halfwidth": self.brier_95ci_halfwidth,
            "calibration_bins": self.calibration_bins,
        }
