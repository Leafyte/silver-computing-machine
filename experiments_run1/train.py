"""Leakage-safe RUN 1 training utilities and synthetic replay labels.

This module consumes routes already verified by ``experiments_run1.native``.
    It never supplies a static geometry travel substitute to a native evaluator;
    time-dependent replay traces are requested from the native instance. Synthetic
replay labels are benchmark-derived, not observed incident or traffic labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import random
import time
import tracemalloc
from typing import Any, Iterable, Mapping, Sequence

from experiments_run1.model import (
    BASE_VARIANT,
    DEPTH2_VARIANT,
    NON_ENTANGLING_VARIANT,
    FeatureScaler,
    LogisticModel,
    ScaledEstimatorModel,
    VQCModel,
)


DEFAULT_SOFT_CUTOFF_FRACTION = 0.90
DEFAULT_COMMON_DELAY_FRACTION = 0.04
DEFAULT_ROUTE_DELAY_FRACTION = 0.02
DEFAULT_MAX_SAFE_OVERLAY_RATE = 64.0
DEFAULT_PENALTY_RATE = 1.0


class NativeOverlayCapError(RuntimeError):
    """A nominal source route lacks a finite native ATF-domain cap."""
FEATURE_SCHEMA = {
    "feature_count": 8,
    "feature_0": "minimum nominal soft-cutoff slack divided by source-native horizon",
    "feature_1": "native nominal route duration divided by source-native horizon",
    "feature_2": "mean nominal soft-cutoff slack divided by source-native horizon",
    "feature_3": "standard deviation of nominal soft-cutoff slack divided by source-native horizon",
    "feature_4": "route stop count divided by instance customer count",
    "feature_5": "total route demand divided by vehicle capacity",
    "feature_6": "maximum leg travel time divided by route duration",
    "feature_7": "service duration divided by route duration",
    "scale": "train-only min-max to [0,1], validation/test clipped",
    "unavailable": ["observed traffic", "departure-time sensitivity as a separate input"],
    "model_contract": "expanded 8-feature input for GBDT; VQC retains 2-feature subset",
}
LABEL_CONTRACT = {
    "synthetic": True,
    "observed_incident_data": False,
    "target": "binary route-level synthetic soft-SLA breach after native replay",
    "label": "route has any service start after the soft cutoff in the synthetic replay",
    "severity": "sum of late time units across route stops",
    "soft_cutoff": "ready + 0.90 * (hard due - ready)",
    "protocol_version": "RUN1-v2-safe-capped-all-source",
    "uncertainty_model": "seeded positive common and route-specific fractional additive overlay on each native leg travel time; fixed nominal departure; no departure re-optimization",
    "uncertainty_domain_cap": "Dabia requested common+route rate clipped per nominal route to a precomputed native ATF-domain-valid cap before replay seeds; clipping requested/applied rate and mass are recorded; no censored rows are relabeled or dropped",
    "common_effect": "seeded positive common fractional additive native-leg travel delay",
    "route_effect": "seeded positive route-specific fractional additive native-leg travel delay",
    "service_delay": "zero in v1 replay; native service times remain unchanged",
    "scenario_seed_policy": "common shock seed derived from replay seed and topology ID; route shock additionally keyed by route ID",
    "observed_outcomes": False,
    "currency_calibrated": False,
}


@dataclass(frozen=True)
class RouteSample:
    """One route under one frozen synthetic replay seed."""

    source: str
    topology_id: str
    instance_id: str
    replay_seed: int
    route_id: str
    raw_features: tuple[float, ...]
    label: int
    lateness_severity: float
    forecast_cost: float
    penalty_rate: float = DEFAULT_PENALTY_RATE
    hard_window_violations: int = 0
    native_duration: float = 0.0
    forecast_travel_duration: float = 0.0
    forecast_duty_duration: float = 0.0
    realized_travel_duration: float = 0.0
    realized_duty_duration: float = 0.0
    realized_total_cost: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("route label must be binary")
        if not self.raw_features or not all(math.isfinite(float(v)) for v in self.raw_features):
            raise ValueError("raw_features must contain finite values")
        if not math.isfinite(self.lateness_severity) or self.lateness_severity < 0:
            raise ValueError("lateness_severity must be finite and non-negative")
        if not math.isfinite(self.forecast_cost) or self.forecast_cost < 0:
            raise ValueError("forecast_cost must be finite and non-negative")
        if not math.isfinite(self.penalty_rate) or self.penalty_rate < 0:
            raise ValueError("penalty_rate must be finite and non-negative")
        for name in ("native_duration", "forecast_travel_duration", "forecast_duty_duration",
                     "realized_travel_duration", "realized_duty_duration", "realized_total_cost"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class SyntheticFinancialRates:
    """Frozen hypothetical cost rates; not calibrated financial parameters."""

    transport_per_native_travel_time_unit: float = 0.50
    labor_per_native_duty_time_unit: float = 0.25
    penalty_per_late_native_time_unit: float = 1.0
    unit_name: str = "synthetic_cost_units_v1"

    def __post_init__(self) -> None:
        rates = (self.transport_per_native_travel_time_unit,
                 self.labor_per_native_duty_time_unit,
                 self.penalty_per_late_native_time_unit)
        if any(not math.isfinite(value) or value < 0.0 for value in rates):
            raise ValueError("synthetic financial rates must be finite and non-negative")
        if not self.unit_name:
            raise ValueError("unit_name must be non-empty")

    def base_cost(self, travel_duration: float, duty_duration: float) -> float:
        return (self.transport_per_native_travel_time_unit * travel_duration
                + self.labor_per_native_duty_time_unit * duty_duration)

    def total_cost(self, travel_duration: float, duty_duration: float, late_time: float) -> float:
        return (self.base_cost(travel_duration, duty_duration)
                + self.penalty_per_late_native_time_unit * late_time)

    def to_dict(self) -> dict[str, Any]:
        return {"unit_name": self.unit_name, "currency": None,
                "calibration": "assumed synthetic rates; not real-world currency calibration",
                "transport_per_native_travel_time_unit": self.transport_per_native_travel_time_unit,
                "labor_per_native_duty_time_unit": self.labor_per_native_duty_time_unit,
                "penalty_per_late_native_time_unit": self.penalty_per_late_native_time_unit}


@dataclass(frozen=True)
class PlanOutcome:
    route_ids: tuple[str, ...]
    realized_cost: float
    hard_window_violations: int = 0
    objective_unit: str = "synthetic_cost_units_v1"

    def __post_init__(self) -> None:
        if not self.route_ids:
            raise ValueError("plan must select at least one route")
        if not math.isfinite(self.realized_cost) or self.realized_cost < 0:
            raise ValueError("realized_cost must be finite and non-negative")
        if not self.objective_unit:
            raise ValueError("objective_unit must be documented")


@dataclass(frozen=True)
class DecisionScenario:
    """Plan alternatives for one topology/replay; outcomes are training labels."""

    scenario_id: str
    topology_id: str
    split: str
    routes: tuple[RouteSample, ...]
    plans: tuple[PlanOutcome, ...]

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if not self.routes or not self.plans:
            raise ValueError("decision scenarios need routes and plan alternatives")
        route_ids = {route.route_id for route in self.routes}
        if any(not set(plan.route_ids).issubset(route_ids) for plan in self.plans):
            raise ValueError("every plan route id must exist in the scenario route set")
        if any(route.topology_id != self.topology_id for route in self.routes):
            raise ValueError("scenario may contain only routes from its topology group")


@dataclass(frozen=True)
class FitResult:
    model: VQCModel
    history: tuple[Mapping[str, float], ...]
    selected_step: int
    wall_seconds: float
    cpu_seconds: float
    peak_memory_bytes: int
    circuit_evaluations: int


@dataclass(frozen=True)
class ClassicalFitResult:
    model: Any
    selected_step: int
    train_loss: float
    validation_loss: float
    wall_seconds: float
    cpu_seconds: float
    configuration: Mapping[str, Any]


def topology_fingerprint(instance: Any, *, prefix_customers: int = 25) -> str:
    """Hash ordered customer coordinates, ignoring windows and replay rows.

    The pinned Solomon/Dabia topology audit establishes 25/50/100 prefix
    identity, exact cross-source copies, and affine-equivalent R1/R2 and
    RC1/RC2 geometries. Those aliases are grouped by audited geometry family
    below. For other fixture objects, use the exact ordered-coordinate prefix
    digest so unit tests and future sources remain deterministic.
    """
    instance_id = str(getattr(instance, "instance_id", ""))
    if instance_id:
        name = instance_id.upper()
        if name.startswith("RC1") or name.startswith("RC2"):
            return "topology:audited:RC"
        if name.startswith("R1") or name.startswith("R2"):
            return "topology:audited:R"
        if name.startswith("C1"):
            return "topology:audited:C1"
        if name.startswith("C2"):
            return "topology:audited:C2"
    coordinates = getattr(instance, "coordinates", None)
    if coordinates is None:
        raw = getattr(instance, "raw", {})
        coordinates = raw.get("coordinates") if isinstance(raw, Mapping) else None
    if coordinates is None:
        raise ValueError("native instance must expose canonical coordinates")
    customers = _coordinate_rows(coordinates)
    if not customers:
        raise ValueError("native instance has no customer coordinates")
    payload = [[_number_text(x), _number_text(y)] for _, x, y in customers[:prefix_customers]]
    digest = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
    return f"topology:{digest}"


def _coordinate_rows(coordinates: Any) -> list[tuple[int, float, float]]:
    rows: list[tuple[int, float, float]] = []
    if isinstance(coordinates, Mapping):
        for key, value in coordinates.items():
            try:
                node_id = int(key)
                if isinstance(value, Mapping):
                    x, y = value["x"], value["y"]
                else:
                    x, y = value[0], value[1]
                rows.append((node_id, float(x), float(y)))
            except (TypeError, ValueError, IndexError):
                continue
    elif isinstance(coordinates, Sequence) and not isinstance(coordinates, (str, bytes)):
        for index, value in enumerate(coordinates):
            try:
                if isinstance(value, Mapping):
                    node_id = int(value.get("id", index))
                    x, y = value["x"], value["y"]
                else:
                    node_id, x, y = index, value[0], value[1]
                rows.append((node_id, float(x), float(y)))
            except (TypeError, ValueError, KeyError, IndexError):
                continue
    return sorted((row for row in rows if row[0] != 0), key=lambda row: row[0])


def _number_text(value: float) -> str:
    return format(float(value), ".12g")


def assign_topology_splits(
    topology_ids: Iterable[str], *, seed: int = 20261029,
    train_fraction: float = 0.60, validation_fraction: float = 0.20,
) -> dict[str, str]:
    """Assign whole topology groups to splits, never replay rows individually."""
    ids = sorted(set(str(value) for value in topology_ids))
    if len(ids) < 3:
        raise ValueError("at least three independent topology groups are required for train/validation/test")
    if not (0.0 < train_fraction < 1.0 and 0.0 < validation_fraction < 1.0 and
            train_fraction + validation_fraction < 1.0):
        raise ValueError("split fractions must be positive and sum to less than one")
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = max(1, min(n - 2, round(n * train_fraction)))
    n_validation = max(1, min(n - n_train - 1, round(n * validation_fraction)))
    mapping = {group_id: "train" for group_id in ids[:n_train]}
    mapping.update({group_id: "validation" for group_id in ids[n_train:n_train + n_validation]})
    mapping.update({group_id: "test" for group_id in ids[n_train + n_validation:]})
    return mapping


def assert_group_split_integrity(samples: Iterable[RouteSample], split_map: Mapping[str, str]) -> None:
    seen: dict[str, str] = {}
    for sample in samples:
        split = split_map.get(sample.topology_id)
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"no split assigned for topology {sample.topology_id}")
        previous = seen.setdefault(sample.topology_id, split)
        if previous != split:
            raise ValueError(f"topology leakage across splits: {sample.topology_id}")


def generate_route_replays(
    instance: Any,
    routes: Sequence[Sequence[int]],
    replay_seeds: Sequence[int],
    *,
    source: str,
    soft_cutoff_fraction: float = DEFAULT_SOFT_CUTOFF_FRACTION,
    common_delay_fraction: float = DEFAULT_COMMON_DELAY_FRACTION,
    route_delay_fraction: float = DEFAULT_ROUTE_DELAY_FRACTION,
    financial_rates: SyntheticFinancialRates = SyntheticFinancialRates(),
    safe_overlay_caps: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[RouteSample]:
    """Create source-specific synthetic departure-shock labels from native traces.

    Native optimized-departure evaluation defines forecast features. Each
    replay keeps that nominal departure fixed, adds the same seeded common
    travel-delay fraction plus a route-specific fraction to native leg travel
    times, and asks the adapter to trace the route with those overlays. Dabia
    first caps the requested total rate using a fixed-departure native ATF-domain
    probe based only on nominal route data; common and route fractions are
    proportionally scaled and clipping is recorded. Solomon uses its static
    native trace without a finite ATF cap. A soft cutoff is frozen at 90% of
    each hard window by default. These labels are synthetic stress labels, not
    observed congestion or real contractual SLAs.
    """
    if not (0.0 < soft_cutoff_fraction <= 1.0):
        raise ValueError("soft_cutoff_fraction must lie in (0, 1]")
    if min(common_delay_fraction, route_delay_fraction) < 0.0:
        raise ValueError("delay fractions must be non-negative")
    if not hasattr(instance, "evaluate_route") or not hasattr(instance, "trace_route_at_departure"):
        raise TypeError("native instance must provide evaluate_route and trace_route_at_departure")

    topology_id = topology_fingerprint(instance)
    instance_id = str(getattr(instance, "instance_id", getattr(instance, "base_id", "unknown")))
    windows = _time_window_map(instance)
    horizon = max((due - ready for ready, due in windows.values()), default=1.0)
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("native instance must have positive finite time-window horizon")

    nominal: dict[str, tuple[Sequence[int], Any, tuple[float, float], Mapping[str, Any]]] = {}
    for route in routes:
        route_seq = tuple(int(node) for node in route)
        result = instance.evaluate_route(route_seq)
        if not bool(_get(result, "feasible", False)):
            continue
        duration = float(_get(result, "duration"))
        route_id = _stable_route_id(route_seq)
        slacks = [float(value) for value in _get(result, "window_slacks", ())]
        service_starts = [float(value) for value in _get(result, "service_starts", ())]
        if len(service_starts) != len(route_seq) or len(slacks) != len(route_seq):
            raise ValueError(f"native trace for {route_id} does not align with route stops")
        soft_slacks = []
        for index, node in enumerate(route_seq):
            ready, due = windows[node]
            soft_due = ready + soft_cutoff_fraction * (due - ready)
            soft_slacks.append(soft_due - service_starts[index])
        # --- Expanded 8-feature extraction (was 2-feature) ---
        _demands = getattr(instance, "demands", None)
        _capacity = getattr(instance, "capacity", 1)
        _service_times = getattr(instance, "service_times", None)
        _num_customers = getattr(instance, "num_customers", len(route_seq))
        _leg_travel_times = [float(v) for v in _get(result, "leg_travel_times", ())]
        _service_duration_val = float(_get(result, "service_duration", 0.0))

        _mean_soft_slack = sum(soft_slacks) / len(soft_slacks) if soft_slacks else 0.0
        _std_soft_slack = (
            (sum((s - _mean_soft_slack) ** 2 for s in soft_slacks) / len(soft_slacks)) ** 0.5
            if len(soft_slacks) > 1 else 0.0
        )
        _route_len_frac = len(route_seq) / max(_num_customers, 1)
        _demand_frac = (
            sum(float(_demands[n]) for n in route_seq) / float(_capacity)
            if _demands is not None and _capacity > 0 else 0.0
        )
        _max_leg_frac = (
            max(_leg_travel_times) / duration if _leg_travel_times and duration > 0.0 else 0.0
        )
        _service_ratio = _service_duration_val / duration if duration > 0.0 else 0.0

        raw_features = (
            min(soft_slacks) / horizon,          # feat 0: original
            duration / horizon,                   # feat 1: original
            _mean_soft_slack / horizon,           # feat 2: average slack buffer
            _std_soft_slack / horizon,            # feat 3: slack heterogeneity
            _route_len_frac,                      # feat 4: route length fraction
            _demand_frac,                         # feat 5: capacity utilisation
            _max_leg_frac,                        # feat 6: longest leg fraction
            _service_ratio,                       # feat 7: service time ratio
        )
        if source == "dabia2013":
            cap_info = (safe_overlay_caps.get(route_id) if safe_overlay_caps is not None
                        else compute_safe_overlay_caps(instance, [route_seq]).get(route_id))
            if not isinstance(cap_info, Mapping) or cap_info.get("status") not in {
                "upper_bound_safe", "safe_bisection_lower_bound"
            }:
                raise NativeOverlayCapError(f"no native-domain-valid replay cap for {route_id}: {cap_info}")
            cap_value = float(cap_info["max_fraction"])
            if not math.isfinite(cap_value) or cap_value < 0.0:
                raise NativeOverlayCapError(f"invalid native replay cap for {route_id}: {cap_info}")
            cap_info = dict(cap_info)
        else:
            cap_info = {"max_fraction": None, "status": "uncapped_static_source",
                        "method": "Solomon Euclidean static travel has no ATF input domain"}
        nominal[route_id] = (route_seq, result, raw_features, cap_info)

    samples: list[RouteSample] = []
    source_name = str(source).lower()
    for seed in sorted(set(int(value) for value in replay_seeds)):
        scenario_rng = random.Random(_seeded_int(f"{seed}:{topology_id}:common"))
        common_rate = _positive_normal(scenario_rng, common_delay_fraction)
        for route_id, (route, nominal_result, raw_features, cap_info) in nominal.items():
            route_rng = random.Random(_seeded_int(f"{seed}:{topology_id}:{route_id}:route"))
            route_rate = _positive_normal(route_rng, route_delay_fraction)
            leg_times = [float(value) for value in _get(nominal_result, "leg_travel_times", ())]
            if len(leg_times) != len(route) + 1:
                raise ValueError(f"native leg trace for {route_id} does not include each leg and depot return")
            requested_rate = common_rate + route_rate
            cap_value = cap_info.get("max_fraction")
            applied_rate = (requested_rate if cap_value is None
                            else min(requested_rate, float(cap_value)))
            scale = applied_rate / requested_rate if requested_rate > 0.0 else 1.0
            applied_common_rate = common_rate * scale
            applied_route_rate = route_rate * scale
            travel_delays = [leg_time * applied_rate for leg_time in leg_times]
            service_delays = [0.0] * len(route)
            selected_departure = _get(nominal_result, "optimized_departure_time", None)
            if selected_departure is None:
                # Solomon's static benchmark has no departure optimizer; its
                # frozen dispatch departure is the depot-ready time.
                selected_departure = windows.get(0, (0.0, 0.0))[0]
                departure_rule = "fixed_depot_ready_time"
            else:
                departure_rule = "canonical_nominal_optimized_departure"
            departure = float(selected_departure)
            replay = instance.trace_route_at_departure(
                route, departure, travel_delays=travel_delays, service_delays=service_delays
            )
            replay_status = str(_get(replay, "status", "valid"))
            replay_duration = _get(replay, "duration", None)
            replay_travel_duration = _get(replay, "travel_duration", None)
            if replay_duration is None or replay_travel_duration is None:
                # A time-dependent trace that leaves its source ATF domain is
                # censored, not a negative label and never extrapolated.
                error = str(_get(replay, "error", ""))
                raise ValueError(
                    f"native replay is censored for {route_id}, seed={seed}: "
                    f"status={replay_status}; error={error}"
                )
            service_starts = [float(value) for value in _get(replay, "service_starts", ())]
            if len(service_starts) != len(route):
                raise ValueError(f"synthetic replay trace for {route_id} does not align with route stops")
            late_minutes = []
            hard_violations = 0
            for index, node in enumerate(route):
                ready, due = windows[node]
                soft_due = ready + soft_cutoff_fraction * (due - ready)
                late_minutes.append(max(0.0, service_starts[index] - soft_due))
                hard_violations += int(service_starts[index] > due)
            severity = sum(late_minutes)
            realized_travel = float(replay_travel_duration)
            realized_duty = float(replay_duration)
            forecast_duty = float(_get(nominal_result, "duration"))
            forecast_travel = float(_get(nominal_result, "travel_duration"))
            realized_total_cost = financial_rates.total_cost(realized_travel, realized_duty, severity)
            samples.append(RouteSample(
                source=source_name,
                topology_id=topology_id,
                instance_id=instance_id,
                replay_seed=seed,
                route_id=route_id,
                raw_features=raw_features,
                label=int(severity > 0.0),
                lateness_severity=severity,
                forecast_cost=financial_rates.base_cost(forecast_travel, forecast_duty),
                penalty_rate=financial_rates.penalty_per_late_native_time_unit,
                hard_window_violations=hard_violations,
                native_duration=forecast_duty,
                forecast_travel_duration=forecast_travel,
                forecast_duty_duration=forecast_duty,
                realized_travel_duration=realized_travel,
                realized_duty_duration=realized_duty,
                realized_total_cost=realized_total_cost,
                metadata={
                    "label_source": "synthetic_additive_leg_travel_overlay",
                    "uncertainty_protocol_version": "RUN1-v2-safe-capped-all-source",
                    "nominal_departure_fixed": True,
                    "soft_cutoff_fraction": soft_cutoff_fraction,
                    "soft_cutoff_definition": "ready + fraction * (hard_due - ready)",
                    "synthetic_cost_units": financial_rates.unit_name,
                    "financial_rate_contract": financial_rates.to_dict(),
                    "forecast_transport_cost": financial_rates.transport_per_native_travel_time_unit * forecast_travel,
                    "forecast_labor_cost": financial_rates.labor_per_native_duty_time_unit * forecast_duty,
                    "realized_transport_cost": financial_rates.transport_per_native_travel_time_unit * realized_travel,
                    "realized_labor_cost": financial_rates.labor_per_native_duty_time_unit * realized_duty,
                    "realized_sla_penalty": financial_rates.penalty_per_late_native_time_unit * severity,
                    "realized_total_cost": realized_total_cost,
                    "common_effect": "seeded common fractional travel-time overlay",
                    "common_delay_rate": common_rate,
                    "route_delay_rate": route_rate,
                    "requested_total_overlay_rate": requested_rate,
                    "applied_common_delay_rate": applied_common_rate,
                    "applied_route_delay_rate": applied_route_rate,
                    "applied_total_overlay_rate": applied_rate,
                    "safe_overlay_cap": cap_value,
                    "safe_overlay_cap_metadata": dict(cap_info),
                    "overlay_clipped_rate_mass": max(0.0, requested_rate - applied_rate),
                    "overlay_clipped_fraction": (max(0.0, requested_rate - applied_rate) / requested_rate
                                                 if requested_rate > 0.0 else 0.0),
                    "travel_delays_by_leg": travel_delays,
                    "service_delays_by_stop": service_delays,
                    "departure_time": departure,
                    "departure_rule": departure_rule,
                    "native_objective": str(getattr(instance, "native_objective", "unavailable")),
                    "dataset_size": getattr(instance, "size", None),
                    "replay_engine": "native_static_trace" if source_name == "solomon" else "native_time_dependent_trace",
                },
            ))
    return samples


def compute_safe_overlay_caps(
    instance: Any, routes: Sequence[Sequence[int]], *, upper_rate: float = DEFAULT_MAX_SAFE_OVERLAY_RATE,
) -> dict[str, dict[str, Any]]:
    """Compute deterministic Dabia ATF-domain caps from nominal route data only."""
    caps: dict[str, dict[str, Any]] = {}
    if str(getattr(instance, "source", "")).lower() != "dabia2013":
        return caps
    safe_rate = getattr(instance, "safe_overlay_rate", None)
    if not callable(safe_rate):
        raise NativeOverlayCapError("Dabia native adapter lacks safe_overlay_rate domain cap")
    for route in routes:
        normalized = tuple(int(node) for node in route)
        route_id = _stable_route_id(normalized)
        info = safe_rate(normalized, upper_rate=upper_rate)
        if not isinstance(info, Mapping) or info.get("status") not in {
            "upper_bound_safe", "safe_bisection_lower_bound"
        }:
            raise NativeOverlayCapError(f"native domain cap failed for {route_id}: {info}")
        fraction = info.get("max_fraction")
        if fraction is None or not math.isfinite(float(fraction)) or float(fraction) < 0.0:
            raise NativeOverlayCapError(f"native domain cap is not finite for {route_id}: {info}")
        caps[route_id] = dict(info)
    return caps


def _time_window_map(instance: Any) -> dict[int, tuple[float, float]]:
    windows = getattr(instance, "time_windows", None)
    if windows is None:
        raw = getattr(instance, "raw", {})
        windows = raw.get("time_windows") if isinstance(raw, Mapping) else None
    if isinstance(windows, Mapping):
        result = {int(key): _window_pair(value) for key, value in windows.items()}
    elif isinstance(windows, Sequence):
        start_index = 0 if len(windows) == int(getattr(instance, "num_customers", len(windows) - 1)) + 1 else 1
        result = {index + start_index: _window_pair(value) for index, value in enumerate(windows)}
    else:
        raise ValueError("native instance must expose time_windows")
    return result


def _window_pair(value: Any) -> tuple[float, float]:
    if isinstance(value, Mapping):
        return float(value.get("ready_time", value.get("ready"))), float(value.get("due_time", value.get("due")))
    return float(value[0]), float(value[1])


def _get(value: Any, key: str, default: Any = ... ) -> Any:
    if isinstance(value, Mapping):
        if default is ...:
            return value[key]
        return value.get(key, default)
    if default is ...:
        return getattr(value, key)
    return getattr(value, key, default)


def _stable_route_id(route: Sequence[int]) -> str:
    return "route:" + ">".join(str(value) for value in route)


def _seeded_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _positive_normal(rng: random.Random, scale: float) -> float:
    if scale == 0.0:
        return 0.0
    return max(0.0, rng.gauss(scale, scale * 0.5))


def fit_train_only_scaler(samples: Sequence[RouteSample], split_map: Mapping[str, str]) -> FeatureScaler:
    assert_group_split_integrity(samples, split_map)
    training = [sample for sample in samples if split_map[sample.topology_id] == "train"]
    if not training:
        raise ValueError("training split contains no route samples")
    return FeatureScaler.fit(
        (sample.raw_features for sample in training),
        group_ids=(sample.topology_id for sample in training),
    )


def binary_metrics(labels: Sequence[int], probabilities: Sequence[float], *, bins: int = 10) -> dict[str, Any]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("metrics require equally sized non-empty labels and probabilities")
    if bins < 1:
        raise ValueError("bins must be positive")
    pairs = [(int(y), float(p)) for y, p in zip(labels, probabilities)]
    if any(y not in (0, 1) or not math.isfinite(p) or not 0.0 <= p <= 1.0 for y, p in pairs):
        raise ValueError("metrics received invalid labels or probabilities")
    count = len(pairs)
    brier = sum((p - y) ** 2 for y, p in pairs) / count
    log_loss = -sum(y * math.log(max(p, 1e-15)) + (1 - y) * math.log(max(1.0 - p, 1e-15)) for y, p in pairs) / count
    ece = 0.0
    bin_rows = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [(y, p) for y, p in pairs if low <= p < high or (index == bins - 1 and p == 1.0)]
        if not members:
            continue
        mean_confidence = sum(p for _, p in members) / len(members)
        mean_label = sum(y for y, _ in members) / len(members)
        ece += len(members) / count * abs(mean_confidence - mean_label)
        bin_rows.append({"lower": low, "upper": high, "count": len(members),
                         "mean_probability": mean_confidence, "event_rate": mean_label})
    return {"n": count, "positive_count": sum(y for y, _ in pairs), "positive_rate": sum(y for y, _ in pairs) / count,
            "brier": brier, "log_loss": log_loss, "ece": ece, "ece_bins": bin_rows}


def verify_qpiai_prediction_parity(
    model: VQCModel, raw_feature_rows: Sequence[Sequence[float]], *, tolerance: float = 1e-10
) -> dict[str, Any]:
    """Compare custom reference and QpiAI local statevector probabilities."""
    if not raw_feature_rows or tolerance <= 0.0:
        raise ValueError("parity verification needs rows and a positive tolerance")
    from backend.qpiai_vqc import sdk_versions

    checks = []
    for row in raw_feature_rows:
        reference = model.predict_proba(row)
        sdk = model.predict_proba_qpiai_local(row)
        checks.append({"raw_features": [float(value) for value in row],
                       "reference_probability": reference,
                       "qpiai_local_probability": sdk,
                       "absolute_error": abs(reference - sdk)})
    max_error = max(check["absolute_error"] for check in checks)
    if max_error > tolerance:
        raise AssertionError(f"QpiAI local prediction parity error {max_error} exceeds {tolerance}")
    return {
        "execution_type": "QpiAI Quantum local StatevectorSimulator",
        "remote_calls": 0,
        "qpu_jobs": 0,
        "sdk_versions": sdk_versions(),
        "row_count": len(checks),
        "max_absolute_error": max_error,
        "tolerance": tolerance,
        "model_sha256": model.sha256,
        "scaler_sha256": model.scaler.sha256 if model.scaler else None,
        "checks": checks,
    }


def verify_finite_difference_gradient(
    model: VQCModel, scaled_features: Sequence[float], *, step: float = 1e-5
) -> dict[str, Any]:
    if step <= 0 or not math.isfinite(step):
        raise ValueError("finite-difference step must be finite and positive")
    _, analytic, evaluations = model.probability_and_gradient(scaled_features)
    numeric = []
    for index in range(len(model.parameters)):
        plus, minus = list(model.parameters), list(model.parameters)
        plus[index] += step
        minus[index] -= step
        numeric.append((
            model.with_parameters(plus).predict_proba_scaled(scaled_features)
            - model.with_parameters(minus).predict_proba_scaled(scaled_features)
        ) / (2.0 * step))
    errors = [abs(a - n) for a, n in zip(analytic, numeric)]
    return {"analytic_gradient": list(analytic), "finite_difference_gradient": numeric,
            "max_absolute_error": max(errors, default=0.0), "step": step,
            "circuit_evaluations_for_analytic": evaluations,
            "model_sha256": model.sha256}


def fit_logistic_baseline(
    train_samples: Sequence[RouteSample],
    validation_samples: Sequence[RouteSample],
    *,
    scaler: FeatureScaler,
    learning_rate: float = 0.35,
    max_steps: int = 500,
    patience: int = 30,
    l2: float = 1e-4,
) -> ClassicalFitResult:
    """Fit dependency-free logistic regression; validation selects iterations."""
    _assert_disjoint_group_sets(train_samples, validation_samples)
    if not train_samples or not validation_samples or learning_rate <= 0 or max_steps < 1 or l2 < 0:
        raise ValueError("invalid logistic training inputs or settings")
    dim = scaler.feature_count
    weights = [0.0] * dim
    bias = 0.0
    best_model = LogisticModel(tuple(weights), 0.0, scaler)
    best_validation = float("inf")
    best_step = 0
    stale = 0
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    for step in range(max_steps + 1):
        model = LogisticModel(tuple(weights), bias, scaler)
        train_loss = _classical_log_loss(model, train_samples) + 0.5 * l2 * sum(w * w for w in weights)
        validation_loss = _classical_log_loss(model, validation_samples)
        if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
            raise FloatingPointError("non-finite logistic loss")
        if validation_loss < best_validation - 1e-9:
            best_validation, best_model, best_step, stale = validation_loss, model, step, 0
        else:
            stale += 1
        if step == max_steps or stale >= patience:
            break
        grad_w = [0.0] * dim
        grad_b = 0.0
        for sample in train_samples:
            x = scaler.transform(sample.raw_features)
            p = model.predict_proba(sample.raw_features)
            error = p - sample.label
            for k in range(dim):
                grad_w[k] += error * x[k]
            grad_b += error
        scale = 1.0 / len(train_samples)
        weights = [weights[k] - learning_rate * (grad_w[k] * scale + l2 * weights[k]) for k in range(dim)]
        bias -= learning_rate * grad_b * scale
    return ClassicalFitResult(
        model=best_model, selected_step=best_step,
        train_loss=_classical_log_loss(best_model, train_samples),
        validation_loss=best_validation,
        wall_seconds=time.perf_counter() - start_wall, cpu_seconds=time.process_time() - start_cpu,
        configuration={"optimizer": "full_batch_sgd", "learning_rate": learning_rate,
                       "max_steps": max_steps, "patience": patience, "l2": l2},
    )


def fit_newton_gbdt_baseline(
    train_samples: Sequence[RouteSample],
    validation_samples: Sequence[RouteSample],
    *,
    scaler: FeatureScaler,
    n_estimators: int = 40,
    max_depth: int = 3,
    learning_rate: float = 0.15,
    l2_reg: float = 1.5,
) -> ClassicalFitResult:
    """Reuse the reviewed V1.1 Newton GBDT; no new tree implementation."""
    from types import SimpleNamespace
    from experiments_v1_1.decision_quality import NewtonGBDT

    _assert_disjoint_group_sets(train_samples, validation_samples)
    if not train_samples or not validation_samples:
        raise ValueError("GBDT training requires train and validation samples")
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    train_rows = [SimpleNamespace(features=scaler.transform(sample.raw_features), train_label=sample.label)
                  for sample in train_samples]
    train_scenarios = [SimpleNamespace(routes=train_rows)]
    model = NewtonGBDT(n_estimators=n_estimators, max_depth=max_depth,
                       learning_rate=learning_rate, l2_reg=l2_reg).fit(train_scenarios)
    wrapped = ScaledEstimatorModel(model, scaler)
    return ClassicalFitResult(
        model=wrapped, selected_step=n_estimators,
        train_loss=_classical_log_loss(wrapped, train_samples),
        validation_loss=_classical_log_loss(wrapped, validation_samples),
        wall_seconds=time.perf_counter() - start_wall, cpu_seconds=time.process_time() - start_cpu,
        configuration={"estimator": "experiments_v1_1.NewtonGBDT", "n_estimators": n_estimators,
                       "max_depth": max_depth, "learning_rate": learning_rate, "l2_reg": l2_reg},
    )


def _classical_log_loss(model: Any, samples: Sequence[RouteSample]) -> float:
    if not samples:
        raise ValueError("log loss requires non-empty samples")
    losses = []
    for sample in samples:
        probability = min(1.0 - 1e-12, max(1e-12, float(model.predict_proba(sample.raw_features))))
        losses.append(-sample.label * math.log(probability) - (1 - sample.label) * math.log(1.0 - probability))
    return sum(losses) / len(losses)


def _bce_loss_and_gradient(model: VQCModel, samples: Sequence[RouteSample]) -> tuple[float, tuple[float, ...], int]:
    if not samples:
        raise ValueError("cannot fit on empty route sample sequence")
    grad = [0.0] * len(model.parameters)
    loss = 0.0
    evaluations = 0
    for sample in samples:
        features = model.transform(sample.raw_features)
        p, pg, circuit_evals = model.probability_and_gradient(features)
        y = sample.label
        p_safe = min(1.0 - 1e-12, max(1e-12, p))
        loss -= y * math.log(p_safe) + (1 - y) * math.log(1.0 - p_safe)
        coeff = (p - y) / max(p * (1.0 - p), 1e-12)
        for k, value in enumerate(pg):
            grad[k] += coeff * value
        evaluations += circuit_evals
    scale = 1.0 / len(samples)
    return loss * scale, tuple(value * scale for value in grad), evaluations


def predictive_loss(model: VQCModel, samples: Sequence[RouteSample]) -> float:
    return _bce_loss_and_gradient(model, samples)[0]


def train_predictive_vqc(
    train_samples: Sequence[RouteSample],
    validation_samples: Sequence[RouteSample],
    *,
    scaler: FeatureScaler,
    initial_parameters: Sequence[float] | None = None,
    learning_rate: float = 0.08,
    max_steps: int = 100,
    patience: int = 12,
    min_delta: float = 1e-7,
    model_version: str = "debugleaf-vqc-v1.1-run1-predictive",
    circuit_variant: str = BASE_VARIANT,
) -> FitResult:
    """Full-batch predictive pretraining with validation-only early stopping."""
    _assert_disjoint_group_sets(train_samples, validation_samples)
    if initial_parameters is None:
        initial_parameters = _initial_parameters(circuit_variant)
    return _fit_vqc(
        train_samples, validation_samples, scaler=scaler, initial_parameters=initial_parameters,
        learning_rate=learning_rate, max_steps=max_steps, patience=patience, min_delta=min_delta,
        model_version=model_version, objective="predictive", circuit_variant=circuit_variant,
    )


def train_decision_focused_vqc(
    train_samples: Sequence[RouteSample],
    validation_samples: Sequence[RouteSample],
    train_scenarios: Sequence[DecisionScenario],
    validation_scenarios: Sequence[DecisionScenario],
    *,
    scaler: FeatureScaler,
    pretrained_model: VQCModel,
    severity_by_source: Mapping[str, float],
    learning_rate: float = 0.03,
    max_steps: int = 50,
    patience: int = 8,
    temperature: float = 0.25,
    decision_weight: float = 0.85,
    model_version: str = "debugleaf-vqc-v1.1-run1-decision",
) -> FitResult:
    """Fine-tune from predictive VQC weights using V1.1's soft-plan surrogate.

    No QUBO/QAOA solve is performed in a training gradient. Each scenario's
    exact feasible alternatives and synthetic realized costs are prepared by
    the native/dispatch layer and are consumed as immutable supervision.
    """
    if pretrained_model.circuit_variant not in {BASE_VARIANT, DEPTH2_VARIANT}:
        raise ValueError("decision fine-tuning supports only the V1.1 baseline or independent-angle two-block extension")
    if pretrained_model.scaler != scaler:
        raise ValueError("predictive pretraining and decision fine-tuning must share the train-only scaler")
    _assert_disjoint_group_sets(train_samples, validation_samples)
    if any(scenario.split != "train" for scenario in train_scenarios):
        raise ValueError("decision gradient accepts train scenarios only")
    if any(scenario.split != "validation" for scenario in validation_scenarios):
        raise ValueError("decision model selection accepts validation scenarios only")
    if {scenario.topology_id for scenario in train_scenarios} & {
        scenario.topology_id for scenario in validation_scenarios
    }:
        raise ValueError("decision train/validation scenarios share a topology group")
    if not train_scenarios or not validation_scenarios:
        raise ValueError("decision training requires non-empty train and validation scenarios")
    if temperature <= 0.0 or not (0.0 <= decision_weight <= 1.0):
        raise ValueError("invalid decision loss temperature or weight")

    params = list(pretrained_model.parameters)
    circuit_variant = pretrained_model.circuit_variant
    model = VQCModel(tuple(params), scaler, model_version, circuit_variant)
    history: list[Mapping[str, float]] = []
    best_params = tuple(params)
    best_val = float("inf")
    best_step = 0
    stale = 0
    circuit_evals = 0
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    tracemalloc.start()
    try:
        for step in range(max_steps + 1):
            train_loss, grad, evals = decision_loss_and_gradient(
                model, train_scenarios, temperature=temperature, decision_weight=decision_weight,
                severity_by_source=severity_by_source,
            )
            circuit_evals += evals
            val_loss, _, val_evals = decision_loss_and_gradient(
                model, validation_scenarios, temperature=temperature, decision_weight=decision_weight,
                severity_by_source=severity_by_source,
            )
            circuit_evals += val_evals
            if not math.isfinite(train_loss) or not math.isfinite(val_loss) or not all(math.isfinite(g) for g in grad):
                raise FloatingPointError("non-finite loss or gradient in decision VQC training")
            history.append({"step": float(step), "train_loss": train_loss, "validation_loss": val_loss,
                            "gradient_l2": math.sqrt(sum(g * g for g in grad))})
            if val_loss < best_val - 1e-8:
                best_val, best_params, best_step, stale = val_loss, tuple(params), step, 0
            else:
                stale += 1
            if step == max_steps or stale >= patience:
                break
            params = [params[k] - learning_rate * grad[k] for k in range(len(params))]
            if not all(math.isfinite(value) for value in params):
                raise FloatingPointError("non-finite VQC parameter during decision training")
            model.parameters = tuple(params)
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return FitResult(
        model=VQCModel(best_params, scaler, model_version, circuit_variant), history=tuple(history), selected_step=best_step,
        wall_seconds=time.perf_counter() - start_wall, cpu_seconds=time.process_time() - start_cpu,
        peak_memory_bytes=peak, circuit_evaluations=circuit_evals,
    )


def decision_loss_and_gradient(
    model: VQCModel,
    scenarios: Sequence[DecisionScenario],
    *,
    temperature: float,
    decision_weight: float,
    severity_by_source: Mapping[str, float],
) -> tuple[float, tuple[float, ...], int]:
    """V1.1 expected realized plan cost plus BCE anchor and exact VQC gradient."""
    if not scenarios:
        raise ValueError("decision objective needs scenarios")
    if any(scenario.split == "test" for scenario in scenarios):
        raise ValueError("test outcomes cannot enter the training/selection objective")
    if temperature <= 0.0 or not 0.0 <= decision_weight <= 1.0:
        raise ValueError("invalid decision objective settings")
    grad = [0.0] * len(model.parameters)
    loss = 0.0
    evals = 0
    for scenario in scenarios:
        probs: dict[str, float] = {}
        pgrads: dict[str, tuple[float, ...]] = {}
        for route in scenario.routes:
            p, pg, count = model.probability_and_gradient(model.transform(route.raw_features))
            probs[route.route_id], pgrads[route.route_id] = p, pg
            evals += count
        route_map = {route.route_id: route for route in scenario.routes}
        plan_costs = [
            sum(route_map[rid].forecast_cost + probs[rid] * _source_severity(route_map[rid], severity_by_source)
                * route_map[rid].penalty_rate
                for rid in plan.route_ids)
            for plan in scenario.plans
        ]
        realized = [plan.realized_cost for plan in scenario.plans]
        base = min(plan_costs)
        # Subtracting the minimum stabilizes the softmax; exponent inputs are
        # non-positive, so ordinary IEEE underflow to zero is safe. Clipping
        # at a finite magnitude would change the derivative in saturated tails.
        unnormalized = [math.exp(-(cost - base) / temperature) for cost in plan_costs]
        norm = sum(unnormalized)
        if norm <= 0.0 or not math.isfinite(norm):
            raise FloatingPointError("invalid soft-plan normalization")
        weights = [value / norm for value in unnormalized]
        expected_realized = sum(w * outcome for w, outcome in zip(weights, realized))
        loss += decision_weight * expected_realized
        cost_grad = [-(w * (outcome - expected_realized)) / temperature
                     for w, outcome in zip(weights, realized)]
        for route in scenario.routes:
            # Decision costs use train-only expected severity, never the
            # replay outcome attached to this route sample.
            sensitivity = _source_severity(route, severity_by_source) * route.penalty_rate
            dloss_dp = sum(
                coefficient * sensitivity
                for plan, coefficient in zip(scenario.plans, cost_grad)
                if route.route_id in plan.route_ids
            )
            for k, value in enumerate(pgrads[route.route_id]):
                grad[k] += decision_weight * dloss_dp * value
            label = route.label
            if (1.0 - decision_weight) > 0.0:
                p = min(1.0 - 1e-12, max(1e-12, probs[route.route_id]))
                weight = 1.0 / len(scenario.routes)
                bce = -label * math.log(p) - (1 - label) * math.log(1.0 - p)
                loss += (1.0 - decision_weight) * weight * bce
                coeff = (1.0 - decision_weight) * weight * (p - label) / max(p * (1.0 - p), 1e-12)
                for k, value in enumerate(pgrads[route.route_id]):
                    grad[k] += coeff * value
    scale = 1.0 / len(scenarios)
    return loss * scale, tuple(value * scale for value in grad), evals


@dataclass(frozen=True)
class SeverityEstimate:
    by_source: Mapping[str, float]
    fallback_sources: tuple[str, ...]
    training_split_sha256: str
    training_topology_ids: tuple[str, ...]
    training_sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {"by_source": dict(self.by_source), "fallback_sources": list(self.fallback_sources), "fit_split": "train",
                "training_split_sha256": self.training_split_sha256,
                "training_topology_ids": list(self.training_topology_ids),
                "training_sample_count": self.training_sample_count}


def estimate_conditional_severity(
    samples: Sequence[RouteSample], split_map: Mapping[str, str]
) -> SeverityEstimate:
    """Estimate E[late minutes | breach] from train rows only, with split proof."""
    assert_group_split_integrity(samples, split_map)
    training = [sample for sample in samples if split_map[sample.topology_id] == "train"]
    if not training:
        raise ValueError("severity estimation requires training samples")
    source_values: dict[str, list[float]] = {}
    for sample in training:
        if sample.label == 1:
            source_values.setdefault(sample.source, []).append(sample.lateness_severity)
    all_sources = sorted({sample.source for sample in samples})
    positive_rows = [sample.lateness_severity for sample in training if sample.label == 1]
    if not positive_rows:
        raise ValueError("no positive train labels; conditional severity is not estimable")
    global_mean = sum(positive_rows) / len(positive_rows)
    fallback_sources = sorted(set(all_sources) - set(source_values))
    for source in fallback_sources:
        source_values[source] = [global_mean]
    train_groups = tuple(sorted({sample.topology_id for sample in training}))
    return SeverityEstimate(
        by_source={source: sum(values) / len(values) for source, values in source_values.items()},
        fallback_sources=tuple(fallback_sources),
        training_split_sha256=split_sha256(split_map, "train"),
        training_topology_ids=train_groups,
        training_sample_count=len(training),
    )


def split_sha256(split_map: Mapping[str, str], split: str) -> str:
    group_ids = sorted(group for group, assigned in split_map.items() if assigned == split)
    if not group_ids:
        raise ValueError(f"split {split!r} is empty")
    return hashlib.sha256("\n".join(group_ids).encode()).hexdigest()


def _source_severity(route: RouteSample, severity_by_source: Mapping[str, float]) -> float:
    try:
        value = float(severity_by_source[route.source])
    except KeyError as exc:
        raise ValueError(f"missing train-only severity estimate for source {route.source}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("train-only severity estimate must be finite and non-negative")
    return value


def _fit_vqc(
    train_samples: Sequence[RouteSample], validation_samples: Sequence[RouteSample], *, scaler: FeatureScaler,
    initial_parameters: Sequence[float], learning_rate: float, max_steps: int, patience: int,
    min_delta: float, model_version: str, objective: str,
    circuit_variant: str = BASE_VARIANT,
) -> FitResult:
    if not train_samples or not validation_samples:
        raise ValueError("predictive training requires non-empty train and validation samples")
    if learning_rate <= 0 or max_steps < 1 or patience < 1 or min_delta < 0:
        raise ValueError("invalid optimizer settings")
    expected_count = {BASE_VARIANT: 4, DEPTH2_VARIANT: 6, NON_ENTANGLING_VARIANT: 4}.get(circuit_variant)
    if expected_count is None or len(initial_parameters) != expected_count:
        raise ValueError("initial parameter count does not match the requested VQC circuit variant")
    params = [float(value) for value in initial_parameters]
    model = VQCModel(tuple(params), scaler, model_version, circuit_variant)
    best_params = tuple(params)
    best_val = float("inf")
    best_step = 0
    stale = 0
    history: list[Mapping[str, float]] = []
    evals = 0
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    tracemalloc.start()
    try:
        for step in range(max_steps + 1):
            train_loss, grad, count = _bce_loss_and_gradient(model, train_samples)
            evals += count
            val_loss = _bce_loss_and_gradient(model, validation_samples)[0]
            evals += len(validation_samples) * 5
            if not math.isfinite(train_loss) or not math.isfinite(val_loss) or not all(math.isfinite(g) for g in grad):
                raise FloatingPointError("non-finite loss or gradient in predictive VQC training")
            history.append({"step": float(step), "train_bce": train_loss, "validation_bce": val_loss,
                            "gradient_l2": math.sqrt(sum(value * value for value in grad))})
            if val_loss < best_val - min_delta:
                best_val, best_params, best_step, stale = val_loss, tuple(params), step, 0
            else:
                stale += 1
            if step == max_steps or stale >= patience:
                break
            params = [params[k] - learning_rate * grad[k] for k in range(len(params))]
            if not all(math.isfinite(value) for value in params):
                raise FloatingPointError("non-finite VQC parameter during predictive training")
            model.parameters = tuple(params)
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return FitResult(
        model=VQCModel(best_params, scaler, model_version, circuit_variant), history=tuple(history), selected_step=best_step,
        wall_seconds=time.perf_counter() - start_wall, cpu_seconds=time.process_time() - start_cpu,
        peak_memory_bytes=peak, circuit_evaluations=evals,
    )


def _initial_parameters(circuit_variant: str) -> tuple[float, ...]:
    if circuit_variant == BASE_VARIANT:
        return (0.31, -0.47, -3.5, -1.5)
    if circuit_variant == DEPTH2_VARIANT:
        return (0.31, -0.47, 0.18, -0.22, -3.5, -1.5)
    if circuit_variant == NON_ENTANGLING_VARIANT:
        return (0.31, -0.47, -3.5, -1.5)
    raise ValueError(f"unknown VQC circuit variant: {circuit_variant}")


def _assert_disjoint_group_sets(
    training: Sequence[RouteSample], validation: Sequence[RouteSample]
) -> None:
    training_groups = {sample.topology_id for sample in training}
    validation_groups = {sample.topology_id for sample in validation}
    overlap = training_groups & validation_groups
    if overlap:
        raise ValueError(f"train/validation topology leakage: {sorted(overlap)}")

