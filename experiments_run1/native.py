"""Native Solomon and MAMUT benchmark loading/evaluation for RUN 1.

Dabia2013 routes are priced by the pinned MAMUT TD checker. Their travel times
always come from the benchmark's per-arc arrival-time functions (ATFs); no
coordinate-based approximation is used for this source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Literal, Mapping, Sequence


MAMUT_ROUTING_COMMIT = "eca336ff4b087f5c2f7b9ac7c2ab33668ff7af2b"
MAMUT_ROUTING_LIB_COMMIT = "3f8d07705d6d00bba1b94180833f82c005c2f5c6"
MAMUT_ROUTING_LIB_VERSION = "0.12.0"
SUPPORTED_SIZES = (25, 50, 100)


@dataclass(frozen=True)
class NativeRouteResult:
    route: tuple[int, ...]
    feasible: bool
    duration: float | None
    optimized_departure_time: float | None
    departure_time: float | None
    arrivals: tuple[float, ...] = ()
    service_starts: tuple[float, ...] = ()
    window_slacks: tuple[float, ...] = ()
    lateness: tuple[float, ...] = ()
    return_time: float | None = None
    distance: float | None = None
    travel_duration: float | None = None
    service_duration: float = 0.0
    waiting_duration: float = 0.0
    added_travel_delay: float = 0.0
    added_service_delay: float = 0.0
    leg_travel_times: tuple[float, ...] = ()
    waiting_times: tuple[float, ...] = ()
    status: str = "valid"
    error: str = ""
    trace_feasible: bool | None = None
    trace_status: str | None = None
    trace_error: str = ""


@dataclass(frozen=True)
class NativeSolutionResult:
    valid: bool
    status: str
    native_cost: float | None
    num_routes: int
    route_evaluations: tuple[NativeRouteResult, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class NativeInstance:
    source: Literal["solomon", "dabia2013"]
    family: str
    size: int
    instance_id: str
    base_id: str
    raw: Mapping[str, Any]
    bks: Mapping[str, Any] | None
    file_hashes: Mapping[str, str]
    native_objective: str
    _dataset_path: Path = field(repr=False)
    _loaded_td: Any = field(default=None, repr=False, compare=False)
    _mamut: Any = field(default=None, repr=False, compare=False)

    @property
    def num_customers(self) -> int:
        return int(self.raw["num_customers"] if self.source == "dabia2013" else len(self.raw["customers"]) - 1)

    @property
    def customers(self) -> tuple[int, ...]:
        return tuple(range(1, self.num_customers + 1))

    @property
    def depot(self) -> int:
        return int(self.raw.get("depot", 0))

    @property
    def coordinates(self) -> tuple[tuple[float, float], ...]:
        if self.source == "dabia2013":
            return tuple((float(x), float(y)) for x, y in self.raw["coordinates"])
        rows = sorted(self.raw["customers"], key=lambda row: int(row["id"]))
        return tuple((float(row["x"]), float(row["y"])) for row in rows)

    @property
    def demands(self) -> tuple[float, ...]:
        if self.source == "dabia2013":
            return tuple(float(v) for v in self.raw["demands"])
        rows = sorted(self.raw["customers"], key=lambda row: int(row["id"]))
        return tuple(float(row["demand"]) for row in rows)

    @property
    def service_times(self) -> tuple[float, ...]:
        if self.source == "dabia2013":
            return tuple(float(v) for v in self.raw["service_times"])
        rows = sorted(self.raw["customers"], key=lambda row: int(row["id"]))
        return tuple(float(row["service_time"]) for row in rows)

    @property
    def time_windows(self) -> tuple[tuple[float, float], ...]:
        if self.source == "dabia2013":
            return tuple((float(a), float(b)) for a, b in self.raw["time_windows"])
        rows = sorted(self.raw["customers"], key=lambda row: int(row["id"]))
        return tuple((float(row["ready_time"]), float(row["due_time"])) for row in rows)

    @property
    def capacity(self) -> float:
        if self.source == "dabia2013":
            return float(self.raw["vehicle_capacity"])
        return float(self.raw["vehicle"]["capacity"])

    @property
    def num_vehicles(self) -> int:
        if self.source == "dabia2013":
            return int(self.raw["num_vehicles"])
        return int(self.raw["vehicle"]["number"])

    @property
    def horizon(self) -> tuple[float, float] | None:
        if self.source == "dabia2013":
            return tuple(float(x) for x in self.raw["horizon"])  # type: ignore[return-value]
        return None

    def arrival_time(self, origin: int, destination: int, departure_time: float) -> float:
        """Evaluate native travel from one node to another at time ``t``."""
        if not 0 <= origin <= self.num_customers or not 0 <= destination <= self.num_customers:
            raise ValueError("node indices must be depot 0 or a one-based customer index")
        if origin == destination:
            raise ValueError("self arcs are not present in the benchmark")
        if self.source == "dabia2013":
            atf = self._loaded_td.atfs.arcs[(origin, destination)]
            return float(atf.evaluate(float(departure_time), slope_one_exact=True))
        return float(departure_time) + _euclidean(self.coordinates[origin], self.coordinates[destination])

    def trace_route_at_departure(
        self,
        route: Sequence[int],
        departure_time: float,
        *,
        travel_delays: Sequence[float] | None = None,
        service_delays: Sequence[float] | None = None,
    ) -> NativeRouteResult:
        """Replay a route at a fixed depot departure without optimizing on outcomes.

        For Dabia2013 every leg is evaluated with the canonical ATF. Time-window
        waiting and service are then applied in visit order. This is the replay
        hook for disclosed departure-time and overlay uncertainty scenarios.
        Optional delay arrays are synthetic nonnegative increments: one per
        traversed leg (including return to depot) and one per customer service.
        They are applied after evaluating each canonical ATF, never by replacing
        its time-dependent travel function. Callers own scenario seeds and may
        supply common or route-specific arrays.
        """
        nodes = _validate_route(route, self.num_customers)
        departure = float(departure_time)
        if not math.isfinite(departure):
            raise ValueError("departure_time must be finite")
        travel_noise = _validated_delays(travel_delays, len(nodes) + 1, "travel_delays")
        service_noise = _validated_delays(service_delays, len(nodes), "service_delays")
        if self.source == "dabia2013":
            horizon = self.horizon
            assert horizon is not None
            if not horizon[0] <= departure <= horizon[1]:
                return _failed_route(nodes, "departure_outside_horizon", "departure is outside the ATF horizon")
        else:
            departure = max(departure, self.time_windows[self.depot][0])

        time = departure
        previous = self.depot
        arrivals: list[float] = []
        service_starts: list[float] = []
        slacks: list[float] = []
        lateness: list[float] = []
        leg_travel_times: list[float] = []
        waiting_times: list[float] = []
        distance = 0.0
        added_travel = 0.0
        added_service = 0.0
        feasible = True
        reason = ""
        for stop_index, customer in enumerate(nodes):
            if self.source == "solomon":
                leg_distance = _euclidean(self.coordinates[previous], self.coordinates[customer])
                distance += leg_distance
                arrival = time + leg_distance
            else:
                try:
                    arrival = self.arrival_time(previous, customer, time)
                except Exception as exc:
                    return _partial_failed_route(
                        nodes, departure, arrivals, service_starts, slacks, lateness,
                        leg_travel_times, waiting_times, self.service_times, added_travel, added_service,
                        "atf_domain_exceeded", f"ATF cannot be evaluated for arc ({previous}, {customer}) at {time!r}: {type(exc).__name__}",
                    )
            arrival += travel_noise[stop_index]
            added_travel += travel_noise[stop_index]
            if not math.isfinite(arrival):
                return _partial_failed_route(
                    nodes, departure, arrivals, service_starts, slacks, lateness,
                    leg_travel_times, waiting_times, self.service_times, added_travel, added_service,
                    "non_finite_arrival", "native travel replay produced a non-finite arrival",
                )
            leg_travel_times.append(arrival - time)
            ready, due = self.time_windows[customer]
            start = max(arrival, ready)
            late = max(0.0, start - due)
            if late > 0.0:
                feasible = False
                reason = reason or "time_window_violated"
            waiting_times.append(start - arrival)
            arrivals.append(arrival)
            service_starts.append(start)
            slacks.append(due - start)
            lateness.append(late)
            added_service += service_noise[stop_index]
            time = start + self.service_times[customer] + service_noise[stop_index]
            previous = customer

        if self.source == "solomon":
            leg_distance = _euclidean(self.coordinates[previous], self.coordinates[self.depot])
            distance += leg_distance
            return_time = time + leg_distance
        else:
            try:
                return_time = self.arrival_time(previous, self.depot, time)
            except Exception as exc:
                return _partial_failed_route(
                    nodes, departure, arrivals, service_starts, slacks, lateness,
                    leg_travel_times, waiting_times, self.service_times, added_travel, added_service,
                    "atf_domain_exceeded", f"ATF cannot be evaluated for return arc ({previous}, {self.depot}) at {time!r}: {type(exc).__name__}",
                )
        return_time += travel_noise[-1]
        added_travel += travel_noise[-1]
        if not math.isfinite(return_time):
            return _partial_failed_route(
                nodes, departure, arrivals, service_starts, slacks, lateness,
                leg_travel_times, waiting_times, added_travel, added_service,
                "non_finite_arrival", "native travel replay produced a non-finite return arrival",
            )
        leg_travel_times.append(return_time - time)
        depot_due = self.time_windows[self.depot][1]
        if return_time > depot_due:
            feasible = False
            reason = reason or "depot_due_time_violated"
        return NativeRouteResult(
            route=nodes,
            feasible=feasible,
            duration=return_time - departure,
            optimized_departure_time=None,
            departure_time=departure,
            arrivals=tuple(arrivals),
            service_starts=tuple(service_starts),
            window_slacks=tuple(slacks),
            lateness=tuple(lateness),
            return_time=return_time,
            distance=distance if self.source == "solomon" else None,
            travel_duration=sum(leg_travel_times),
            service_duration=sum(self.service_times[node] for node in nodes),
            waiting_duration=sum(waiting_times),
            added_travel_delay=added_travel,
            added_service_delay=added_service,
            leg_travel_times=tuple(leg_travel_times),
            waiting_times=tuple(waiting_times),
            status="valid" if feasible else reason,
            error="" if feasible else reason,
            trace_feasible=feasible,
            trace_status="valid" if feasible else reason,
            trace_error="" if feasible else reason,
        )

    def safe_overlay_rate(
        self,
        route: Sequence[int],
        *,
        upper_rate: float = 64.0,
        bisection_iterations: int = 32,
    ) -> dict[str, Any]:
        """Return a deterministic cap for multiplicative travel-delay overlays.

        For TD sources, the nominal route's canonical optimized departure is
        frozen before replay. The nominal per-leg travel times then define a
        scalar overlay ``delay_i = rate * leg_time_i``. Bisection finds the
        largest tested safe lower endpoint whose fixed-departure ATF trace has
        finite duration and one finite service start per customer. Hard
        time-window violations remain valid scenario outcomes and do not
        reduce this domain-only cap. No replay noise or labels are consulted.

        Static Solomon travel is unbounded by an ATF domain, so it returns a
        ``None`` cap with ``not_applicable_static_source`` status.
        """
        nodes = _validate_route(route, self.num_customers)
        if isinstance(upper_rate, bool) or not isinstance(upper_rate, (int, float)):
            raise ValueError("upper_rate must be a finite nonnegative number")
        upper = float(upper_rate)
        if not math.isfinite(upper) or upper < 0.0:
            raise ValueError("upper_rate must be a finite nonnegative number")
        if isinstance(bisection_iterations, bool) or not isinstance(bisection_iterations, int):
            raise ValueError("bisection_iterations must be an integer in [1, 64]")
        if not 1 <= bisection_iterations <= 64:
            raise ValueError("bisection_iterations must be an integer in [1, 64]")
        if self.source != "dabia2013":
            return {
                "max_fraction": None,
                "method": "no_cap_static_travel",
                "status": "not_applicable_static_source",
                "departure_time": self.time_windows[self.depot][0],
                "nominal_leg_travel_times": None,
                "upper_rate": upper,
                "iterations": 0,
                "tested_safe_upper_bound": True,
                "trace_status": "static_euclidean_domain_unbounded",
            }

        nominal = self.evaluate_route(nodes)
        if nominal.optimized_departure_time is None or not math.isfinite(nominal.optimized_departure_time):
            return {
                "max_fraction": 0.0,
                "method": "fixed_nominal_departure_scalar_bisection",
                "status": "no_canonical_nominal_departure",
                "departure_time": nominal.departure_time,
                "nominal_leg_travel_times": None,
                "upper_rate": upper,
                "iterations": 0,
                "tested_safe_upper_bound": False,
                "trace_status": nominal.status,
            }
        departure = float(nominal.optimized_departure_time)
        baseline = self.trace_route_at_departure(nodes, departure)
        leg_times = tuple(float(value) for value in baseline.leg_travel_times)
        if (baseline.duration is None or not math.isfinite(float(baseline.duration)) or
                len(baseline.service_starts) != len(nodes) or
                not all(math.isfinite(float(value)) for value in baseline.service_starts) or
                len(leg_times) != len(nodes) + 1 or
                not all(math.isfinite(value) and value >= 0.0 for value in leg_times)):
            return {
                "max_fraction": 0.0,
                "method": "fixed_nominal_departure_scalar_bisection",
                "status": "nominal_trace_not_finite_or_complete",
                "departure_time": departure,
                "nominal_leg_travel_times": leg_times,
                "upper_rate": upper,
                "iterations": 0,
                "tested_safe_upper_bound": False,
                "trace_status": baseline.status,
            }

        def trace_at(rate: float) -> NativeRouteResult:
            return self.trace_route_at_departure(
                nodes,
                departure,
                travel_delays=tuple(rate * leg for leg in leg_times),
                service_delays=(0.0,) * len(nodes),
            )

        def is_domain_safe(trace: NativeRouteResult) -> bool:
            # Deliberately do not require trace.feasible: finite hard-window
            # lateness is a valid label/outcome, unlike ATF-domain failure.
            return (
                trace.status != "atf_domain_exceeded" and
                trace.duration is not None and math.isfinite(float(trace.duration)) and
                trace.return_time is not None and math.isfinite(float(trace.return_time)) and
                len(trace.arrivals) == len(nodes) and
                len(trace.service_starts) == len(nodes) and
                all(math.isfinite(float(value)) for value in trace.arrivals) and
                all(math.isfinite(float(value)) for value in trace.service_starts)
            )

        upper_trace = trace_at(upper)
        if is_domain_safe(upper_trace):
            return {
                "max_fraction": upper,
                "method": "fixed_nominal_departure_scalar_bisection",
                "status": "upper_bound_safe",
                "departure_time": departure,
                "nominal_leg_travel_times": leg_times,
                "upper_rate": upper,
                "iterations": 0,
                "tested_safe_upper_bound": True,
                "trace_status": upper_trace.status,
            }

        low = 0.0  # The nominal baseline above proved this endpoint is safe.
        high = upper
        low_trace = baseline
        iterations = 0
        for iterations in range(1, bisection_iterations + 1):
            midpoint = low + (high - low) / 2.0
            candidate = trace_at(midpoint)
            if is_domain_safe(candidate):
                low, low_trace = midpoint, candidate
            else:
                high = midpoint
        # Preserve the last tested finite low endpoint exactly; no epsilon is
        # added to the returned rate or used to alter benchmark feasibility.
        return {
            "max_fraction": low,
            "method": "fixed_nominal_departure_scalar_bisection",
            "status": "safe_bisection_lower_bound",
            "departure_time": departure,
            "nominal_leg_travel_times": leg_times,
            "upper_rate": upper,
            "iterations": iterations,
            "tested_safe_upper_bound": False,
            "first_tested_unsafe_rate": high,
            "trace_status": low_trace.status,
            "trace_duration": low_trace.duration,
            "finite_service_start_count": len(low_trace.service_starts),
        }

    def evaluate_route(self, route: Sequence[int], *, departure_time: float | None = None) -> NativeRouteResult:
        """Evaluate a route natively; optimize Dabia departure only nominally.

        A fixed ``departure_time`` invokes a direct benchmark replay. Without
        one, Dabia uses MAMUT's canonical departure-optimized route checker;
        Solomon uses its static Euclidean objective and the depot ready time.
        """
        nodes = _validate_route(route, self.num_customers)
        load = sum(self.demands[node] for node in nodes)
        if load > self.capacity:
            return _failed_route(nodes, "vehicle_capacity_exceeded", "route load exceeds vehicle capacity")
        if departure_time is not None:
            return self.trace_route_at_departure(nodes, departure_time)
        if self.source == "dabia2013":
            _, _, compute_route_duration, _ = self._mamut
            result = compute_route_duration(self._loaded_td.instance, self._loaded_td.atfs, list(nodes))
            if not result.feasible:
                return _failed_route(nodes, "time_window_violated", "route has no feasible depot departure")
            assert result.duration is not None and result.departure_time is not None
            trace = self.trace_route_at_departure(nodes, result.departure_time)
            return NativeRouteResult(
                route=nodes, feasible=bool(result.feasible), duration=float(result.duration),
                optimized_departure_time=float(result.departure_time),
                departure_time=float(result.departure_time), arrivals=trace.arrivals,
                service_starts=trace.service_starts, window_slacks=trace.window_slacks,
                lateness=trace.lateness, return_time=trace.return_time,
                distance=trace.distance, travel_duration=trace.travel_duration,
                service_duration=trace.service_duration, waiting_duration=trace.waiting_duration,
                added_travel_delay=trace.added_travel_delay, added_service_delay=trace.added_service_delay,
                leg_travel_times=trace.leg_travel_times, waiting_times=trace.waiting_times,
                status="valid" if result.feasible else "time_window_violated",
                trace_feasible=trace.feasible, trace_status=trace.status,
                trace_error=trace.error,
            )
        result = self.trace_route_at_departure(nodes, self.time_windows[self.depot][0])
        return NativeRouteResult(
            route=result.route, feasible=result.feasible, duration=result.duration,
            optimized_departure_time=None, departure_time=result.departure_time,
            arrivals=result.arrivals, service_starts=result.service_starts,
            window_slacks=result.window_slacks, lateness=result.lateness,
            return_time=result.return_time, distance=result.distance,
            travel_duration=result.travel_duration, service_duration=result.service_duration,
            waiting_duration=result.waiting_duration, leg_travel_times=result.leg_travel_times,
            added_travel_delay=result.added_travel_delay, added_service_delay=result.added_service_delay,
            waiting_times=result.waiting_times,
            status=result.status, error=result.error,
            trace_feasible=result.feasible, trace_status=result.status, trace_error=result.error,
        )

    def evaluate_solution(self, routes: Sequence[Sequence[int]]) -> NativeSolutionResult:
        """Validate full customer coverage and aggregate the native objective."""
        normalized = tuple(_validate_route(route, self.num_customers) for route in routes)
        if not normalized:
            return NativeSolutionResult(False, "empty_solution", None, 0, error="at least one route is required")
        if len(normalized) > self.num_vehicles:
            return NativeSolutionResult(False, "too_many_vehicles_used", None, len(normalized))
        seen: set[int] = set()
        for route in normalized:
            for node in route:
                if node in seen:
                    return NativeSolutionResult(False, "customer_served_multiple_times", None, len(normalized))
                seen.add(node)
        if seen != set(self.customers):
            return NativeSolutionResult(False, "not_all_customers_served", None, len(normalized))
        if self.source == "dabia2013":
            BenchmarkSolution, check_td_solution, _, _ = self._mamut
            solution = BenchmarkSolution(instance_name=self.instance_id, routes=[list(r) for r in normalized])
            check = check_td_solution(self._loaded_td, solution)
            if not check.is_valid():
                return NativeSolutionResult(False, check.status.value, None, len(normalized), error=check.error_message)
            evaluations = tuple(self.evaluate_route(route) for route in normalized)
            return NativeSolutionResult(True, check.status.value, float(check.routing_cost), len(normalized), evaluations)
        evaluations = tuple(self.evaluate_route(route) for route in normalized)
        bad = next((item for item in evaluations if not item.feasible), None)
        if bad is not None:
            return NativeSolutionResult(False, bad.status, None, len(normalized), evaluations, bad.error)
        cost = sum(float(item.distance or 0.0) for item in evaluations)
        return NativeSolutionResult(True, "valid", cost, len(normalized), evaluations)


def load_instance(
    source: Literal["solomon", "dabia2013"],
    instance_id: str = "C101",
    size: int = 25,
    *,
    dataset_root: str | Path | None = None,
) -> NativeInstance:
    """Load a Solomon or Dabia instance at the requested customer count."""
    if size not in SUPPORTED_SIZES:
        raise ValueError(f"size must be one of {SUPPORTED_SIZES}")
    if not instance_id or Path(instance_id).name != instance_id:
        raise ValueError("instance_id must be a filename stem")
    repo_root = Path(__file__).resolve().parents[1]
    root = Path(dataset_root) if dataset_root is not None else repo_root / ".runtime" / "datasets"
    hashes: dict[str, str] = {}
    if source == "dabia2013":
        directory = root / "Dabia2013" / f"n={size}"
        instance_path = directory / f"{instance_id}.vrp.json"
        if not instance_path.is_file():
            raise FileNotFoundError(instance_path)
        raw = _read_json(instance_path)
        atf_path = directory / str(raw["td"]["atf_path"])
        bks_path = directory / f"{instance_id}.bks.Duration.json"
        bks = _read_json(bks_path) if bks_path.is_file() else None
        hashes["instance"] = _checked_sha256(instance_path, root / "Dabia2013", f"n={size}/{instance_path.name}")
        hashes["atf"] = _checked_sha256(atf_path, root / "Dabia2013", f"n={size}/{atf_path.name}")
        if bks is not None:
            hashes["bks"] = _checked_sha256(bks_path, root / "Dabia2013", f"n={size}/{bks_path.name}")
        loaded_td, mamut = _load_mamut_td(instance_path)
        family = "Dabia2013"
        source_name = "dabia2013"
        native_objective = "Duration"
        origin = str(raw.get("instance_origin", "Dabia2013"))
    elif source == "solomon":
        directory = root / f"solomon_{size}"
        instance_path = directory / f"{instance_id}.json"
        if not instance_path.is_file():
            raise FileNotFoundError(f"Solomon n={size} not found at {instance_path}")
        raw = _read_json(instance_path)
        bks = None
        hashes["instance"] = _checked_sha256(instance_path, directory, instance_path.name)
        loaded_td = None
        mamut = None
        family = "Solomon1987"
        source_name = "solomon"
        native_objective = "EuclideanDistance"
        origin = "Solomon1987"
        _validate_solomon_payload(raw, size)
    else:
        raise ValueError("source must be 'solomon' or 'dabia2013'")
    base_id = f"solomon1987:{instance_id}" if origin.casefold().startswith("solomon") else f"{family}:{instance_id}"
    return NativeInstance(
        source=source_name, family=family, size=size, instance_id=instance_id,
        base_id=base_id, raw=raw, bks=bks, file_hashes=hashes,
        native_objective=native_objective, _dataset_path=instance_path,
        _loaded_td=loaded_td, _mamut=mamut,
    )


def _load_mamut_td(instance_path: Path):
    source_root = Path(__file__).resolve().parents[1] / ".runtime" / "mamut-routing-lib-source"
    expected = source_root / f"MAMUT-routing-lib-{MAMUT_ROUTING_LIB_COMMIT}" / "src"
    override = os.environ.get("MAMUT_ROUTING_LIB_SRC")
    candidates = [Path(override)] if override else []
    candidates.append(expected)
    src = next((path.resolve() for path in candidates if path.is_dir()), None)
    if src is None:
        raise RuntimeError(f"pinned MAMUT-routing-lib source is unavailable: {expected}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from mamut_routing_lib import BenchmarkSolution
    from mamut_routing_lib.td import check_td_solution, compute_route_duration, load_td_instance
    import mamut_routing_lib
    if mamut_routing_lib.__version__ not in ("0.0.0+unknown", MAMUT_ROUTING_LIB_VERSION):
        raise RuntimeError(f"unexpected MAMUT-routing-lib version: {mamut_routing_lib.__version__}")
    loaded = load_td_instance(instance_path, verify_sha256=True, validate_complete=True)
    return loaded, (BenchmarkSolution, check_td_solution, compute_route_duration, load_td_instance)


def _checked_sha256(path: Path, dataset_root: Path, relative: str) -> str:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    manifest_path = dataset_root / "DATASET-MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_path = {row["path"]: row for row in manifest.get("files", [])}
        record = by_path.get(relative)
        if record is not None:
            if record.get("sha256") != digest or record.get("bytes") != len(data):
                raise ValueError(f"dataset size/SHA-256 mismatch for {relative}")
            blob = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
            if record.get("git_blob_sha1") != blob:
                raise ValueError(f"dataset Git blob SHA-1 mismatch for {relative}")
    return digest


def _read_json(path: Path) -> dict[str, Any]:
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            value = json.load(stream)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _validate_solomon_payload(raw: Mapping[str, Any], size: int) -> None:
    vehicles, customers = raw.get("vehicle"), raw.get("customers")
    if not isinstance(vehicles, dict) or not isinstance(customers, list) or len(customers) != size + 1:
        raise ValueError(f"invalid Solomon payload: expected vehicle object and {size + 1} nodes")
    ids = [int(row["id"]) for row in customers]
    if sorted(ids) != list(range(size + 1)):
        raise ValueError("Solomon customer ids must be contiguous from depot 0")


def _validate_route(route: Sequence[int], num_customers: int) -> tuple[int, ...]:
    nodes = tuple(route)
    if not nodes:
        raise ValueError("route must contain at least one customer")
    if any(isinstance(node, bool) or not isinstance(node, int) or not 1 <= node <= num_customers for node in nodes):
        raise ValueError("route nodes must be one-based customer indices")
    if len(set(nodes)) != len(nodes):
        raise ValueError("route must not repeat a customer")
    return nodes


def _failed_route(route: tuple[int, ...], status: str, error: str) -> NativeRouteResult:
    return NativeRouteResult(route, False, None, None, None, status=status, error=error,
                             trace_feasible=False, trace_status=status, trace_error=error)


def _validated_delays(values: Sequence[float] | None, expected: int, name: str) -> tuple[float, ...]:
    if values is None:
        return (0.0,) * expected
    delays = tuple(float(value) for value in values)
    if len(delays) != expected or any(not math.isfinite(value) or value < 0.0 for value in delays):
        raise ValueError(f"{name} must contain {expected} finite non-negative values")
    return delays


def _partial_failed_route(
    route: tuple[int, ...],
    departure: float,
    arrivals: list[float],
    service_starts: list[float],
    slacks: list[float],
    lateness: list[float],
    leg_travel_times: list[float],
    waiting_times: list[float],
    service_times: Sequence[float],
    added_travel: float,
    added_service: float,
    status: str,
    error: str,
) -> NativeRouteResult:
    completed = len(service_starts)
    return NativeRouteResult(
        route=route, feasible=False, duration=None, optimized_departure_time=None,
        departure_time=departure, arrivals=tuple(arrivals), service_starts=tuple(service_starts),
        window_slacks=tuple(slacks), lateness=tuple(lateness),
        service_duration=sum(service_times[node] for node in route[:completed]),
        waiting_duration=sum(waiting_times), added_travel_delay=added_travel,
        added_service_delay=added_service, travel_duration=sum(leg_travel_times),
        leg_travel_times=tuple(leg_travel_times), waiting_times=tuple(waiting_times),
        status=status, error=error, trace_feasible=False, trace_status=status, trace_error=error,
    )


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

