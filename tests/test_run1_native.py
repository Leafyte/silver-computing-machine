from __future__ import annotations

import math

import pytest

from experiments_run1.native import load_instance


def test_dabia_c101_bks_passes_canonical_time_dependent_checker():
    instance = load_instance("dabia2013", "C101", 25)
    assert instance.num_customers == 25
    assert instance.customers == tuple(range(1, 26))
    assert instance.base_id == "solomon1987:C101"
    assert instance.native_objective == "Duration"
    assert len(instance._loaded_td.atfs.arcs) == 26 * 25

    checked = instance.evaluate_solution(instance.bks["routes"])
    assert checked.valid
    assert checked.status == "valid"
    assert checked.num_routes == 4
    assert checked.native_cost == instance.bks["cost"] == 24709.17715754648
    assert [item.optimized_departure_time for item in checked.route_evaluations] == [
        0.0, 0.0, 14.28595918297377, 7222.49975624939,
    ]
    assert math.isclose(sum(item.duration for item in checked.route_evaluations), checked.native_cost, abs_tol=1e-9)


def test_dabia_fixed_departure_replay_uses_atfs_and_keeps_scenario_departure():
    instance = load_instance("dabia2013", "C101", 25)
    route = instance.bks["routes"][0]
    nominal = instance.evaluate_route(route)
    fixed = instance.trace_route_at_departure(
        route,
        nominal.optimized_departure_time + 60.0,
        travel_delays=[1.0] * (len(route) + 1),
        service_delays=[0.5] * len(route),
    )
    assert fixed.departure_time == nominal.optimized_departure_time + 60.0
    assert fixed.optimized_departure_time is None
    assert len(fixed.arrivals) == len(route)
    assert len(fixed.service_starts) == len(route)
    assert len(fixed.window_slacks) == len(route)
    assert len(fixed.leg_travel_times) == len(route) + 1
    assert all(service >= arrival for service, arrival in zip(fixed.service_starts, fixed.arrivals))
    assert fixed.added_travel_delay == len(route) + 1
    assert fixed.added_service_delay == 0.5 * len(route)
    assert math.isclose(
        fixed.duration,
        fixed.travel_duration + fixed.service_duration + fixed.waiting_duration + fixed.added_service_delay,
        abs_tol=1e-8,
    )
    # At least one ATF is time-dependent; the adapter must not substitute Euclidean travel.
    assert instance.arrival_time(0, route[0], 0.0) != instance.arrival_time(0, route[0], 6000.0) - 6000.0


def test_dabia_replay_reports_atf_domain_overflow_without_extrapolation():
    instance = load_instance("dabia2013", "C101", 25)
    result = instance.trace_route_at_departure([1], instance.horizon[1])
    assert not result.feasible
    assert result.status == "atf_domain_exceeded"
    assert all(math.isfinite(value) for value in result.arrivals + result.lateness)


def test_dabia_replay_rejects_invalid_synthetic_delay_array():
    instance = load_instance("dabia2013", "C101", 25)
    with pytest.raises(ValueError, match="travel_delays"):
        instance.trace_route_at_departure([1], 0.0, travel_delays=[-1.0, 0.0])


def test_dabia_safe_overlay_rate_caps_only_atf_domain_not_window_outcomes():
    instance = load_instance("dabia2013", "C101", 25)
    cap = instance.safe_overlay_rate([1], upper_rate=64.0)
    assert cap["method"] == "fixed_nominal_departure_scalar_bisection"
    assert cap["status"] == "safe_bisection_lower_bound"
    assert cap["iterations"] == 32
    assert cap["tested_safe_upper_bound"] is False
    assert 0.0 < cap["max_fraction"] < 64.0
    assert cap["first_tested_unsafe_rate"] > cap["max_fraction"]

    nominal = instance.evaluate_route([1])
    assert nominal.optimized_departure_time == cap["departure_time"]
    leg_times = cap["nominal_leg_travel_times"]
    safe_trace = instance.trace_route_at_departure(
        [1], cap["departure_time"],
        travel_delays=[cap["max_fraction"] * value for value in leg_times],
        service_delays=[0.0],
    )
    assert safe_trace.duration is not None and math.isfinite(safe_trace.duration)
    assert len(safe_trace.service_starts) == 1
    # A finite late arrival remains an outcome; the domain cap does not filter it.
    assert not safe_trace.feasible
    assert safe_trace.status == "time_window_violated"

    unsafe_trace = instance.trace_route_at_departure(
        [1], cap["departure_time"],
        travel_delays=[cap["first_tested_unsafe_rate"] * value for value in leg_times],
        service_delays=[0.0],
    )
    assert unsafe_trace.status == "atf_domain_exceeded"
    assert unsafe_trace.duration is None


def test_solomon_safe_overlay_rate_is_uncapped_static_travel():
    instance = load_instance("solomon", "C101", 25)
    cap = instance.safe_overlay_rate([1], upper_rate=64.0)
    assert cap["max_fraction"] is None
    assert cap["status"] == "not_applicable_static_source"
    assert cap["tested_safe_upper_bound"] is True


def test_dabia_invalid_solution_is_reported_by_canonical_checker():
    instance = load_instance("dabia2013", "C101", 25)
    invalid = instance.evaluate_solution(instance.bks["routes"][:-1])
    assert not invalid.valid
    assert invalid.status == "not_all_customers_served"


def test_solomon_c101_stays_a_distinct_100_customer_instance():
    instance = load_instance("solomon", "C101", 100)
    assert instance.num_customers == 100
    assert instance.base_id == "solomon1987:C101"
    route = instance.evaluate_route([1])
    assert route.feasible
    assert route.distance is not None
    assert route.travel_duration is not None
    assert route.service_starts[0] >= instance.time_windows[1][0]


@pytest.mark.parametrize("size", [25, 50])
def test_solomon_matching_customer_buckets_are_loaded(size):
    instance = load_instance("solomon", "C101", size)
    assert instance.num_customers == size
    assert instance.capacity > 0
    assert len(instance.coordinates) == size + 1
    assert len(instance.time_windows) == size + 1
    assert instance.file_hashes["instance"]


def test_same_c101_topology_is_grouped_across_size_and_source():
    solomon = load_instance("solomon", "C101", 25)
    dabia = load_instance("dabia2013", "C101", 25)
    solomon_100 = load_instance("solomon", "C101", 100)
    assert solomon.base_id == dabia.base_id == solomon_100.base_id
    assert solomon.num_customers == dabia.num_customers == 25
    assert solomon_100.num_customers == 100
