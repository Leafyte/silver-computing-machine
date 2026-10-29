from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from backend.india_eta.optimization import optimize_dispatch
from backend.india_eta.projection import project_remaining


@pytest.fixture
def scenario():
    return json.loads((Path(__file__).parents[2] / "fixtures" / "dispatch_tiny.json").read_text())


def weighted(scenario):
    slower = copy.deepcopy(scenario["travel"])
    for origin in slower:
        for destination in slower[origin]:
            if origin != destination:
                slower[origin][destination] += 2
    return [{"id": "base", "weight": 0.5, "travel": copy.deepcopy(scenario["travel"]),
             "source_label": "synthetic_overlay", "prediction_target": "travel_time"},
            {"id": "slow", "weight": 0.5, "travel": slower,
             "source_label": "synthetic_overlay", "prediction_target": "travel_time"}]


def states_for(plan, *, actual_vehicle=None, actual_position=None, prefix=(), elapsed=25, available=27):
    states = []
    for route in plan["routes"]:
        if route["vehicle_id"] == actual_vehicle:
            position, mode, completed = actual_position, "actual", prefix
            end_time = elapsed
        else:
            position, mode, completed = "D", "simulated", ()
            end_time = 0
        states.append({"vehicle_id": route["vehicle_id"], "position": position,
                       "position_type": mode, "elapsed_minutes": end_time,
                       "available_at": max(end_time, available if mode == "actual" else 0),
                       "completed_prefix": [{"order_id": oid, "completed_at": 23 + i}
                                            for i, oid in enumerate(completed)]})
    return states


def test_projection_starts_at_observed_position_and_keeps_completed_prefix(scenario):
    output = optimize_dispatch(scenario, weighted(scenario), quantum=False)
    route = output["plan"]["routes"][0]
    assert len(route["order_ids"]) >= 1
    completed = route["order_ids"][:1]
    position = completed[-1]
    states = states_for(output["plan"], actual_vehicle=route["vehicle_id"],
                        actual_position=position, prefix=completed)
    projection = project_remaining(scenario, output["plan"], states, weighted(scenario))
    assert projection["status"] == "projected"
    assert projection["state_mode"][route["vehicle_id"]] == "actual"
    projected = projection["plan"]["routes"][0]
    assert projected["remaining_stops"][0]["immutable"]
    assert projected["remaining_stops"][0]["completed_at"] == 23
    if projected["future_stops"]:
        assert min(stop["arrival"] for stop in projected["future_stops"]) >= 27
        assert all(stop["scenario_id"] in {"base", "slow"} for stop in projected["future_stops"])


def test_projection_requires_explicit_state_and_prefix_match(scenario):
    output = optimize_dispatch(scenario, weighted(scenario), quantum=False)
    assert project_remaining(scenario, output["plan"], [], weighted(scenario))["status"] == "unsupported_missing_vehicle_state"
    route = output["plan"]["routes"][0]
    bad = states_for(output["plan"])
    bad[0]["completed_prefix"] = [{"order_id": "not-in-route", "completed_at": 0}]
    with pytest.raises(ValueError, match="exactly match"):
        project_remaining(scenario, output["plan"], bad, weighted(scenario))


def test_invalid_clock_and_timestamp_order_rejected(scenario):
    output = optimize_dispatch(scenario, weighted(scenario), quantum=False)
    route = output["plan"]["routes"][0]
    states = states_for(output["plan"])
    states[0]["elapsed_minutes"] = -1
    with pytest.raises(ValueError, match="elapsed_minutes"):
        project_remaining(scenario, output["plan"], states, weighted(scenario))


def test_projection_reports_hard_window_violation_frequency_across_joint_scenarios():
    scenario = {"depot": "D", "vehicles": [{"id": "V", "capacity": 1}],
                "orders": [{"id": "A", "demand": 1, "window": [0, 5], "service": 0,
                            "penalty": 3, "hard_window": True}],
                "travel": {"D": {"D": 0, "A": 2}, "A": {"D": 2, "A": 0}}}
    delayed = {"D": {"D": 0, "A": 8}, "A": {"D": 2, "A": 0}}
    weighted_scenarios = [{"id": "nominal", "weight": 0.5, "travel": scenario["travel"],
                           "source_label": "synthetic_overlay", "prediction_target": "travel_time"},
                          {"id": "delayed", "weight": 0.5, "travel": delayed,
                           "source_label": "synthetic_overlay", "prediction_target": "travel_time"}]
    plan = optimize_dispatch(scenario, weighted_scenarios, quantum=False)["plan"]
    state = [{"vehicle_id": "V", "position": "D", "position_type": "simulated",
              "elapsed_minutes": 0, "available_at": 0, "completed_prefix": []}]
    projected = project_remaining(scenario, plan, state, weighted_scenarios)
    assert projected["expected_remaining_cost"]["hard_window_violation_frequency"] == pytest.approx(0.5)
    flagged = [stop for stop in projected["plan"]["routes"][0]["future_stops"]
               if stop["hard_window_violation"]]
    assert len(flagged) == 1 and flagged[0]["scenario_id"] == "delayed"


def test_projection_refreshes_stale_nominal_metrics_but_preserves_completed_snapshot():
    original = {"depot": "D", "vehicles": [{"id": "V", "capacity": 2}],
                "orders": [{"id": "A", "demand": 1, "window": [0, 50], "service": 0, "penalty": 2},
                           {"id": "B", "demand": 1, "window": [0, 50], "service": 0, "penalty": 2}],
                "travel": {"D": {"D": 0, "A": 2, "B": 3},
                           "A": {"D": 1, "A": 0, "B": 4},
                           "B": {"D": 1, "A": 4, "B": 0}}}
    original_weight = [{"id": "old", "weight": 1.0, "travel": copy.deepcopy(original["travel"]),
                        "source_label": "synthetic_overlay", "prediction_target": "travel_time"}]
    saved = optimize_dispatch(original, original_weight, quantum=False)["plan"]
    assert saved["routes"][0]["order_ids"] == ["A", "B"]
    updated = copy.deepcopy(original)
    updated["travel"]["D"]["A"] = 9
    updated_weight = [{"id": "event", "weight": 1.0, "travel": copy.deepcopy(updated["travel"]),
                       "source_label": "synthetic_overlay", "prediction_target": "travel_time"}]
    state = [{"vehicle_id": "V", "position": "A", "position_type": "simulated",
              "elapsed_minutes": 10, "available_at": 10,
              "completed_prefix": [{"order_id": "A", "completed_at": 3}]}]
    projection = project_remaining(updated, saved, state, updated_weight)
    assert projection["status"] == "projected"
    completed = projection["plan"]["routes"][0]["remaining_stops"][0]
    assert completed["completed_at"] == 3
    assert completed["arrival"] == 2  # preserved from the saved, already-completed snapshot
    future = projection["plan"]["routes"][0]["future_stops"][0]
    assert future["arrival"] == 14  # 10 at A + current A->B duration 4


def test_projection_rejects_position_that_skips_an_uncompleted_route_stop():
    scenario = {"depot": "D", "vehicles": [{"id": "V", "capacity": 3}],
                "orders": [{"id": oid, "demand": 1, "window": [0, 100], "service": 0, "penalty": 1}
                           for oid in ("A", "B", "C")],
                "travel": {"D": {"D": 0, "A": 1, "B": 1, "C": 1},
                           "A": {"D": 1, "A": 0, "B": 1, "C": 1},
                           "B": {"D": 1, "A": 1, "B": 0, "C": 1},
                           "C": {"D": 1, "A": 1, "B": 1, "C": 0}}}
    weighted_scenarios = [{"id": "world", "weight": 1.0, "travel": scenario["travel"],
                           "source_label": "synthetic_overlay", "prediction_target": "travel_time"}]
    plan = optimize_dispatch(scenario, weighted_scenarios, quantum=False)["plan"]
    route = plan["routes"][0]
    assert len(route["order_ids"]) == 3
    state = [{"vehicle_id": "V", "position": route["order_ids"][2], "position_type": "simulated",
              "elapsed_minutes": 1, "available_at": 1, "completed_prefix": []}]
    with pytest.raises(ValueError, match="skips an uncompleted"):
        project_remaining(scenario, plan, state, weighted_scenarios)


def test_all_completed_projection_preserves_snapshots_and_prices_only_return_leg():
    scenario = {"depot": "D", "vehicles": [{"id": "V", "capacity": 2}],
                "orders": [{"id": "A", "demand": 1, "window": [0, 100], "service": 0, "penalty": 3},
                           {"id": "B", "demand": 1, "window": [0, 100], "service": 0, "penalty": 3}],
                "travel": {"D": {"D": 0, "A": 2, "B": 4},
                           "A": {"D": 5, "A": 0, "B": 6},
                           "B": {"D": 7, "A": 2, "B": 0}}}
    weighted_scenarios = [{"id": "world", "weight": 1.0, "travel": scenario["travel"],
                           "source_label": "synthetic_overlay", "prediction_target": "travel_time"}]
    plan = optimize_dispatch(scenario, weighted_scenarios, quantum=False)["plan"]
    route_ids = plan["routes"][0]["order_ids"]
    final_stop = route_ids[-1]
    state = [{"vehicle_id": "V", "position": "B", "position_type": "simulated",
              "elapsed_minutes": 40, "available_at": 42,
              "completed_prefix": [{"order_id": oid, "completed_at": 15 + i * 10}
                                   for i, oid in enumerate(route_ids)]}]
    state[0]["position"] = final_stop
    result = project_remaining(scenario, plan, state, weighted_scenarios, operating_rate=2.0)
    route = result["plan"]["routes"][0]
    assert len(route["remaining_stops"]) == 2
    assert all(stop["immutable"] and stop["completed"] for stop in route["remaining_stops"])
    assert route["future_stops"] == []
    return_minutes = scenario["travel"][final_stop][scenario["depot"]]
    assert result["evaluation"]["expected_cost"]["amount"] == pytest.approx(return_minutes * 2.0)
    assert result["expected_remaining_cost"]["remaining_travel_minutes"] == pytest.approx(return_minutes)
