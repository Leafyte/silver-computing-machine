from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from backend.domain import generate_routes
from backend.india_eta.optimization import evaluate_plan, optimize_dispatch


@pytest.fixture
def scenario():
    return json.loads((Path(__file__).parents[2] / "fixtures" / "dispatch_tiny.json").read_text())


def weighted(scenario):
    first = copy.deepcopy(scenario["travel"])
    second = copy.deepcopy(scenario["travel"])
    for a in second:
        for b in second[a]:
            if a != b:
                second[a][b] *= 1.5
    return [{"id": "synthetic-low", "weight": 0.5, "travel": first,
             "source_label": "synthetic_overlay", "prediction_target": "travel_time"},
            {"id": "synthetic-high", "weight": 0.5, "travel": second,
             "source_label": "synthetic_overlay", "prediction_target": "travel_time"}]


def test_optimizer_records_qubo_isings_audit_and_separates_raw_from_repairs(scenario):
    result = optimize_dispatch(scenario, weighted(scenario), quantum=True, seed=7)
    assert result["schema_version"] == "debugleaf.india_eta.optimization.v1"
    assert result["validation"]["valid"]
    assert result["stochastic_validation"]["valid"]
    assert len(result["coefficients"]["candidate_costs"]) == len(generate_routes(scenario))
    if len(result["coefficients"]["candidate_costs"]) > 8:
        assert result["quantum"]["status"] == "unsupported_candidate_pool_size"
        assert result["fallback"]["used"]
    audit = result["coefficients"]["qubo"]["exhaustive_audit"]
    assert audit["all_energy_checks_passed"]
    assert audit["assignment_count"] == 2 ** len(result["coefficients"]["qubo"]["variable_order"])
    if result["quantum"]["status"] == "success":
        assert result["quantum"]["lp_warm_start"]["method"] == "v1.1_augmented_lagrangian_set_partitioning_relaxation"
    assert all("independent_validation" in row for row in result["quantum"]["raw_samples"])
    assert all("edits" in row and "independent_validation" in row
               for row in result["quantum"]["repaired_samples"])


def test_scenario_cost_is_recomputed_and_expected_lateness_uses_minutes(scenario):
    plan = optimize_dispatch(scenario, weighted(scenario), quantum=False)["plan"]
    evaluated = evaluate_plan(scenario, plan, weighted(scenario), currency="INR", operating_rate=2.0)
    assert evaluated["valid"]
    rows = evaluated["scenario_costs"]
    manual = sum(row["weight"] * (row["travel_minutes"] * 2 + row["late_penalty_cost"]) for row in rows)
    assert evaluated["expected_cost"]["total"] == pytest.approx(manual)


def test_locked_prefix_stays_with_vehicle_and_does_not_enter_remaining_qubo(scenario):
    scenario["vehicles"] = [{"id": "V1", "capacity": 3}]
    scenario["orders"] = scenario["orders"][:2]
    result = optimize_dispatch(scenario, weighted(scenario), locked_prefixes={"V1": [scenario["orders"][0]["id"]]}, quantum=False)
    assert result["validation"]["valid"]
    route = result["plan"]["routes"][0]
    assert route["order_ids"][0] == scenario["orders"][0]["id"]
    assert result["coefficients"]["qubo"]["variable_order"]
    assert any(row["stops"] == [scenario["orders"][0]["id"]]
               for row in result["coefficients"]["candidate_costs"])


def test_completed_prefix_remains_covered_when_other_vehicle_takes_remaining_work():
    scenario = {"depot": "D", "vehicles": [{"id": "V1", "capacity": 2}, {"id": "V2", "capacity": 1}],
                "orders": [{"id": "A", "demand": 1, "window": [0, 100], "service": 0, "penalty": 2},
                           {"id": "B", "demand": 1, "window": [0, 100], "service": 0, "penalty": 2}],
                "travel": {"D": {"D": 0, "A": 2, "B": 1},
                           "A": {"D": 1, "A": 0, "B": 50},
                           "B": {"D": 1, "A": 1, "B": 0}}}
    rows = [{"id": "world", "weight": 1.0, "travel": scenario["travel"],
             "source_label": "synthetic_overlay", "prediction_target": "travel_time"}]
    states = [{"vehicle_id": "V1", "position": "A", "position_type": "simulated",
               "elapsed_minutes": 10, "available_at": 10,
               "completed_prefix": [{"order_id": "A", "completed_at": 5}]},
              {"vehicle_id": "V2", "position": "D", "position_type": "simulated",
               "elapsed_minutes": 10, "available_at": 10, "completed_prefix": []}]
    result = optimize_dispatch(scenario, rows, locked_prefixes={"V1": ["A"]},
                               vehicle_states=states, quantum=False)
    assigned = {route["vehicle_id"]: route["order_ids"] for route in result["plan"]["routes"]}
    assert assigned == {"V1": ["A"], "V2": ["B"]}
    assert result["validation"]["valid"]
    assert {stop for route in result["plan"]["routes"] for stop in route["order_ids"]} == {"A", "B"}


def test_invalid_weight_or_incomplete_matrix_rejected(scenario):
    rows = weighted(scenario)
    rows[0]["weight"] = 0.4
    with pytest.raises(ValueError, match="sum to 1"):
        evaluate_plan(scenario, {"status": "feasible", "routes": []}, rows)
    rows = weighted(scenario)
    rows[0]["travel"].pop(scenario["depot"])
    with pytest.raises(ValueError, match="missing"):
        evaluate_plan(scenario, {"status": "feasible", "routes": []}, rows)


def test_small_candidate_pool_records_complete_v11_lp_vector():
    tiny = {"depot": "D", "vehicles": [{"id": "V", "capacity": 1}],
            "orders": [{"id": "A", "demand": 1, "window": [0, 10], "service": 0, "penalty": 2}],
            "travel": {"D": {"D": 0, "A": 2}, "A": {"D": 2, "A": 0}}}
    result = optimize_dispatch(tiny, [{"id": "fixture", "weight": 1.0, "travel": tiny["travel"],
        "source_label": "synthetic_overlay", "prediction_target": "travel_time"}], quantum=True)
    warm = result["quantum"]["lp_warm_start"]
    assert result["quantum"]["status"] == "success"
    assert result["execution_type"] == "local_python_statevector_simulator"
    assert len(result["quantum"]["raw_samples"]) == 256
    assert len(warm["vector"]) == len(result["coefficients"]["qubo"]["variable_order"]) == 1
    assert warm["vector"][0] == pytest.approx(1.0)
    assert warm["steps"] == 5000
    assert warm["mapping"] == "candidate_rows order equals QUBO variable order"
    candidate_cost = result["coefficients"]["candidate_costs"][0]["expected_cost"]
    assert result["classical_reference"]["expected_cost"] == pytest.approx(candidate_cost)
    raw_feasible = [sample for sample in result["quantum"]["raw_samples"]
                    if sample["qubo_validation"]["feasible"] and sample["independent_validation"]["valid"]]
    assert raw_feasible
    assert all(sample["energy"] == pytest.approx(candidate_cost) for sample in raw_feasible)


def test_all_completed_optimizer_keeps_full_plan_but_prices_only_return_leg():
    scenario = {"depot": "D", "vehicles": [{"id": "V", "capacity": 2}],
                "orders": [{"id": "A", "demand": 1, "window": [0, 100], "service": 0, "penalty": 3},
                           {"id": "B", "demand": 1, "window": [0, 100], "service": 0, "penalty": 3}],
                "travel": {"D": {"D": 0, "A": 2, "B": 4},
                           "A": {"D": 5, "A": 0, "B": 6},
                           "B": {"D": 7, "A": 2, "B": 0}}}
    rows = [{"id": "world", "weight": 1.0, "travel": scenario["travel"],
             "source_label": "synthetic_overlay", "prediction_target": "travel_time"}]
    state = [{"vehicle_id": "V", "position": "B", "position_type": "simulated",
              "elapsed_minutes": 40, "available_at": 42,
              "completed_prefix": [{"order_id": "A", "completed_at": 15},
                                   {"order_id": "B", "completed_at": 25}]}]
    result = optimize_dispatch(scenario, rows, vehicle_states=state,
                               locked_prefixes={"V": ["A", "B"]}, quantum=False)
    assert result["validation"]["valid"]
    assert result["plan"]["routes"][0]["order_ids"] == ["A", "B"]
    assert result["expected_cost"]["amount"] == pytest.approx(7.0)
    assert result["remaining_cost_validation"]["historical_prefix_excluded"]


def test_state_aware_cost_skips_completed_history_and_reprices_from_position():
    scenario = {"depot": "D", "vehicles": [{"id": "V", "capacity": 2}],
                "orders": [{"id": "A", "demand": 1, "window": [0, 30], "service": 0, "penalty": 2},
                           {"id": "B", "demand": 1, "window": [0, 25], "service": 0, "penalty": 2}],
                "travel": {"D": {"D": 0, "A": 2, "B": 10},
                           "A": {"D": 0, "A": 0, "B": 9},
                           "B": {"D": 0, "A": 1, "B": 0}}}
    rows = [{"id": "world", "weight": 1.0, "travel": scenario["travel"],
             "source_label": "synthetic_overlay", "prediction_target": "travel_time"}]
    state = [{"vehicle_id": "V", "position": "A", "position_type": "simulated",
              "elapsed_minutes": 18, "available_at": 20,
              "completed_prefix": [{"order_id": "A", "completed_at": 15}]}]
    result = optimize_dispatch(scenario, rows, vehicle_states=state,
                               locked_prefixes={"V": ["A"]}, quantum=False)
    candidate_cost = next(row for row in result["coefficients"]["candidate_costs"]
                          if row["stops"] == ["A", "B"])["expected_cost"]
    assert candidate_cost == pytest.approx(17.0)  # 9 travel + 4 late minutes * 2
    assert result["expected_cost"]["amount"] == pytest.approx(17.0)
    assert result["state_validation"]["historical_prefix_excluded_from_expected_cost"]


def test_whole_delivery_prediction_cannot_be_used_as_a_travel_arc(scenario):
    row = weighted(scenario)[0]
    row["source_label"] = "india_full_delivery_duration_model"
    row["prediction_target"] = "total_delivery_time"
    with pytest.raises(ValueError, match="whole-delivery"):
        evaluate_plan(scenario, {"status": "feasible", "routes": []}, [row, weighted(scenario)[1]])


def test_unlabeled_scenario_matrices_are_rejected(scenario):
    rows = weighted(scenario)
    rows[0].pop("source_label")
    with pytest.raises(ValueError, match="source_label is required"):
        evaluate_plan(scenario, {"status": "feasible", "routes": []}, rows)
