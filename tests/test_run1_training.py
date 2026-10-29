from __future__ import annotations

import math
from types import SimpleNamespace
from pathlib import Path

import pytest

from experiments_run1.model import (
    DEPTH2_VARIANT,
    NON_ENTANGLING_VARIANT,
    FeatureScaler,
    VQCModel,
)
from experiments_run1.train import (
    DecisionScenario,
    PlanOutcome,
    RouteSample,
    assign_topology_splits,
    decision_loss_and_gradient,
    estimate_conditional_severity,
    fit_logistic_baseline,
    fit_newton_gbdt_baseline,
    fit_train_only_scaler,
    generate_route_replays,
    train_decision_focused_vqc,
    topology_fingerprint,
    train_predictive_vqc,
)


def _sample(group: str, route_id: str, features: tuple[float, float], label: int, *, source: str = "solomon") -> RouteSample:
    return RouteSample(
        source=source,
        topology_id=group,
        instance_id=group,
        replay_seed=7,
        route_id=route_id,
        raw_features=features,
        label=label,
        lateness_severity=1.5 if label else 0.0,
        forecast_cost=10.0,
    )


def _central_difference(function, parameters: tuple[float, ...], step: float = 1e-6) -> tuple[float, ...]:
    result = []
    for index in range(len(parameters)):
        plus, minus = list(parameters), list(parameters)
        plus[index] += step
        minus[index] -= step
        result.append((function(tuple(plus)) - function(tuple(minus))) / (2.0 * step))
    return tuple(result)


def test_v11_affine_readout_gradient_matches_centered_finite_difference():
    parameters = (0.31, -0.47, -3.5, -1.5)
    model = VQCModel(parameters)
    features = (0.24, 0.73)
    _, analytic, evaluations = model.probability_and_gradient(features)
    numeric = _central_difference(
        lambda theta: VQCModel(theta).predict_proba_scaled(features), parameters
    )
    assert evaluations == 5
    assert all(abs(a - n) < 2e-7 for a, n in zip(analytic, numeric))
    assert all(abs(value) > 1e-9 for value in analytic)


def test_saturated_clipped_readout_has_zero_gradient():
    model = VQCModel((0.31, -0.47, 2.0, 40.0))
    probability, gradient, _ = model.probability_and_gradient((0.2, 0.7))
    assert probability == pytest.approx(1.0)
    assert gradient == (0.0, 0.0, 0.0, 0.0)


def test_both_existing_features_influence_the_readout():
    model = VQCModel((0.31, -0.47, -3.5, -1.5))
    baseline = model.predict_proba_scaled((0.24, 0.73))
    perturbed_0 = model.predict_proba_scaled((0.29, 0.73))
    perturbed_1 = model.predict_proba_scaled((0.24, 0.78))
    assert abs(baseline - perturbed_0) > 1e-6
    assert abs(baseline - perturbed_1) > 1e-6


def test_v1_one_block_two_block_and_non_entangling_ablation_profiles():
    one = VQCModel((0.31, -0.47, -3.5, -1.5))
    two = VQCModel((0.31, -0.47, 0.18, -0.22, -3.5, -1.5), circuit_variant=DEPTH2_VARIANT)
    no_entanglement = VQCModel((0.31, -0.47, -3.5, -1.5), circuit_variant=NON_ENTANGLING_VARIANT)
    assert one.circuit_spec["depth"] == 4
    assert one.circuit_spec["gate_counts"] == {"RY": 4, "CNOT": 2}
    assert two.circuit_spec["depth"] == 7
    assert two.circuit_spec["gate_counts"] == {"RY": 6, "CNOT": 4}
    assert no_entanglement.circuit_spec["depth"] == 2
    assert no_entanglement.circuit_spec["gate_counts"] == {"RY": 4, "CNOT": 0}
    features = (0.24, 0.73)
    _, gradient, _ = two.probability_and_gradient(features)
    assert all(math.isfinite(value) for value in gradient)
    numeric = _central_difference(
        lambda params: VQCModel(params, circuit_variant=DEPTH2_VARIANT).predict_proba_scaled(features),
        two.parameters,
    )
    assert all(abs(a - n) < 3e-7 for a, n in zip(gradient, numeric))
    _, unentangled_gradient, _ = no_entanglement.probability_and_gradient((0.24, 0.73))
    assert unentangled_gradient[1] == pytest.approx(0.0, abs=1e-12)
    assert any(abs(value) > 1e-8 for value in gradient[:4])


def test_train_only_scaler_clips_validation_features_without_refitting():
    train = [_sample("g-train", "a", (2.0, 10.0), 0), _sample("g-train", "b", (4.0, 20.0), 1)]
    validation = [_sample("g-validation", "c", (100.0, -20.0), 0)]
    scaler = fit_train_only_scaler(train + validation, {"g-train": "train", "g-validation": "validation"})
    assert scaler.minimum == (2.0, 10.0)
    assert scaler.maximum == (4.0, 20.0)
    assert scaler.fit_group_ids == ("g-train",)
    assert scaler.transform(validation[0].raw_features) == (1.0, 0.0)


def test_topology_fingerprint_ignores_windows_and_links_size_prefixes():
    coords_25 = {0: (0, 0), **{i: (i, i * 2) for i in range(1, 26)}}
    coords_100 = {0: (0, 0), **{i: (i, i * 2) for i in range(1, 101)}}
    first = SimpleNamespace(coordinates=coords_25)
    larger = SimpleNamespace(coordinates=coords_100)
    changed_geometry = SimpleNamespace(coordinates={0: (0, 0), **{i: (i + 0.1, i * 2) for i in range(1, 101)}})
    assert topology_fingerprint(first) == topology_fingerprint(larger)
    assert topology_fingerprint(first) != topology_fingerprint(changed_geometry)


def test_audited_r_and_rc_family_aliases_share_topology_groups():
    audited = [
        SimpleNamespace(instance_id="R101", coordinates=[(0, 0)] + [(i, i + 1) for i in range(1, 26)]),
        SimpleNamespace(instance_id="R201", coordinates=[(0, 0)] + [(i, i + 1) for i in range(1, 26)]),
        SimpleNamespace(instance_id="RC101", coordinates=[(0, 0)] + [(i, i + 1) for i in range(1, 26)]),
        SimpleNamespace(instance_id="RC201", coordinates=[(0, 0)] + [(i, i + 1) for i in range(1, 26)]),
        SimpleNamespace(instance_id="C101", coordinates=[(0, 0)] + [(i, i + 1) for i in range(1, 26)]),
    ]
    groups = [topology_fingerprint(item) for item in audited]
    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    assert len(set(groups)) == 3


def test_grouped_split_is_stable_and_never_splits_topology():
    groups = [f"topology-{index}" for index in range(8)]
    first = assign_topology_splits(groups, seed=11)
    second = assign_topology_splits(reversed(groups), seed=11)
    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}
    rows = [_sample(group, str(i), (i / 10.0, i / 20.0), i % 2) for group in groups for i in range(2)]
    for row in rows:
        assert first[row.topology_id] in {"train", "validation", "test"}
    with pytest.raises(ValueError, match="three independent"):
        assign_topology_splits(groups[:2])


def test_predictive_training_is_bounded_and_returns_finite_validation_selection():
    train = [_sample("train-topology", f"r{i}", (i / 10.0, 1.0 - i / 10.0), i % 2) for i in range(10)]
    validation = [_sample("validation-topology", f"v{i}", (i / 10.0, 1.0 - i / 10.0), i % 2) for i in range(6)]
    scaler = FeatureScaler.fit([row.raw_features for row in train], group_ids=["train-topology"])
    fitted = train_predictive_vqc(
        train, validation, scaler=scaler, max_steps=8, patience=3, learning_rate=0.04
    )
    assert 0 <= fitted.selected_step <= 8
    assert fitted.circuit_evaluations > 0
    assert fitted.wall_seconds >= 0 and fitted.cpu_seconds >= 0 and fitted.peak_memory_bytes > 0
    assert all(math.isfinite(value) for value in fitted.model.parameters)
    assert all(0.0 <= fitted.model.predict_proba(row.raw_features) <= 1.0 for row in validation)


def test_classical_logistic_and_v11_gbdt_share_train_scaler():
    train = [_sample("train-topology", f"r{i}", (i / 10.0, 1.0 - i / 10.0), int(i >= 5)) for i in range(10)]
    validation = [_sample("validation-topology", f"v{i}", (i / 10.0, 1.0 - i / 10.0), int(i >= 5)) for i in range(6)]
    scaler = FeatureScaler.fit([row.raw_features for row in train], group_ids=["train-topology"])
    logistic = fit_logistic_baseline(train, validation, scaler=scaler, max_steps=12, patience=4)
    gbdt = fit_newton_gbdt_baseline(train, validation, scaler=scaler, n_estimators=4, max_depth=2)
    assert logistic.model.scaler == scaler
    assert gbdt.model.scaler == scaler
    assert 0 <= logistic.model.predict_proba(validation[0].raw_features) <= 1
    assert 0 <= gbdt.model.predict_proba(validation[0].raw_features) <= 1
    assert logistic.wall_seconds >= 0 and gbdt.wall_seconds >= 0


def test_severity_estimator_reads_only_frozen_training_groups():
    rows = [
        _sample("train", "r0", (0.1, 0.2), 1),
        _sample("train", "r1", (0.2, 0.3), 1),
        _sample("validation", "r2", (0.3, 0.4), 1),
        _sample("test", "r3", (0.4, 0.5), 1),
    ]
    rows[0] = RouteSample(**{**rows[0].__dict__, "lateness_severity": 2.0})
    rows[1] = RouteSample(**{**rows[1].__dict__, "lateness_severity": 4.0})
    rows[2] = RouteSample(**{**rows[2].__dict__, "lateness_severity": 100.0})
    rows[3] = RouteSample(**{**rows[3].__dict__, "lateness_severity": 1000.0})
    split_map = {"train": "train", "validation": "validation", "test": "test"}
    estimate = estimate_conditional_severity(rows, split_map)
    assert estimate.by_source == {"solomon": 3.0}
    assert estimate.training_sample_count == 2
    assert estimate.training_topology_ids == ("train",)
    assert estimate.training_split_sha256 == __import__("hashlib").sha256(b"train").hexdigest()


def test_repeated_instance_ids_are_separated_by_dataset_size():
    from experiments_run1.run import _group_samples_by_instance

    n25 = RouteSample("solomon", "topology:C1", "C101", 17, "route:1>2", (0.2, 0.3),
                      0, 0.0, 10.0, metadata={"dataset_size": 25})
    n50 = RouteSample("solomon", "topology:C1", "C101", 17, "route:1>2", (0.2, 0.3),
                      1, 3.0, 20.0, realized_total_cost=23.0, metadata={"dataset_size": 50})
    grouped = _group_samples_by_instance([n25, n50])
    assert set(grouped) == {("solomon", 25, "C101", 17), ("solomon", 50, "C101", 17)}
    assert grouped[("solomon", 25, "C101", 17)][0].realized_total_cost == 0.0
    assert grouped[("solomon", 50, "C101", 17)][0].realized_total_cost == 23.0


def test_decision_surrogate_gradient_matches_finite_difference_and_uses_train_severity():
    routes = (
        _sample("train-topology", "r1", (0.2, 0.7), 1),
        _sample("train-topology", "r2", (0.8, 0.3), 0),
    )
    scenario = DecisionScenario(
        scenario_id="train-replay-1",
        topology_id="train-topology",
        split="train",
        routes=routes,
        plans=(PlanOutcome(("r1",), 12.0), PlanOutcome(("r2",), 10.0)),
    )
    scaler = FeatureScaler((0.0, 0.0), (1.0, 1.0), ("train-topology",))
    parameters = (0.31, -0.47, -3.5, -1.5)
    model = VQCModel(parameters, scaler)
    settings = {"temperature": 0.8, "decision_weight": 0.7, "severity_by_source": {"solomon": 1.5}}
    loss, analytic, evaluations = decision_loss_and_gradient(model, (scenario,), **settings)
    numeric = _central_difference(
        lambda theta: decision_loss_and_gradient(
            VQCModel(theta, scaler), (scenario,), **settings
        )[0], parameters
    )
    assert math.isfinite(loss)
    assert evaluations > 0
    assert all(abs(a - n) < 3e-6 for a, n in zip(analytic, numeric))


def test_decision_objective_rejects_test_outcomes():
    sample = _sample("test-topology", "r1", (0.2, 0.7), 1)
    scenario = DecisionScenario("test-only", "test-topology", "test", (sample,), (PlanOutcome(("r1",), 10.0),))
    model = VQCModel((0.31, -0.47, -3.5, -1.5), FeatureScaler((0, 0), (1, 1)))
    with pytest.raises(ValueError, match="test outcomes"):
        decision_loss_and_gradient(model, (scenario,), temperature=0.5, decision_weight=0.8,
                                   severity_by_source={"solomon": 1.5})


def test_decision_temperature_uses_train_only_predicted_plan_cost_gaps():
    from experiments_run1.run import _training_decision_temperature

    routes = (_sample("train-topology", "r1", (0.2, 0.7), 1),
              _sample("train-topology", "r2", (0.8, 0.3), 0))
    scenario = DecisionScenario("train-replay", "train-topology", "train", routes,
                                (PlanOutcome(("r1",), 12.0), PlanOutcome(("r2",), 10.0)))
    scaler = FeatureScaler((0.0, 0.0), (1.0, 1.0), ("train-topology",))
    model = VQCModel((0.31, -0.47, -3.5, -1.5), scaler)
    estimate = _training_decision_temperature([scenario], model, {"solomon": 2.0})
    assert estimate["fit_split"] == "train_only"
    assert estimate["positive_plan_gap_count"] == 1
    assert estimate["temperature_scu"] == estimate["gap_median_scu"] >= 1e-3
    validation_routes = (_sample("validation-topology", "r1", (0.2, 0.7), 1),
                         _sample("validation-topology", "r2", (0.8, 0.3), 0))
    validation = DecisionScenario("validation-replay", "validation-topology", "validation", validation_routes,
                                  (PlanOutcome(("r1",), 12.0), PlanOutcome(("r2",), 10.0)))
    with pytest.raises(ValueError, match="train scenarios"):
        _training_decision_temperature([validation], model, {"solomon": 2.0})


def test_two_block_decision_finetune_preserves_six_parameter_architecture():
    train_routes = (_sample("train-topology", "a", (0.2, 0.7), 1),
                    _sample("train-topology", "b", (0.8, 0.3), 0))
    validation_routes = (_sample("validation-topology", "c", (0.3, 0.6), 1),
                         _sample("validation-topology", "d", (0.7, 0.4), 0))
    train_scenario = DecisionScenario("train", "train-topology", "train", train_routes,
                                      (PlanOutcome(("a",), 12.0), PlanOutcome(("b",), 10.0)))
    validation_scenario = DecisionScenario("validation", "validation-topology", "validation", validation_routes,
                                           (PlanOutcome(("c",), 12.0), PlanOutcome(("d",), 10.0)))
    scaler = FeatureScaler((0.0, 0.0), (1.0, 1.0), ("train-topology",))
    pretrained = VQCModel((0.31, -0.47, 0.18, -0.22, -3.5, -1.5), scaler,
                          "test-two-block-pretrain", DEPTH2_VARIANT)
    fit = train_decision_focused_vqc(
        train_routes, validation_routes, [train_scenario], [validation_scenario],
        scaler=scaler, pretrained_model=pretrained, severity_by_source={"solomon": 2.0},
        max_steps=1, patience=1, temperature=1.0,
    )
    assert fit.model.circuit_variant == DEPTH2_VARIANT
    assert len(fit.model.parameters) == 6


def test_real_solomon_replay_uses_fixed_depot_departure_when_optimizer_is_absent():
    from experiments_run1.native import load_instance
    from experiments_run1.run import _service_start_features

    data_root = Path(__file__).resolve().parents[1] / ".runtime" / "datasets"
    if not (data_root / "solomon_100" / "R101.json").is_file():
        pytest.skip("pinned Solomon data are not installed in this checkout")
    instance = load_instance("solomon", "R101", 100, dataset_root=data_root)
    route = (1,)
    nominal = instance.evaluate_route(route)
    assert nominal.feasible
    assert nominal.optimized_departure_time is None
    samples = generate_route_replays(instance, [route], [20261029], source="solomon")
    assert len(samples) == 1
    assert samples[0].metadata["departure_rule"] == "fixed_depot_ready_time"
    assert samples[0].metadata["nominal_departure_fixed"] is True
    assert samples[0].metadata["dataset_size"] == 100
    assert all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
               for a, b in zip(_service_start_features(instance, route), samples[0].raw_features))
    replay = instance.trace_route_at_departure(
        route, instance.time_windows[0][0],
        travel_delays=samples[0].metadata["travel_delays_by_leg"],
        service_delays=samples[0].metadata["service_delays_by_stop"],
    )
    assert replay.departure_time == instance.time_windows[0][0]
    assert replay.feasible
    assert math.isclose(samples[0].realized_duty_duration, replay.duration, rel_tol=1e-12)
