from __future__ import annotations

import json
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

from experiments_run1.model import (
    BASE_VARIANT,
    DEPTH2_VARIANT,
    NON_ENTANGLING_VARIANT,
    FeatureScaler,
    VQCModel,
)
from experiments_run1.train import _bce_loss_and_gradient


def _iter_json_array(path: Path):
    """Incrementally read rows from the large frozen training JSON fixture."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as stream:
        buffer = ""
        position = 0
        eof = False
        while True:
            if position >= len(buffer) and not eof:
                chunk = stream.read(65536)
                eof = not chunk
                buffer = chunk
                position = 0
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer) and buffer[position] in "[,]":
                position += 1
                continue
            if position >= len(buffer):
                if eof:
                    return
                continue
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = stream.read(65536)
                eof = not chunk
                buffer = buffer[position:] + chunk
                position = 0
                continue
            if not isinstance(value, dict):
                raise ValueError("training fixture must contain sample objects")
            yield value
            position = end
            if position >= 65536:
                buffer = buffer[position:]
                position = 0


def _eight_identical_real_train_rows() -> list[SimpleNamespace]:
    repo = Path(__file__).resolve().parents[1]
    fixture = repo / ".runtime/run1/20261029T083317Z_run1_v2_full/route_samples_train.json"
    if not fixture.is_file():
        raise AssertionError(f"frozen RUN 1 training fixture is missing: {fixture}")
    by_features: dict[tuple[float, ...], list[dict]] = {}
    for row in _iter_json_array(fixture):
        features = tuple(float(value) for value in row["raw_features"])
        group = by_features.setdefault(features, [])
        group.append(row)
        if len(group) == 8:
            return [SimpleNamespace(raw_features=features, label=int(item["label"])) for item in group]
    raise AssertionError("frozen training fixture has no eight-row duplicate feature group")


def _real_train_scaler() -> FeatureScaler:
    """Recover the exact train-only min/max ranges without loading labels."""
    repo = Path(__file__).resolve().parents[1]
    fixture = repo / ".runtime/run1/20261029T083317Z_run1_v2_full/route_samples_train.json"
    minimum = [math.inf, math.inf]
    maximum = [-math.inf, -math.inf]
    reading = False
    coordinate = 0
    row_count = 0
    with fixture.open("r", encoding="utf-8") as stream:
        for line in stream:
            if '"raw_features": [' in line:
                reading = True
                coordinate = 0
                continue
            if not reading:
                continue
            value = line.strip().rstrip(",")
            if value == "]":
                reading = False
                row_count += 1
                continue
            number = float(value)
            minimum[coordinate] = min(minimum[coordinate], number)
            maximum[coordinate] = max(maximum[coordinate], number)
            coordinate += 1
            if coordinate > 2:
                raise AssertionError("real train feature row has more than two fields")
    if not row_count or not all(math.isfinite(value) for value in minimum + maximum):
        raise AssertionError("real train feature scaler could not be recovered")
    return FeatureScaler(tuple(minimum), tuple(maximum), ("real-train-fixture",))


def _c202_train_native_pool_scenario():
    from backend.qubo import build_route_qubo, enumerate_qubo
    from experiments_run1.native import load_instance
    from experiments_run1.run import _service_start_features
    from experiments_run1.train import DecisionScenario, PlanOutcome, RouteSample

    repo = Path(__file__).resolve().parents[1]
    pool_data = json.loads((repo / ".runtime/run1/pool-inventory.json").read_text(encoding="utf-8"))
    record = next(row for row in pool_data["eligible"]
                  if row["source"] == "solomon" and row["size"] == 25 and row["instance_id"] == "C202")
    instance = load_instance("solomon", "C202", 25)
    candidates = []
    for source_row in record["candidate_routes"]:
        row = dict(source_row)
        row["features"] = list(_service_start_features(instance, row["route"]))
        candidates.append(row)
    fixed_routes = []
    for source_row in record["fixed_routes"]:
        row = dict(source_row)
        row["features"] = list(_service_start_features(instance, row["route"]))
        fixed_routes.append(row)
    fixed_ids = [str(row["id"]) for row in fixed_routes]
    required_ids = {str(row["id"]) for row in candidates} | set(fixed_ids)
    fixture = repo / ".runtime/run1/20261029T083317Z_run1_v2_full/route_samples_train.json"
    by_seed: dict[int, dict[str, dict]] = {}
    selected_rows = None
    for row in _iter_json_array(fixture):
        metadata = row.get("metadata", {})
        if (row.get("source") != "solomon" or row.get("instance_id") != "C202" or
                metadata.get("dataset_size") != 25):
            continue
        route_id = str(row["route_id"])
        if route_id not in required_ids:
            continue
        seed = int(row["replay_seed"])
        group = by_seed.setdefault(seed, {})
        group[route_id] = row
        if required_ids.issubset(group):
            selected_rows = group
            break
    if selected_rows is None:
        raise AssertionError("no complete C202 n=25 native-pool scenario in train-only fixture")

    samples_by_id = {}
    for route_id in required_ids:
        row = selected_rows[route_id]
        sample = RouteSample(
            source=str(row["source"]), topology_id=str(row["topology_id"]),
            instance_id=str(row["instance_id"]), replay_seed=int(row["replay_seed"]),
            route_id=route_id, raw_features=tuple(float(value) for value in row["raw_features"]),
            label=int(row["label"]), lateness_severity=float(row["lateness_severity"]),
            forecast_cost=float(row["forecast_cost"]), penalty_rate=float(row["penalty_rate"]),
            hard_window_violations=int(row["hard_window_violations"]),
            native_duration=float(row["native_duration"]),
            forecast_travel_duration=float(row["forecast_travel_duration"]),
            forecast_duty_duration=float(row["forecast_duty_duration"]),
            realized_travel_duration=float(row["realized_travel_duration"]),
            realized_duty_duration=float(row["realized_duty_duration"]),
            realized_total_cost=float(row["realized_total_cost"]), metadata=row.get("metadata", {}),
        )
        samples_by_id[route_id] = sample
    routes = tuple(samples_by_id[str(row["id"])] for row in candidates + fixed_routes)
    qubo_rows = [{"id": row["id"], "vehicle": row["vehicle"],
                  "stops": row["stops"], "cost": row["forecast_cost"]} for row in candidates]
    qubo = build_route_qubo(qubo_rows, record["required_stops"])
    plans = []
    for enumeration in enumerate_qubo(qubo):
        if not enumeration["feasible"]:
            continue
        selected = set(str(value) for value in enumeration["selected_route_ids"])
        candidate_plan = [row for row in candidates if str(row["id"]) in selected]
        native = instance.evaluate_solution(
            [row["route"] for row in fixed_routes + candidate_plan]
        )
        if not native.valid:
            continue
        route_ids = tuple([str(row["id"]) for row in candidate_plan] + fixed_ids)
        selected_samples = [samples_by_id[route_id] for route_id in route_ids]
        plans.append(PlanOutcome(
            route_ids=route_ids,
            realized_cost=sum(sample.realized_total_cost for sample in selected_samples),
            hard_window_violations=sum(sample.hard_window_violations for sample in selected_samples),
            objective_unit=str(selected_samples[0].metadata["financial_rate_contract"]["unit_name"]),
        ))
    if len(plans) < 2:
        raise AssertionError("native pool did not provide at least two train scenario plans")
    topology_ids = {sample.topology_id for sample in routes}
    if len(topology_ids) != 1:
        raise AssertionError("train fixture route samples do not share one topology ID")
    return DecisionScenario(
        scenario_id="train-fixture:solomon:C202:n25:one-seed",
        topology_id=next(iter(topology_ids)), split="train", routes=routes, plans=tuple(plans),
    )


def test_exact_memoization_matches_real_fixture_loss_gradient_and_one_step_update():
    samples = _eight_identical_real_train_rows()
    candidate_path = Path(__file__).resolve().parents[1] / ".runtime/run1/optimization_candidate.py"
    spec = importlib.util.spec_from_file_location("run1_optimization_candidate", candidate_path)
    assert spec is not None and spec.loader is not None
    optimization_candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(optimization_candidate)
    # Recover the actual train-only min/max scaler from train raw_features only;
    # validation/test files and labels do not contribute to scaling.
    scaler = _real_train_scaler()
    initial = {
        BASE_VARIANT: (0.31, -0.47, -3.5, -1.5),
        DEPTH2_VARIANT: (0.31, -0.47, 0.18, -0.22, -3.5, -1.5),
        NON_ENTANGLING_VARIANT: (0.31, -0.47, -3.5, -1.5),
    }

    for variant, parameters in initial.items():
        model = VQCModel(parameters, scaler, circuit_variant=variant)
        cached = optimization_candidate.MemoizedProbabilityGradient(model)
        scaled = model.transform(samples[0].raw_features)
        probability, gradient, baseline_eval_count = model.probability_and_gradient(scaled)
        cached_probability, cached_gradient, cached_eval_count = cached.probability_and_gradient(scaled)
        assert (cached_probability, cached_gradient, cached_eval_count) == (
            probability, gradient, baseline_eval_count
        )
        # Seven repeated calls return identical probability/gradient and report
        # zero new circuit evaluations while retaining all eight loss rows.
        for _ in range(7):
            assert cached.probability_and_gradient(scaled) == (probability, gradient, 0)
        assert cached.cache_hits == 7
        assert cached.cache_misses == 1
        cached.clear()

        baseline_loss, baseline_gradient, baseline_evals = _bce_loss_and_gradient(model, samples)
        cached_loss, cached_gradient, cached_evals = _bce_loss_and_gradient(cached, samples)
        assert cached_loss == baseline_loss
        assert cached_gradient == baseline_gradient
        assert len(samples) == 8
        assert baseline_evals == 8 * baseline_eval_count
        assert cached_evals == cached.actual_circuit_evaluations == baseline_eval_count
        assert cached.cache_misses == 1
        assert cached.cache_hits == 7

        learning_rate = 0.04
        updated_parameters = tuple(
            value - learning_rate * derivative
            for value, derivative in zip(model.parameters, baseline_gradient)
        )
        cached_updated_parameters = tuple(
            value - learning_rate * derivative
            for value, derivative in zip(model.parameters, cached_gradient)
        )
        baseline_updated = model.with_parameters(updated_parameters)
        cached_updated = model.with_parameters(cached_updated_parameters)
        assert baseline_updated.parameters == cached_updated.parameters
        assert all(math.isfinite(value) for value in baseline_updated.parameters)
        assert baseline_updated.predict_proba(samples[0].raw_features) == cached_updated.predict_proba(
            samples[0].raw_features
        )


def test_memoization_key_changes_with_parameters_and_circuit_variant():
    from pathlib import Path
    import importlib.util

    candidate_path = Path(__file__).resolve().parents[1] / ".runtime/run1/optimization_candidate.py"
    spec = importlib.util.spec_from_file_location("run1_optimization_candidate_key_test", candidate_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = VQCModel((0.31, -0.47, -3.5, -1.5))
    cache = module.MemoizedProbabilityGradient(model)
    features = (0.24, 0.73)
    cache.probability_and_gradient(features)

    model.parameters = (0.32, -0.47, -3.5, -1.5)
    expected = model.probability_and_gradient(features)
    assert cache.probability_and_gradient(features) == expected

    model.circuit_variant = NON_ENTANGLING_VARIANT
    expected = model.probability_and_gradient(features)
    assert cache.probability_and_gradient(features) == expected
    assert cache.cache_misses == 3
    assert cache.cache_hits == 0


def test_decision_objective_memoization_matches_native_train_scenario_and_update():
    from pathlib import Path
    import importlib.util
    from experiments_run1.train import decision_loss_and_gradient

    samples = _c202_train_native_pool_scenario()
    scenario = samples
    scaler = _real_train_scaler()
    severity_file = Path(__file__).resolve().parents[1] / ".runtime/run1/20261029T083317Z_run1_v2_full/severity_estimate.json"
    severity = json.loads(severity_file.read_text(encoding="utf-8"))
    assert severity["fit_split"] == "train"
    assert scenario.topology_id in severity["training_topology_ids"]
    severity_by_source = {"solomon": float(severity["by_source"]["solomon"])}

    candidate_path = Path(__file__).resolve().parents[1] / ".runtime/run1/optimization_candidate.py"
    spec = importlib.util.spec_from_file_location("run1_optimization_candidate_decision", candidate_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    initial = {
        BASE_VARIANT: (0.31, -0.47, -3.5, -1.5),
        DEPTH2_VARIANT: (0.31, -0.47, 0.18, -0.22, -3.5, -1.5),
    }
    for variant, parameters in initial.items():
        model = VQCModel(parameters, scaler, circuit_variant=variant)
        cache = module.MemoizedProbabilityGradient(model)
        settings = {"temperature": 500.0, "decision_weight": 0.85,
                    "severity_by_source": severity_by_source}
        baseline_loss, baseline_gradient, baseline_evaluations = decision_loss_and_gradient(
            model, (scenario,), **settings
        )
        cached_loss, cached_gradient, cached_evaluations = decision_loss_and_gradient(
            cache, (scenario,), **settings
        )
        scaled_features = {model.transform(route.raw_features) for route in scenario.routes}
        expected_per_call = 1 + 2 * model._circuit_angle_count
        assert cached_loss == baseline_loss
        assert cached_gradient == baseline_gradient
        assert len(scenario.routes) == len({route.route_id for route in scenario.routes})
        assert baseline_evaluations == len(scenario.routes) * expected_per_call
        assert cache.cache_misses == len(scaled_features)
        assert cache.cache_hits == len(scenario.routes) - len(scaled_features)
        assert cached_evaluations == cache.actual_circuit_evaluations == len(scaled_features) * expected_per_call

        learning_rate = 0.03
        base_params = tuple(value - learning_rate * derivative
                            for value, derivative in zip(model.parameters, baseline_gradient))
        cached_params = tuple(value - learning_rate * derivative
                              for value, derivative in zip(model.parameters, cached_gradient))
        base_updated = model.with_parameters(base_params)
        cached_updated = model.with_parameters(cached_params)
        assert base_updated.parameters == cached_updated.parameters
        assert base_updated.predict_proba(scenario.routes[0].raw_features) == cached_updated.predict_proba(
            scenario.routes[0].raw_features
        )
