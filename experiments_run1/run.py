"""RUN 1 pilot and frozen-inventory training entry points.

The pilot is deliberately isolated from final model selection and held-out
evaluation. It measures the existing V1.1 gradient path using only a topology
group assigned to training before replay labels are generated.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import statistics
import tracemalloc
from typing import Any, Mapping
from collections import defaultdict
from itertools import combinations

from experiments_run1.dispatch import build_native_candidate_pool
from experiments_run1.dispatch import run_dispatch
from backend.qubo import build_route_qubo, enumerate_qubo
from experiments_run1.model import BASE_VARIANT, VQCModel
from experiments_run1.native import load_instance
from experiments_run1.train import (
    LABEL_CONTRACT,
    FEATURE_SCHEMA,
    NativeOverlayCapError,
    SyntheticFinancialRates,
    _bce_loss_and_gradient,
    assign_topology_splits,
    binary_metrics,
    compute_safe_overlay_caps,
    decision_loss_and_gradient,
    estimate_conditional_severity,
    fit_train_only_scaler,
    fit_logistic_baseline,
    fit_newton_gbdt_baseline,
    generate_route_replays,
    split_sha256,
    train_decision_focused_vqc,
    train_predictive_vqc,
    topology_fingerprint,
    verify_finite_difference_gradient,
    verify_qpiai_prediction_parity,
)


RUN_SCHEMA = "run1.run_manifest.v2"
SPLIT_SEED = 20261029
PILOT_CONFIG: dict[str, Any] = {
    "purpose": "bounded compute and native replay seam pilot; not model selection",
    "pilot_only": True,
    "split_seed": SPLIT_SEED,
    "split_unit": "ordered first-25 customer-coordinate topology fingerprint; exact prefixes linked",
    "replay_seeds": list(range(20261029, 20261045)),
    "soft_cutoff_fraction": 0.90,
    "common_delay_fraction": 0.04,
    "route_delay_fraction": 0.02,
    "financial_rates": SyntheticFinancialRates().to_dict(),
    "learning_rate": 0.03,
    "gradient_steps": 1,
    "model_variant": BASE_VARIANT,
    "model_parameters_initial": [0.31, -0.47, -3.5, -1.5],
}


def _family(instance_id: str) -> str:
    upper = instance_id.upper()
    if upper.startswith("RC1"):
        return "RC1"
    if upper.startswith("RC2"):
        return "RC2"
    if upper.startswith("R1"):
        return "R1"
    if upper.startswith("R2"):
        return "R2"
    if upper.startswith("C1"):
        return "C1"
    if upper.startswith("C2"):
        return "C2"
    return "unknown"


def _route_id(route: Any) -> str:
    return "route:" + ">".join(str(int(node)) for node in route)


def _service_start_features(instance: Any, route: Any) -> tuple[float, ...]:
    """Recompute inference features matching the expanded 8-feature schema."""
    nodes = tuple(int(value) for value in route)
    result = instance.evaluate_route(nodes)
    if not result.feasible or len(result.service_starts) != len(nodes):
        raise ValueError(f"cannot build service-start features for native route {nodes}")
    windows = instance.time_windows
    horizon = max(float(due) - float(ready) for ready, due in windows)
    if horizon <= 0.0:
        raise ValueError("native source has no positive time-window horizon")
    soft_slacks = [float(windows[node][0]) + 0.90 *
                   (float(windows[node][1]) - float(windows[node][0])) - float(service_start)
                   for node, service_start in zip(nodes, result.service_starts)]
    duration = float(result.duration)

    _demands = getattr(instance, "demands", None)
    _capacity = getattr(instance, "capacity", 1)
    _num_customers = getattr(instance, "num_customers", len(nodes))
    _leg_travel_times = [float(v) for v in getattr(result, "leg_travel_times", ())]
    _service_duration_val = float(getattr(result, "service_duration", 0.0))

    _mean_soft_slack = sum(soft_slacks) / len(soft_slacks) if soft_slacks else 0.0
    _std_soft_slack = (
        (sum((s - _mean_soft_slack) ** 2 for s in soft_slacks) / len(soft_slacks)) ** 0.5
        if len(soft_slacks) > 1 else 0.0
    )
    _route_len_frac = len(nodes) / max(_num_customers, 1)
    _demand_frac = (
        sum(float(_demands[n]) for n in nodes) / float(_capacity)
        if _demands is not None and _capacity > 0 else 0.0
    )
    _max_leg_frac = (
        max(_leg_travel_times) / duration if _leg_travel_times and duration > 0.0 else 0.0
    )
    _service_ratio = _service_duration_val / duration if duration > 0.0 else 0.0

    return (
        min(soft_slacks) / horizon,
        duration / horizon,
        _mean_soft_slack / horizon,
        _std_soft_slack / horizon,
        _route_len_frac,
        _demand_frac,
        _max_leg_frac,
        _service_ratio,
    )


def _corrected_pool(instance: Any, pool: Mapping[str, Any]) -> dict[str, Any]:
    """Align cached dispatch features to RUN1 service-start training features."""
    corrected = dict(pool)
    for key in ("candidate_routes", "fixed_routes"):
        corrected[key] = []
        for raw in pool[key]:
            row = dict(raw)
            row["features"] = list(_service_start_features(instance, row["route"]))
            row["feature_source"] = "RUN1 native nominal service-start; same schema used for training"
            corrected[key].append(row)
    return corrected


def _pool_index(path: Path) -> tuple[dict[tuple[str, int, str], dict[str, Any]], str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"native candidate-pool inventory is missing: {path}")
    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    payload = json.loads(raw_bytes)
    if payload.get("complete") is not True:
        raise RuntimeError(f"candidate-pool inventory is not complete: {path}")
    rows = {}
    for record in payload.get("eligible", []):
        key = (str(record["source"]), int(record["size"]), str(record["instance_id"]))
        rows[key] = record
    return rows, digest, payload


def _route_samples_and_scenarios(instance: Any, pool: Mapping[str, Any], seeds: list[int], split: str,
                                 *, common_delay: float, route_delay: float):
    pool = _corrected_pool(instance, pool)
    candidate_routes = [dict(row) for row in pool["candidate_routes"]]
    fixed_routes = [dict(row) for row in pool["fixed_routes"]]
    all_routes = candidate_routes + fixed_routes
    native_routes = [tuple(int(node) for node in row["route"]) for row in all_routes]
    samples = generate_route_replays(
        instance, native_routes, seeds, source=instance.source,
        common_delay_fraction=common_delay, route_delay_fraction=route_delay,
    )
    sample_features = {sample.route_id: sample.raw_features for sample in samples}
    for row in all_routes:
        sample_feature = sample_features.get(str(row["id"]))
        if sample_feature is None or len(row["features"]) != len(sample_feature) or any(
                abs(float(a) - float(b)) > 1e-12
                for a, b in zip(row["features"], sample_feature)):
            raise RuntimeError(f"candidate/training feature contract mismatch for {instance.base_id}:{row['id']}")
    by_seed: dict[int, dict[str, Any]] = defaultdict(dict)
    for sample in samples:
        by_seed[sample.replay_seed][sample.route_id] = sample

    q_candidates = [{"id": str(row["id"]), "vehicle": str(row["vehicle"]),
                     "stops": tuple(int(node) for node in row["stops"]),
                     "cost": float(row["forecast_cost"])} for row in candidate_routes]
    qubo = build_route_qubo(q_candidates, pool["required_stops"])
    raw_feasible_rows = [row for row in enumerate_qubo(qubo) if row["feasible"]]
    if not raw_feasible_rows:
        raise RuntimeError(f"no feasible exact-cover QUBO plans for {instance.base_id} n={instance.size}")
    fixed_ids = tuple(_route_id(row["route"]) for row in fixed_routes)
    feasible_rows = []
    for result in raw_feasible_rows:
        selected = tuple(str(value) for value in result["selected_route_ids"])
        selected_candidates = [next(row for row in candidate_routes if str(row["id"]) == route_id)
                               for route_id in selected]
        selected_native = [tuple(int(node) for node in row["route"]) for row in selected_candidates]
        native_solution = instance.evaluate_solution(
            [tuple(int(node) for node in row["route"]) for row in fixed_routes] + selected_native
        )
        if native_solution.valid:
            feasible_rows.append({"selected_route_ids": selected, "route_ids": selected + fixed_ids,
                                  "qubo_bits": result["bits"], "native_cost": native_solution.native_cost})
    if not feasible_rows:
        raise RuntimeError(f"no independently native-valid exact-cover plans for {instance.base_id} n={instance.size}")
    scenarios = []
    for seed in seeds:
        route_map = by_seed.get(seed, {})
        if any(str(row["id"]) not in route_map for row in candidate_routes) or any(rid not in route_map for rid in fixed_ids):
            raise RuntimeError(f"missing native replay sample for {instance.base_id} seed={seed}")
        scenario_routes = tuple(route_map[str(row["id"])] for row in candidate_routes) + tuple(
            route_map[rid] for rid in fixed_ids
        )
        outcomes = []
        for result in feasible_rows:
            all_ids = result["route_ids"]
            chosen_samples = [route_map[route_id] for route_id in all_ids]
            outcomes.append({
                "route_ids": all_ids,
                "realized_cost": sum(sample.realized_total_cost for sample in chosen_samples),
                "hard_window_violations": sum(sample.hard_window_violations for sample in chosen_samples),
                "objective_unit": SyntheticFinancialRates().unit_name,
            })
        if not outcomes:
            continue
        from experiments_run1.train import DecisionScenario, PlanOutcome
        scenarios.append(DecisionScenario(
            scenario_id=f"{instance.source}:{instance.size}:{instance.instance_id}:seed:{seed}",
            topology_id=topology_fingerprint(instance), split=split, routes=scenario_routes,
            plans=tuple(PlanOutcome(tuple(outcome["route_ids"]), float(outcome["realized_cost"]),
                                    int(outcome["hard_window_violations"]), str(outcome["objective_unit"]))
                        for outcome in outcomes),
        ))
    return samples, scenarios, q_candidates


def _metric_buckets(samples: list[Any], model: Any) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        grouped[f"source:{sample.source}"].append(sample)
        grouped[f"family:{_family(sample.instance_id)}"].append(sample)
        grouped[f"source_family:{sample.source}:{_family(sample.instance_id)}"].append(sample)
        grouped[f"source_size:{sample.source}:n{sample.metadata.get('dataset_size')}"].append(sample)
    result = {}
    for key, members in sorted(grouped.items()):
        probs = [float(model.predict_proba(row.raw_features)) for row in members]
        result[key] = binary_metrics([row.label for row in members], probs, bins=10)
    return result


def _serialize_tree(node: Any) -> dict[str, Any]:
    return {"leaf_value": float(node.leaf_value), "feature": node.feature, "threshold": node.threshold,
            "left": _serialize_tree(node.left) if node.left is not None else None,
            "right": _serialize_tree(node.right) if node.right is not None else None}


def _model_payload(model: Any) -> dict[str, Any]:
    if isinstance(model, VQCModel):
        value = model.to_dict()
        value.update({"model_sha256": model.sha256,
                      "scaler_sha256": model.scaler.sha256 if model.scaler else None})
        return value
    if hasattr(model, "weights") and hasattr(model, "bias"):
        return {"model_version": model.model_version, "weights": list(model.weights), "bias": model.bias,
                "scaler": model.scaler.to_dict(), "scaler_sha256": model.scaler.sha256,
                "model_sha256": _sha256({"weights": model.weights, "bias": model.bias,
                                          "scaler_sha256": model.scaler.sha256})}
    if hasattr(model, "estimator"):
        estimator = model.estimator
        return {"model_version": model.model_version,
                "estimator": {"n_estimators": estimator.n_estimators, "max_depth": estimator.max_depth,
                              "learning_rate": estimator.learning_rate, "l2_reg": estimator.l2_reg,
                              "min_child_hessian": estimator.min_child_hessian,
                              "base_logit": estimator.base_logit,
                              "trees": [_serialize_tree(tree) for tree in estimator.trees]},
                "scaler": model.scaler.to_dict(), "scaler_sha256": model.scaler.sha256,
                "model_sha256": _sha256({"base_logit": estimator.base_logit,
                                          "trees": [_serialize_tree(tree) for tree in estimator.trees],
                                          "scaler_sha256": model.scaler.sha256})}
    raise TypeError(f"cannot serialize model type {type(model).__name__}")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_inventory(data_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Load local native sources and freeze hashes/topology groups before labels."""
    repo_root = Path(__file__).resolve().parents[1]
    inventory: dict[str, str] = {}
    by_topology: dict[str, list[dict[str, Any]]] = {}
    source_dirs = [("solomon", size, data_root / f"solomon_{size}")
                   for size in (25, 50, 100)]
    source_dirs.extend(("dabia2013", size, data_root / "Dabia2013" / f"n={size}")
                       for size in (25, 50, 100))
    missing = []
    for source, size, directory in source_dirs:
        files = sorted(path for path in directory.glob("*.json")
                       if path.name != "DATASET-MANIFEST.json") if directory.is_dir() else []
        if not files:
            missing.append(f"{source}:{size} ({directory})")
            continue
        for path in files:
            stem = path.name.removesuffix(".vrp.json")
            if source == "solomon":
                stem = path.stem
            try:
                instance = load_instance(source, stem, size, dataset_root=data_root)
            except (ValueError, FileNotFoundError, RuntimeError) as exc:
                # Dabia listing has auxiliary JSON files; only the canonical
                # instance documents are inventory members.
                if source == "dabia2013" and not path.name.endswith(".vrp.json"):
                    continue
                raise RuntimeError(f"could not load declared native source {path}: {exc}") from exc
            topology_id = topology_fingerprint(instance)
            key = f"{source}:{size}:{stem}"
            inventory[key] = instance.file_hashes["instance"]
            by_topology.setdefault(topology_id, []).append({
                "source": source, "size": size, "instance_id": stem,
                "base_id": instance.base_id,
                "file_hashes": dict(instance.file_hashes),
            })
    if missing:
        raise FileNotFoundError("full inventory is incomplete; missing: " + ", ".join(missing))
    topology_splits = {
        "topology:audited:R": "train",
        "topology:audited:C2": "train",
        "topology:audited:RC": "validation",
        "topology:audited:C1": "test",
    }
    if set(by_topology) != set(topology_splits):
        raise RuntimeError(
            "audited topology inventory does not match frozen C1/C2/R/RC geometry groups: "
            f"{sorted(by_topology)}"
        )
    audit_path = repo_root / ".runtime" / "run1" / "topology-audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"independent coordinate audit missing: {audit_path}")
    audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    return inventory, {
        "schema_version": RUN_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(data_root.resolve()),
        "source_count": len(inventory),
        "topology_group_count": len(by_topology),
        "topology_instances": by_topology,
        "split_by_topology": topology_splits,
        "split_topology_ids": {
            split: sorted(group for group, value in topology_splits.items() if value == split)
            for split in ("train", "validation", "test")
        },
        "split_sha256": {split: hashlib.sha256("\n".join(sorted(
            group for group, value in topology_splits.items() if value == split
        )).encode()).hexdigest() for split in ("train", "validation", "test")},
        "inventory_sha256": _sha256(inventory),
        "near_duplicate_audit": {
            "method": "independent pinned-coordinate audit: exact cross-source and nested-size matches; unordered coordinate multisets; affine translation/orthogonal transform/uniform scale fit",
            "numerical_similarity_threshold_normalized_rms": 1e-12,
            "audit_sha256": audit_sha,
            "audit_path": str(audit_path),
            "status": "audited groups merged: C1, C2, R1/R2, RC1/RC2; C1/C2 nearest normalized RMS 0.302-0.604 are distinct",
        },
    }


def run_pilot(*, source: str = "solomon", size: int = 100, instance_id: str = "R101",
              data_root: Path | None = None, output_root: Path | None = None,
              replay_count: int = 16) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = (data_root or repo_root / ".runtime" / "datasets").resolve()
    output_root = (output_root or repo_root / ".runtime" / "run1").resolve()
    instance = load_instance(source, instance_id, size, dataset_root=data_root)

    # Freeze current complete local source inventory and splits first. If other
    # buckets are pending, this manifest remains marked provisional and cannot
    # be reused for the final run.
    try:
        inventory, manifest = _source_inventory(data_root)
        manifest_status = "complete_local_inventory_provisional_until_acquisition_closed"
    except FileNotFoundError as exc:
        inventory = {f"{source}:{size}:{instance_id}": instance.file_hashes["instance"]}
        manifest = {
            "schema_version": RUN_SCHEMA,
            "dataset_root": str(data_root),
            "source_count": 1,
            "split_by_topology": {},
            "topology_instances": {},
            "near_duplicate_audit": {"status": "pending"},
            "inventory_missing": str(exc),
        }
        raise RuntimeError("cannot freeze train-only pilot assignment before inventory: " + str(exc))

    topology_id = topology_fingerprint(instance)
    split = manifest["split_by_topology"].get(topology_id)
    if split != "train":
        train_groups = set(manifest["split_topology_ids"]["train"])
        eligible = [key for key, rows in manifest["topology_instances"].items()
                    if key in train_groups for row in rows if row["source"] == source and row["size"] == size]
        if not eligible:
            raise RuntimeError("no available instance belongs to a frozen training topology group")
        selected_id = sorted(manifest["topology_instances"][eligible[0]],
                             key=lambda row: row["instance_id"])[0]["instance_id"]
        instance = load_instance(source, selected_id, size, dataset_root=data_root)
        topology_id = topology_fingerprint(instance)
        instance_id = selected_id
        split = manifest["split_by_topology"][topology_id]
    if split != "train":
        raise RuntimeError("pilot instance is not assigned to train before labels")

    pilot_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{source}_{instance_id}_n{size}"
    out = output_root / f"pilot_{pilot_id}"
    out.mkdir(parents=True, exist_ok=False)
    config = {**PILOT_CONFIG, "replay_seeds": list(range(SPLIT_SEED, SPLIT_SEED + replay_count)),
              "source": source, "instance_id": instance_id, "size": size,
              "topology_id": topology_id, "provisional_split": split,
              "inventory_sha256": manifest.get("inventory_sha256"),
              "native_file_hashes": dict(instance.file_hashes),
              "objective_unit": SyntheticFinancialRates().unit_name}
    config_hash = _sha256(config)
    _write_json(out / "frozen_pilot_config.json", {**config, "config_sha256": config_hash})

    # Candidate routes are source-native and do not consult BKS or any replay.
    pool = build_native_candidate_pool(instance, severity_by_source={source: 0.0})
    route_sequences = [row["route"] for row in pool["candidate_routes"]]
    if not route_sequences:
        raise RuntimeError("native candidate generator returned no quantum route candidates")
    samples = generate_route_replays(instance, route_sequences, config["replay_seeds"], source=source)
    if not samples:
        raise RuntimeError("native replay seam produced no pilot samples")
    split_map = {topology_id: "train"}
    scaler = fit_train_only_scaler(samples, split_map)
    model = VQCModel(tuple(config["model_parameters_initial"]), scaler)
    tracemalloc.start()
    measured_wall_start, measured_cpu_start = time.perf_counter(), time.process_time()
    try:
        initial_loss, initial_gradient, gradient_evals = _bce_loss_and_gradient(model, samples)
        updated = model.with_parameters(tuple(
            param - config["learning_rate"] * grad
            for param, grad in zip(model.parameters, initial_gradient)
        ))
        post_step_loss, _, post_evals = _bce_loss_and_gradient(updated, samples)
        _, pilot_peak_memory_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    measured_wall_seconds = time.perf_counter() - measured_wall_start
    measured_cpu_seconds = time.process_time() - measured_cpu_start
    representative = scaler.transform(samples[0].raw_features)
    finite_difference = verify_finite_difference_gradient(model, representative)
    parity_rows = [sample.raw_features for sample in samples[:min(4, len(samples))]]
    sdk_parity = verify_qpiai_prediction_parity(model, parity_rows)
    balance = {"n": len(samples), "positive": sum(s.label for s in samples),
               "negative": sum(1 - s.label for s in samples),
               "positive_rate": sum(s.label for s in samples) / len(samples)}
    try:
        label_metrics = binary_metrics([s.label for s in samples], [model.predict_proba(s.raw_features) for s in samples])
    except ValueError:
        label_metrics = None
    pilot = {
        "schema_version": "run1.pilot.v1", "pilot_only": True,
        "selection_or_heldout_evaluation": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source, "instance_id": instance_id, "size": size,
        "topology_id": topology_id, "split": split,
        "instance_file_hashes": dict(instance.file_hashes),
        "native_objective": instance.native_objective,
        "candidate_pool": {"candidate_routes": route_sequences,
                           "fixed_routes": [row["route"] for row in pool["fixed_routes"]],
                           "construction": pool["construction"]},
        "replay_count": replay_count, "route_samples": len(samples),
        "class_balance": balance, "predictive_metrics_on_pilot_training_rows_not_generalization": label_metrics,
        "feature_schema": FEATURE_SCHEMA, "label_contract": LABEL_CONTRACT,
        "financial_rates": SyntheticFinancialRates().to_dict(),
        "scaler": scaler.to_dict(), "scaler_sha256": scaler.sha256,
        "model_initial": model.to_dict(), "model_initial_sha256": model.sha256,
        "gradient_step": {"learning_rate": config["learning_rate"], "steps": 1,
                          "initial_bce": initial_loss, "post_step_bce": post_step_loss,
                          "gradient_l2": math.sqrt(sum(g * g for g in initial_gradient)),
                          "gradient": list(initial_gradient),
                          "circuit_evaluations": gradient_evals + post_evals},
        "gradient_check": finite_difference,
        "qpiai_local_parity": sdk_parity,
        "compute": {"python": sys.version, "platform": platform.platform(),
                    "measured_wall_seconds_gradient_plus_one_update": measured_wall_seconds,
                    "measured_cpu_seconds_gradient_plus_one_update": measured_cpu_seconds,
                    "measured_peak_python_memory_bytes": pilot_peak_memory_bytes,
                    "samples_in_measured_step": len(samples),
                    "circuit_evaluations_in_measured_step": gradient_evals + post_evals,
                    "projected_60_step_wall_seconds_linear_only": measured_wall_seconds * 60,
                    "projection_is_not_measured_full_training": True,
                    "note": "one measured full-batch gradient/update; the pilot does not select final model weights"},
        "manifest_status": manifest_status,
        "manifest": manifest,
        "config_sha256": config_hash,
    }
    _write_json(out / "pilot.json", pilot)
    _write_json(out / "route_samples.json", [asdict(sample) for sample in samples])
    return out


def _balance(samples: list[Any]) -> dict[str, Any]:
    positives = sum(int(row.label) for row in samples)
    return {"n": len(samples), "positive": positives, "negative": len(samples) - positives,
            "positive_rate": positives / len(samples) if samples else None}


def _feature_clipping(samples: list[Any], scaler: Any) -> dict[str, Any]:
    clipped = [0] * len(scaler.minimum)
    total = len(samples)
    for row in samples:
        for index, value in enumerate(row.raw_features):
            if value < scaler.minimum[index] or value > scaler.maximum[index]:
                clipped[index] += 1
    return {"total_rows": total, "per_feature_clipped_count": clipped,
            "per_feature_clipped_rate": [value / total if total else None for value in clipped],
            "scaler_sha256": scaler.sha256}


def _group_balance(samples: list[Any]) -> dict[str, Any]:
    buckets: dict[str, list[Any]] = defaultdict(list)
    for row in samples:
        buckets[f"source:{row.source}"].append(row)
        buckets[f"family:{_family(row.instance_id)}"].append(row)
        buckets[f"source_family:{row.source}:{_family(row.instance_id)}"].append(row)
        buckets[f"source_size:{row.source}:n{row.metadata.get('dataset_size')}"].append(row)
    return {key: _balance(rows) for key, rows in sorted(buckets.items())}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(_jsonable(row), sort_keys=True, allow_nan=False) + "\n" for row in rows),
                    encoding="utf-8")


def _group_samples_by_instance(samples: list[Any]) -> dict[tuple[str, int, str, int], list[Any]]:
    result: dict[tuple[str, int, str, int], list[Any]] = defaultdict(list)
    for row in samples:
        size = row.metadata.get("dataset_size")
        if size is None:
            raise ValueError(f"route sample {row.route_id} lacks dataset_size metadata")
        result[(row.source, int(size), row.instance_id, row.replay_seed)].append(row)
    return result


def _replay_component_summary(samples: list[Any], model: Any, severity: Mapping[str, float]) -> dict[str, Any]:
    predicted = realized_transport = realized_labor = realized_penalty = realized_total = 0.0
    hard_violations = labels = 0
    for sample in samples:
        p = float(model.predict_proba(sample.raw_features))
        predicted += sample.forecast_cost + p * float(severity.get(sample.source, 0.0)) * sample.penalty_rate
        realized_transport += float(sample.metadata.get("realized_transport_cost", 0.0))
        realized_labor += float(sample.metadata.get("realized_labor_cost", 0.0))
        realized_penalty += float(sample.metadata.get("realized_sla_penalty", 0.0))
        realized_total += sample.realized_total_cost
        hard_violations += sample.hard_window_violations
        labels += sample.label
    n = len(samples)
    return {"route_samples": n,
            "expected_risk_adjusted_cost_scu_sum": predicted,
            "realized_transport_cost_scu_sum": realized_transport,
            "realized_labor_cost_scu_sum": realized_labor,
            "realized_sla_penalty_scu_sum": realized_penalty,
            "realized_total_cost_scu_sum": realized_total,
            "realized_total_cost_scu_mean": realized_total / n if n else None,
            "synthetic_sla_breach_count": labels,
            "synthetic_sla_breach_rate": labels / n if n else None,
            "hard_window_violation_count": hard_violations,
            "native_duration_is_reported_separately": True,
            "currency_calibrated": False}


def _exact_pool_plan(candidate_routes: list[dict[str, Any]], pool: Mapping[str, Any], model: Any,
                     severity: float) -> tuple[list[str], list[dict[str, Any]]]:
    priced = []
    for row in candidate_routes:
        p = float(model.predict_proba(row["features"]))
        cost = float(row["forecast_cost"]) + p * severity * float(row["penalty_rate"])
        priced.append({"id": str(row["id"]), "vehicle": str(row["vehicle"]),
                       "stops": tuple(row["stops"]), "cost": cost})
    qubo = build_route_qubo(priced, pool["required_stops"])
    feasible = [row for row in enumerate_qubo(qubo) if row["feasible"]]
    if not feasible:
        raise RuntimeError("risk-priced candidate pool has no exact-cover plan")
    chosen = min(feasible, key=lambda row: (row["business_cost"], row["bits"]))
    return list(chosen["selected_route_ids"]), feasible


def _training_decision_temperature(scenarios: list[Any], model: VQCModel,
                                   severity_by_source: Mapping[str, float]) -> dict[str, Any]:
    """Set a soft-plan temperature from TRAIN-only predicted SCU gaps.

    The protocol uses no realized outcomes: for each training scenario it prices
    plan routes with the predictive model and train-only conditional severity,
    then takes the median positive gap from that scenario's minimum plan cost.
    """
    if not scenarios or any(scenario.split != "train" for scenario in scenarios):
        raise ValueError("decision temperature can be estimated only from non-empty train scenarios")
    gaps = []
    for scenario in scenarios:
        route_map = {route.route_id: route for route in scenario.routes}
        predicted_costs = []
        for plan in scenario.plans:
            cost = 0.0
            for route_id in plan.route_ids:
                route = route_map[route_id]
                severity = float(severity_by_source.get(route.source, 0.0))
                cost += route.forecast_cost + float(model.predict_proba(route.raw_features)) * severity * route.penalty_rate
            predicted_costs.append(cost)
        if not predicted_costs:
            continue
        base = min(predicted_costs)
        gaps.extend(value - base for value in predicted_costs if value - base > 1e-12)
    if not gaps:
        raise ValueError("train-only candidate plans have no positive predicted SCU gaps")
    temperature = max(1e-3, float(statistics.median(gaps)))
    return {"temperature_scu": temperature, "fit_split": "train_only",
            "scenario_count": len(scenarios), "positive_plan_gap_count": len(gaps),
            "gap_min_scu": min(gaps), "gap_median_scu": statistics.median(gaps),
            "gap_max_scu": max(gaps),
            "protocol": "median positive predicted plan-cost gap on train scenarios at predictive-pretrained model; floors 0.001 SCU; excludes realized plan outcomes"}


def run_full(*, data_root: Path | None = None, output_root: Path | None = None,
             replay_count: int = 8, train_steps: int = 60, decision_steps: int = 24,
             dispatch_count: int = 6) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = (data_root or repo_root / ".runtime" / "datasets").resolve()
    output_root = (output_root or repo_root / ".runtime" / "run1").resolve()
    # Keep the preliminary v1 sentinel immutable. V2 explicitly qualifies the
    # same C1 topology as retrospective fresh-seed evidence after that pilot.
    test_sentinel = output_root / "TEST-EVALUATION-OPENED-v2.json"
    if test_sentinel.exists():
        raise RuntimeError(f"RUN 1 test labels have already been opened once: {test_sentinel}")
    inventory, manifest = _source_inventory(data_root)
    pool_path = repo_root / ".runtime" / "run1" / "pool-inventory.json"
    pools, pool_sha, pool_manifest = _pool_index(pool_path)
    print(f"[run1] frozen inventory={len(inventory)} topology_groups={manifest['topology_group_count']} pools={len(pools)} sha256={pool_sha}", flush=True)
    if len(inventory) != 336:
        raise RuntimeError(f"expected all 336 pinned source instances, got {len(inventory)}")
    if not pool_manifest.get("complete"):
        raise RuntimeError("native pool inventory is incomplete")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_run1_v2_full"
    out = output_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "models").mkdir()
    (out / "predictions").mkdir()
    (out / "evaluations").mkdir()
    train_split_sha = manifest["split_sha256"]["train"]
    validation_split_sha = manifest["split_sha256"]["validation"]
    test_split_sha = manifest["split_sha256"]["test"]
    config = {
        "protocol_revision": "RUN1-v2-safe-capped-all-source",
        "evaluation_qualification": "retrospective fresh-seed evaluation on C1 topology previously opened in invalid Solomon-only v1; not a pristine untouched holdout",
        "run_id": run_id, "seed": SPLIT_SEED,
        "replay_seeds": list(range(202610290100, 202610290100 + replay_count)),
        "soft_cutoff_fraction": 0.90, "common_delay_fraction": 0.04, "route_delay_fraction": 0.02,
        "financial_rates": SyntheticFinancialRates().to_dict(),
        "predictive_vqc": {"max_steps": train_steps, "learning_rate": 0.08, "patience": 10},
        "decision_vqc": {"max_steps": decision_steps, "learning_rate": 0.03,
                          "patience": 6, "temperature_protocol": "train-only median positive predicted plan-cost gap in SCU",
                          "decision_weight": 0.85},
        "classical": {"logistic_steps": 300, "gbdt_estimators": 100, "gbdt_depth": 4},
        "ablation_steps": 30, "dispatch": {"evaluation_cases": dispatch_count, "shots": 64,
                                               "grid_size": 2, "max_qaoa_seconds": 12.0},
        "minimum_training_class_support": {"positive": 20, "negative": 20},
        "objective_unit": SyntheticFinancialRates().unit_name,
        "feature_schema": FEATURE_SCHEMA, "label_contract": LABEL_CONTRACT,
    }
    config_sha = _sha256(config)
    manifest.update({
        "run_id": run_id, "run_config": config, "run_config_sha256": config_sha,
        "run_seed": SPLIT_SEED, "feature_schema": FEATURE_SCHEMA, "label_contract": LABEL_CONTRACT,
        "source_instances": [
            {**row, "topology_id": topology, "file_sha256": inventory[
                f"{row['source']}:{row['size']}:{row['instance_id']}"],
             "split": manifest["split_by_topology"][topology]}
            for topology, rows in sorted(manifest["topology_instances"].items()) for row in rows
        ],
        "pool_inventory_sha256": pool_sha,
        "pool_inventory_complete": bool(pool_manifest.get("complete")),
        "candidate_pool_eligible_count": len(pools),
        "test_labels_not_model_inputs": True,
        "test_label_visibility": "sealed; generation only after model artifacts are frozen",
        "test_evaluation_started": False,
        "severity_training_split_sha256": train_split_sha,
        "split_hashes": {"train": train_split_sha, "validation": validation_split_sha, "test": test_split_sha},
        "source_provenance": {"datasets": [
            {"source": "Solomon1987", "url": "https://github.com/gaopan812/VRPTW_solomon_instances",
             "revision": "c2af99436bb81a8277dd98a9e0521f44a769710d", "native_objective": "EuclideanDistance"},
            {"source": "Dabia2013", "url": "https://github.com/ANR-MAMUT/MAMUT-routing",
             "revision": "eca336ff4b087f5c2f7b9ac7c2ab33668ff7af2b", "native_objective": "Duration"},
        ]},
        "code_sha256": {
            name: hashlib.sha256((repo_root / relative).read_bytes()).hexdigest()
            for name, relative in {
                "experiments_run1/run.py": "experiments_run1/run.py",
                "experiments_run1/train.py": "experiments_run1/train.py",
                "experiments_run1/model.py": "experiments_run1/model.py",
                "experiments_run1/native.py": "experiments_run1/native.py",
                "experiments_run1/dispatch.py": "experiments_run1/dispatch.py",
            }.items()
        },
    })
    manifest["near_duplicate_audit"]["status"] = "complete"
    manifest["near_duplicate_audit"]["cross_split_pairs"] = []
    manifest["near_duplicate_audit"]["group_ids"] = ["topology:audited:C1", "topology:audited:C2",
                                                       "topology:audited:R", "topology:audited:RC"]
    manifest["near_duplicate_audit"]["split_rationale"] = (
        "Preserve pilot-exposed R/C2 groups in train; RC validation; C1 used for qualified retrospective fresh-seed evaluation because v1 opened it prematurely; "
        "R1/R2 and RC1/RC2 geometry aliases merged from independent audit."
    )
    source_rows_by_key = {
        (str(row["source"]), int(row["size"]), str(row["instance_id"])): row
        for row in manifest["source_instances"]
    }
    _write_json(out / "run_manifest.json", manifest)
    print(f"[run1] output={out} split=train:{len(manifest['split_topology_ids']['train'])},validation:{len(manifest['split_topology_ids']['validation'])},test:{len(manifest['split_topology_ids']['test'])}", flush=True)

    # Native validation gates (Dabia canonical BKS, where supplied) finish
    # before synthetic target generation. Solomon route feasibility is gated by
    # the independently checked candidate-pool baseline record.
    native_gate = {"checked": 0, "bks_checked": 0, "bks_exact_matches": 0,
                   "candidate_pool_cases": len(pools), "failures": []}
    for key, digest in sorted(inventory.items()):
        source, size_text, instance_id = key.split(":", 2)
        size = int(size_text)
        instance = load_instance(source, instance_id, size, dataset_root=data_root)
        native_gate["checked"] += 1
        if source == "dabia2013" and instance.bks:
            bks = instance.bks
            routes = bks.get("routes", [])
            check = instance.evaluate_solution(routes)
            native_gate["bks_checked"] += 1
            if check.valid and check.native_cost is not None and float(check.native_cost) == float(bks.get("cost", 0.0)):
                native_gate["bks_exact_matches"] += 1
            else:
                native_gate["failures"].append({"source_key": key, "status": check.status,
                                                "expected_cost": bks.get("cost"), "actual_cost": check.native_cost})
    _write_json(out / "native_gate.json", native_gate)
    print(f"[run1] native gate checked={native_gate['checked']} exact BKS={native_gate['bks_exact_matches']}/{native_gate['bks_checked']}", flush=True)
    if native_gate["failures"] or native_gate["checked"] != 336:
        raise RuntimeError(f"native full-inventory gate failed; see {out / 'native_gate.json'}")

    # Compute held-out Dabia ATF caps from nominal source routes before any
    # stochastic replay labels or model fitting. These source-only caps are
    # sealed with the run configuration and reused after model freeze.
    test_pools: dict[tuple[str, int, str], dict[str, Any]] = {}
    test_overlay_caps: dict[str, Any] = {}
    for key, raw_pool in sorted(pools.items()):
        source_row = source_rows_by_key[key]
        if source_row["split"] != "test":
            continue
        source, size, instance_id = key
        instance = load_instance(source, instance_id, size, dataset_root=data_root)
        pool = _corrected_pool(instance, raw_pool)
        routes = [tuple(row["route"]) for row in pool["candidate_routes"] + pool["fixed_routes"]]
        caps = compute_safe_overlay_caps(instance, routes)
        pool["safe_overlay_caps"] = caps
        test_pools[key] = pool
        test_overlay_caps[f"{source}:{size}:{instance_id}"] = {
            "topology_id": source_row["topology_id"], "status": "precomputed_before_labels",
            "route_caps": caps,
        }
    _write_json(out / "test_route_overlay_caps.json", {
        "computed_before_training_and_test_labels": True,
        "method": "native fixed-departure ATF domain cap; common+route rate proportionally clipped per route",
        "upper_rate": 64.0, "instances": test_overlay_caps,
    })
    manifest["test_overlay_caps_sha256"] = hashlib.sha256(
        (out / "test_route_overlay_caps.json").read_bytes()).hexdigest()
    manifest["test_overlay_caps_precomputed_before_labels"] = True
    _write_json(out / "run_manifest.json", manifest)

    train_samples: list[Any] = []
    validation_samples: list[Any] = []
    train_scenarios: list[Any] = []
    validation_scenarios: list[Any] = []
    pool_exclusions = list(pool_manifest.get("excluded", []))
    run_start = time.perf_counter()
    print("[run1] generating only train and validation synthetic replay labels", flush=True)
    for key, pool in sorted(pools.items()):
        source, size, instance_id = key
        source_row = source_rows_by_key[key]
        topology_id, split = source_row["topology_id"], source_row["split"]
        if split == "test":
            continue
        instance = load_instance(source, instance_id, size, dataset_root=data_root)
        try:
            samples, scenarios, _ = _route_samples_and_scenarios(
                instance, pool, config["replay_seeds"], split,
                common_delay=config["common_delay_fraction"], route_delay=config["route_delay_fraction"],
            )
        except NativeOverlayCapError:
            raise
        except (ValueError, RuntimeError) as exc:
            pool_exclusions.append({"source": source, "size": size, "instance_id": instance_id,
                                    "split": split, "stage": "replay_or_plan_outcomes",
                                    "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if split == "train":
            train_samples.extend(samples)
            train_scenarios.extend(scenarios)
        else:
            validation_samples.extend(samples)
            validation_scenarios.extend(scenarios)
    if not train_samples or not validation_samples or not train_scenarios or not validation_scenarios:
        raise RuntimeError("source-native training/validation pools are not both usable; no model fit was run")
    balance = {"train": _group_balance(train_samples), "validation": _group_balance(validation_samples)}
    total_train_balance = _balance(train_samples)
    _write_json(out / "label_balance.json", {"by_split": balance, "train_total": total_train_balance,
                                               "validation_total": _balance(validation_samples),
                                               "pool_exclusions": pool_exclusions})
    if total_train_balance["positive"] < 20 or total_train_balance["negative"] < 20:
        raise RuntimeError("train class support below predeclared 20 positives / 20 negatives; no model fit was run")
    print(f"[run1] replay rows train={len(train_samples)} validation={len(validation_samples)} train positives={total_train_balance['positive']}", flush=True)
    scaler = fit_train_only_scaler(train_samples,
                                   {group: "train" for group in manifest["split_topology_ids"]["train"]})
    validation_groups = {sample.topology_id for sample in validation_samples}
    for group in validation_groups:
        if group in scaler.fit_group_ids:
            raise RuntimeError("validation geometry leaked into scaler fit groups")
    _write_json(out / "route_samples_train.json", [asdict(sample) for sample in train_samples])
    _write_json(out / "route_samples_validation.json", [asdict(sample) for sample in validation_samples])
    severity_estimate = estimate_conditional_severity(train_samples + validation_samples,
                                                      manifest["split_by_topology"])
    _write_json(out / "severity_estimate.json", severity_estimate.to_dict())

    predictive = train_predictive_vqc(
        train_samples, validation_samples, scaler=scaler, max_steps=train_steps, patience=10,
        learning_rate=0.08, model_version="debugleaf-vqc-v1.1-run1-predictive",
    )
    print(f"[run1] predictive V1.1 fit selected_step={predictive.selected_step}", flush=True)
    decision_temperature = _training_decision_temperature(
        train_scenarios, predictive.model, severity_estimate.by_source
    )
    ablations = {}
    for variant, tag in (("run1_two_ansatz_blocks", "two_blocks"), ("run1_non_entangling_ablation", "non_entangling")):
        fit = train_predictive_vqc(train_samples, validation_samples, scaler=scaler,
                                   max_steps=train_steps if tag == "two_blocks" else 30,
                                   patience=10 if tag == "two_blocks" else 7, learning_rate=0.06,
                                   model_version=f"run1-ablation-{tag}", circuit_variant=variant)
        ablations[tag] = fit
    decision_component_checks = {}
    for tag, model in (("v1_1_base", predictive.model),
                       ("two_blocks", ablations["two_blocks"].model)):
        component_loss, component_gradient, component_evals = decision_loss_and_gradient(
            model, train_scenarios, temperature=decision_temperature["temperature_scu"],
            decision_weight=1.0, severity_by_source=severity_estimate.by_source,
        )
        decision_component_checks[tag] = {
            "loss": component_loss,
            "gradient_l2": math.sqrt(sum(value * value for value in component_gradient)),
            "gradient": list(component_gradient), "circuit_evaluations": component_evals,
            "fit_split": "train_only", "split": "train", "training_split_sha256": train_split_sha,
            "training_topology_ids": sorted({scenario.topology_id for scenario in train_scenarios}),
            "scenario_count": len(train_scenarios), "decision_weight": 1.0,
            "bce_anchor_included": False,
            "stage": "predictive_pretrained_initialization_before_decision_finetuning",
        }
    decision_base = train_decision_focused_vqc(
        train_samples, validation_samples, train_scenarios, validation_scenarios,
        scaler=scaler, pretrained_model=predictive.model,
        severity_by_source=severity_estimate.by_source, max_steps=decision_steps,
        patience=6, learning_rate=0.03, temperature=decision_temperature["temperature_scu"], decision_weight=0.85,
        model_version="debugleaf-vqc-v1.1-run1-decision-base",
    )
    decision_two_blocks = train_decision_focused_vqc(
        train_samples, validation_samples, train_scenarios, validation_scenarios,
        scaler=scaler, pretrained_model=ablations["two_blocks"].model,
        severity_by_source=severity_estimate.by_source, max_steps=decision_steps,
        patience=6, learning_rate=0.03, temperature=decision_temperature["temperature_scu"], decision_weight=0.85,
        model_version="run1-two-blocks-decision",
    )
    decision_choices = {"v1_1_base": decision_base, "two_blocks": decision_two_blocks}
    selected_decision_name, decision = min(
        decision_choices.items(), key=lambda pair: min(float(row["validation_loss"]) for row in pair[1].history)
    )
    print(f"[run1] decision variants fitted; validation selected={selected_decision_name}", flush=True)
    logistic = fit_logistic_baseline(train_samples, validation_samples, scaler=scaler,
                                     max_steps=300, patience=30)
    gbdt = fit_newton_gbdt_baseline(train_samples, validation_samples, scaler=scaler,
                                   n_estimators=100, max_depth=4)
    models = {"vqc_predictive": predictive.model,
              "vqc_predictive_two_blocks": ablations["two_blocks"].model,
              "vqc_decision": decision.model,
              "logistic": logistic.model, "newton_gbdt": gbdt.model}
    fits = {"vqc_predictive": predictive, "vqc_decision": decision,
            "logistic": logistic, "newton_gbdt": gbdt}
    for name, model in models.items():
        _write_json(out / "models" / f"{name}.json", _model_payload(model))
    _write_json(out / "models" / "vqc_decision_base.json", _model_payload(decision_base.model))
    _write_json(out / "models" / "vqc_decision_two_blocks.json", _model_payload(decision_two_blocks.model))
    for tag, fit in ablations.items():
        _write_json(out / "models" / f"ablation_{tag}.json", _model_payload(fit.model))

    validation_metrics = {name: {"overall": binary_metrics(
        [row.label for row in validation_samples], [model.predict_proba(row.raw_features) for row in validation_samples], bins=10),
        "by_source_and_family": _metric_buckets(validation_samples, model)} for name, model in models.items()}
    validation_prediction_rows = []
    for row in validation_samples:
        validation_prediction_rows.append({"source": row.source, "size": row.metadata.get("dataset_size"),
                                           "family": _family(row.instance_id),
                                           "instance_id": row.instance_id, "topology_id": row.topology_id,
                                           "replay_seed": row.replay_seed, "route_id": row.route_id,
                                           "raw_features": list(row.raw_features), "label": row.label,
                                           "model_probabilities": {name: model.predict_proba(row.raw_features)
                                                                   for name, model in models.items()}})
    _write_jsonl(out / "predictions" / "validation.jsonl", validation_prediction_rows)
    _write_json(out / "validation_metrics.json", validation_metrics)
    _write_json(out / "training_summary.json", {
        "run_seconds_before_test": time.perf_counter() - run_start,
        "train_samples": len(train_samples), "validation_samples": len(validation_samples),
        "train_scenarios": len(train_scenarios), "validation_scenarios": len(validation_scenarios),
        "class_balance": {"train": total_train_balance, "validation": _balance(validation_samples)},
        "feature_clipping_validation": _feature_clipping(validation_samples, scaler),
        "decision_temperature": decision_temperature,
        "decision_component_gradient_checks": decision_component_checks,
        "fit_results": {name: {"selected_step": fit.selected_step, "wall_seconds": fit.wall_seconds,
                                "cpu_seconds": fit.cpu_seconds,
                                "peak_memory_bytes": getattr(fit, "peak_memory_bytes", None),
                                "circuit_evaluations": getattr(fit, "circuit_evaluations", None),
                                "train_loss": getattr(fit, "train_loss", None),
                                "validation_loss": getattr(fit, "validation_loss", None),
                                "configuration": getattr(fit, "configuration", None),
                                "history": getattr(fit, "history", None)} for name, fit in fits.items()},
        "ablation_validation_bce": {name: min(float(row["validation_bce"]) for row in fit.history)
                                    for name, fit in ablations.items()},
        "architecture_selection": {
            "predictive_vqc_validation_bce": {
                "v1_1_base": min(float(row["validation_bce"]) for row in predictive.history),
                "two_blocks": min(float(row["validation_bce"]) for row in ablations["two_blocks"].history),
                "selected_for_decision_pretraining": ("two_blocks" if
                    min(float(row["validation_bce"]) for row in ablations["two_blocks"].history) <
                    min(float(row["validation_bce"]) for row in predictive.history) else "v1_1_base"),
            },
            "decision_validation_loss": {
                name: min(float(row["validation_loss"]) for row in fit.history)
                for name, fit in decision_choices.items()},
            "selected_decision_architecture": selected_decision_name,
            "selection_split": "validation only",
            "test_used_for_selection": False,
        },
    })

    # Freeze the entire model/scaler family before opening test replay outcomes.
    frozen_hash = hashlib.sha256()
    for path in sorted((out / "models").glob("*.json")):
        frozen_hash.update(path.name.encode())
        frozen_hash.update(path.read_bytes())
    frozen_digest = frozen_hash.hexdigest()
    _write_json(out / "models_frozen.json", {"frozen": True, "models_sha256": frozen_digest,
                                               "train_split_sha256": train_split_sha,
                                               "validation_split_sha256": validation_split_sha,
                                               "test_split_sha256": test_split_sha,
                                               "test_labels_not_model_inputs": True,
                                               "frozen_utc": datetime.now(timezone.utc).isoformat()})
    manifest["models_frozen_sha256"] = frozen_digest
    manifest["models_frozen_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["test_evaluation_started"] = True
    manifest["test_label_visibility"] = "opened once after frozen model hash"
    test_sentinel.parent.mkdir(parents=True, exist_ok=True)
    _write_json(test_sentinel, {"run_id": run_id, "test_split_sha256": test_split_sha,
                                "models_frozen_sha256": frozen_digest,
                                "opened_utc": datetime.now(timezone.utc).isoformat(),
                                "re_evaluation_blocked": True})
    _write_json(out / "run_manifest.json", manifest)
    print(f"[run1] frozen_models_sha256={frozen_digest}; opening held-out labels once", flush=True)

    test_samples: list[Any] = []
    for key, pool in sorted(test_pools.items()):
        source, size, instance_id = key
        instance = load_instance(source, instance_id, size, dataset_root=data_root)
        routes = [tuple(row["route"]) for row in pool["candidate_routes"] + pool["fixed_routes"]]
        try:
            test_samples.extend(generate_route_replays(
                instance, routes, config["replay_seeds"], source=source,
                common_delay_fraction=config["common_delay_fraction"],
                route_delay_fraction=config["route_delay_fraction"],
                safe_overlay_caps=pool["safe_overlay_caps"],
            ))
        except NativeOverlayCapError:
            raise
        except (ValueError, RuntimeError) as exc:
            pool_exclusions.append({"source": source, "size": size, "instance_id": instance_id,
                                    "split": "test", "stage": "sealed_test_evaluation",
                                    "reason": f"{type(exc).__name__}: {exc}"})
    if not test_samples:
        raise RuntimeError("held-out C1 topology produced no post-freeze replay samples")
    test_metrics = {name: {"overall": binary_metrics(
        [row.label for row in test_samples], [model.predict_proba(row.raw_features) for row in test_samples], bins=10),
        "by_source_and_family": _metric_buckets(test_samples, model)} for name, model in models.items()}
    test_prediction_rows = []
    for row in test_samples:
        test_prediction_rows.append({"source": row.source, "size": row.metadata.get("dataset_size"),
                                     "family": _family(row.instance_id),
                                     "instance_id": row.instance_id, "topology_id": row.topology_id,
                                     "replay_seed": row.replay_seed, "route_id": row.route_id,
                                     "raw_features": list(row.raw_features),
                                     "model_probabilities": {name: model.predict_proba(row.raw_features)
                                                             for name, model in models.items()}})
    _write_jsonl(out / "predictions" / "test.jsonl", test_prediction_rows)
    _write_json(out / "test_metrics.json", {
        "independent_test_topology_count": len(manifest["split_topology_ids"]["test"]),
        "test_topology_ids": manifest["split_topology_ids"]["test"],
        "metrics": test_metrics, "class_balance": _group_balance(test_samples),
        "aggregate_class_balance": _balance(test_samples),
        "by_model_cost_components": {name: _replay_component_summary(test_samples, model,
                                                                       severity_estimate.by_source)
                                     for name, model in models.items()},
        "feature_clipping": _feature_clipping(test_samples, scaler),
        "pool_exclusions": pool_exclusions,
        "limitation": "one independent held-out geometry; rows and replay seeds are correlated, no IID confidence intervals",
    })
    _write_json(out / "route_samples_test_sealed_after_eval.json", [asdict(sample) for sample in test_samples])

    # Reload the frozen selected decision model and verify its actual circuit
    # against QpiAI's local statevector before publishing the integrity record.
    vqc_payload = json.loads((out / "models" / "vqc_decision.json").read_text(encoding="utf-8"))
    scaler_payload = vqc_payload["scaler"]
    from experiments_run1.model import FeatureScaler
    reloaded_scaler = FeatureScaler(tuple(scaler_payload["minimum"]), tuple(scaler_payload["maximum"]),
                                    tuple(scaler_payload.get("fit_group_ids", ())))
    reloaded_decision = VQCModel(tuple(vqc_payload["parameters"]), reloaded_scaler,
                                 vqc_payload["model_version"], vqc_payload["circuit_variant"])
    if reloaded_decision.sha256 != vqc_payload["model_sha256"]:
        raise RuntimeError("reloaded decision VQC model hash mismatch")
    parity = verify_qpiai_prediction_parity(reloaded_decision,
                                             [row.raw_features for row in validation_samples[:4]])
    gradient_check = verify_finite_difference_gradient(reloaded_decision,
                                                       reloaded_scaler.transform_pair(validation_samples[0].raw_features))
    sensitivity = {}
    for feature_index in range(2):
        base = [0.5, 0.5]
        low, high = list(base), list(base)
        low[feature_index], high[feature_index] = 0.49, 0.51
        sensitivity[f"feature_{feature_index}_abs_delta_probability"] = abs(
            reloaded_decision.predict_proba_scaled(high) - reloaded_decision.predict_proba_scaled(low))
    _write_json(out / "integrity.json", {
        "models_frozen_sha256": frozen_digest, "vqc_decision_reloaded_sha256": reloaded_decision.sha256,
        "gradient_checks": {"predictive_vqc": gradient_check,
                            "decision_vqc": verify_finite_difference_gradient(
                                models["vqc_decision"], scaler.transform_pair(validation_samples[0].raw_features))},
        "decision_component_gradient": decision_component_checks[selected_decision_name],
        "feature_sensitivity": sensitivity,
        "qpiai_parity": {key: parity[key] for key in
                          ("max_absolute_error", "tolerance", "execution_type", "row_count", "sdk_versions",
                           "remote_calls", "qpu_jobs")},
        "source_inventory_sha256": manifest["inventory_sha256"],
        "pool_inventory_sha256": pool_sha, "scaler_sha256": scaler.sha256,
        "test_label_generation_after_freeze": True,
    })

    # Dispatch a bounded six-case cross-source/size slice from the single
    # held-out C1 geometry. All outcome costs are paired with the same replay
    # seed and remain synthetic SCU; native Distance/Duration is reported apart.
    dispatch_rows = []
    dispatch_keys = [(source, size, "C101") for source in ("solomon", "dabia2013")
                     for size in (25, 50, 100)][:dispatch_count]
    for dispatch_index, key in enumerate(dispatch_keys):
        pool = test_pools.get(key)
        if pool is None:
            dispatch_rows.append({"source_key": key, "status": "not_in_eligible_pool_inventory"})
            continue
        source, size, instance_id = key
        instance = load_instance(source, instance_id, size, dataset_root=data_root)
        candidate_routes = [dict(row) for row in pool["candidate_routes"]]
        fixed_routes = [dict(row) for row in pool["fixed_routes"]]
        for row in candidate_routes + fixed_routes:
            row["expected_breach_severity"] = float(severity_estimate.by_source.get(source, 0.0))
            row["severity_estimate_split"] = "train"
        route_samples = [row for row in test_samples if row.source == source and
                         row.instance_id == instance_id and row.replay_seed == config["replay_seeds"][0]]
        sample_by_route = {(int(row.metadata["dataset_size"]), _parse_route_id(row.route_id)): row
                           for row in route_samples}

        def test_native_evaluator(routes: list[Any]) -> dict[str, Any]:
            solution = instance.evaluate_solution(routes)
            if not solution.valid:
                return {"valid": False, "feasible": False, "status": solution.status,
                        "native_cost": solution.native_cost}
            selected_samples = [sample_by_route[(size, tuple(route))] for route in routes]
            return {"valid": True, "feasible": True, "status": solution.status,
                    "native_cost": solution.native_cost,
                    "native_objective": instance.native_objective,
                    "realized_decision_cost": sum(sample.realized_total_cost for sample in selected_samples),
                    "decision_objective_unit": SyntheticFinancialRates().unit_name,
                    "realized_transport_cost_scu": sum(float(s.metadata["realized_transport_cost"]) for s in selected_samples),
                    "realized_labor_cost_scu": sum(float(s.metadata["realized_labor_cost"]) for s in selected_samples),
                    "realized_sla_penalty_scu": sum(float(s.metadata["realized_sla_penalty"]) for s in selected_samples),
                    "synthetic_sla_breaches": sum(s.label for s in selected_samples),
                    "hard_window_violations": sum(s.hard_window_violations for s in selected_samples),
                    "financial_currency_claim": False}

        metadata = {
            "model_id": "debugleaf-vqc-v1.1-run1-decision", "model_version": models["vqc_decision"].model_version,
            "dataset_id": f"{source}:{instance_id}:n{size}",
            "dataset_sha256": _sha256(instance.file_hashes), "replicate_id": run_id,
            "train_split_sha256": train_split_sha, "validation_split_sha256": validation_split_sha,
            "test_split_sha256": test_split_sha, "evaluation_split": "test",
            "scenario_id": f"{source}:{instance_id}:n{size}:seed{config['replay_seeds'][0]}",
            "objective_unit": SyntheticFinancialRates().unit_name,
            "feature_schema": FEATURE_SCHEMA, "severity_fit_split_sha256": train_split_sha,
            "all_source_customer_ids": list(instance.customers),
            "source_revision": "Solomon:c2af99436bb81a8277dd98a9e0521f44a769710d;Dabia:eca336ff4b087f5c2f7b9ac7c2ab33668ff7af2b",
            "source_files_sha256": dict(instance.file_hashes), "model_sha256": models["vqc_decision"].sha256,
            "scaler_sha256": scaler.sha256, "target_specification": LABEL_CONTRACT,
            "circuit_specification": models["vqc_decision"].circuit_spec,
            "is_reference_fixture": False, "family_seed": SPLIT_SEED, "split_seed": SPLIT_SEED,
            "brier_score_validation": validation_metrics["vqc_decision"]["overall"]["brier"],
            "ece_validation": validation_metrics["vqc_decision"]["overall"]["ece"],
            "financial_unit": SyntheticFinancialRates().unit_name,
            "candidate_pool_construction": pool["construction"],
            "native_bks_duration": (float(instance.bks["cost"])
                                     if source == "dabia2013" and instance.bks and
                                     instance.native_objective == "Duration" else None),
        }
        try:
            result = run_dispatch(
                candidate_routes, pool["required_stops"], models["vqc_decision"], test_native_evaluator,
                metadata=metadata, fixed_routes=fixed_routes, fixed_stops=pool["fixed_stops"],
                seed=SPLIT_SEED + dispatch_index, shots=64, grid_size=2, max_qaoa_seconds=12.0,
            )
            dispatch_rows.append(result)
        except Exception as exc:
            dispatch_rows.append({"source_key": key, "status": "dispatch_failed_after_test_open",
                                  "error_type": type(exc).__name__, "error": str(exc)})
    _write_json(out / "evaluations" / "dispatch.json", dispatch_rows)

    # Same-replay exact candidate-set oracle regret for the classifier-picked
    # plan (separate from dispatch's native Distance/Duration comparison).
    regret_rows = []
    for key, pool in sorted(test_pools.items()):
        source, size, instance_id = key
        relevant = [row for row in test_samples if row.source == source and row.instance_id == instance_id
                    and int(row.metadata["dataset_size"]) == size]
        if not relevant:
            continue
        candidate_routes = [dict(row) for row in pool["candidate_routes"]]
        fixed_ids = [_route_id(row["route"]) for row in pool["fixed_routes"]]
        for seed in config["replay_seeds"]:
            sample_map = {row.route_id: row for row in relevant if row.replay_seed == seed}
            if not sample_map:
                continue
            for model_name in ("vqc_predictive", "vqc_decision", "logistic", "newton_gbdt"):
                selected, feasible = _exact_pool_plan(candidate_routes, pool, models[model_name],
                                                      severity_estimate.by_source.get(source, 0.0))
                plans = [tuple(row["selected_route_ids"] + fixed_ids) for row in feasible]
                costs = [sum(sample_map[rid].realized_total_cost for rid in plan) for plan in plans
                         if all(rid in sample_map for rid in plan)]
                selected_plan_ids = selected + fixed_ids
                missing_ids = set(selected_plan_ids) - set(sample_map)
                if missing_ids:
                    raise RuntimeError(f"partial realized cost refused; missing route samples: {sorted(missing_ids)}")
                selected_cost = sum(sample_map[rid].realized_total_cost for rid in selected_plan_ids)
                if costs:
                    regret_rows.append({"source": source, "size": size, "instance_id": instance_id,
                                        "topology_id": source_rows_by_key[key]["topology_id"],
                                        "replay_seed": seed, "model": model_name,
                                        "objective_unit": SyntheticFinancialRates().unit_name,
                                        "selected_same_replay_realized_cost_scu": selected_cost,
                                        "same_candidate_set_oracle_cost_scu": min(costs),
                                        "same_candidate_set_regret_scu": selected_cost - min(costs),
                                        "candidate_plan_count": len(costs)})
    _write_json(out / "evaluations" / "same_replay_regret.json", {
        "oracle": "minimum realized SCU over all native-feasible exact-cover QUBO selections in same candidate pool and same replay seed",
        "rows": regret_rows,
        "mean_regret_by_model": {name: (sum(row["same_candidate_set_regret_scu"] for row in regret_rows
                                             if row["model"] == name) / max(1, sum(row["model"] == name for row in regret_rows)))
                                 for name in ("vqc_predictive", "vqc_decision", "logistic", "newton_gbdt")},
    })
    artifact_hashes = {}
    for path in sorted(item for item in out.rglob("*") if item.is_file() and
                       item.name not in {"run_manifest.json", "artifact_hashes.json"}):
        artifact_hashes[str(path.relative_to(out)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json(out / "artifact_hashes.json", artifact_hashes)
    manifest["artifacts"] = artifact_hashes
    manifest["artifact_hashes_sha256"] = hashlib.sha256((out / "artifact_hashes.json").read_bytes()).hexdigest()
    manifest["manifest_hash_scope"] = "run_manifest.json excluded from artifact table to avoid self-reference"
    _write_json(out / "run_manifest.json", manifest)
    return out


def _parse_route_id(route_id: str) -> tuple[int, ...]:
    return tuple(int(value) for value in route_id.removeprefix("route:").split(">") if value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Leakage-safe RUN 1 pilot / training entry point")
    sub = parser.add_subparsers(dest="command", required=True)
    pilot = sub.add_parser("pilot", help="run bounded one-step native-source training pilot")
    pilot.add_argument("--source", choices=("solomon", "dabia2013"), default="solomon")
    pilot.add_argument("--instance", default="R101")
    pilot.add_argument("--size", type=int, choices=(25, 50, 100), default=100)
    pilot.add_argument("--data-root", type=Path)
    pilot.add_argument("--output-root", type=Path)
    pilot.add_argument("--replay-count", type=int, default=16)
    full = sub.add_parser("full", help="train after complete source inventory and native gates")
    full.add_argument("--data-root", type=Path)
    full.add_argument("--output-root", type=Path)
    full.add_argument("--replay-count", type=int, default=8)
    full.add_argument("--train-steps", type=int, default=60)
    full.add_argument("--decision-steps", type=int, default=24)
    full.add_argument("--dispatch-count", type=int, default=6)
    args = parser.parse_args(argv)
    if args.command == "pilot":
        if args.replay_count < 2 or args.replay_count > 64:
            parser.error("--replay-count must be in [2, 64]")
        out = run_pilot(source=args.source, instance_id=args.instance, size=args.size,
                        data_root=args.data_root, output_root=args.output_root,
                        replay_count=args.replay_count)
        print(out)
        return 0
    if args.replay_count < 4 or args.replay_count > 32:
        parser.error("--replay-count must be in [4, 32]")
    if args.train_steps < 1 or args.train_steps > 100 or args.decision_steps < 1 or args.decision_steps > 40:
        parser.error("training budgets must be train_steps<=100 and decision_steps<=40")
    if args.dispatch_count < 0 or args.dispatch_count > 6:
        parser.error("--dispatch-count must be in [0, 6]")
    out = run_full(data_root=args.data_root, output_root=args.output_root,
                   replay_count=args.replay_count, train_steps=args.train_steps,
                   decision_steps=args.decision_steps, dispatch_count=args.dispatch_count)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
