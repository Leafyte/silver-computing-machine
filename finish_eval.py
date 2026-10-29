"""Resume evaluation on frozen RUN1 v2 full run after model training."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from experiments_run1.dispatch import run_dispatch
from experiments_run1.model import FeatureScaler, LogisticModel, ScaledEstimatorModel, VQCModel
from experiments_run1.native import load_instance
from experiments_run1.run import (
    FEATURE_SCHEMA,
    LABEL_CONTRACT,
    SPLIT_SEED,
    _balance,
    _corrected_pool,
    _exact_pool_plan,
    _family,
    _feature_clipping,
    _group_balance,
    _metric_buckets,
    _model_payload,
    _parse_route_id,
    _pool_index,
    _replay_component_summary,
    _route_id,
    _sha256,
    _write_json,
    _write_jsonl,
    binary_metrics,
    generate_route_replays,
    load_instance,
    verify_finite_difference_gradient,
    verify_qpiai_prediction_parity,
    NativeOverlayCapError,
    SyntheticFinancialRates,
)
from experiments_run1.train import RouteSample, SeverityEstimate, compute_safe_overlay_caps
from experiments_v1_1.decision_quality import NewtonGBDT


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    data_root = repo_root / ".runtime" / "datasets"
    out = Path(".runtime/run1/20261029T100322Z_run1_v2_full")
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    config = manifest["run_config"]
    run_id = manifest["run_id"]
    train_split_sha = manifest["split_sha256"]["train"]
    validation_split_sha = manifest["split_sha256"]["validation"]
    test_split_sha = manifest["split_sha256"]["test"]
    pool_exclusions = list(manifest.get("pool_exclusions", []))
    dispatch_count = 1

    source_rows_by_key = {
        (row["source"], int(row["size"]), row["instance_id"]): row
        for row in manifest["source_instances"]
    }
    pool_path = repo_root / ".runtime" / "run1" / "pool-inventory.json"
    pools, pool_sha, pool_manifest = _pool_index(pool_path)
    test_pools = {}
    for key, raw_pool in pools.items():
        if source_rows_by_key[key]["split"] != "test":
            continue
        source, size, instance_id = key
        instance = load_instance(source, instance_id, size, dataset_root=data_root)
        pool = _corrected_pool(instance, raw_pool)
        routes = [tuple(row["route"]) for row in pool["candidate_routes"] + pool["fixed_routes"]]
        caps = compute_safe_overlay_caps(instance, routes)
        pool["safe_overlay_caps"] = caps
        test_pools[key] = pool

    # Load scaler
    vqc_payload = json.loads((out / "models" / "vqc_decision.json").read_text(encoding="utf-8"))
    s_payload = vqc_payload["scaler"]
    scaler = FeatureScaler(
        tuple(s_payload["minimum"]),
        tuple(s_payload["maximum"]),
        tuple(s_payload.get("fit_group_ids", ())),
    )

    # Load severity estimate
    sev_data = json.loads((out / "severity_estimate.json").read_text(encoding="utf-8"))
    severity_estimate = SeverityEstimate(
        by_source=sev_data["by_source"],
        fallback_sources=tuple(sev_data.get("fallback_sources", ())),
        training_split_sha256=sev_data["training_split_sha256"],
        training_topology_ids=tuple(sev_data["training_topology_ids"]),
        training_sample_count=sev_data["training_sample_count"],
    )

    # Load models
    models: dict[str, Any] = {}
    vqc_pred_data = json.loads((out / "models" / "vqc_predictive.json").read_text(encoding="utf-8"))
    models["vqc_predictive"] = VQCModel(tuple(vqc_pred_data["parameters"]), scaler,
                                        vqc_pred_data["model_version"], vqc_pred_data["circuit_variant"])

    vqc_two_data = json.loads((out / "models" / "vqc_predictive_two_blocks.json").read_text(encoding="utf-8"))
    models["vqc_predictive_two_blocks"] = VQCModel(tuple(vqc_two_data["parameters"]), scaler,
                                                   vqc_two_data["model_version"], vqc_two_data["circuit_variant"])

    models["vqc_decision"] = VQCModel(tuple(vqc_payload["parameters"]), scaler,
                                      vqc_payload["model_version"], vqc_payload["circuit_variant"])

    log_data = json.loads((out / "models" / "logistic.json").read_text(encoding="utf-8"))
    models["logistic"] = LogisticModel(tuple(log_data["weights"]), float(log_data["bias"]), scaler,
                                       log_data["model_version"])

    gbdt_data = json.loads((out / "models" / "newton_gbdt.json").read_text(encoding="utf-8"))
    from experiments_v1_1.decision_quality import _GBDTNode

    def _deserialize_tree(node_dict: dict[str, Any]) -> _GBDTNode:
        return _GBDTNode(
            leaf_value=float(node_dict["leaf_value"]),
            feature=node_dict["feature"],
            threshold=node_dict["threshold"],
            left=_deserialize_tree(node_dict["left"]) if node_dict["left"] is not None else None,
            right=_deserialize_tree(node_dict["right"]) if node_dict["right"] is not None else None,
        )

    est_data = gbdt_data["estimator"]
    gbdt_raw = NewtonGBDT(
        n_estimators=est_data["n_estimators"],
        max_depth=est_data["max_depth"],
        learning_rate=est_data["learning_rate"],
        l2_reg=est_data["l2_reg"],
        min_child_hessian=est_data["min_child_hessian"],
    )
    gbdt_raw.base_logit = est_data["base_logit"]
    gbdt_raw.trees = [_deserialize_tree(t) for t in est_data["trees"]]
    models["newton_gbdt"] = ScaledEstimatorModel(gbdt_raw, scaler, gbdt_data["model_version"])

    # Load validation samples for parity/gradient checks
    val_raw = json.loads((out / "route_samples_validation.json").read_text(encoding="utf-8"))
    validation_samples = [RouteSample(
        source=row["source"], topology_id=row["topology_id"], instance_id=row["instance_id"],
        replay_seed=row["replay_seed"], route_id=row["route_id"],
        raw_features=tuple(row["raw_features"]), label=row["label"],
        lateness_severity=row["lateness_severity"], forecast_cost=row["forecast_cost"],
        penalty_rate=row.get("penalty_rate", 1.0), hard_window_violations=row.get("hard_window_violations", 0),
        native_duration=row.get("native_duration", 0.0), forecast_travel_duration=row.get("forecast_travel_duration", 0.0),
        forecast_duty_duration=row.get("forecast_duty_duration", 0.0), realized_travel_duration=row.get("realized_travel_duration", 0.0),
        realized_duty_duration=row.get("realized_duty_duration", 0.0), realized_total_cost=row.get("realized_total_cost", 0.0),
        metadata=row.get("metadata", {}),
    ) for row in val_raw]

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
    test_sentinel = Path(".runtime/run1/TEST-EVALUATION-OPENED-v2.json")
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

    print("--- TEST METRICS ---")
    for name, metric in test_metrics.items():
        print(f"  {name:25s} Brier: {metric['overall']['brier']:.6f}  ECE: {metric['overall']['ece']:.6f}  LogLoss: {metric['overall']['log_loss']:.6f}")

    # Parity and gradient verification
    reloaded_decision = models["vqc_decision"]
    parity = verify_qpiai_prediction_parity(reloaded_decision,
                                             [row.raw_features for row in validation_samples[:4]])
    gradient_check = verify_finite_difference_gradient(reloaded_decision,
                                                       scaler.transform_pair(validation_samples[0].raw_features))
    sensitivity = {}
    for feature_index in range(2):
        base = [0.5, 0.5]
        low, high = list(base), list(base)
        low[feature_index], high[feature_index] = 0.49, 0.51
        sensitivity[f"feature_{feature_index}_abs_delta_probability"] = abs(
            reloaded_decision.predict_proba_scaled(high) - reloaded_decision.predict_proba_scaled(low))

    dec_grad_check = verify_finite_difference_gradient(
        models["vqc_decision"], scaler.transform_pair(validation_samples[0].raw_features))
    _write_json(out / "integrity.json", {
        "models_frozen_sha256": frozen_digest, "vqc_decision_reloaded_sha256": reloaded_decision.sha256,
        "gradient_checks": {"predictive_vqc": gradient_check,
                            "decision_vqc": dec_grad_check},
        "feature_sensitivity": sensitivity,
        "qpiai_parity": {key: parity[key] for key in
                          ("max_absolute_error", "tolerance", "execution_type", "row_count", "sdk_versions",
                           "remote_calls", "qpu_jobs")},
        "source_inventory_sha256": manifest["inventory_sha256"],
        "pool_inventory_sha256": pool_sha, "scaler_sha256": scaler.sha256,
        "test_label_generation_after_freeze": True,
    })

    # Dispatch
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

    # Oracle regret
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
    print("[run1] Evaluation complete and frozen artifacts verified.")


if __name__ == "__main__":
    main()
