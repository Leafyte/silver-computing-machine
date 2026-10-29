"""Independent RUN 1 artifact, leakage, reload, gradient, and dispatch audit."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from experiments_run1.model import FeatureScaler, LogisticModel, ScaledEstimatorModel, VQCModel
from experiments_run1.train import verify_finite_difference_gradient


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status(name: str, passed: bool | None, detail: str) -> dict[str, Any]:
    return {"check": name, "status": "PASS" if passed is True else "FAIL" if passed is False else "NOT RUN",
            "detail": detail}


def _file_manifest(manifest: Mapping[str, Any]) -> list[tuple[str, str]]:
    entries = manifest.get("artifacts", manifest.get("files", {}))
    if isinstance(entries, Mapping):
        output = []
        for rel, record in entries.items():
            digest = record.get("sha256") if isinstance(record, Mapping) else record
            output.append((str(rel), str(digest)))
        return output
    if isinstance(entries, list):
        return [(str(row["path"]), str(row["sha256"])) for row in entries
                if isinstance(row, Mapping) and "path" in row and "sha256" in row]
    return []


def _safe_artifact(root: Path, rel: str) -> Path:
    path = (root / rel).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"artifact path escapes run directory: {rel}")
    return path


def _binary_metrics(labels: list[int], probabilities: list[float], *, bins: int = 10) -> dict[str, float | int]:
    if not labels or len(labels) != len(probabilities):
        raise ValueError("cannot recompute metrics for empty/mismatched labels and probabilities")
    n = len(labels)
    brier = sum((p - y) ** 2 for y, p in zip(labels, probabilities)) / n
    log_loss = -sum(y * math.log(max(p, 1e-15)) + (1 - y) * math.log(max(1 - p, 1e-15))
                    for y, p in zip(labels, probabilities)) / n
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        rows = [(y, p) for y, p in zip(labels, probabilities)
                if low <= p < high or (index == bins - 1 and p == 1.0)]
        if rows:
            ece += len(rows) / n * abs(sum(p for _, p in rows) / len(rows) -
                                       sum(y for y, _ in rows) / len(rows))
    return {"n": n, "positive_count": sum(labels), "brier": brier, "log_loss": log_loss, "ece": ece}


def _metrics_match(recorded: Mapping[str, Any], actual: Mapping[str, float | int], tolerance: float = 1e-9) -> bool:
    for name in ("n", "positive_count"):
        if name in actual and recorded.get(name) != actual[name]:
            return False
    return all(name in recorded and math.isfinite(float(recorded[name])) and
               abs(float(recorded[name]) - float(actual[name])) <= tolerance
               for name in ("brier", "log_loss", "ece"))


def _object_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_classical_artifact(name: str, data: Mapping[str, Any]) -> Any:
    scaler_data = data["scaler"]
    scaler = FeatureScaler(tuple(float(x) for x in scaler_data["minimum"]),
                           tuple(float(x) for x in scaler_data["maximum"]),
                           tuple(str(x) for x in scaler_data.get("fit_group_ids", ())))
    if name == "logistic":
        model = LogisticModel(tuple(float(x) for x in data["weights"]), float(data["bias"]), scaler,
                              str(data.get("model_version", "unknown")))
        expected_sha = _object_sha256({"weights": model.weights, "bias": model.bias,
                                       "scaler_sha256": scaler.sha256})
    elif name == "newton_gbdt":
        from experiments_v1_1.decision_quality import NewtonGBDT, _GBDTNode

        estimator_data = data["estimator"]
        estimator = NewtonGBDT(
            n_estimators=int(estimator_data["n_estimators"]),
            max_depth=int(estimator_data["max_depth"]),
            learning_rate=float(estimator_data["learning_rate"]),
            l2_reg=float(estimator_data["l2_reg"]),
            min_child_hessian=float(estimator_data["min_child_hessian"]),
        )
        estimator.base_logit = float(estimator_data["base_logit"])

        def tree(node: Mapping[str, Any]) -> Any:
            return _GBDTNode(
                leaf_value=float(node["leaf_value"]), feature=node.get("feature"),
                threshold=(float(node["threshold"]) if node.get("threshold") is not None else None),
                left=tree(node["left"]) if node.get("left") is not None else None,
                right=tree(node["right"]) if node.get("right") is not None else None,
            )

        estimator.trees = [tree(node) for node in estimator_data["trees"]]
        model = ScaledEstimatorModel(estimator, scaler, str(data.get("model_version", "unknown")))
        expected_sha = _object_sha256({"base_logit": estimator.base_logit,
                                       "trees": estimator_data["trees"],
                                       "scaler_sha256": scaler.sha256})
    else:
        raise ValueError(f"unsupported classical model artifact: {name}")
    return model, scaler, expected_sha


def load_vqc_artifact(path: Path) -> tuple[VQCModel, Mapping[str, Any]]:
    """Reconstruct a VQC from JSON without pickle/joblib execution."""
    data = _load_json(path)
    model_data = data.get("model", data)
    scaler_data = model_data.get("scaler")
    scaler = None
    if scaler_data is not None:
        scaler = FeatureScaler(
            minimum=tuple(float(value) for value in scaler_data["minimum"]),
            maximum=tuple(float(value) for value in scaler_data["maximum"]),
            fit_group_ids=tuple(str(value) for value in scaler_data.get("fit_group_ids", ())),
        )
    model = VQCModel(
        tuple(float(value) for value in model_data["parameters"]), scaler,
        model_version=str(model_data.get("model_version", "unknown")),
        circuit_variant=str(model_data.get("circuit_variant", "v1_1_base")),
    )
    expected_model_hash = data.get("model_sha256", model_data.get("model_sha256"))
    expected_scaler_hash = data.get("scaler_sha256", model_data.get("scaler_sha256"))
    if expected_model_hash is not None and expected_model_hash != model.sha256:
        raise ValueError("reloaded model SHA-256 does not match artifact")
    if scaler is not None and expected_scaler_hash is not None and expected_scaler_hash != scaler.sha256:
        raise ValueError("reloaded scaler SHA-256 does not match artifact")
    return model, data


def _split_map(manifest: Mapping[str, Any]) -> dict[str, str]:
    mapping = manifest.get("split_by_topology", manifest.get("topology_split_map", {}))
    if isinstance(mapping, Mapping):
        return {str(key): str(value) for key, value in mapping.items()}
    result = {}
    topology_ids = manifest.get("topology_ids_by_split", {})
    if isinstance(topology_ids, Mapping):
        for split in ("train", "validation", "test"):
            for group in topology_ids.get(split, ()):
                result[str(group)] = split
    return result


def _validate_split_integrity(manifest: Mapping[str, Any]) -> tuple[bool | None, str]:
    mapping = _split_map(manifest)
    if not mapping:
        return None, "manifest has no topology split mapping"
    invalid = {key: value for key, value in mapping.items() if value not in {"train", "validation", "test"}}
    by_split = {split: {key for key, value in mapping.items() if value == split}
                for split in ("train", "validation", "test")}
    pair_overlaps = [sorted(by_split[left] & by_split[right])
                     for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))]
    ok = not invalid and not any(pair_overlaps)
    return ok, ("whole-topology assignments are mutually exclusive" if ok else
                f"invalid assignments={invalid}, overlaps={pair_overlaps}")


def validate_run_directory(run_dir: str | Path) -> dict[str, Any]:
    """Audit a RUN 1 export; missing evidence is NOT RUN, never a pass."""
    root = Path(run_dir).resolve()
    checks: list[dict[str, Any]] = []
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        return {"status": "BLOCKED", "run_dir": str(root),
                "checks": [_status("manifest_present", False, "run_manifest.json is missing")],
                "errors": ["manifest missing"]}
    manifest = _load_json(manifest_path)
    checks.append(_status("manifest_schema", manifest.get("schema_version") in {
        "run1.run_manifest.v1", "run1.run_manifest.v2"},
                          str(manifest.get("schema_version"))))
    run_config = manifest.get("run_config", {})
    qualification = (run_config.get("evaluation_qualification")
                     if isinstance(run_config, Mapping) else None)
    if qualification is None:
        checks.append(_status("original_untouched_holdout", None,
                              "manifest does not state whether test topology was previously exposed"))
    else:
        qualification_text = str(qualification).lower()
        pristine = not any(token in qualification_text for token in
                           ("retrospective", "previously opened", "previously exposed", "not a pristine"))
        checks.append(_status("original_untouched_holdout", pristine,
                              str(qualification)))
    if isinstance(run_config, Mapping) and run_config.get("protocol_revision") == "RUN1-v2-safe-capped-all-source":
        expected_seeds = list(range(202610290100, 202610290108))
        predictive_budget = run_config.get("predictive_vqc", {}).get("max_steps")
        decision_budget = run_config.get("decision_vqc", {}).get("max_steps")
        replay_seeds = run_config.get("replay_seeds")
        dispatch_cases = run_config.get("dispatch", {}).get("evaluation_cases")
        budget_ok = (predictive_budget == 60 and decision_budget == 24 and
                     replay_seeds == expected_seeds and dispatch_cases == 6)
        checks.append(_status("v2_frozen_budget_and_seed_policy", budget_ok,
                              f"predictive={predictive_budget}/60; decision={decision_budget}/24; "
                              f"replay_seeds={replay_seeds}; dispatch_cases={dispatch_cases}/6"))
        caps_path = root / "test_route_overlay_caps.json"
        if caps_path.is_file():
            caps = _load_json(caps_path)
            instances = caps.get("instances", {})
            cap_ok = (caps.get("computed_before_training_and_test_labels") is True and
                      isinstance(instances, Mapping) and len(instances) >= 54 and
                      all(isinstance(entry, Mapping) and
                          (not str(key).startswith("dabia2013:") or
                           (isinstance(entry.get("route_caps"), Mapping) and bool(entry.get("route_caps"))))
                          for key, entry in instances.items()))
            checks.append(_status("dabia_overlay_caps_prelabel", cap_ok,
                                  f"precomputed={caps.get('computed_before_training_and_test_labels')}; "
                                  f"instance_entries={len(instances)}"))
        else:
            checks.append(_status("dabia_overlay_caps_prelabel", False,
                                  "v2 run lacks test_route_overlay_caps.json"))
    else:
        checks.append(_status("v2_frozen_budget_and_seed_policy", None,
                              "v2 protocol revision is missing"))
        checks.append(_status("dabia_overlay_caps_prelabel", None,
                              "no v2 prelabel overlay-cap evidence"))

    records = _file_manifest(manifest)
    if not records:
        checks.append(_status("artifact_hashes", None, "manifest has no artifact SHA-256 records"))
    else:
        hash_errors = []
        for relative, expected in records:
            try:
                path = _safe_artifact(root, relative)
            except ValueError as exc:
                hash_errors.append(str(exc))
                continue
            if not path.is_file():
                hash_errors.append(f"missing artifact: {relative}")
            elif _sha256(path).lower() != expected.lower():
                hash_errors.append(f"SHA-256 mismatch: {relative}")
        checks.append(_status("artifact_hashes", not hash_errors,
                              "all listed artifacts match" if not hash_errors else "; ".join(hash_errors)))

    split_ok, split_detail = _validate_split_integrity(manifest)
    checks.append(_status("topology_group_splits", split_ok, split_detail))
    source_rows = manifest.get("source_instances", [])
    if source_rows:
        split_map = _split_map(manifest)
        conflicts = []
        for row in source_rows:
            topology = str(row.get("topology_id", ""))
            declared = str(row.get("split", ""))
            if not topology or declared != split_map.get(topology):
                conflicts.append({"instance": row.get("instance_id"), "topology": topology,
                                  "declared_split": declared, "group_split": split_map.get(topology)})
        checks.append(_status("source_instance_group_assignment", not conflicts,
                              "all source instances inherit their topology split" if not conflicts else str(conflicts)))
    else:
        checks.append(_status("source_instance_group_assignment", None,
                              "manifest has no source_instances inventory"))
    near_dupes = manifest.get("near_duplicate_audit")
    if isinstance(near_dupes, Mapping):
        pair_conflicts = near_dupes.get("cross_split_pairs", [])
        near_complete = near_dupes.get("status") == "complete"
        checks.append(_status("near_duplicate_topology_audit",
                              near_complete and not pair_conflicts,
                              f"status={near_dupes.get('status')}; cross_split_pairs={len(pair_conflicts)}"))
    else:
        checks.append(_status("near_duplicate_topology_audit", None,
                              "no near_duplicate_audit evidence was recorded"))
    source_provenance = manifest.get("source_provenance")
    if isinstance(source_provenance, Mapping) and source_provenance.get("datasets"):
        rows = source_provenance["datasets"]
        malformed = [row for row in rows if not row.get("source") or not row.get("url") or
                     not row.get("revision") or not row.get("native_objective")]
        checks.append(_status("source_provenance", not malformed,
                              f"datasets={len(rows)}; missing required source/revision/objective={len(malformed)}"))
    else:
        checks.append(_status("source_provenance", None,
                              "manifest has no source URL/revision/native-objective inventory"))
    label_contract = manifest.get("label_contract")
    if isinstance(label_contract, Mapping):
        label_ok = (label_contract.get("synthetic") is True and
                    label_contract.get("observed_incident_data") is False and
                    bool(label_contract.get("target")) and bool(label_contract.get("uncertainty_model")) and
                    bool(label_contract.get("scenario_seed_policy")))
        checks.append(_status("synthetic_label_contract", label_ok,
                              "synthetic target, uncertainty mechanism, and seed policy are documented"))
    else:
        checks.append(_status("synthetic_label_contract", None, "label_contract is missing"))
    train_hash = manifest.get("split_hashes", {}).get("train")
    checks.append(_status("severity_uses_training_split",
                          manifest.get("severity_training_split_sha256") == train_hash and bool(train_hash),
                          "severity provenance must equal the frozen train split hash"))
    test_input_flag = manifest.get("test_labels_not_model_inputs")
    checks.append(_status("test_outcomes_excluded_from_model_inputs",
                          test_input_flag is True,
                          "manifest must declare test_labels_not_model_inputs=true"))
    test_metrics_path = root / "test_metrics.json"
    if test_metrics_path.is_file():
        test_metrics = _load_json(test_metrics_path)
        test_samples_path = root / "route_samples_test_sealed_after_eval.json"
        test_samples = _load_json(test_samples_path) if test_samples_path.is_file() else []
        observed_buckets = {f"{row.get('source')}:n{row.get('metadata', {}).get('dataset_size')}"
                            for row in test_samples if isinstance(row, Mapping)}
        expected_buckets = {f"{source}:n{size}" for source in ("solomon", "dabia2013")
                            for size in (25, 50, 100)}
        exclusions = [row for row in test_metrics.get("pool_exclusions", [])
                      if row.get("split") == "test" or row.get("stage") == "sealed_test_evaluation"]
        coverage_ok = (observed_buckets == expected_buckets and not exclusions and
                       all(isinstance(row, Mapping) and row.get("label") in (0, 1)
                           for row in test_samples))
        checks.append(_status(
            "heldout_source_size_coverage", coverage_ok,
            f"sample_rows={len(test_samples)}, observed={sorted(observed_buckets)}, "
            f"expected={sorted(expected_buckets)}, test_exclusions={len(exclusions)}",
        ))
    else:
        checks.append(_status("heldout_source_size_coverage", None,
                              "test_metrics.json is missing; cannot verify both sources and 25/50/100 buckets"))

    model_paths = manifest.get("model_paths", {
        "vqc_predictive": "models/vqc_predictive.json",
        "vqc_decision": "models/vqc_decision.json",
        "logistic": "models/logistic.json",
        "newton_gbdt": "models/newton_gbdt.json",
    })
    loaded_models = {}
    train_groups = {key for key, value in _split_map(manifest).items() if value == "train"}
    for name, relative in model_paths.items():
        if str(name) not in {"vqc_predictive", "vqc_decision", "logistic", "newton_gbdt"}:
            continue
        path = root / str(relative)
        if not path.is_file():
            checks.append(_status(f"model_reload_{name}", False, f"missing: {relative}"))
            continue
        try:
            if str(name).startswith("vqc"):
                model, artifact = load_vqc_artifact(path)
                scaler = model.scaler
                expected = artifact.get("model_sha256", artifact.get("model", {}).get("model_sha256"))
                hash_ok = expected == model.sha256
            else:
                artifact = _load_json(path)
                model, scaler, expected = _load_classical_artifact(str(name), artifact)
                hash_ok = expected == artifact.get("model_sha256")
            loaded_models[str(name)] = model
            scaler_groups = set(scaler.fit_group_ids) if scaler else set()
            scaler_ok = scaler is not None and scaler_groups == train_groups
            model_digest = getattr(model, "sha256", expected)
            checks.append(_status(f"model_reload_{name}", True,
                                  f"reloaded {model.model_version}; model_sha256={model_digest}"))
            checks.append(_status(f"scaler_train_only_{name}", scaler_ok,
                                  f"scaler groups={len(scaler_groups)}, train groups={len(train_groups)}"))
            if str(name).startswith("vqc"):
                hash_ok = expected == model.sha256
            checks.append(_status(f"model_hash_{name}", hash_ok,
                                  "serialized model hash matches reloaded model" if hash_ok else
                                  f"serialized={artifact.get('model_sha256')}; reloaded={model_digest}"))
        except Exception as exc:
            checks.append(_status(f"model_reload_{name}", False, f"{type(exc).__name__}: {exc}"))

    test_samples_path = root / "route_samples_test_sealed_after_eval.json"
    if test_metrics_path.is_file() and test_samples_path.is_file() and loaded_models:
        try:
            test_metrics = _load_json(test_metrics_path)
            test_samples = _load_json(test_samples_path)
            metric_errors = []
            checked_buckets = set()
            for name, model in loaded_models.items():
                if name not in {"vqc_predictive", "vqc_decision", "logistic", "newton_gbdt"}:
                    continue
                model_metrics = test_metrics.get("metrics", {}).get(name)
                if not isinstance(model_metrics, Mapping):
                    metric_errors.append(f"missing stored test metrics for {name}")
                    continue
                labels = [int(row["label"]) for row in test_samples]
                probabilities = [float(model.predict_proba(row["raw_features"])) for row in test_samples]
                recomputed = _binary_metrics(labels, probabilities)
                if not _metrics_match(model_metrics.get("overall", {}), recomputed):
                    metric_errors.append(f"{name}: overall metrics mismatch")
                grouped: dict[str, list[int]] = {}
                for index, row in enumerate(test_samples):
                    key = f"source_size:{row['source']}:n{int(row['metadata']['dataset_size'])}"
                    grouped.setdefault(key, []).append(index)
                recorded_buckets = model_metrics.get("by_source_and_family", {})
                for key, indices in grouped.items():
                    checked_buckets.add(key.removeprefix("source_size:"))
                    actual = _binary_metrics([labels[index] for index in indices],
                                             [probabilities[index] for index in indices])
                    if not _metrics_match(recorded_buckets.get(key, {}), actual):
                        metric_errors.append(f"{name}: {key} metrics mismatch")
            metrics_ok = bool(loaded_models) and not metric_errors and len(checked_buckets) == 6
            checks.append(_status("test_metrics_recomputed_from_reloaded_models", metrics_ok,
                                  f"recomputed model/bucket metrics; buckets={sorted(checked_buckets)}; "
                                  f"errors={metric_errors[:8]}"))
        except Exception as exc:
            checks.append(_status("test_metrics_recomputed_from_reloaded_models", False,
                                  f"{type(exc).__name__}: {exc}"))
    else:
        checks.append(_status("test_metrics_recomputed_from_reloaded_models", None,
                              "sealed test route outcomes or reloadable VQC models are missing"))

    integrity_path = root / "integrity.json"
    if integrity_path.is_file():
        integrity = _load_json(integrity_path)
        gradients = integrity.get("gradient_checks", [])
        if isinstance(gradients, Mapping):
            gradients = list(gradients.values())
        if gradients:
            bad = [row for row in gradients if not row.get("nonzero_gradient_path", True) or
                   float(row.get("max_absolute_error", math.inf)) > float(row.get("tolerance", 2e-6))]
            checks.append(_status("gradient_checks", not bad,
                                  f"{len(gradients)} checks; failures={len(bad)}"))
        else:
            checks.append(_status("gradient_checks", None, "integrity.json contains no gradient_checks"))
        decision_component = integrity.get("decision_component_gradient")
        if isinstance(decision_component, Mapping):
            try:
                decision_norm = float(decision_component.get("gradient_l2", math.nan))
                decision_weight = float(decision_component.get("decision_weight", 0.0))
                decision_ok = (math.isfinite(decision_norm) and decision_norm > 1e-10 and
                               decision_weight == 1.0 and
                               decision_component.get("split") == "train")
            except (TypeError, ValueError):
                decision_norm, decision_ok = math.nan, False
            checks.append(_status(
                "decision_component_gradient",
                decision_ok,
                f"train-only decision-only gradient L2={decision_norm}; required >1e-10, weight=1.0",
            ))
        else:
            checks.append(_status(
                "decision_component_gradient", None,
                "integrity.json lacks an isolated train-only decision objective gradient norm",
            ))
        sensitivities = integrity.get("feature_sensitivity", [])
        if sensitivities:
            if isinstance(sensitivities, Mapping):
                flat = [float(value) for value in sensitivities.values()
                        if isinstance(value, (int, float))]
                flat.extend(float(value) for row in sensitivities.values()
                            if isinstance(row, Mapping)
                            for value in row.get("absolute_delta", []))
            else:
                flat = [float(value) for row in sensitivities for value in row.get("absolute_delta", [])]
            checks.append(_status("feature_sensitivity", bool(flat) and any(value > 0 for value in flat),
                                  f"tested {len(flat)} feature perturbations"))
        else:
            checks.append(_status("feature_sensitivity", None, "integrity.json contains no feature_sensitivity"))
        parity = integrity.get("qpiai_parity", {})
        if parity:
            checks.append(_status("qpiai_local_parity",
                                  float(parity.get("max_absolute_error", math.inf)) <= float(parity.get("tolerance", 0.0)),
                                  f"execution={parity.get('execution_type')}; rows={parity.get('row_count')}"))
            decision_model = loaded_models.get("vqc_decision")
            parity_model_hash = parity.get("model_sha256", integrity.get("vqc_decision_reloaded_sha256"))
            if decision_model is None or parity_model_hash is None:
                selected_parity_ok = None
                selected_parity_detail = "no parity hash for the reloaded decision model; baseline parity does not cover selected model"
            else:
                selected_parity_ok = parity_model_hash == decision_model.sha256
                selected_parity_detail = (f"parity model SHA matches decision model={selected_parity_ok}; "
                                          f"variant={decision_model.circuit_variant}")
            checks.append(_status("selected_decision_model_sdk_parity", selected_parity_ok,
                                  selected_parity_detail))
        else:
            checks.append(_status("qpiai_local_parity", None, "SDK circuit parity was not recorded"))
            checks.append(_status("selected_decision_model_sdk_parity", None,
                                  "SDK circuit parity was not recorded"))
    else:
        checks.extend((_status("gradient_checks", None, "integrity.json is missing"),
                       _status("feature_sensitivity", None, "integrity.json is missing"),
                       _status("qpiai_local_parity", None, "integrity.json is missing"),
                       _status("selected_decision_model_sdk_parity", None, "integrity.json is missing")))

    dispatch_path = root / "dispatch.json"
    if not dispatch_path.is_file():
        dispatch_path = root / "evaluations" / "dispatch.json"
    if dispatch_path.is_file():
        dispatch = _load_json(dispatch_path)
        dispatch_rows = dispatch if isinstance(dispatch, list) else [dispatch]
        expected_dispatches = (run_config.get("dispatch", {}).get("evaluation_cases")
                               if isinstance(run_config, Mapping) else None)
        usable = [row for row in dispatch_rows if isinstance(row, Mapping) and row.get("status") in {
            "validated_qaoa", "validated_fallback"}]
        native_failures = [row for row in usable if row.get("selected_plan", {}).get(
            "native_validation", {}).get("valid") is not True]
        bound_failures = []
        currency_failures = []
        for index, row in enumerate(usable):
            variables = row.get("artifact", {}).get("embedded_contract", {}).get(
                "qubo_and_constraints", {}).get("n_variables")
            if not isinstance(variables, int) or not 0 < variables <= 8:
                bound_failures.append((index, variables))
            adapter = row.get("artifact", {}).get("compatibility_mapping", {})
            objective_unit = adapter.get("run1_objective_unit")
            if objective_unit == "SCU" and not (
                    adapter.get("financial_currency_claim") is False and
                    adapter.get("currency_conversion_performed") is False):
                currency_failures.append((index, objective_unit, adapter.get("financial_currency_claim")))
        checks.append(_status("full_source_native_dispatch",
                              bool(usable) and not native_failures and len(usable) == len(dispatch_rows),
                              f"dispatches={len(dispatch_rows)}, usable={len(usable)}, native_invalid={len(native_failures)}"))
        checks.append(_status("quantum_variable_bound", bool(usable) and not bound_failures,
                              f"dispatches={len(usable)}, violations={bound_failures}"))
        checks.append(_status("objective_unit_and_currency_boundary",
                              bool(usable) and not currency_failures,
                              f"SCU currency-boundary violations={currency_failures}"))
        if expected_dispatches is not None:
            count_ok = len(dispatch_rows) == int(expected_dispatches)
            checks.append(_status("dispatch_case_count", count_ok,
                                  f"dispatch rows={len(dispatch_rows)}, configured={expected_dispatches}"))
        if isinstance(run_config, Mapping) and run_config.get("protocol_revision") == "RUN1-v2-safe-capped-all-source":
            native_errors = []
            native_source_size = set()
            try:
                from experiments_run1.native import load_instance
                repo_root = Path(__file__).resolve().parents[1]
                data_root = Path(str(manifest.get("dataset_root", repo_root / ".runtime" / "datasets")))
                for index, row in enumerate(dispatch_rows):
                    scenario = str(row.get("scenario_id", ""))
                    parts = scenario.split(":")
                    if len(parts) < 4 or not parts[2].startswith("n"):
                        native_errors.append(f"dispatch[{index}] has no parseable source/instance/size scenario id")
                        continue
                    source, instance_id, size = parts[0], parts[1], int(parts[2][1:])
                    native_source_size.add(f"{source}:n{size}")
                    route_map = {str(item.get("id")): item.get("route_payload")
                                 for item in row.get("candidate_routes_and_expected_costs", [])}
                    route_map.update({str(item.get("id")): item.get("route_payload")
                                      for item in row.get("fixed_routes", [])})
                    plan = row.get("selected_plan", {})
                    ids = list(plan.get("selected_route_ids", []))
                    fixed_ids = list(plan.get("fixed_route_ids", []))
                    if not set(fixed_ids).issubset(set(ids)):
                        ids.extend(fixed_ids)
                    if not ids or any(route_id not in route_map or not route_map[route_id] for route_id in ids):
                        native_errors.append(f"dispatch[{index}] selected route payloads are incomplete")
                        continue
                    routes = [route_map[route_id] for route_id in ids]
                    instance = load_instance(source, instance_id, size, dataset_root=data_root)
                    evaluated = instance.evaluate_solution(routes)
                    if not bool(getattr(evaluated, "valid", False)):
                        native_errors.append(f"dispatch[{index}] canonical recheck failed: {getattr(evaluated, 'status', None)}")
                    if set(int(node) for route in routes for node in route) != set(instance.customers):
                        native_errors.append(f"dispatch[{index}] routes do not exactly cover native source customers")
                    if row.get("selected_plan", {}).get("native_validation", {}).get("valid") is not True:
                        native_errors.append(f"dispatch[{index}] serialized native validity flag is false")
                expected_cells = {f"{source}:n{size}" for source in ("solomon", "dabia2013")
                                  for size in (25, 50, 100)}
                if native_source_size != expected_cells:
                    native_errors.append(f"dispatch source/size cells={sorted(native_source_size)}")
                checks.append(_status("dispatch_native_recomputed_from_source", not native_errors,
                                      f"rechecked={len(dispatch_rows)}; cells={sorted(native_source_size)}; "
                                      f"errors={native_errors[:8]}"))
            except Exception as exc:
                checks.append(_status("dispatch_native_recomputed_from_source", False,
                                      f"{type(exc).__name__}: {exc}"))
    else:
        checks.append(_status("full_source_native_dispatch", None, "dispatch.json is missing"))

    failures = [row for row in checks if row["status"] == "FAIL"]
    unrun = [row for row in checks if row["status"] == "NOT RUN"]
    overall = "FAIL" if failures else "PARTIAL" if unrun else "PASS"
    strict_gate_names = {"original_untouched_holdout"}
    technical_failures = [row for row in failures if row["check"] not in strict_gate_names]
    technical_unrun = [row for row in unrun if row["check"] not in strict_gate_names]
    technical_status = ("FAIL" if technical_failures else "PARTIAL" if technical_unrun else "PASS")
    return {"schema_version": "run1.acceptance_audit.v1", "status": overall,
            "technical_status": technical_status,
            "run_dir": str(root), "checks": checks,
            "errors": [row["detail"] for row in failures],
            "open_checks": [row["check"] for row in unrun]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_run_directory(args.run_dir), indent=2, sort_keys=True))
