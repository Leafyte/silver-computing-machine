from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments_run1.acceptance import _binary_metrics, validate_run_directory
from experiments_run1.model import FeatureScaler, VQCModel


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(root: Path) -> None:
    (root / "models").mkdir(parents=True)
    scaler = FeatureScaler((0.0, -1.0), (1.0, 1.0), ("train-a",))
    model = VQCModel((0.31, -0.47, -3.5, -1.5), scaler)
    payload = model.to_dict()
    payload["model_sha256"] = model.sha256
    payload["scaler_sha256"] = scaler.sha256
    (root / "models" / "vqc_decision.json").write_text(json.dumps(payload), encoding="utf-8")
    (root / "integrity.json").write_text(json.dumps({
        "gradient_checks": [{"max_absolute_error": 1e-8, "tolerance": 2e-7,
                             "nonzero_gradient_path": True}],
        "decision_component_gradient": {"gradient_l2": 0.01, "decision_weight": 1.0,
                                         "split": "train"},
        "feature_sensitivity": [{"absolute_delta": [0.01, 0.02]}],
        "qpiai_parity": {"execution_type": "local_simulator", "row_count": 3,
                         "max_absolute_error": 1e-12, "tolerance": 1e-10,
                         "model_sha256": model.sha256},
    }), encoding="utf-8")
    (root / "dispatch.json").write_text(json.dumps({
        "status": "validated_qaoa",
        "selected_plan": {"native_validation": {"valid": True}},
        "artifact": {
            "embedded_contract": {"qubo_and_constraints": {"n_variables": 8}},
            "compatibility_mapping": {"run1_objective_unit": "SCU",
                                      "financial_currency_claim": False,
                                      "currency_conversion_performed": False},
        },
    }), encoding="utf-8")
    samples = []
    for index, (source, size) in enumerate((source, size) for source in ("solomon", "dabia2013")
                                            for size in (25, 50, 100)):
        samples.append({"source": source, "label": index % 2,
                        "raw_features": [0.2 + 0.1 * index, 0.7 - 0.05 * index],
                        "metadata": {"dataset_size": size}})
    (root / "route_samples_test_sealed_after_eval.json").write_text(json.dumps(samples), encoding="utf-8")
    labels = [row["label"] for row in samples]
    probabilities = [model.predict_proba(row["raw_features"]) for row in samples]
    overall = _binary_metrics(labels, probabilities)
    metric_buckets = {}
    for index, row in enumerate(samples):
        key = f"source_size:{row['source']}:n{row['metadata']['dataset_size']}"
        metric_buckets[key] = _binary_metrics([labels[index]], [probabilities[index]])
    (root / "test_metrics.json").write_text(json.dumps({
        "class_balance": {}, "pool_exclusions": [],
        "metrics": {"vqc_decision": {"overall": overall,
                                      "by_source_and_family": metric_buckets}},
    }), encoding="utf-8")
    caps = {f"{source}:{size}:C10{i}": {"route_caps": {f"route-{i}": {"max_fraction": 0.1}}}
            for source in ("solomon", "dabia2013")
            for size in (25, 50, 100) for i in range(1, 10)}
    (root / "test_route_overlay_caps.json").write_text(json.dumps({
        "computed_before_training_and_test_labels": True, "instances": caps}), encoding="utf-8")
    split_hash = hashlib.sha256(b"train-groups").hexdigest()
    files = ["models/vqc_decision.json", "integrity.json", "dispatch.json", "test_metrics.json",
             "test_route_overlay_caps.json", "route_samples_test_sealed_after_eval.json"]
    (root / "run_manifest.json").write_text(json.dumps({
        "schema_version": "run1.run_manifest.v1",
        "run_config": {"evaluation_qualification": "fresh topology first exposure",
                       "protocol_revision": "fixture-run1",
                       "predictive_vqc": {"max_steps": 60},
                       "decision_vqc": {"max_steps": 24},
                       "replay_seeds": list(range(202610290100, 202610290108)),
                       "dispatch": {"evaluation_cases": 1}},
        "artifacts": {relative: _digest(root / relative) for relative in files},
        "split_by_topology": {"train-a": "train", "validation-a": "validation", "test-a": "test"},
        "split_hashes": {"train": split_hash},
        "severity_training_split_sha256": split_hash,
        "test_labels_not_model_inputs": True,
        "source_instances": [
            {"topology_id": "train-a", "instance_id": "A", "split": "train"},
            {"topology_id": "validation-a", "instance_id": "B", "split": "validation"},
            {"topology_id": "test-a", "instance_id": "C", "split": "test"},
        ],
        "near_duplicate_audit": {"status": "complete", "threshold": "exact ordered 25-customer prefix", "cross_split_pairs": []},
        "source_provenance": {"datasets": [
            {"source": "Solomon", "url": "https://example.invalid", "revision": "fixture-sha",
             "native_objective": "EuclideanDistance"},
        ]},
        "label_contract": {"synthetic": True, "observed_incident_data": False,
                           "target": "synthetic breach", "uncertainty_model": "fixture additive delay",
                           "scenario_seed_policy": "fixed fixture seed"},
        "model_paths": {"vqc_decision": "models/vqc_decision.json"},
    }), encoding="utf-8")


def test_independent_run_acceptance_reloads_and_checks_hashes(tmp_path):
    _write_run(tmp_path)
    result = validate_run_directory(tmp_path)
    assert result["status"] == "PARTIAL"
    assert not result["errors"]
    assert not any(row["status"] == "FAIL" for row in result["checks"])
    assert result["technical_status"] == "PARTIAL"


def test_independent_run_acceptance_detects_bad_group_split_and_hash_tamper(tmp_path):
    _write_run(tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_by_topology"]["train-a"] = "heldout_unknown"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with (tmp_path / "integrity.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    result = validate_run_directory(tmp_path)
    statuses = {row["check"]: row["status"] for row in result["checks"]}
    assert result["status"] == "FAIL"
    assert statuses["artifact_hashes"] == "FAIL"
    assert statuses["topology_group_splits"] == "FAIL"


def test_independent_run_acceptance_does_not_call_missing_sdk_evidence_a_pass(tmp_path):
    _write_run(tmp_path)
    integrity_path = tmp_path / "integrity.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity.pop("qpiai_parity")
    integrity_path.write_text(json.dumps(integrity), encoding="utf-8")
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["integrity.json"] = _digest(integrity_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_run_directory(tmp_path)
    statuses = {row["check"]: row["status"] for row in result["checks"]}
    assert result["status"] == "PARTIAL"
    assert statuses["qpiai_local_parity"] == "NOT RUN"


def test_independent_run_acceptance_rejects_near_zero_decision_gradient(tmp_path):
    _write_run(tmp_path)
    integrity_path = tmp_path / "integrity.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity["decision_component_gradient"]["gradient_l2"] = 2.89e-33
    integrity_path.write_text(json.dumps(integrity), encoding="utf-8")
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["integrity.json"] = _digest(integrity_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_run_directory(tmp_path)
    statuses = {row["check"]: row["status"] for row in result["checks"]}
    assert result["status"] == "FAIL"
    assert statuses["decision_component_gradient"] == "FAIL"
