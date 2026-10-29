"""Write the immutable model/policy/source freeze used by the one-shot scorer."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "india_eta_vqc4_run2"
OUT = RUN / "PRETEST_FREEZE.json"
RELEASE = ROOT / "artifacts" / "run2_india" / "india_eta_vqc4_cpu_release_v1"
RELEASE_ZIP = ROOT / "artifacts" / "run2_india" / "india_eta_vqc4_cpu_release_v1.zip"
PROTOCOL = ROOT / "experiments_india_vqc4" / "retrospective_test_protocol.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists() or (RUN / "TEST-EVALUATION-STARTED-vqc4.json").exists():
        raise FileExistsError("pretest freeze already exists or test scoring has already started")
    acceptance_path = RUN / "independent_frozen_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    training = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    runtime_status = json.loads((RUN / "runtime_status.json").read_text(encoding="utf-8"))
    resource = json.loads((ROOT / "artifacts" / "india_eta_vqc4_run2_repro_metrics" / "reproduction_metrics.json").read_text(encoding="utf-8"))
    package_manifest = json.loads((RELEASE / "release_manifest.json").read_text(encoding="utf-8"))
    policy = json.loads((RELEASE / "fusion_policy.json").read_text(encoding="utf-8"))
    if acceptance.get("status") != "PASS_DEVELOPMENT_CHECKS_TEST_TARGETS_NOT_ACCESSED_FEATURE_ACCESS_INCIDENT":
        raise RuntimeError("independent development acceptance did not pass or incident disclosure is missing")
    if acceptance.get("test_target_columns_requested_or_read_by_this_audit") is not False:
        raise RuntimeError("test target data was read by the independent development acceptance")
    if not acceptance.get("test_partition_feature_materialization_incident", {}).get("occurred"):
        raise RuntimeError("test feature materialization incident must be disclosed")
    if training.get("status") != "frozen_models_selection_only_no_test_read" or training.get("selected_alpha") != 0.0:
        raise RuntimeError("models and alpha=0 operational policy are not frozen")
    if acceptance.get("run_manifest_sha256") != sha256(RUN / "run_manifest.json"):
        raise RuntimeError("independent acceptance is not bound to the current training manifest")
    if acceptance.get("selection_choice") != "alpha0_gbdt_only" or acceptance.get("selected_alpha") != 0.0:
        raise RuntimeError("development-only acceptance did not independently reproduce alpha=0")
    if policy.get("selected_variant") is not None or policy.get("alpha") != 0.0:
        raise RuntimeError("portable fusion policy differs from the accepted alpha=0 fallback")
    if package_manifest.get("test_status") != "retrospective_exposed_partition":
        raise RuntimeError("portable release does not disclose prior test exposure")
    if resource.get("status") != "both_frozen_checkpoints_reproduced_bitwise" or resource.get("test_partition_read") is not False:
        raise RuntimeError("resource reproduction did not exactly reproduce frozen weights without test access")
    if runtime_status.get("original_training", {}).get("test_partition_read_by_this_training_process") is not False:
        raise RuntimeError("original training provenance does not attest test isolation")
    if protocol.get("interpretation", {}).get("may_claim_pristine_holdout") is not False:
        raise RuntimeError("retrospective test protocol permits an invalid pristine claim")

    paths = {
        "run_manifest": RUN / "run_manifest.json",
        "independent_acceptance": acceptance_path,
        "protocol": PROTOCOL,
        "circuit_code": ROOT / "experiments_india_vqc4" / "circuit.py",
        "trainer_code": ROOT / "experiments_india_vqc4" / "train.py",
        "scorer_code": ROOT / "experiments_india_vqc4" / "evaluate_retrospective_test.py",
        "portable_code": ROOT / "experiments_india_vqc4" / "portable.py",
        "package_builder_code": ROOT / "experiments_india_vqc4" / "package_release.py",
        "runtime_status": RUN / "runtime_status.json",
        "reproduction_metrics": ROOT / "artifacts" / "india_eta_vqc4_run2_repro_metrics" / "reproduction_metrics.json",
        "baseline60_checkpoint": RUN / "baseline60_best.json",
        "extended76_checkpoint": RUN / "extended76_best.json",
        "release_manifest": RELEASE / "release_manifest.json",
        "release_archive": RELEASE_ZIP,
    }
    expected_checkpoints = {
        "baseline60": training["stages"]["baseline60"]["checkpoint_sha256"],
        "extended76": training["stages"]["extended76"]["checkpoint_sha256"],
    }
    if expected_checkpoints != acceptance["checkpoint_hashes"] or expected_checkpoints != resource["reproduced_checkpoint_sha256"]:
        raise RuntimeError("the independently checked/reproduced model weights do not match the training manifest")
    freeze = {
        "format": "india_eta_vqc4_pretest_freeze_v1",
        "status": "FROZEN_BEFORE_RETROSPECTIVE_TEST_SCORE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": "40bd7fad0166b35b40bbc116e997cbac474e2c22",
        "hashes": {name: sha256(path) for name, path in paths.items()},
        "model_checkpoints": expected_checkpoints,
        "fusion_policy": {"selected_variant": None, "alpha": 0.0,
                           "policy_sha256": sha256(RELEASE / "fusion_policy.json"),
                           "choice": "GBDT only; chosen on validation model-selection couriers, not calibration or test"},
        "resource_reproduction_sha256": sha256(ROOT / "artifacts" / "india_eta_vqc4_run2_repro_metrics" / "reproduction_metrics.json"),
        "portable_release": {"zip_sha256": sha256(RELEASE_ZIP),
                             "manifest_sha256": sha256(RELEASE / "release_manifest.json"),
                             "contains_raw_data_or_oof": False},
        "evaluation_contract": {
            "expected_test_rows": 5941,
            "expected_courier_groups": 176,
            "single_scoring_attempt": True,
            "methods": ["gbdt_only", "ridge_control", "baseline60_candidate", "extended76_candidate", "operational_alpha0"],
            "metrics": ["MAE_minutes", "RMSE_minutes", "median_absolute_error_minutes"],
            "selection_after_test": False,
            "nonnegative_eta_clip": "max(0, GBDT + correction)",
        },
        "test_qualification": {
            "pristine_or_unseen": False,
            "reason": "The same original test partition was opened/scored in a prior 2Q RUN2 workflow; a later 4Q source-feature filter failure also materialized feature columns for all 5,941 test rows, without target or synthetic labels.",
            "new_4q_test_target_access_before_this_freeze": False,
            "no_test_values_used_for_model_or_policy_selection": True,
            "required_interpretation": "retrospective descriptive evaluation only; never claim an untouched holdout",
        },
    }
    OUT.write_text(json.dumps(freeze, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": freeze["status"], "pretest_freeze": str(OUT),
                      "sha256": sha256(OUT), "test_rows_opened": False,
                      "alpha": freeze["fusion_policy"]["alpha"]}, indent=2))


if __name__ == "__main__":
    main()
