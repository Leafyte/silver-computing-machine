from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments_india_qml.independent_acceptance import (
    SOURCE_CSV_SHA256,
    audit_classical_recovery,
    keyed_join,
    regression_metrics,
    validate_feature_names,
    validate_oof_rows,
    validate_validation_roles,
    validation_roles_for_couriers,
    verify_committed_csv_source,
    verify_manifest_hashes,
)


ROOT = Path(__file__).resolve().parents[1]


def test_source_csv_committed_hash_accounts_for_crlf_checkout() -> None:
    result = verify_committed_csv_source(ROOT)
    assert result["git_blob_sha256"] == SOURCE_CSV_SHA256
    assert result["canonical_lf_sha256"] == SOURCE_CSV_SHA256
    assert result["worktree_clean"] is True


def test_oof_contract_enforces_minute_residual_and_courier_isolation() -> None:
    rows = [
        {"delivery_record_id": "a", "courier_id": "c1", "split_partition": "train",
         "fold": 0, "y_true_min": 12.0, "gbdt_pred_min": 10.5, "residual_true_min": 1.5},
        {"delivery_record_id": "b", "courier_id": "c2", "split_partition": "train",
         "fold": 1, "y_true_min": 20.0, "gbdt_pred_min": 21.0, "residual_true_min": -1.0},
    ]
    result = validate_oof_rows(rows, expected_train_count=2, fold_train_couriers={0: {"c2"}, 1: {"c1"}})
    assert result["passed"] is True
    with pytest.raises(ValueError, match="trained on its courier"):
        validate_oof_rows(rows, expected_train_count=2, fold_train_couriers={0: {"c1"}, 1: {"c1"}})
    corrupt = [dict(rows[0], residual_true_min=2.0), rows[1]]
    with pytest.raises(ValueError, match="residual arithmetic"):
        validate_oof_rows(corrupt, expected_train_count=2)


def test_validation_role_split_is_frozen_and_courier_disjoint() -> None:
    couriers = [f"courier-{i}" for i in range(18)]
    role_map = validation_roles_for_couriers(couriers)
    rows = [
        {"delivery_record_id": f"id-{i}", "courier_id": courier,
         "split_partition": "val", "validation_role": role_map[courier]}
        for i, courier in enumerate(couriers)
    ]
    result = validate_validation_roles(rows)
    assert result["couriers"]["model_selection"] == 12
    assert result["couriers"]["fusion_calibration"] == 6
    changed = [dict(row) for row in rows]
    changed[0]["validation_role"] = "fusion_calibration" if changed[0]["validation_role"] == "model_selection" else "model_selection"
    with pytest.raises(ValueError, match="role mismatch"):
        validate_validation_roles(changed)


def test_join_uses_record_identity_not_row_position() -> None:
    left = [
        {"delivery_record_id": "x", "split_partition": "train", "value": 1},
        {"delivery_record_id": "y", "split_partition": "train", "value": 2},
    ]
    right = [
        {"delivery_record_id": "y", "split_partition": "train", "value": 20},
        {"delivery_record_id": "x", "split_partition": "train", "value": 10},
    ]
    joined = keyed_join(left, right)
    assert [(a["value"], b["value"]) for a, b in joined] == [(1, 10), (2, 20)]
    with pytest.raises(ValueError, match="keyed join mismatch"):
        keyed_join(left, right[:1])


def test_feature_schema_rejects_target_metadata_and_order_drift() -> None:
    validate_feature_names(("vqc_x0_distance", "vqc_x1_congestion"), allowed=("vqc_x0_distance", "vqc_x1_congestion"))
    with pytest.raises(ValueError, match="target fields"):
        validate_feature_names(("vqc_x0_distance", "delivery_duration_min"))
    with pytest.raises(ValueError, match="frozen schema"):
        validate_feature_names(("vqc_x1_congestion", "vqc_x0_distance"), allowed=("vqc_x0_distance", "vqc_x1_congestion"))


def test_minute_regression_metrics_and_artifact_hash_loader_contract(tmp_path: Path) -> None:
    stats = regression_metrics([10, 20, 30], [11, 18, 30])
    assert stats == {"n": 3, "mae_min": 1.0, "rmse_min": pytest.approx((5 / 3) ** 0.5),
                     "median_absolute_error_min": 1.0, "mean_error_min": -1 / 3}
    artifact = tmp_path / "model.json"
    artifact.write_text(json.dumps({"frozen": True}), encoding="utf-8")
    import hashlib
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert verify_manifest_hashes(tmp_path, {"artifacts": {"model.json": {"sha256": digest}}}) == {"model.json": digest}
    with pytest.raises(ValueError, match="escapes release root"):
        verify_manifest_hashes(tmp_path, {"artifacts": {"..\\outside": {"sha256": digest}}})


def test_frozen_classical_recovery_passes_independent_row_level_audit() -> None:
    if not (ROOT / "artifacts" / "india_eta_recovery" / "recovery_manifest.json").is_file():
        pytest.skip("authorized GBDT recovery artifacts are not present")
    pytest.importorskip("pandas")
    pytest.importorskip("fastparquet")
    pytest.importorskip("sklearn")
    result = audit_classical_recovery(ROOT)
    assert result["status"] == "PASS"
    assert result["test_targets_read"] is False
    assert result["oof"]["rows"] == 29_592
    assert result["validation_role_assignment"]["couriers"] == {
        "model_selection": 117, "fusion_calibration": 58
    }
    assert result["reloaded_model_validation_prediction_max_abs_error_minutes"] == 0.0

