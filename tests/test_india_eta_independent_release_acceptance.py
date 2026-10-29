from __future__ import annotations

from pathlib import Path

import pytest

from experiments_india_eta.independent_release_acceptance import audit_classical_release


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts" / "india_eta_classical_release_v1"


def test_classical_release_directory_and_zip_pass_independent_portability_audit() -> None:
    if not RELEASE.is_dir():
        pytest.skip("classical release bundle is not present")
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    result = audit_classical_release(
        RELEASE,
        ROOT / "artifacts" / "india_eta_classical_release_v1.zip",
        source_schema_path=ROOT / "data" / "feature_schema.json",
        source_checkpoint_path=ROOT / "artifacts" / "india_eta_recovery" / "gbdt" / "india_eta_hgb_regressor.joblib",
    )
    assert result["status"] == "PASS"
    assert result["model_matches_frozen_checkpoint"] is True
    assert result["zip"]["sha256"] == "f110fd17a52b742e92c3320ab6b88abc6cd1efa4ac463212eb8a40da54efdfee"
    assert result["feature_contract"]["ordered_schema_matches_source"] is True
    assert result["portable_loader_reference_max_abs_error_min"] == 0.0
    assert result["independent_joblib_reference_max_abs_error_min"] == 0.0
    assert result["tamper_fail_closed"]["manifest_path_traversal_variants_rejected"] == 3
    assert result["tamper_fail_closed"]["tampered_model_rejected_before_deserialization"] is True
    assert result["isolated_zip_extraction"]["loader_and_reference_passed"] is True
    assert result["real_dataset_rows_read"] is False
    assert result["sealed_test_opened"] is False
