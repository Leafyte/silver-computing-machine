from __future__ import annotations

from pathlib import Path

import pytest

from experiments_india_eta.independent_residual_control_acceptance import audit_residual_control


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "artifacts" / "india_eta_residual_control_v1"


def test_residual_control_recomputes_from_oof_and_selection_only() -> None:
    if not CONTROL.is_dir():
        pytest.skip("classical residual control artifacts are not present")
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    pytest.importorskip("fastparquet")
    result = audit_residual_control(ROOT)
    assert result["status"] == "PASS"
    assert result["test_targets_read"] is False
    assert result["training_rows"] == 29_592
    assert result["selection_rows"] == 4_124
    assert result["selection_couriers"] == 117
    assert result["calibration_rows"] == 1_865
    assert result["calibration_couriers"] == 58
    assert result["selected_alpha_by_independent_selection_only_recompute"] == 0.01
    assert result["saved_validation_prediction_max_abs_error_min"] == 0.0
    assert result["saved_validation_correction_max_abs_error_min"] == 0.0
    assert result["status_is_experimental_not_activated"] is True
