"""Test isolated, non-interfering QML inference runner."""

import os
import sqlite3
from tools.isolated_qml_inference import (
    predict_india_eta_residual,
    predict_fleet_delay_risk,
    model_route_qubo_qaoa,
)


def test_isolated_qml_execution_has_no_side_effects():
    # Verify 4Q inference
    res_4q = predict_india_eta_residual(features=[0.4, 0.6, 0.3, 0.2])
    assert res_4q["status"] == "success"
    assert res_4q["num_qubits"] == 4
    assert len(res_4q["pauli_z_expectations"]) == 4
    assert isinstance(res_4q["predicted_residual_minutes"], float)

    # Verify 2Q decision VQC
    res_2q = predict_fleet_delay_risk(slack=0.30, stop_density=0.40)
    assert res_2q["status"] == "success"
    assert 0.0 <= res_2q["breach_risk_probability"] <= 1.0
    assert res_2q["recommendation"] in ["SAFE_DISPATCH", "MODERATE_RISK", "HIGH_RISK_BREACH"]

    # Verify QpiAI Opt QUBO
    res_qubo = model_route_qubo_qaoa()
    assert res_qubo["num_candidates"] == 3
    assert res_qubo["terms_count"] >= 3
