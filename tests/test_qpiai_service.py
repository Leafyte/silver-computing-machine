"""Tests for QpiAIService unified integration."""

import os
from backend.qpiai_service import QpiAIService


def test_qpiai_service_local_vqc():
    service = QpiAIService()
    res = service.predict_route_risk(
        features=[0.4, 0.2],
        theta=[0.31, -0.47],
        shots=100,
        use_cloud=False,
    )
    assert res["status"] == "success"
    assert res["execution_backend"] == "QpiAI-QSV-Local"
    assert "breach_risk_probability" in res
    assert 0.0 <= res["breach_risk_probability"] <= 1.0
    assert res["dispatch_recommendation"] in ["SAFE_DISPATCH", "MODERATE_RISK", "HIGH_RISK_BREACH"]
    assert len(res["state_probabilities"]) == 4


def test_qpiai_service_routing_qubo():
    service = QpiAIService()
    costs = [10.0, 15.0, 8.0]
    overlap = {(0, 1): 12.0, (1, 2): 10.0}
    relaxation = [0.5, 0.3, 0.8]

    res = service.build_routing_qubo(
        candidate_costs=costs,
        overlap_matrix=overlap,
        relaxed_solution=relaxation,
    )
    assert res["num_candidates"] == 3
    assert res["terms_count"] >= 3
    assert res["warm_start_circuit"] is not None
    assert res["warm_start_circuit"]["num_qubits"] == 3


def test_qpiai_service_diagnostics():
    service = QpiAIService()
    diag = service.check_connection()
    assert "quantum" in diag
    assert "opt" in diag
    assert diag["quantum"]["configured"] is True
    assert diag["opt"]["configured"] is True
