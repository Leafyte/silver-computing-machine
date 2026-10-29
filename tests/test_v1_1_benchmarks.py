"""Unit and regression tests for V1.1 Quantum Optimization, Decision Quality, and Contract modules."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.domain import generate_routes
from backend.qaoa import qaoa_state
from backend.qubo import build_route_qubo
from experiments_v1_1.artifact_contract import (
    CONTRACT_SCHEMA_VERSION,
    HybridDispatchArtifact,
    validate_hybrid_artifact,
)
from experiments_v1_1.decision_quality import (
    NewtonGBDT,
    build_experiment_records,
    decision_loss_and_gradient,
    estimate_train_conditional_severity,
    evaluate_plan_financial,
    risk_probability,
    run_comparator_benchmarks,
    select_plan_exact,
    train_decision_focused_vqc,
)
from experiments_v1_1.mathematical_verification import (
    run_independent_mathematical_audit,
    verify_qubo_isomorphism,
)
from experiments_v1_1.quantum_optimization import (
    execute_qaoa_benchmark,
    fast_qaoa_state,
    get_circuit_profile,
    lp_augmented_relaxation,
    precompute_hamiltonian_spectrum,
    solve_classical_exact,
    solve_classical_heuristic_sampler,
)


@pytest.fixture
def tiny_qubo():
    candidates = [
        {"id": "r0", "vehicle": "A", "stops": ["a"], "cost": 2.0},
        {"id": "r1", "vehicle": "A", "stops": ["a", "b"], "cost": 3.0},
        {"id": "r2", "vehicle": "B", "stops": ["b"], "cost": 2.0},
        {"id": "r3", "vehicle": "C", "stops": ["a", "b"], "cost": 6.0},
    ]
    return build_route_qubo(candidates, ["a", "b"], coverage_penalty=20.0, vehicle_penalty=15.0)


@pytest.fixture
def dispatch_qubo():
    fix_path = Path(__file__).resolve().parents[1] / "experiments_qaoa" / "dispatch_fixture.json"
    scenario = json.loads(fix_path.read_text(encoding="utf-8"))
    routes = generate_routes(scenario)
    candidates = [
        {
            "id": f"{r['vehicle_id']}:{'>'.join(r['order_ids'])}",
            "vehicle": r["vehicle_id"],
            "stops": r["order_ids"],
            "cost": float(r["travel_cost"]),
        }
        for r in routes
    ]
    return build_route_qubo(candidates, [o["id"] for o in scenario["orders"]])


def test_mathematical_isomorphism_4_8_12_var():
    audit = run_independent_mathematical_audit()
    for key, expected_n in (("tiny_4var", 4), ("dispatch_8var", 8), ("prospective_12var", 12)):
        res = audit[key]
        assert res["n_variables"] == expected_n
        assert res["total_assignments_checked"] == (1 << expected_n)
        assert res["max_qubo_direct_discrepancy"] == pytest.approx(0.0, abs=1e-10)
        assert res["max_ising_direct_discrepancy"] == pytest.approx(0.0, abs=1e-10)
        assert res["ground_state_is_feasible"] is True
        assert res["penalty_margin_gap"] > 0.0

    # Low penalty violates ground-state feasibility; sufficient penalty restores it
    sweep = audit["penalty_threshold_study"]
    assert sweep["penalty_1.5"]["ground_state_is_feasible"] is False
    assert sweep["penalty_29.0"]["ground_state_is_feasible"] is True


def test_fast_qaoa_state_matches_reference_backend(tiny_qubo):
    energies, _ = precompute_hamiltonian_spectrum(tiny_qubo)
    gammas = [0.23, 0.41]
    betas = [0.37, 0.19]
    probs = [0.15, 0.80, 0.25, 0.10]

    ref_std = qaoa_state(tiny_qubo, gammas, betas, probabilities=None)
    fast_std = fast_qaoa_state(tiny_qubo.n, energies, gammas, betas, probabilities=None)
    for a, b in zip(ref_std, fast_std):
        assert abs(a.real**2 + a.imag**2 - (b.real**2 + b.imag**2)) < 1e-11

    ref_warm = qaoa_state(tiny_qubo, gammas, betas, probabilities=probs)
    fast_warm = fast_qaoa_state(tiny_qubo.n, energies, gammas, betas, probabilities=probs)
    for a, b in zip(ref_warm, fast_warm):
        assert abs(a.real**2 + a.imag**2 - (b.real**2 + b.imag**2)) < 1e-11


def test_circuit_profile_and_classical_baselines(dispatch_qubo):
    p1_std = get_circuit_profile(dispatch_qubo, layers=1, warm_start=False)
    p2_warm = get_circuit_profile(dispatch_qubo, layers=2, warm_start=True)
    assert p1_std.qubits == 8
    assert p2_warm.total_gates > p1_std.total_gates

    exact = solve_classical_exact(dispatch_qubo)
    assert exact["feasible"] is True
    assert exact["best_objective"] == pytest.approx(4.0)

    heur = solve_classical_heuristic_sampler(dispatch_qubo, shots=256, seed=7)
    assert heur["shots"] == 256
    assert 0.0 <= heur["feasible_sample_probability"] <= 1.0


def test_lp_augmented_warm_start_beats_v1_box_and_standard_qaoa(dispatch_qubo):
    lp_vec = lp_augmented_relaxation(dispatch_qubo)
    assert len(lp_vec) == 8
    # Optimal routes on dispatch_8var are indices 2, 3, 6, 7 (cost=2.0 each -> plan cost=4.0)
    assert sum(lp_vec[i] for i in (2, 3, 6, 7)) > 0.95

    std_p1 = execute_qaoa_benchmark(
        dispatch_qubo, layers=1, warm_start=False, grid_size=8, shots=256, seed=42, certified_optimum=4.0
    )
    box_p1 = execute_qaoa_benchmark(
        dispatch_qubo, layers=1, warm_start=True, relaxation_mode="box_v1", grid_size=8, shots=256, seed=42, certified_optimum=4.0
    )
    lp_p1 = execute_qaoa_benchmark(
        dispatch_qubo, layers=1, warm_start=True, relaxation_mode="lp_augmented", grid_size=8, shots=256, seed=42, certified_optimum=4.0
    )
    lp_p2 = execute_qaoa_benchmark(
        dispatch_qubo, layers=2, warm_start=True, relaxation_mode="lp_augmented", optimizer_method="pattern", max_iter=40, shots=256, seed=42, certified_optimum=4.0
    )

    assert lp_p1["feasible_sample_probability"] > box_p1["feasible_sample_probability"]
    assert lp_p1["feasible_sample_probability"] > std_p1["feasible_sample_probability"]
    assert lp_p1["optimality_gap"] == pytest.approx(0.0)
    assert lp_p2["feasible_sample_probability"] > lp_p1["feasible_sample_probability"]
    assert lp_p2["approximation_ratio"] == pytest.approx(1.0)


def test_vqc_4param_decision_gradient_matches_finite_differences():
    records = build_experiment_records(n_replicates=1, cases_per_replicate=10, family_seed=101, split_seed=202)
    train_set = [r for r in records if r.split == "train"]
    sev_map = estimate_train_conditional_severity(train_set)
    theta = [0.42, -0.33, -3.1, -1.2]

    _, analytic_grad = decision_loss_and_gradient(
        theta, train_set, temperature=0.25, decision_weight=0.85, severity_map=sev_map
    )
    eps = 1e-5
    for k in range(4):
        tp = list(theta)
        tm = list(theta)
        tp[k] += eps
        tm[k] -= eps
        lp, _ = decision_loss_and_gradient(tp, train_set, temperature=0.25, decision_weight=0.85, severity_map=sev_map)
        lm, _ = decision_loss_and_gradient(tm, train_set, temperature=0.25, decision_weight=0.85, severity_map=sev_map)
        fd = (lp - lm) / (2.0 * eps)
        assert analytic_grad[k] == pytest.approx(fd, abs=1e-6)


def test_newton_gbdt_and_7way_comparator():
    records = build_experiment_records(n_replicates=1, cases_per_replicate=10, family_seed=42, split_seed=43)
    train_set = [r for r in records if r.split == "train"]
    test_set = [r for r in records if r.split == "test"]

    gbdt_a = NewtonGBDT(n_estimators=20, max_depth=3).fit(train_set)
    gbdt_b = NewtonGBDT(n_estimators=20, max_depth=3).fit(train_set)
    for rt in test_set[0].routes:
        pa = gbdt_a.predict_proba(rt.features)
        pb = gbdt_b.predict_proba(rt.features)
        assert pa == pytest.approx(pb, abs=1e-12)
        assert 0.0 < pa < 1.0

    theta_df = train_decision_focused_vqc(train_set, steps=8, lr=0.08, temperature=0.25)
    assert len(theta_df) == 4

    probs = {r.route_id: risk_probability(r.features, theta_df) for r in test_set[0].routes}
    sel = select_plan_exact(test_set[0], probs)
    ev = evaluate_plan_financial(test_set[0].realized, sel["route_ids"])
    assert ev["valid"] is True
    assert ev["hard_violations"] == 0
    assert ev["transport_cost"] > 0.0
    assert ev["labor_cost"] > 0.0

    comp = run_comparator_benchmarks(records, qaoa_shots=64, qaoa_grid_size=4)
    expected_pipes = {
        "A_classical_ml_exact",
        "B_std_vqc_exact",
        "C_df_vqc_exact",
        "D_std_vqc_std_qaoa",
        "E_df_vqc_std_qaoa",
        "F_df_vqc_warm_qaoa",
        "G_classical_ml_warm_qaoa",
    }
    assert set(comp["summary_table"].keys()) == expected_pipes
    for p_metrics in comp["summary_table"].values():
        assert p_metrics["valid_plan_rate"] == 1.0
        assert p_metrics["hard_constraint_violations"] == 0
        assert "raw_solver_success_rate" in p_metrics
        assert "fallback_rate" in p_metrics


def test_hybrid_artifact_contract_validation():
    artifact = HybridDispatchArtifact(
        schema_version=CONTRACT_SCHEMA_VERSION,
        model_and_dataset_version={
            "model_id": "df-vqc-4p",
            "model_version": "1.1.0",
            "dataset_id": "test-ds",
            "dataset_sha256": "ABC123",
            "is_reference_fixture": True,
        },
        split_identifiers={
            "replicate_id": "rep01",
            "train_split_sha256": "T1",
            "validation_split_sha256": "V1",
            "test_split_sha256": "E1",
            "evaluation_split": "test",
        },
        scenario_and_seed_Provenance={
            "scenario_id": "rep01-case00",
            "family_seed": 1,
            "split_seed": 2,
            "qaoa_seed": 7,
        },
        risk_predictions_and_calibration={
            "route_probabilities": {"R1:P": 0.12, "R2:Q>R": 0.44},
            "readout_type": "trainable_pauli_z0_affine_sigmoid",
            "brier_score_validation": 0.048,
            "ece_validation": 0.052,
        },
        candidate_routes_and_coefficients={
            "candidate_route_ids": ["R1:P", "R2:Q>R"],
            "transport_cost_usd": {"R1:P": 2.0, "R2:Q>R": 3.0},
            "labor_cost_usd": {"R1:P": 1.2, "R2:Q>R": 1.8},
            "expected_sla_penalty_usd": {"R1:P": 0.5, "R2:Q>R": 1.1},
            "total_route_coefficients_usd": {"R1:P": 3.7, "R2:Q>R": 5.9},
        },
        qubo_and_constraints={
            "n_variables": 2,
            "variable_order": ["R1:P", "R2:Q>R"],
            "linear": [-10.0, -8.0],
            "quadratic": {"0,1": 0.0},
            "offset": 20.0,
            "coverage_penalty": 10.0,
            "vehicle_penalty": 10.0,
            "required_stops": ["P", "Q", "R"],
        },
        execution_backend_and_config={
            "backend_name": "local_exact_statevector_cpu",
            "execution_mode": "warm_start_lp_augmented_qaoa",
            "is_simulated": True,
            "qpu_authorized": False,
            "layers": 1,
            "shots": 256,
        },
    )
    payload = artifact.to_dict()
    ok = validate_hybrid_artifact(payload)
    assert ok["valid"] is True
    assert ok["errors"] == []

    # Tampering with probability breaks hash and probability bounds
    tampered = json.loads(json.dumps(payload))
    tampered["risk_predictions_and_calibration"]["route_probabilities"]["R1:P"] = 1.5
    bad = validate_hybrid_artifact(tampered)
    assert bad["valid"] is False
    assert any("probability out of [0, 1]" in e for e in bad["errors"])
    assert any("artifact_sha256" in e for e in bad["errors"])
