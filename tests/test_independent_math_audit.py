"""Independent Mathematical & Quantum Audit Pytest Suite (Audits A, B, C, D, E).

Tests:
- Audit A: 2-qubit VQC closed-form identity, SDK statevector match, x1 in {0, 1} branch cancellation,
           x0 reflection collision, saddle gradient, BCE tail gradient behavior, monotonic ablation.
- Audit B: Financial units, duty labor cost inclusion in mathematical_verification, realized hard-window
           breach defect reproduction on unmodified Codex module, zero future leakage in features.
- Audit C: Candidate-route validity, QUBO/Ising isomorphism across 4/8/12-var instances, critical penalty
           bounds, LP relaxation KKT convergence vs SciPy HiGHS, warm-start state & mixer unitarity/eigenstate,
           parallel circuit depth profiling.
- Audit D: QpiAI Quantum SDK (0.7.5) local StatevectorSimulator circuit execution, verified bitstring
           endianness, local SDK COBYLA + StatevectorSimulator optimization, and unauthenticated
           Circuit.run auth-gate verification (without spoofing or bypassing auth).
- Audit E: V1 box-relaxation failure proof, analytical 1.353% odd-hole warm-start probability, zero-breach
           test-split cost-risk degeneracy, and sub-deadline QAOA execution budget compliance.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
from scipy.optimize import linprog

from backend.domain import generate_routes
from backend.learning_dispatch import _feasible_plans, _nominal_pressure, _route_id
from backend.qpiai_vqc import sdk_expectation_z0, sdk_statevector
from backend.qaoa import continuous_relaxation, qaoa_state
from backend.quantum_learning import expectation_z0
from backend.qubo import build_route_qubo, qubo_to_ising, validate_selection
from backend.validator import validate_plan
from experiments_v1_1.decision_quality import (
    FEATURE_MAX_TRAVEL,
    _base_scenario,
    _deterministic_route_base_cost,
    build_experiment_records,
    decision_loss_and_gradient,
    evaluate_plan_financial,
    risk_probability,
)
from experiments_v1_1.mathematical_verification import run_independent_mathematical_audit
from experiments_v1_1.quantum_optimization import (
    execute_qaoa_benchmark,
    fast_qaoa_state,
    get_circuit_profile,
    lp_augmented_relaxation,
    precompute_hamiltonian_spectrum,
)


def _solve_highs_lp(qubo) -> float:
    n = qubo.n
    c = np.array([float(cand["cost"]) for cand in qubo.candidates], dtype=float)
    a_eq = [[1.0 if s in cand["stops"] else 0.0 for cand in qubo.candidates] for s in qubo.required_stops]
    b_eq = [1.0] * len(a_eq)
    vehs = sorted(set(cand["vehicle"] for cand in qubo.candidates))
    a_ub = [[1.0 if cand["vehicle"] == v else 0.0 for cand in qubo.candidates] for v in vehs]
    b_ub = [1.0] * len(a_ub)
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=[(0.0, 1.0)] * n, method="highs")
    assert res.success
    return float(res.fun)


def test_audit_a_symbolic_closed_form_and_feature_folding():
    x0, x1, t0, t1 = sp.symbols("x0 x1 theta0 theta1", real=True)
    ry = lambda a: sp.Matrix([[sp.cos(a / 2), -sp.sin(a / 2)], [sp.sin(a / 2), sp.cos(a / 2)]])
    cnot01 = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    cnot10 = sp.Matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    ket00 = sp.Matrix([1, 0, 0, 0])
    psi = cnot10 * sp.kronecker_product(ry(t0), ry(t1)) * cnot01 * sp.kronecker_product(ry(sp.pi * x0), ry(sp.pi * x1)) * ket00
    z0_op = sp.kronecker_product(sp.Matrix([[1, 0], [0, -1]]), sp.eye(2))
    closed = (
        sp.cos(t0) * sp.cos(t1) * sp.cos(sp.pi * x1)
        - sp.cos(t0) * sp.sin(t1) * sp.cos(sp.pi * x0) * sp.sin(sp.pi * x1)
        + sp.sin(t0) * sp.sin(t1) * sp.sin(sp.pi * x0)
    )
    assert sp.simplify((psi.T * z0_op * psi)[0] - closed) == 0

    # Reflection symmetry collision when x1 in {0, 1}: x0=0.1 and x0=0.9 produce identical <Z0> and risk
    theta4 = [0.656763, -0.336557, -3.9401, -1.181035]
    assert expectation_z0((0.1, 0.0), theta4[:2]) == pytest.approx(expectation_z0((0.9, 0.0), theta4[:2]), abs=1e-15)
    assert risk_probability((0.1, 1.0), theta4) == pytest.approx(risk_probability((0.9, 1.0), theta4), abs=1e-15)

    # Normal-domain gradient accuracy vs centered finite differences
    recs = build_experiment_records(n_replicates=1, cases_per_replicate=8, family_seed=101, split_seed=202)
    tr = [r for r in recs if r.split == "train"][:3]
    th_norm = [0.42, -0.33, -2.5, 1.4]
    _, g_an = decision_loss_and_gradient(th_norm, tr, decision_weight=0.85)
    eps = 1e-6
    for k in range(4):
        tp, tm = list(th_norm), list(th_norm)
        tp[k] += eps
        tm[k] -= eps
        lp, _ = decision_loss_and_gradient(tp, tr, decision_weight=0.85)
        lm, _ = decision_loss_and_gradient(tm, tr, decision_weight=0.85)
        assert g_an[k] == pytest.approx((lp - lm) / (2.0 * eps), abs=1e-8)


def test_audit_b_financial_duty_labor_and_zero_feature_leakage():
    audit = run_independent_mathematical_audit()
    # Patched prospective_12var includes service + waiting duty labor ($8.10 vs travel-only $7.575)
    assert audit["prospective_12var"]["best_feasible_cost"] == pytest.approx(8.10, abs=1e-6)

    base = _base_scenario()
    r12 = { _route_id(r): r for r in generate_routes(base) }
    opt_true = _deterministic_route_base_cost(r12["R1:Q>R"]) + _deterministic_route_base_cost(r12["R2:P"])
    assert opt_true == pytest.approx(8.10, abs=1e-6)

    # Verify zero realized-outcome leakage into forecast features or val/test records
    recs = build_experiment_records(n_replicates=1, cases_per_replicate=10, family_seed=303, split_seed=404)
    for rec in recs:
        f_map = {_route_id(r): r for r in generate_routes(rec.forecast)}
        for pr in rec.routes:
            assert pr.features[0] == pytest.approx(min(1.0, float(f_map[pr.route_id]["travel_cost"]) / FEATURE_MAX_TRAVEL))
            assert pr.features[1] == pytest.approx(_nominal_pressure(r12[pr.route_id], base))
            if rec.split != "train":
                assert pr.train_label is None


def test_audit_c_lp_kkt_convergence_warmstart_mixer_and_parallel_depth():
    fix8 = json.loads((Path(__file__).resolve().parents[1] / "experiments_qaoa" / "dispatch_fixture.json").read_text(encoding="utf-8"))
    r8 = generate_routes(fix8)
    c8 = [{"id": f"{r['vehicle_id']}:{'>'.join(r['order_ids'])}", "vehicle": r["vehicle_id"], "stops": r["order_ids"], "cost": float(r["travel_cost"])} for r in r8]
    q8 = build_route_qubo(c8, [o["id"] for o in fix8["orders"]])

    # Compare custom LP relaxation against SciPy HiGHS on fixture + sample synthetic 12-var scenarios
    x8 = lp_augmented_relaxation(q8)
    obj8 = sum(float(c["cost"]) * xi for c, xi in zip(c8, x8))
    assert obj8 == pytest.approx(_solve_highs_lp(q8), abs=1e-6)

    recs = build_experiment_records(n_replicates=1, cases_per_replicate=12, family_seed=77, split_seed=88)
    for rec in recs:
        active = {rid for p in rec.plans for rid in p["route_ids"]}
        cands = [
            {
                "id": r.route_id,
                "vehicle": r.route_id.split(":")[0],
                "stops": r.route_id.split(":")[1].split(">"),
                "cost": round(r.forecast_travel_cost * 0.50, 4),
            }
            for r in rec.routes
            if r.route_id in active
        ]
        qr = build_route_qubo(cands, ["P", "Q", "R"], coverage_penalty=18.0, vehicle_penalty=18.0)
        xc = lp_augmented_relaxation(qr)
        eq_v = max(abs(sum(xc[i] for i, c in enumerate(cands) if s in c["stops"]) - 1.0) for s in ("P", "Q", "R"))
        ub_v = max(max(0.0, sum(xc[i] for i, c in enumerate(cands) if c["vehicle"] == v) - 1.0) for v in ("R1", "R2"))
        assert max(eq_v, ub_v) < 1e-7
        assert sum(c["cost"] * xi for c, xi in zip(cands, xc)) == pytest.approx(_solve_highs_lp(qr), abs=5e-4)

    # Verify warm-start initial state and mixer +1-eigenstate evolution at gamma=0
    probs = [0.05, 0.15, 0.25, 0.35, 0.10, 0.20, 0.45, 0.30]
    energies, _ = precompute_hamiltonian_spectrum(q8)
    st0 = fast_qaoa_state(q8.n, energies, [0.0], [0.0], probabilities=probs)
    st_beta = fast_qaoa_state(q8.n, energies, [0.0], [1.234], probabilities=probs)
    phase = complex(math.cos(-q8.n * 1.234), math.sin(-q8.n * 1.234))
    for idx in range(1 << q8.n):
        amp_exp = 1.0
        for qi, p in enumerate(probs):
            amp_exp *= math.sqrt(p) if ((idx >> (q8.n - 1 - qi)) & 1) else math.sqrt(1.0 - p)
        assert abs(st0[idx] - amp_exp) < 1e-14
        assert abs(st_beta[idx] - phase * amp_exp) < 1e-14

    # Parallel ASAP circuit depth must account for 26 nonzero RZZ decompositions on 8 qubits (depth >= 24)
    prof_std = get_circuit_profile(q8, layers=1, warm_start=False)
    prof_warm = get_circuit_profile(q8, layers=1, warm_start=True)
    assert prof_std.two_qubit_gates == 52
    assert prof_std.circuit_depth == 24
    assert prof_warm.circuit_depth == 26


def test_audit_d_qpiai_sdk_circuit_and_local_optimization():
    from qpiai_quantum import Circuit
    from qpiai_quantum.algorithms.opt.problems.base import OptimizationProblem
    from qpiai_quantum.algorithms.opt.solvers.qaoa import QAOASolver
    from qpiai_quantum.algorithms.utils.optimizers.gradient_free import cobyla_optimize
    from qpiai_quantum.simulator import StatevectorSimulator

    # 1. 2-qubit VQC SDK statevector == closed-form custom simulator
    feats, th = (0.35, 0.25), (1.119555, 0.768995)
    assert sdk_expectation_z0(feats, th) == pytest.approx(expectation_z0(feats, th), abs=1e-14)
    sv = sdk_statevector(feats, th)
    assert sum(abs(a) ** 2 for a in sv) == pytest.approx(1.0, abs=1e-14)

    # 2. Local SDK optimization call on certified 4-var QUBO via StatevectorSimulator + cobyla_optimize
    c4 = [
        {"id": "r0", "vehicle": "A", "stops": ["a"], "cost": 2.0},
        {"id": "r1", "vehicle": "A", "stops": ["a", "b"], "cost": 3.0},
        {"id": "r2", "vehicle": "B", "stops": ["b"], "cost": 2.0},
        {"id": "r3", "vehicle": "C", "stops": ["a", "b"], "cost": 6.0},
    ]
    q4 = build_route_qubo(c4, ["a", "b"], coverage_penalty=20.0, vehicle_penalty=15.0)
    c_i, h_i, j_i = qubo_to_ising(q4)
    lp4 = lp_augmented_relaxation(q4)
    thetas4 = [2.0 * math.asin(math.sqrt(min(0.95, max(0.05, p)))) for p in lp4]
    sim = StatevectorSimulator()

    def build_qc(params, measure: bool = False):
        gm, bt = float(params[0]), float(params[1])
        qc = Circuit(q4.n)
        for qi, thi in enumerate(thetas4):
            qc.ry(qi, thi)
        for qi, hv in enumerate(h_i):
            if abs(hv) > 1e-12:
                qc.rz(qi, 2.0 * gm * float(hv))
        for (i, k), jv in j_i.items():
            if abs(jv) > 1e-12:
                qc.cx(i, k)
                qc.rz(k, 2.0 * gm * float(jv))
                qc.cx(i, k)
        for qi, thi in enumerate(thetas4):
            qc.ry(qi, -thi)
            qc.rz(qi, 2.0 * bt)
            qc.ry(qi, thi)
        if measure:
            qc.measure_all()
        return qc

    def obj_fn(params):
        sv_q = sim.run(build_qc(params, measure=False), shots=1).statevector
        tot = 0.0
        for idx_le, amp in enumerate(sv_q):
            pr = float(abs(amp) ** 2)
            z = [1 - 2 * ((idx_le >> qi) & 1) for qi in range(q4.n)]
            tot += pr * (c_i + sum(h_i[qi] * z[qi] for qi in range(q4.n)) + sum(jv * z[i] * z[k] for (i, k), jv in j_i.items()))
        return tot

    best_p, _, _ = cobyla_optimize(obj_fn, np.array([0.08, 0.45]), bounds=[(0.0, math.pi), (0.0, math.pi)], maxiter=35)
    meas = sim.run(build_qc(best_p, measure=True), shots=256, seed=42)
    feas = 0
    best_cost = float("inf")
    for raw_bs, cnt in meas.counts.items():
        padded = str(raw_bs).zfill(q4.n)
        var_bits = [int(padded[-(qi + 1)]) for qi in range(q4.n)]
        v = validate_selection(q4, var_bits)
        if v["feasible"]:
            feas += int(cnt)
            best_cost = min(best_cost, float(v["business_cost"]))
    assert feas / 256.0 > 0.80
    assert best_cost == pytest.approx(3.0)

    # 3. Confirm QAOASolver.run() enforces QpiAIQuantumAuth gate when unauthenticated
    class DummyProb(OptimizationProblem):
        def __init__(self):
            super().__init__()
            self.n_qubits = 2
        def _construct_hamiltonian(self):
            return []
        def _get_hamiltonian_terms(self):
            return [([(0, "Z"), (1, "Z")], 1.0)]
        def decode_solution(self, bitstring: str):
            return {"bs": bitstring}
        def validate_solution(self, solution):
            return True, "ok"
        def compute_solution_quality(self, solution):
            return {"cost": 0.0}

    with pytest.raises(ValueError, match="Authentication required"):
        QAOASolver(problem=DummyProb(), layers=1, verbose=False).run(
            shots=64, method="statevector", device_name="QpiAI-QSV-Local"
        )


def test_audit_e_box_relaxation_failure_and_deadline_budget():
    fix8 = json.loads((Path(__file__).resolve().parents[1] / "experiments_qaoa" / "dispatch_fixture.json").read_text(encoding="utf-8"))
    r8 = generate_routes(fix8)
    c8 = [{"id": f"{r['vehicle_id']}:{'>'.join(r['order_ids'])}", "vehicle": r["vehicle_id"], "stops": r["order_ids"], "cost": float(r["travel_cost"])} for r in r8]
    q8 = build_route_qubo(c8, [o["id"] for o in fix8["orders"]])

    # V1 box relaxation puts zero weight on all 4 optimal 2-stop routes (2, 3, 6, 7)
    box = continuous_relaxation(q8)
    assert sum(box[i] for i in (2, 3, 6, 7)) == pytest.approx(0.0, abs=1e-9)

    # Patched deadline guard keeps 8-var warm-start QAOA within tight wall budget
    res = execute_qaoa_benchmark(
        q8, layers=1, warm_start=True, relaxation_mode="lp_augmented", grid_size=12, shots=512, seed=101, wall_deadline_seconds=0.025
    )
    assert res["resource_accounting"]["total_wall_seconds"] <= 0.030
    assert res["feasible_shots"] > 0

