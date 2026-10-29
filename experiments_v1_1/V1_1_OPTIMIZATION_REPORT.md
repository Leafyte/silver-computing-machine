# DebugLeaf V1.1 — Dual-Objective Quantum Optimization & Decision-Quality Report

**Branch:** `feature/macbook-quantum-optimization` (branched from `main` @ `6b1480a027a839fa2ebe683b34270a761a4a8449`)  
**Machine Role:** MacBook (`Apple M3`, `arm64`, 16.0 GB RAM) — Quantum Optimization & Scientific Evaluation Engine  
**Data & Execution Provenance:** Synthetic paired forecast/realized contract-logistics scenarios (`150` scenarios across `5` independent replicates: `90` train, `30` validation, `30` held-out test) + `experiments_qaoa/dispatch_fixture.json`. Executed strictly via local exact CPU statevector simulation (`cloud_qpu_authorized = false`).

---

## 1. MacBook Environment & Reproducibility Summary

- **OS & Kernel:** macOS `15.7.3` (`Darwin 24.6.0`, `arm64`)
- **CPU / Memory:** Apple M3 (`8` logical cores), `16.0 GB` unified memory
- **Accelerator Audit:** Apple M3 Integrated 8-core GPU (`Metal 3`), `cuda_available = false`, `qpiai_metal_accelerated = false` (QpiAI Quantum `StatevectorSimulator` executes on CPU only)
- **Python Runtime:** Isolated `.venv` running CPython `3.12.13` (`/Users/karthikm/Desktop/29nmit/silver-computing-machine/.venv/bin/python`)
- **SDK Compatibility Pins:** Verified against repository evidence (`qpiai-opt==0.0.9`, `qpiai-quantum==0.3.0 / 0.7.5`, `ortools==9.15.6755`). Fast Cooley-Tukey statevector evolution (`fast_qaoa_state`) is unit-tested against `backend.qaoa.qaoa_state` to `< 1e-11` probability discrepancy.
- **V1 Reproduction Note:** Canonical SHA-256 dataset digest (`DB95F5F1...`) and all `90/90` frozen V1 plans in `experiments_matched_budget/` reproduce with `0.0` objective discrepancy; raw file-byte hashes in `validate_saved_results.py` differ solely due to Windows `CRLF` vs macOS Git `LF` checkout line endings.
- **Exact Reproduction Commands:**
  ```bash
  .venv/bin/python -m experiments_v1_1.run_benchmarks
  .venv/bin/python -m pytest tests/test_v1_1_benchmarks.py tests/test_domain.py tests/test_qubo.py tests/test_qaoa_dispatch.py tests/test_quantum_learning.py tests/test_financial.py tests/test_risk.py tests/test_recovery.py tests/test_strong_baseline.py
  ```

---

## 2. P0 — Decision-Quality Benchmark Table (7-Way Held-Out Comparator, N = 30 Test Scenarios)

Explicit financial contract: `Transport = $0.50 / travel_min`, `Driver Labor = $0.25 / duty_min`, `Contractual SLA Penalty = order_penalty_USD * late_min`. Hyperparameters selected strictly on the 30-scenario validation split (`trainable_readout=True, tau=0.15, lr=0.08, decision_weight=0.50, use_empirical_train_severity=False`).

| Pipeline | Predictor | Optimizer | Total Cost (\$) | Transport (\$) | Labor (\$) | SLA Penalty (\$) | Mean Regret ± 95% CI (\$) | SLA Breach Rate | Valid Plan Rate | Raw Solver Success | Fallback Rate | Mean QAOA Shot Feas. | Subproblem Gap (\$) | Decision CPU (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | Classical GBDT (Newton XGBoost) | Exact / MILP | 10.3790 | 6.5693 | 3.8097 | 0.0000 | **0.1385 ± 0.0497** | 0.0% | 100.0% | 100.0% | 0.0% | N/A (1.0) | 0.0000 | 0.00001 |
| **B** | De-saturated Std-VQC (4-param) | Exact / MILP | 10.3949 | 6.5800 | 3.8150 | 0.0000 | 0.1545 ± 0.0497 | 0.0% | 100.0% | 100.0% | 0.0% | N/A (1.0) | 0.0000 | <0.00001 |
| **C** | Decision-Focused VQC (4-param) | Exact / MILP | 10.3949 | 6.5800 | 3.8150 | 0.0000 | 0.1545 ± 0.0497 | 0.0% | 100.0% | 100.0% | 0.0% | N/A (1.0) | 0.0000 | <0.00001 |
| **D** | De-saturated Std-VQC (4-param) | Standard QAOA (`p=1`) | 10.4428 | 6.6119 | 3.8309 | 0.0000 | 0.2023 ± 0.0774 | 0.0% | 100.0% | **16.7%** | **83.3%** | 0.09% | 0.0672 | 5.38786 |
| **E** | Decision-Focused VQC (4-param) | Standard QAOA (`p=1`) | 10.4202 | 6.5968 | 3.8234 | 0.0000 | 0.1797 ± 0.0704 | 0.0% | 100.0% | **16.7%** | **83.3%** | 0.08% | 0.0464 | 5.34682 |
| **F** | Decision-Focused VQC (4-param) | **LP Warm-Start QAOA (`p=1`)** | **10.3901** | 6.5767 | 3.8134 | 0.0000 | **0.1496 ± 0.0614** | 0.0% | 100.0% | **100.0%** | **0.0%** | **2.02%** | 0.0396 | 5.26772 |
| **G** | Classical GBDT (Newton XGBoost) | **LP Warm-Start QAOA (`p=1`)** | 10.3910 | 6.5773 | 3.8137 | 0.0000 | 0.1505 ± 0.0507 | 0.0% | 100.0% | **100.0%** | **0.0%** | **7.76%** | **0.0080** | 5.30877 |

### Held-Out Calibration Audit (`420` Evaluated Route-Scenarios)

| Predictor | Brier Score (↓) | Expected Calibration Error — ECE (↓) | Diagnostic Note |
| :--- | :---: | :---: | :--- |
| V1 Fixed-Readout VQC (`sigmoid(-2<Z0>)`) | 0.06567 | 0.13478 | Trapped in `[0.1192, 0.8808]` saturation band |
| Classical Logistic Regression | 0.06741 | 0.11402 | Linear logit underfits route-pressure threshold |
| Classical Newton GBDT (`40` trees, `depth=3`) | 0.04993 | **0.03678** | Strong non-linear classical benchmark |
| **V1.1 De-saturated Standard VQC (`4-param`)** | **0.04692** | 0.05283 | **Best overall Brier score (-28.6% vs V1 VQC)** |
| **V1.1 Decision-Focused VQC (`4-param`)** | **0.04713** | 0.05434 | Retains probability calibration while coupling to dispatch cost |

---

## 3. P1 — Quantum Optimization Quality Benchmark Table

### 3A. Exhaustive QUBO / Ising Mathematical Isomorphism Audit

| Instance | Qubits (`n`) | Basis States Checked | Feasible States | Max `|QUBO - Direct|` | Max `|Ising - Direct|` | Penalty Margin Gap | Ground State Feasible? | Certified Optimum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `tiny_4var` | 4 | 16 | 3 (18.75%) | `0.0` | `0.0` | `+19.00` | Yes (`0100`) | `3.000` |
| `dispatch_8var` | 8 | 256 | 6 (2.34%) | `0.0` | `0.0` | `+27.00` | Yes (`00000001`) | `4.000` |
| `prospective_12var` | 12 | 4,096 | 8 (0.195%) | `2.84e-14` | `4.55e-13` | `+13.95` | Yes (`000001001000`) | `7.575` |

### 3B. Solver Comparison on 8-Qubit Dispatch Benchmark (`512` Shots, Seed `7` + `5`-Seed Summary)

| Method | Circuit (`q`, `p`, 1Q/2Q Gates, Depth) | Circuit Evals | Feasible Prob. (Seed 7) | 5-Seed Mean Feas. ± 95% CI | 5-Seed Global-Opt Shot Prob. | Expected QUBO Energy | Mean Feas. Obj. | Opt. Gap | TTS-99% Feas. (s) | Total CPU (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Classical Exact (`enumerate`) | N/A | 256 states | 100.0% (det.) | 100.0% | 100.0% | 4.00 | 4.0000 | 0.000 | 0.0153 | 0.0152 |
| Classical Heuristic Sampler | N/A | 512 samples | 13.87% | 13.87% | — | 4.00 (best) | 4.3944 | 0.000 | — | 0.0125 |
| Standard QAOA `p=1` (`grid`) | `(8, 1, 50/52, d=17)` | 101 | 22.07% ± 3.59% | 6.21% ± 0.82% | 4.96% | 176.00 | 4.0000 | 0.000 | 0.5123 | 0.5205 |
| Standard QAOA `p=2` (`INTERP+pattern`) | `(8, 2, 92/104, d=33)` | 213 | 30.27% ± 3.98% | 28.48% ± 0.62% | 21.41% | 39.73 | 4.4774 | 0.000 | 1.4808 | 1.4828 |
| V1 Warm-Start QAOA `p=1` (`box_v1`) | `(8, 1, 66/52, d=19)` | 101 | 21.48% ± 3.56% | 22.70% ± 0.67% | **8.79%** | 35.39 | 5.1091 | 0.000 | 0.5200 | 0.5267 |
| **V1.1 LP Warm-Start QAOA `p=1`** | `(8, 1, 66/52, d=19)` | 101 | **61.52% ± 4.21%** | **60.12% ± 2.25%** | **59.81%** | 31.07 | 4.0127 | **0.000** | **0.5179** | **0.5248** |
| **V1.1 LP Warm-Start QAOA `p=2`** | `(8, 2, 124/104, d=37)` | 222 | **57.81% ± 4.28%** | **59.96% ± 2.35%** | **59.61%** | **27.57** | **4.0068** | **0.000** | 1.2499 | 1.2544 |

---

## 4. Matched-Budget Evaluation & Phase-Level Resource Accounting

### 4A. Operational Equal-Deadline Curves (`256` Shots)

| Decision Deadline Target | `std_qaoa_p1` Feas. Prob. (CPU s / Evals) | `v1_box_warm_p1` Feas. Prob. (CPU s / Evals) | `v1_1_lp_warm_p1` Feas. Prob. (CPU s / Evals) | `v1_1_lp_warm_p2` Feas. Prob. (CPU s / Evals) |
| :--- | :---: | :---: | :---: | :---: |
| **`0.02s` cap** | 1.95% (`0.0378s` / 5) | 8.98% (`0.0307s` / 3) | **32.03%** (`0.0354s` / 5) | **32.03%** (`0.0368s` / 5) |
| **`0.05s` cap** | 1.95% (`0.0679s` / 11) | 8.98% (`0.0644s` / 9) | **32.03%** (`0.0624s` / 10) | **32.03%** (`0.0702s` / 11) |
| **`0.12s` cap** | 33.20% (`0.1350s` / 24) | 8.98% (`0.1363s` / 20) | 32.03% (`0.1317s` / 22) | **60.94%** (`0.1374s` / 24) |

### 4B. 5-Stage Resource Breakdown (`warm_lp_augmented_qaoa_p1_grid`, 512 shots)
1. **Preprocessing + LP Relaxation + Spectrum:** `0.0068s` CPU (`1.3%`), peak memory `48.55 KB`
2. **Classical Parameter Optimization (`100` iters):** `0.5031s` CPU (`95.9%`)
3. **Final Statevector Evolution:** `0.0052s` CPU (`1.0%`)
4. **Finite-Shot Sampling (`512` shots):** `0.0017s` CPU (`0.3%`)
5. **Independent Decoding & Feasibility Validation:** `0.0080s` CPU (`1.5%`)

---

## 5. Rejected Optimization Attempts

1. **SPSA Stochastic Optimizer on Noiseless Small-Circuit Warm-QAOA (`spsa_warm_start_qaoa_depth_2_rejected`):** Degraded expected QUBO energy from `27.57` to `69.44` and feasible-sample probability from `57.81%` to `29.88%` due to simultaneous random perturbation variance across anisotropic `(gamma, beta)` scales.
2. **Tight Warm-Start Bloch Clamping `epsilon = 0.005` on 12-Qubit Instances (`tight_epsilon_clamp_0_005_rejected`):** While raising 8-qubit `p=1` feasibility to `70.31%` when LP coordinates are quarter-integral (`0.25`), clamping `0.0 -> 0.005` on 12-qubit instances where 8/12 LP coordinates are zero collapses mixer transverse strength (`nx = 2*sqrt(p*(1-p)) = 0.141`), dropping 12-qubit feasible probability from `2.34%` down to `0.39%`. Retained `(0.05, 0.95)` across all pipelines for uniform cross-scale stability.
3. **Small-Sample Per-Route Empirical Severity (`noisy_per_route_empirical_severity_rejected`) & `3x` Over-Penalized Severity (`over_penalized_3x_severity_rejected`):** On 18 training scenarios per replicate, estimating separate conditional lateness per route ID (`14` routes, `0–3` breaches each) injected high estimator variance and degraded validation mean regret from `$0.1091` (`Brier = 0.08566`) to `$0.1390` (`Brier = 0.09228`); `3x` severity inflation degraded validation Brier to `0.09821` (`$0.1300` regret). Rejected via validation-split selection before touching the test split.
4. **Un-normalized Penalty Gradient Descent (`lr=0.005`) for Continuous Relaxation:** Diverged/collapsed when penalty curvature `2 * P = 130` exceeded `2 / lr`; replaced with Lipschitz-normalized (`alpha = 0.8 / (2 * rho * (|S| + 1))`) Primal-Dual Augmented-Lagrangian LP relaxation.

---

## 6. Cross-Team Artifact Contract (`debugleaf.hybrid_artifact.v1.1`)

Implemented in `experiments_v1_1/artifact_contract.py` and exported at `experiments_v1_1/artifacts/exported_hybrid_contract_sample.json` (`SHA-256: 54FF0477F5E505C2F18AFEA5EE2F40436179075EDE52E526F875B4A11AB95A0E`, `valid = true`).

