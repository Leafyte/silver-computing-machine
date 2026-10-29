# Product Requirements Document (PRD) — ArQ

> **Project Name:** **ArQ** (internal Python/SQLite/env namespace: `DebugLeaf` / `DEBUGLEAF_*`)  
> **Challenge Alignment:** Quant-A-Maze 3.0 — Track I (`QML-01`: Delay-Risk-Aware Fleet Dispatch for Contract Logistics)  
> **Document Role:** Canonical product requirements grounded in the existing `backend/` and `experiments_v1_1/` codebase, defining both verified capabilities and remaining V1.1+ milestones.

---

## 1. Overview & Problem Statement

Contract logistics operators dispatch finite, heterogeneous vehicle fleets across customer stops subject to vehicle capacities, delivery time windows (`[open, close]`), service durations, and contractual Service-Level Agreement (SLA) lateness penalties. In real-world operations, minimizing nominal travel distance or duration often selects tight-margin routes where travel-time variance triggers cascading SLA breaches and severe financial penalties. Conversely, predictive-only ML models trained strictly on Binary Cross-Entropy (BCE) or Brier score often misallocate accuracy toward routes that do not alter downstream fleet-assignment decisions.

**ArQ** is a decision-focused hybrid quantum-classical fleet dispatch backend built around four non-negotiable engineering principles established in the repository:
1. **Decision-Focused Quantum Learning:** Parameterized Variational Quantum Circuits (VQCs) predict route-level SLA breach probabilities trained end-to-end through a differentiable softmin exact-cover dispatch surrogate via exact parameter-shift gradients (paired with trainable affine readout de-saturation in V1.1).
2. **Warm-Start Quantum Combinatorial Optimization:** Candidate vehicle routes are compiled into exact Set-Partitioning QUBO and Ising Hamiltonians and solved via **Primal-Dual Augmented-Lagrangian LP Warm-Start QAOA** (alongside QpiAI Opt `0.0.9` / QpiAI Quantum `0.3.0` local statevector adapters) with explicit classical exact fallback if raw quantum samples fail validation.
3. **Zero-Trust Independent Validation:** No solver, heuristic, or quantum sample is trusted directly. Every published dispatch plan and monetary total must pass an independent validator (`backend/validator.py`, `backend/financial.py`) that recomputes order coverage, vehicle capacity, leg-by-leg arrival/service/departure clocks, hard-window compliance, and SLA exposure from raw scenario inputs.
4. **Reproducible Local-First Execution:** All computations run deterministically on bounded local CPU statevector simulation (`4–12` qubits) with WAL-mode SQLite persistence, SHA-256 provenance digests, leakage-safe forecast/realized splits, and secret-redacted REST boundaries.

---

## 2. Target Users / Personas

| Persona | Role in ArQ (`backend/auth.py`) | Primary Goals & Pain Points |
| :--- | :--- | :--- |
| **Logistics Dispatcher** | `dispatcher` | Needs feasible vehicle-to-route plans, ordered stop ETAs, explicit USD/currency cost vs. SLA exposure breakdowns, and fast disruption re-planning that respects already-committed (`locked_prefixes`) stops without silently overwriting the last valid plan. |
| **QML / OR Researcher** | `researcher` | Needs leakage-safe forecast-vs-realized evaluation, Common-Random-Number (CRN) scenario risk/CVaR replay, raw vs. repaired QAOA bitstring diagnostics, Brier/ECE calibration audits, and reproducible cross-machine JSON contracts (`debugleaf.hybrid_artifact.v1.1`). |
| **Platform / Benchmark Admin** | `admin` | Needs bounded CPU/worker concurrency controls (`429 capacity_reached`, `503 solver_budget_exceeded`), durable async job lifecycles, restart-interrupted recovery, append-only SQLite audit logs, and zero credential leakage. |

---

## 3. Goals & Measurable Success Criteria

*(Note: Criteria distinguish **Verified Local Baseline** already proven in `backend/` and `experiments_v1_1/` from **Next-Stage Targets** for upcoming implementation prompts. All quantum metrics below reflect local exact CPU statevector simulation, not cloud/QPU advantage claims.)*

| ID | Goal | Measurable Success Criterion | Current Codebase State |
| :--- | :--- | :--- | :--- |
| **G-1** | **Zero Invalid Dispatches** | `100%` of plans returned with HTTP `200` or job status `succeeded` pass `backend.validator.validate_plan` (`valid == True`) with `< 1e-9` recomputed timing/cost discrepancy. | **Verified** (`backend/validator.py`, `90/90` & `30/30` held-out plans). |
| **G-2** | **Verified Circuit Decision Gradients** | Analytic parameter-shift VQC gradients through the downstream softmin route-cover loss match centered finite differences (`h = 1e-5`) within `< 1e-9` max absolute error. | **Verified** (`5.31e-11` toy, `2.04e-10` prospective, `8.21e-11` QpiAI SDK). |
| **G-3** | **De-Saturated VQC Calibration** | 4-parameter affine-readout VQC reduces held-out Brier score by $\ge 20\%$ vs. V1 fixed-readout `sigmoid(-2<Z0>)` without touching held-out test scenarios during tuning. | **Verified in `experiments_v1_1/`** (`0.04692` vs `0.06567`, `-28.6%`); API promotion pending. |
| **G-4** | **Exact QUBO/Ising & High-Feasibility Warm QAOA** | `0.0` (`< 1e-12` float) energy discrepancy across all $2^n$ states for $n \in \{4, 8, 12\}$ qubits; LP Warm-Start QAOA (`p=1`) achieves $\ge 50\%$ 5-seed mean feasible shot rate on the 8-qubit dispatch benchmark and `0%` classical fallback across the 30 held-out test scenarios. | **Verified in `experiments_v1_1/`** (`60.12%` mean feasible shots; `100%` raw quantum feasibility on `N=30`); API promotion pending. |
| **G-5** | **Bounded & Crash-Safe API Runtime** | Every endpoint enforces explicit size/work caps, returns non-echoing field errors (`422`), distinguishes `409 infeasible` from `503 solver_budget_exceeded`, and marks stale in-flight SQLite jobs `restart_interrupted` on startup. | **Verified** (`backend/api.py`, `backend/jobs.py`, `backend/recovery_store.py`). |
| **G-6** | **Unified V1.1 API & Branch Convergence** | Merge `origin/main` cooperative cancellation suite and expose V1.1 LP Warm-Start QAOA, 4-param VQC inference, and `debugleaf.hybrid_artifact.v1.1` validation on `/api/v1/*` with `100%` passing `pytest` suite. | **Target for Milestones M0–M2** (`docs/IMPLEMENTATION_PLAN.md`). |

---

## 4. Prioritized Feature List (`MUST` / `SHOULD` / `COULD`)

| ID | Priority | Feature Name | Status in Repo | Description |
| :--- | :---: | :--- | :---: | :--- |
| **F-01** | **MUST** | **Scenario Validation & Capacity-Pruned Route Enumeration** | Built (`backend/domain.py`) | Strict schema checking (`depot`, `vehicles`, `orders`, `travel`), soft vs. `hard_window: true` deadlines, and prefix-pruned ordered route generation ($\le 8$ orders, $\le 16$ vehicles, $\le 10,000$ routes). |
| **F-02** | **MUST** | **Independent Zero-Trust Dispatch & Financial Validator** | Built (`backend/validator.py`, `backend/financial.py`) | Standalone recomputation of order coverage, vehicle uniqueness, load limits, stop ETAs, lateness penalties, and caller-declared currency totals without trusting solver output. |
| **F-03** | **MUST** | **Classical Exact-Cover & Caller-Declared Financial Dispatch** | Built (`backend/domain.py`, `backend/financial.py`) | Deterministic exact-cover search ($100,000$ proxy work budget; $5,000$ financial work budget) with explicit `currency`, `operating_cost_per_travel_minute`, and `penalty_rate_unit`. |
| **F-04** | **MUST** | **Decision-Focused VQC Risk Scoring & Frozen Inference** | Built V1 (`backend/quantum_learning.py`, `learning_inference.py`); V1.1 in `experiments_v1_1/` | 2-qubit statevector circuit (`RY`-`CNOT`-`RY`-`CNOT`, `<Z0>`) with parameter-shift gradients, forecast-only feature extraction, and single-request frozen inference (`/api/v1/scenarios/vqc-plan`). |
| **F-05** | **MUST** | **Route-Selection QUBO/Ising Compilation & Warm-Start QAOA** | Built V1 (`backend/qubo.py`, `qaoa.py`, `qaoa_dispatch.py`); V1.1 LP in `experiments_v1_1/` | Binary route-cover QUBO, constant-preserving Ising map, tilted-mixer warm-start QAOA, raw vs. greedy-repaired sample auditing, and explicit classical fallback labeling. |
| **F-06** | **MUST** | **Credential-Free Local QpiAI Opt & Quantum SDK Execution** | Built (`backend/qpiai_adapter.py`, `backend/qpiai_vqc.py`) | Local `StatevectorSimulator` integration for `qpiai-opt==0.0.9` (`Model`, `QUBOProblem`, `WarmStart`, `QAOA`) and `qpiai-quantum==0.3.0`/`0.7.5` (`Circuit`), capped at $\le 8$ verified QUBO variables with bit-reversal handling. |
| **F-07** | **MUST** | **Durable SQLite Async Jobs, Cancellation & Restart Recovery** | Built (`backend/jobs.py`; cooperative cancel on `origin/main`) | WAL-mode SQLite job lifecycle (`queued`, `running`, `succeeded`, `infeasible`, `failed`, `cancelled`), PID singleton lease, 2-worker admission cap, 30s timeout, and `restart_interrupted` startup recovery. |
| **F-08** | **MUST** | **Locked-Prefix Disruption Recovery & Scoped Audit Store** | Built (`backend/recovery.py`, `backend/recovery_store.py`) | Re-solves dispatch after travel spikes or vehicle loss while preserving committed `locked_prefixes`, appending SHA-256-scoped records so failures never overwrite the last valid plan. |
| **F-09** | **SHOULD** | **Common-Random-Number (CRN) Risk Replay & Empirical CVaR** | Built (`backend/risk.py`) | Replays a validated plan across $1\text{–}100$ realized travel matrices, reporting empirical breach rates, per-order late frequencies, expected cost, and fractional-tail $\text{CVaR}_\alpha$. |
| **F-10** | **SHOULD** | **Cross-Team Hybrid Artifact Contract (`v1.1`)** | Built in `experiments_v1_1/artifact_contract.py` | Versioned JSON schema (`debugleaf.hybrid_artifact.v1.1`) with canonical SHA-256 digest linking QML risk probabilities, route dollar coefficients, QUBO terms, and execution metadata. |
| **F-11** | **SHOULD** | **Opt-In Role-Based Bearer-Token Guard** | Built (`backend/auth.py`) | Localhost-first RBAC (`DEBUGLEAF_AUTH_MODE=token`) enforcing constant-time checks for `dispatcher`, `researcher`, and `admin` roles with fail-closed `503 auth_misconfigured` protection. |
| **F-12** | **SHOULD** | **V1.1 Production API Promotion (`lp_warm_qaoa` & 4-Param VQC)** | Planned (`M1–M2`) | Promote `lp_augmented_relaxation`, Cooley-Tukey `fast_qaoa_state` (`p=1,2`), 4-param affine-readout VQC (`theta` length 4 + labor term), and `/api/v1/artifacts/validate` from `experiments_v1_1/` into `backend/`. |
| **F-13** | **COULD** | **OR-Tools Candidate Pruning (>8 Orders) & CLI/UI Demo Harness** | Planned (`M3–M4`) | Optional offline `ortools` (`pywrapcp`) top-$K$ candidate generator to feed $\le 12$ high-quality routes into QAOA for medium instances (`9–25` orders), plus a lightweight local CLI/HTML inspector. |

---

## 5. Key User Flows (Step-by-Step)

### Flow 1: Risk-Aware VQC + Financial Dispatch (`Dispatcher` / `Researcher`)
1. Client submits `POST /api/v1/scenarios/vqc-plan` with `forecast_scenario`, `nominal_scenario`, frozen parameter vector `theta`, `model_version`, `execution_backend` (`"reference"` or `"qpiai_local"`), and optional `financial_terms` (`currency`, `operating_cost_per_travel_minute`, `penalty_rate_unit`).
2. `backend/api.py` enforces optional RBAC (`TokenAuthMiddleware`), strips/rejects unknown JSON keys, checks that `nominal_scenario` and `forecast_scenario` define the identical fleet/orders/depot, and acquires 1 of 2 synchronous `compute_slots`.
3. `backend/learning_inference.py` enumerates capacity- and `hard_window`-feasible routes ($\le 20$ routes) and computes normalized `(forecast_travel, nominal_window_pressure)` features per route.
4. `backend/quantum_learning.py` (or `backend/qpiai_vqc.py` when `qpiai_local` is requested) executes the 2-qubit statevector circuit for each route and outputs SLA breach risk probability $p_r \in (0, 1)$.
5. Exact-cover selector chooses the feasible route combination minimizing expected operating cost plus risk-weighted SLA penalty exposure.
6. `backend/validator.py` (and `backend/financial.py` when `financial_terms` are supplied) independently recomputes stop ETAs, capacity, coverage, and monetary totals against `forecast_scenario`.
7. API returns HTTP `200` with `plan`, per-route `risk_probability`, SHA-256 parameter fingerprint, `validation` (`valid: true`), and explicit `fallback` metadata (`used: false`, or `used: true` if classical fallback handled a reference-mode size limit).

### Flow 2: Durable Asynchronous Warm-Start QAOA Job (`Researcher` / `Dispatcher`)
1. Client submits `POST /api/v1/jobs/qaoa` with `{scenario, seed, shots, warm_start: true, angle_grid}`.
2. `backend/api.py` validates scenario bounds ($\le 12$ generated route variables, $1 \le \text{shots} \le 512$, $2 \le \text{angle\_grid} \le 10$) and calls `JobManager.submit_qaoa(...)`.
3. `backend/jobs.py` checks worker capacity ($\le 2$ active/queued jobs; else HTTP `429 capacity_reached`), persists a `queued` row with `request_sha256` and runtime metadata in WAL-mode SQLite, and returns HTTP `202 Accepted` with `job_id`.
4. Background worker transitions the row to `running`, builds the route QUBO (`backend/qubo.py`), computes continuous relaxation probabilities $p_i \in [0, 1]$, runs tilted-mixer QAOA (`backend/qaoa.py`), and samples `shots` bitstrings.
5. `backend/qaoa_dispatch.py` records every raw sample, its QUBO energy, greedy repair edits (kept strictly diagnostic), and `validate_plan` outcome. If at least one **raw** quantum sample is domain-valid, the lowest-objective valid raw sample is chosen (`origin: "qaoa_raw_sample"`); otherwise `solve_exact` supplies an explicit classical fallback (`fallback.used: true`).
6. Client polls `GET /api/v1/jobs/{job_id}` (or cancels via `POST /api/v1/jobs/{job_id}/cancel`) and receives terminal status (`succeeded`, `infeasible`, `failed`, or `cancelled`), full sample provenance, and independent `validation`.

### Flow 3: Mid-Shift Disruption Recovery with Scoped Audit Preservation (`Dispatcher`)
1. After a traffic spike or vehicle breakdown, client submits `POST /api/v1/recovery/plan` with `{original_scenario, original_plan, updated_scenario, locked_prefixes, change_penalty}`.
2. `backend/api.py` independently validates `original_plan` against `original_scenario` and verifies every `locked_prefixes[vehicle_id]` is a true ordered prefix of that vehicle's original route.
3. `backend/recovery.py` filters updated candidate routes ($\le 2,000$ routes, $\le 10,000$ search nodes) to preserve all locked prefixes and solves for the minimum-cost exact cover (incorporating `change_penalty` for vehicle reassignments without polluting the physical validator objective).
4. `backend/validator.py` validates `recovered_plan` against `updated_scenario`.
5. `backend/recovery_store.py` appends an immutable audit row keyed by `scope_sha256` (`SHA-256(original_scenario + original_plan)`). If recovery fails (`409 recovery_infeasible` or `locked_work_vehicle_unavailable`), the failed attempt is logged, while `GET /api/v1/recovery/audits/latest?scope_sha256=...` still returns the previous validated plan for that scope.

---

## 6. Out of Scope (Explicit Exclusions)

1. **Paid Cloud / Physical QPU Execution:** Invoking QpiAI QCloud Indus-1 hardware, online `qpiai_opt.Client.solve`, or any credit-consuming remote API is strictly out of scope unless the project owner grants explicit written authorization.
2. **Live Fleet GPS / Telemetry Ingestion:** Recovery and risk modules operate via deterministic depot/time-zero scenario replay; real-time ELD/telematics streaming, turn-by-turn street map routing (OSRM/Google Maps), and driver mobile apps are excluded.
3. **Public Internet / Multi-Node Deployment:** The FastAPI service is designed for `127.0.0.1` loopback operation with a single-process SQLite PID lease (`runtime_lock`). Multi-tenant OAuth/OIDC, TLS termination, horizontal worker clustering across shared SQLite files, and cloud hosting are out of scope.
4. **Empirical Quantum Advantage or Operational Cost Claims:** Synthetic benchmark datasets (`fixtures/`, `experiments*/`) verify mathematical correctness, calibration improvements, and warm-start feasibility rates on `4–12` qubits; they must not be represented as real-world fleet savings or asymptotic quantum speedup.
5. **Modifying Inert Historical Archives:** `_archive/`, `sources/`, `bootstrap/skill-sources/`, and locked V1 result JSONs (`experiments/artifacts/prospective-dispatch-results.json`, `experiments_strong_baseline/results.json`, `experiments_matched_budget/results.json`) are read-only historical/scientific records.

---

## 7. Assumptions & Open Questions

### Assumptions Made from Codebase Inspection
- **A-1 (Project vs. Internal Namespace):** External documentation and new UI/CLI surfaces refer to the system as **ArQ**, while Python package imports (`backend.*`), environment variables (`DEBUGLEAF_*`), default SQLite path (`.runtime/debugleaf.sqlite3`), and artifact schema (`debugleaf.hybrid_artifact.v1.1`) retain `debugleaf` naming for 100% backward compatibility with the existing 17 test suites.
- **A-2 (Primary Python Runtime):** CPython `3.12` (64-bit) managed via isolated `uv` virtual environments (`.venv-api` for FastAPI + core tests; `.venv-sdk` for QpiAI SDKs) is the canonical runtime across macOS (`arm64`) and Windows (`x86_64`), avoiding CPython `3.14` wheel gaps in `numba`/`cvxpy`.
- **A-3 (Backward-Compatible V1.1 Promotion):** Promoting V1.1 models (`4-param` VQC and `lp_augmented` QAOA from `experiments_v1_1/`) into `backend/` will preserve existing V1 request payloads as defaults (e.g., 2-angle `theta` and `relaxation="box_v1"` or explicit `solver_version`) so zero existing V1 API tests break.
- **A-4 (Branch Merge Order):** Merging `origin/main` (`3cb42b4`, cooperative job cancellation) into `master` (`dcf59c6`, V1.1 benchmarks) is assumed to be Milestone `M0` before editing `backend/jobs.py` or `backend/api.py`.

### Open Questions for Owner Review
- **Q-1 (Default QAOA Relaxation on `/api/v1/scenarios/qaoa`):** When promoting V1.1 into `backend/qaoa.py`, should `/api/v1/scenarios/qaoa` default `warm_start=True` to the vastly stronger V1.1 `lp_augmented_relaxation` (`60.1%` vs `22.7%` 8-qubit feasible shot rate) or keep `box_v1` as default behind an optional `"relaxation": "lp_augmented"` field?
- **Q-2 (QpiAI Adapter Variable Cap):** `backend/qpiai_adapter.py` currently enforces `MAX_VERIFIED_QPIAI_VARIABLES = 8` because exhaustive $2^n$ SDK-to-source QUBO equality is checked at runtime. Should Milestone `M3` raise this cap to `12` variables (matching `backend/qaoa_dispatch.py`'s 12-variable limit, $2^{12} = 4,096$ states, already verified in `experiments_v1_1/mathematical_verification.py`)?
- **Q-3 (Authenticated QpiAI Cloud Smoke):** Will the owner authorize a bounded, single-circuit QpiAI QCloud/Opt authenticated smoke test before final hackathon submission (to satisfy Section 8 rubric A2/A3 cloud execution if required), or keep ArQ 100% offline credential-free?

