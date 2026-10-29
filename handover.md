# ArQ — Backend & Workspace Handover Guide

> **Project Name:** **ArQ**  
> **Legacy / Internal Codebase Namespace:** `DebugLeaf` / `Q-VANTIS` (Quant-A-Maze 3.0 — Problem `QML-01`: Delay-Risk-Aware Fleet Dispatch for Contract Logistics)  
> **Purpose of this Document:** Fast, ground-truth orientation for any coding agent or engineer entering this repository, detailing directory layout, every working backend module, REST API contracts, quantum/classical optimization pipelines, environment quirks, test commands, and strict safety/evidence guardrails.

---

## 1. Quick Orientation & Naming Note

- **Project Identity:** This project is **ArQ**. Because it evolved from the DebugLeaf / Terra V1 & V1.1 sprints, internal Python identifiers, SQLite filenames, environment variables (`DEBUGLEAF_DB_PATH`, `DEBUGLEAF_AUTH_MODE`, `DEBUGLEAF_*_TOKEN`), and cross-team schema tags (`debugleaf.hybrid_artifact.v1.1`) use the `debugleaf` prefix. **Keep internal symbol/env names intact** unless performing a coordinated migration across `backend/` and `tests/`.
- **Operating Rules:** Always read [`AGENTS.md`](AGENTS.md) before modifying code or running experiments. Key rules:
  - Keep all runtimes isolated (never install packages globally).
  - No paid cloud/QPU jobs or credit spending without explicit owner authorization (`cloud_qpu_authorized = false` by default; all quantum execution here is **exact local CPU statevector simulation**).
  - Every feasible dispatch plan must pass the **independent zero-trust validator** (`backend/validator.py`), keeping raw solver/circuit outputs, repairs, fallbacks, and validator recomputations separate.
  - **Never overwrite frozen benchmark artifacts** or tune hyperparameters on held-out test splits.

### Git & Branch Topology (Current State)
- **Active branch:** `master` (tracks `origin/master` on `ommd3076/silver-computing-machine`; `url` remote points to `Leafyte/ArQ`).
- **Divergence note vs `origin/main`:** Both `master` and `origin/main` branched from V1 commit `6b1480a`:
  - `master` adds the **V1.1 Dual-Objective Quantum Optimization, 7-Way Decision Comparator, LP Warm-Start QAOA, and Hybrid Artifact Contract** (`experiments_v1_1/`, `tests/test_v1_1_benchmarks.py`).
  - `origin/main` (`3cb42b4`) has 3 cooperative job-cancellation commits (`backend/jobs.py`, `tests/test_api.py`, `tests/test_cancellation.py`).
  - The two branches touch **completely disjoint files** and can be merged cleanly when desired.

---

## 2. Directory Structure

```text
.
├── AGENTS.md                        # Mandatory operating, scientific-claim, and safety rules
├── README.md                        # High-level project overview and evidence boundaries
├── handover.md                      # This agent handover guide
├── TERRA-CHECKPOINT.md              # Chronological V1 engineering & verification log
├── pytest.ini                       # Scoped test discovery (tests/) + workspace basetemp
│
├── backend/                         # Core ArQ dispatch, quantum, risk, recovery & API backend
│   ├── api.py                       # Versioned FastAPI surface (/api/v1/*, 16 endpoints)
│   ├── auth.py                      # Localhost default + opt-in constant-time RBAC token middleware
│   ├── domain.py                    # Scenario schema, capacity-pruned route generator, exact solver
│   ├── validator.py                 # Independent zero-trust plan & ETA/SLA recomputation validator
│   ├── financial.py                 # Caller-declared monetary exact-cover solver & validator
│   ├── quantum_learning.py          # 2-qubit exact CPU statevector VQC + parameter-shift gradients
│   ├── qpiai_vqc.py                 # Local QpiAI Quantum StatevectorSimulator adapter for 2-qubit VQC
│   ├── learning_dispatch.py         # Diagnostic route-candidate VQC + classical dispatch pipeline
│   ├── prospective_dispatch.py      # Leakage-safe forecast-vs-realized VQC training & evaluation
│   ├── learning_inference.py        # Bounded single-request frozen VQC inference engine
│   ├── qubo.py                      # Route-selection QUBO builder, QUBO->Ising mapper, enumerator
│   ├── qaoa.py                      # Pure-Python Standard & Warm-Start QAOA (tilted XY/Z mixer)
│   ├── qaoa_dispatch.py             # End-to-end QAOA route selector + repair audit + exact fallback
│   ├── qpiai_adapter.py             # Credential-free QpiAI Opt 0.0.9 + Quantum 0.3.0 QAOA adapter
│   ├── risk.py                      # Fixed-plan scenario replay, empirical SLA breach & CVaR scorer
│   ├── recovery.py                  # Disruption recovery solver preserving locked route prefixes
│   ├── recovery_store.py            # Append-only SQLite recovery audit store with SHA-256 scoping
│   └── jobs.py                      # Durable WAL-mode SQLite async job queue (exact & QAOA jobs)
│
├── experiments_v1_1/                # V1.1 Quantum Optimization & 7-Way Decision Benchmark Engine
│   ├── quantum_optimization.py      # LP-Relaxed Warm-Start QAOA (p=1,2), Cooley-Tukey fast simulator
│   ├── decision_quality.py          # 4-param de-saturated VQC, Newton GBDT, 7-way financial benchmark
│   ├── mathematical_verification.py # Exhaustive 4/8/12-qubit QUBO<->Ising isomorphism & margin verifier
│   ├── artifact_contract.py         # Cross-team schema validator (debugleaf.hybrid_artifact.v1.1)
│   ├── run_benchmarks.py            # Reproducible V1.1 benchmark runner
│   ├── results_v1_1.json            # Recorded V1.1 benchmark outputs
│   ├── V1_1_OPTIMIZATION_REPORT.md  # Full V1.1 benchmark tables, timings, and rejected ablations
│   └── artifacts/                   # Sample validated hybrid contract JSON
│
├── experiments/                     # V1 VQC gradient & prospective dispatch experiments (FROZEN artifacts)
├── experiments_qaoa/                # 4-var pure-Python & 8-var QpiAI warm-start QAOA fixtures/results
├── experiments_qpiai_vqc/           # QpiAI Quantum 0.3.0 local VQC decision-gradient experiment
├── experiments_strong_baseline/     # Preregistered 5-replicate (150-scenario) VQC vs Logistic vs CART study
├── experiments_matched_budget/      # Common CPU-ceiling comparison + 90/90 saved-plan revalidator
│
├── fixtures/
│   └── dispatch_tiny.json           # Canonical deterministic scenario fixture (2 vehicles, 3 orders)
├── tests/                           # 17 unit, API, persistence, protocol-lock, and stress test suites
├── docs/                            # Detailed specs: API-RUNTIME, FINANCIAL, RISK, RECOVERY, VQC-INFERENCE
├── bootstrap/                       # Audited SDK requirement pins (requirements-sdk-strict.txt) & reviews
├── tools/                           # Environment inspector & credential-free SDK probes (tools/sdk_probes/)
├── sources/                         # Extracted QML-01 problem rubric & legacy reference spec
└── .agents/skills/                  # Reviewed project-local agent skills
```

---

## 3. Everything That Works in the ArQ Backend

Every module in `backend/` (except `backend/api.py`, which wraps them in FastAPI, and the two optional `qpiai_*` vendor adapters) is implemented in **pure Python 3.12 standard library** with deterministic bounds and zero external runtime dependencies.

### 3.1 Classical Dispatch Domain & Independent Validator
- **Domain Engine (`backend/domain.py`):**
  - `validate_scenario(scenario)` validates `depot`, `vehicles` (`id`, `capacity`), `orders` (`id`, `demand`, `window: [open, close]`, `service`, `penalty`, optional `hard_window: bool`), and complete `travel` matrix.
  - `generate_routes(scenario)` enumerates ordered per-vehicle routes, pruning over-capacity subsets before permutation expansion while preserving deterministic ordering.
  - `solve_exact(scenario)` finds the minimum-cost exact cover across all orders (`travel_cost + penalty * lateness`).
  - **Enforced bounds:** `MAX_EXACT_ORDERS = 8`, `MAX_VEHICLES = 16`, `MAX_ROUTE_CANDIDATES = 10_000`, `MAX_EXACT_SEARCH_WORK = 100_000`. Exhausting the work budget raises `SolveBudgetExceeded` (`solver_budget_exceeded`) without publishing an unproven incumbent—strictly distinguished from proven `infeasible`.
  - **Time windows:** Soft SLA close by default (`lateness = max(0, service_start - close)`); setting `"hard_window": true` on an order enforces `service_start <= close`.
- **Independent Zero-Trust Validator (`backend/validator.py`):**
  - `validate_plan(scenario, plan)` recomputes order coverage, single-assignment per vehicle, load limits, stop-by-stop `arrival`, `service_start`, `departure`, `hard_window` compliance, `lateness`, `penalty_exposure`, `travel_cost`, and `objective` directly from raw scenario inputs without calling `domain.generate_routes`.

### 3.2 Caller-Declared Financial Dispatch (`backend/financial.py`)
- `solve_financial_dispatch(scenario, currency=..., operating_cost_per_travel_minute=..., penalty_rate_unit="currency_per_late_minute")` and `validate_financial_plan(...)`.
- Requires explicit 3-letter uppercase currency code and rate from the caller (no silent default currency/rates).
- **Enforced bounds:** $\le 6$ orders, $\le 8$ vehicles, $\le 3,000$ route bound, $\le 5,000$ deterministic search work units. Returns both `proxy_validation` and independent `financial_validation`.

### 3.3 Decision-Focused VQC Learning & Frozen Single-Request Inference
- **2-Qubit VQC Simulator & Gradients (`backend/quantum_learning.py`):**
  - Circuit (`debugleaf.vqc.dispatch.two-qubit.ry-cnot-z0.v1`): encodes normalized route features $(x_0, x_1) \in [0,1]^2$ via `RY(pi * x0)` on $q_0$, `RY(pi * x1)` on $q_1$, `CNOT(0, 1)`, trainable `RY(theta0)` on $q_0$, `RY(theta1)` on $q_1$, `CNOT(1, 0)`, and Pauli readout $\langle Z_0 \rangle$.
  - Propagates exact **parameter-shift circuit gradients** through $\langle Z_0 \rangle \to \sigma(-2\langle Z_0\rangle) \to$ smooth softmin decision surrogate over validated dispatch plans (verified against centered finite differences to $< 2.1 \times 10^{-10}$ max error).
- **Prospective Forecast-Separated Pipeline (`backend/prospective_dispatch.py`):**
  - Enforces strict leakage separation: features use only `forecast` travel duration and `nominal` window pressure; `realized` travel matrices are opened only post-selection for regret/SLA evaluation.
- **Single-Request VQC Inference (`backend/learning_inference.py`):**
  - `predict_dispatch(forecast_scenario, nominal_scenario, theta, model_version=..., execution_backend="reference" | "qpiai_local", financial_terms=...)`.
  - **Enforced bounds:** $\le 3$ orders, $\le 4$ vehicles, $\le 20$ candidate routes, $\le 500$ exact plans, $|\theta_i| \le 10$. Supports `hard_window` orders and optional `financial_terms`.
  - On unsupported size or validation error in default proxy `reference` mode, returns an independently validated classical exact plan with explicit `"fallback": {"used": true, "reason": ...}`. Requesting `execution_backend="qpiai_local"` or `financial_terms` never silently falls back.

### 3.4 Route-Selection QUBO, Ising & Warm-Start QAOA
- **QUBO & Ising Compiler (`backend/qubo.py`):**
  - `build_route_qubo(candidates, required_stops, coverage_penalty, vehicle_penalty)` encodes binary route indicators $x_i \in \{0,1\}$ with exact-once stop-coverage penalties and pairwise at-most-one-route-per-vehicle penalties.
  - `qubo_to_ising(qubo)` maps $x_i = (1 - z_i)/2$ to $E(z) = c + \sum_i h_i z_i + \sum_{i<j} J_{ij} z_i z_j$, preserving the exact constant offset $c$ (verified across all $2^n$ basis states and phase amplitudes $e^{-i\gamma E(x)}$).
- **Pure-Python Standard & Warm-Start QAOA (`backend/qaoa.py` & `backend/qaoa_dispatch.py`):**
  - `run_qaoa(...)` implements both uniform-$|+\rangle^{\otimes n}$ / $X$-mixer QAOA and warm-start QAOA initialized at $\bigotimes_i (\sqrt{1-p_i}|0\rangle + \sqrt{p_i}|1\rangle)$ with tilted single-qubit mixers $(2\sqrt{p_i(1-p_i)})X_i + (1-2p_i)Z_i$.
  - `solve_qaoa_dispatch(scenario, seed=..., shots=..., warm_start=..., angle_grid=...)` enforces $\le 12$ route variables on the API path, records every raw sampled bitstring, QUBO energy, greedy repair trace, and independent `validate_plan` check separately. Only an un-repaired **raw quantum sample** passing `validate_plan` is accepted as `qaoa_raw_sample`; otherwise classical exact solve is returned with `"fallback": {"used": true, ...}`.

### 3.5 Credential-Free Vendor SDK Adapters (`backend/qpiai_adapter.py`, `backend/qpiai_vqc.py`)
- Built for **QpiAI Opt `0.0.9`** + **QpiAI Quantum `0.3.0`** local `StatevectorSimulator` (no API keys read, no cloud/QPU calls).
- `run_qpiai_warm_start(...)` builds a `qpiai_opt` `Model` / `QUBOProblem`, verifies compiled QUBO terms + offset across all $2^n$ assignments ($\le 8$ verified variables), constructs `WarmStart` + `QAOA(reps=1)`, samples locally, and accounts for QpiAI's reversed bitstring endianness.
- `sdk_statevector` / `sdk_expectation_z0` / `sdk_risk_probability` runs the 2-qubit VQC on QpiAI's local `StatevectorSimulator` (verified to $8.21 \times 10^{-11}$ gradient accuracy).

### 3.6 Empirical Risk / CVaR Scoring & Disruption Recovery
- **Risk Replay (`backend/risk.py`):**
  - `evaluate_plan_risk(forecast_scenario, nominal_scenario, plan, realized_scenarios, cvar_alpha=...)` validates `plan` under `forecast_scenario` and replays it across $1\text{–}100$ realized travel matrices, computing SLA-breach frequencies, per-order late rates, expected proxy objective, and equal-weight fractional-tail $\text{CVaR}_\alpha$. Canonical hard-window breaches during replay are tracked as empirical hard-window violations while non-window validation errors fail closed.
- **Disruption Recovery & Audit Store (`backend/recovery.py`, `backend/recovery_store.py`):**
  - `recover_dispatch(original_scenario, original_plan, updated_scenario, locked_prefixes, change_penalty=0.0)` re-solves under updated travel matrices or vehicle removals ($\le 2,000$ routes, $\le 10,000$ nodes) while guaranteeing committed `locked_prefixes` remain intact.
  - `RecoveryAuditStore` persists append-only records in SQLite (`recovery_audits`), scoped by `scope_sha256` (hash of canonical original scenario + plan) so a failed recovery never overwrites the latest valid plan for that scope.

### 3.7 Durable SQLite Async Jobs & FastAPI HTTP Layer
- **Job Manager (`backend/jobs.py`):**
  - Persists classical (`submit`) and QAOA (`submit_qaoa`) jobs in WAL-mode SQLite (`.runtime/debugleaf.sqlite3` or `DEBUGLEAF_DB_PATH`).
  - Enforces a single-process PID lease (`runtime_lock`), bounded worker & queue admission (`max_workers=2` -> HTTP 429 `capacity_reached` when full), 30s timeout, cancellation suppression, SHA-256 request hashing, and startup recovery marking stale active jobs as `failed` (`restart_interrupted`).
- **FastAPI Surface (`backend/api.py` & `backend/auth.py`):**
  - Rejects unknown JSON keys on scenarios, vehicles, orders, and plans; sanitizes 422 errors to field paths only (never echoes secret/sentinel strings into responses or logs).
  - Guards synchronous CPU routes with a 2-slot nonblocking `BoundedSemaphore` (HTTP 429 when saturated).
  - Supports `DEBUGLEAF_AUTH_MODE=local` (default, unauthenticated localhost) and `DEBUGLEAF_AUTH_MODE=token` (requires distinct `DEBUGLEAF_DISPATCHER_TOKEN`, `DEBUGLEAF_RESEARCHER_TOKEN`, `DEBUGLEAF_ADMIN_TOKEN`; fails closed with HTTP 503 `auth_misconfigured` if misconfigured).

#### Summary of `/api/v1` Endpoints (`backend/api.py`)

| Method & Path | Purpose | Key Bounds / Status Codes |
| :--- | :--- | :--- |
| `GET /api/v1/health` | Liveness & API version check | `200 OK` (or `503` if token auth misconfigured) |
| `POST /api/v1/scenarios/validate` | Schema & domain bounds check | `200` valid / `422 invalid_scenario` |
| `POST /api/v1/scenarios/routes` | Enumerate capacity-feasible vehicle routes | $\le 8$ orders, $\le 16$ vehicles, $\le 10,000$ routes |
| `POST /api/v1/scenarios/plan` | Synchronous exact solve + independent validation | `200`, `409 infeasible`, `429`, `503 solver_budget_exceeded` |
| `POST /api/v1/scenarios/financial-plan` | Monetary exact solve + dual validation | $\le 6$ orders, $\le 8$ vehicles, explicit currency & rate |
| `POST /api/v1/scenarios/qaoa` | Synchronous local statevector QAOA + fallback | $\le 12$ route vars, $1\text{–}512$ shots, grid $2\text{–}10$ |
| `POST /api/v1/scenarios/vqc-plan` | Frozen 2-qubit VQC inference (`reference` / `qpiai_local`) | $\le 3$ orders, $\le 4$ vehicles, $\le 20$ routes, $|\theta_i|\le 10$ |
| `POST /api/v1/plans/validate` | Standalone zero-trust `{scenario, plan}` check | Recomputes ETAs, capacity, coverage, hard windows |
| `POST /api/v1/risk/evaluate` | Replay fixed plan across $1\text{–}100$ travel matrices | Reports empirical breach rates & optional `cvar_alpha` |
| `POST /api/v1/recovery/plan` | Disruption re-plan preserving `locked_prefixes` | Writes append-only SQLite audit before returning |
| `GET /api/v1/recovery/audits/latest` | Latest attempt & latest success (optional `scope_sha256`) | Failed attempts never clobber scoped latest success |
| `GET /api/v1/recovery/audits/{audit_id}` | Fetch durable recovery audit record by UUID | `200` or `404 audit_not_found` |
| `POST /api/v1/jobs` (alias `/scenarios/submit`) | Submit durable async classical exact-plan job | `202 Accepted` (or `429 capacity_reached`) |
| `POST /api/v1/jobs/qaoa` | Submit durable async local QAOA dispatch job | `202 Accepted`; requires explicit seed/shots/warm/grid |
| `GET /api/v1/jobs/{job_id}` | Poll async job status, metadata, plan & validation | `queued`, `running`, `succeeded`, `infeasible`, `failed`, `cancelled` |
| `POST /api/v1/jobs/{job_id}/cancel` | Cancel queued job or suppress running publication | `200` with updated job record |

---

## 4. V1.1 Quantum Optimization & Decision-Quality Engine (`experiments_v1_1/`)

Added on the `master` branch, `experiments_v1_1/` upgrades both the quantum learning and quantum combinatorial optimization layers (all pure Python, tested in `tests/test_v1_1_benchmarks.py`):

1. **Primal-Dual Augmented-Lagrangian LP Warm-Start QAOA (`experiments_v1_1/quantum_optimization.py`):**
   - Replaces V1's diagonal-folded box gradient relaxation (`x_i^2 = x_i` artifact) with a Lipschitz-normalized Primal-Dual Augmented-Lagrangian Set-Partitioning LP relaxation (`lp_augmented_relaxation`).
   - Adds a Cooley-Tukey $O(n 2^n)$ statevector simulator (`fast_qaoa_state`, verified to $< 10^{-11}$ probability discrepancy against `backend.qaoa.qaoa_state`) plus `p=1` spectral grid and `p=2` INTERP + coordinate pattern search.
   - **Measured 8-qubit dispatch result (512 shots, 5 seeds):** Lifts mean feasible shot probability from **6.21%** (Standard QAOA $p=1$) and **22.70%** (V1 Box Warm-Start $p=1$) to **60.12% ± 2.25%** (**59.81%** global-optimum shot probability).
2. **De-Saturated 4-Parameter VQC & 7-Way Financial Comparator (`experiments_v1_1/decision_quality.py`):**
   - Adds trainable affine observable readout `sigmoid(scale * <Z0> + bias)` to eliminate V1's `[0.119, 0.881]` sigmoid saturation trap, reducing held-out Brier score by **-28.6%** (`0.06567 -> 0.04692` across 420 route-scenarios).
   - Evaluates 7 end-to-end pipelines (`A–G`) combining Classical Newton GBDT, Standard VQC, and Decision-Focused VQC with Exact/MILP, Standard QAOA, and LP Warm-Start QAOA under explicit USD transport (`$0.50/min`), labor (`$0.25/min`), and SLA penalty terms. LP Warm-Start QAOA achieves **100% raw quantum feasibility (0% classical fallback)** across all 30 held-out test scenarios (vs 83.3% fallback for Standard QAOA).
3. **Exhaustive Mathematical Verifier (`experiments_v1_1/mathematical_verification.py`):**
   - Exhaustively certifies QUBO-to-business-objective equality, QUBO-to-Ising exactness, and positive constraint penalty margins across 4-qubit (16 states), 8-qubit (256 states), and 12-qubit (4,096 states) dispatch instances.
4. **Cross-Team Hybrid Artifact Contract (`experiments_v1_1/artifact_contract.py`):**
   - Implements `HybridDispatchArtifact` and `validate_hybrid_artifact` (`schema_version = "debugleaf.hybrid_artifact.v1.1"`) with canonical SHA-256 tamper verification (`experiments_v1_1/artifacts/exported_hybrid_contract_sample.json`).

---

## 5. Local Environment Setup & Exact Commands (Important Gotchas!)

### 5.1 Current `.venv` Quirks in This Checkout
1. **Invoke via `./.venv/bin/python -m ...` from the repo root:**  
   The existing `.venv` contains CPython `3.12` plus a lightweight offline `pytest.py` runner. Note that `.venv/lib/python3.12/site-packages/debugleaf.pth` contains a stale path from an earlier folder location (`/Users/karthikm/Desktop/29nmit/silver-computing-machine`). Running with `python -m ...` from `/Users/karthikm/Desktop/silver-computing-machine` puts the current working directory first on `sys.path` so everything resolves cleanly (or you can update `debugleaf.pth` to the current repo path).
2. **What runs with zero extra installs right now:**  
   All pure-Python core domain, validator, QUBO, QAOA, VQC, financial, risk, recovery, strong-baseline, and V1.1 benchmark suites (**82 tests across 10 test files**) run and pass out of the box in `./.venv`:
   ```bash
   ./.venv/bin/python -m pytest -q \
     tests/test_domain.py \
     tests/test_qubo.py \
     tests/test_qaoa_dispatch.py \
     tests/test_quantum_learning.py \
     tests/test_learning_inference.py \
     tests/test_financial.py \
     tests/test_risk.py \
     tests/test_recovery.py \
     tests/test_strong_baseline.py \
     tests/test_v1_1_benchmarks.py
   ```
   Re-running the V1.1 benchmarks or V1 standalone pure-Python runners also works immediately:
   ```bash
   ./.venv/bin/python -m experiments_v1_1.run_benchmarks
   ./.venv/bin/python -m experiments.run_quantum_learning
   ./.venv/bin/python -m experiments.run_dispatch_vqc
   ./.venv/bin/python experiments_qaoa/run_tiny.py
   ./.venv/bin/python experiments_matched_budget/validate_saved_results.py
   ```
   *(Note on macOS vs Windows line endings: `experiments_matched_budget/validate_saved_results.py` revalidates 90/90 plans with 0.0 objective difference, though raw file-byte hashes reflect Git `LF` checkout on macOS vs `CRLF` when originally frozen on Windows.)*

### 5.2 Installing Optional API (`FastAPI`) & Vendor SDK (`qpiai-*`) Runtimes
`tests/test_api.py`, `tests/test_auth.py`, `tests/test_recovery_persistence.py`, and `uvicorn` require `fastapi` / `pydantic` / `uvicorn` / `httpx` (which are not pre-installed in the minimal offline `.venv`). Likewise, `tests/test_qpiai_vqc.py` and `experiments_qaoa/run_qpiai_warm_start.py` require the isolated QpiAI SDK environment:

```bash
# 1. Full API + Test Virtual Environment (isolated in .venv-api or .venv)
uv venv --python 3.12 .venv-api
uv pip install --python .venv-api/bin/python \
  "fastapi" "pydantic>=2.0.0" "uvicorn" "httpx" "pytest>=8.0.0" "numpy" "scipy"

# Run the entire test suite (including FastAPI & SQLite persistence suites)
./.venv-api/bin/python -m pytest -q

# Start the localhost ArQ API server (keep bound to 127.0.0.1!)
./.venv-api/bin/python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000

# 2. Optional Isolated Vendor SDK Environment (.venv-sdk)
uv venv --python 3.12 .venv-sdk
uv pip install --python .venv-sdk/bin/python -r bootstrap/requirements-sdk-strict.txt
./.venv-sdk/bin/python -m pytest tests/test_qpiai_vqc.py -q
./.venv-sdk/bin/python experiments_qaoa/run_qpiai_warm_start.py
```

---

## 6. Known Boundaries, Frozen Artifacts & Next Steps for Incoming Agents

### Do NOT Overwrite These Frozen Study Artifacts
- `experiments/artifacts/prospective-dispatch-results.json` (locked under protocol SHA `60CDC7A0...`)
- `experiments_strong_baseline/results.json` and `experiments_strong_baseline/frozen_lock.json` (locked under protocol SHA `18D87045...`)
- `experiments_matched_budget/results.json` and `experiments_matched_budget/frozen_lock.json`

### Honest Capability & Evidence Boundaries
- **Simulator vs Cloud/QPU:** Every quantum result in this repo (`backend/quantum_learning.py`, `backend/qaoa.py`, `backend/qpiai_vqc.py`, `backend/qpiai_adapter.py`, `experiments_v1_1/`) is **credential-free local CPU exact statevector simulation**. QpiAI Cloud (`Client.solve`), remote cloud simulators, and physical QPUs have **NOT** been run and must not be invoked without explicit owner approval.
- **API Integration Gap for V1.1 Models:** The V1.1 4-parameter de-saturated VQC, Newton GBDT baseline, and LP-augmented Warm-Start QAOA (`p=1,2`) live in `experiments_v1_1/` and are validated via `tests/test_v1_1_benchmarks.py`. Wiring them as selectable solver/inference backends inside `backend/api.py` (while keeping V1 defaults backward-compatible) is a natural next engineering task.
- **Branch Convergence:** Merging `origin/main` (`3cb42b4`) into `master` will bring in the cooperative cancellation signals (`backend/jobs.py`) and live HTTP cancellation regression suite (`tests/test_cancellation.py`) without file conflicts.
- **Operational Scope:** Recovery (`backend/recovery.py`) performs static depot/time-zero replay with locked prefixes rather than live GPS/clock telemetry; scenarios and monetary rates are deterministic/synthetic benchmarks rather than calibrated live fleet telemetry; localhost auth (`backend/auth.py`) is a development RBAC prototype, so always bind local servers to `127.0.0.1`.

