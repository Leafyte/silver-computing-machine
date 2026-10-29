# Project Structure & Conventions — ArQ

> **Project Name:** **ArQ**  
> **Rule:** Treat `[EXISTING]` paths as ground truth; `[NEW]` paths mark clean extension points scheduled in `docs/IMPLEMENTATION_PLAN.md`.

---

## 1. Full Repository File Tree (`[EXISTING]` vs. `[NEW]`)

```text
silver-computing-machine/                  # Repository root (Project: ArQ)
├── AGENTS.md                              # [EXISTING] Mandatory agent operating, safety & claim rules
├── README.md                              # [EXISTING] Project overview & evidence boundaries
├── handover.md                            # [EXISTING] Fast technical orientation for AI agents
├── TERRA-CHECKPOINT.md                    # [EXISTING] Chronological V1 sprint verification log
├── pytest.ini                             # [EXISTING] Scoped pytest config (testpaths = tests)
│
├── backend/                               # [EXISTING] Core ArQ backend package
│   ├── __init__.py                        # [EXISTING] Package marker
│   ├── api.py                             # [EXISTING] FastAPI /api/v1/* endpoints & input canonicalizer
│   ├── auth.py                            # [EXISTING] Local mode + constant-time RBAC token middleware
│   ├── domain.py                          # [EXISTING] Scenario validator, route generator & exact solver
│   ├── validator.py                       # [EXISTING] Independent zero-trust plan & ETA recomputer
│   ├── financial.py                       # [EXISTING] Caller-declared currency exact solver & validator
│   ├── quantum_learning.py                # [EXISTING] 2-qubit VQC simulator & parameter-shift gradients
│   ├── qpiai_vqc.py                       # [EXISTING] QpiAI Quantum local StatevectorSimulator VQC adapter
│   ├── learning_dispatch.py               # [EXISTING] Diagnostic VQC + classical dispatch pipeline
│   ├── prospective_dispatch.py            # [EXISTING] Leakage-safe forecast/realized VQC training runner
│   ├── learning_inference.py              # [EXISTING] Bounded single-request frozen VQC inference
│   ├── qubo.py                            # [EXISTING] Route-cover QUBO builder & QUBO->Ising mapper
│   ├── qaoa.py                            # [EXISTING] Pure-Python Standard & Warm-Start QAOA simulator
│   ├── qaoa_dispatch.py                   # [EXISTING] QAOA route selector, repair audit & exact fallback
│   ├── qpiai_adapter.py                   # [EXISTING] Credential-free QpiAI Opt + Quantum QAOA adapter
│   ├── risk.py                            # [EXISTING] CRN travel-matrix replay & empirical CVaR scorer
│   ├── recovery.py                        # [EXISTING] Locked-prefix disruption recovery solver
│   ├── recovery_store.py                  # [EXISTING] Append-only SQLite recovery audit store
│   ├── jobs.py                            # [EXISTING] Durable SQLite WAL async job manager (exact & QAOA)
│   ├── artifact_contract.py               # [NEW - M1] Promoted v1.1 hybrid JSON schema & SHA-256 validator
│   ├── lp_relaxation.py                   # [NEW - M1] Promoted Primal-Dual Augmented-Lagrangian LP solver
│   └── ortools_routes.py                  # [NEW - M3] Optional offline OR-Tools candidate pruner (>8 orders)
│
├── experiments_v1_1/                      # [EXISTING] V1.1 Quantum Optimization & 7-Way Comparator Suite
│   ├── quantum_optimization.py            # [EXISTING] LP warm-start QAOA (p=1,2) & Cooley-Tukey simulator
│   ├── decision_quality.py                # [EXISTING] 4-param affine VQC, Newton GBDT & 7-way benchmark
│   ├── mathematical_verification.py       # [EXISTING] Exhaustive 4/8/12-qubit QUBO/Ising verifier
│   ├── artifact_contract.py               # [EXISTING] V1.1 cross-team hybrid contract implementation
│   ├── run_benchmarks.py                  # [EXISTING] Reproducible V1.1 benchmark CLI entrypoint
│   ├── results_v1_1.json                  # [EXISTING] Saved V1.1 benchmark metrics & resource profiles
│   ├── V1_1_OPTIMIZATION_REPORT.md        # [EXISTING] V1.1 benchmark tables & rejected ablations
│   └── artifacts/
│       └── exported_hybrid_contract_sample.json # [EXISTING] Validated sample v1.1 hybrid artifact
│
├── experiments/                           # [EXISTING] FROZEN V1 VQC gradient & prospective study artifacts
├── experiments_qaoa/                      # [EXISTING] 4-var pure-Python & 8-var QpiAI QAOA fixtures/results
├── experiments_qpiai_vqc/                 # [EXISTING] QpiAI local VQC decision-gradient check
├── experiments_strong_baseline/           # [EXISTING] FROZEN 5-replicate VQC vs Logistic vs CART study
├── experiments_matched_budget/            # [EXISTING] FROZEN CPU-ceiling study & 90/90 plan revalidator
│
├── fixtures/                              # [EXISTING] Deterministic tiny scenario inputs
│   ├── dispatch_tiny.json                 # [EXISTING] Canonical 2-vehicle, 3-order scenario fixture
│   └── dispatch_medium_10orders.json      # [NEW - M3] Deterministic 10-order fixture for OR-Tools pruning
│
├── tests/                                 # [EXISTING] Automated pytest / unittest verification suites
│   ├── stress_api.py                      # [EXISTING] Concurrency, timeout, cancel, restart & redaction check
│   ├── test_api.py                        # [EXISTING] FastAPI route, error-code & SQLite lifecycle tests
│   ├── test_auth.py                       # [EXISTING] Local vs RBAC bearer-token policy tests
│   ├── test_cancellation.py               # [EXISTING on origin/main -> M0] Cooperative HTTP cancel suite
│   ├── test_domain.py                     # [EXISTING] Scenario validation, route pruning & exact budget tests
│   ├── test_financial.py                  # [EXISTING] Monetary exact-cover & dual-validator tests
│   ├── test_inspection_and_probes.py      # [EXISTING] Environment & offline SDK probe tests
│   ├── test_learning_inference.py         # [EXISTING] Frozen VQC inference & qpiai_local parity tests
│   ├── test_qaoa_dispatch.py              # [EXISTING] QAOA statevector, warm-mixer & fallback tests
│   ├── test_qpiai_vqc.py                  # [EXISTING] QpiAI Quantum 0.3.0 VQC analytic gradient tests
│   ├── test_quantum_learning.py           # [EXISTING] Parameter-shift & prospective label semantics tests
│   ├── test_qubo.py                       # [EXISTING] Exhaustive QUBO & Ising offset equivalence tests
│   ├── test_recovery.py                   # [EXISTING] Locked-prefix disruption recovery unit tests
│   ├── test_recovery_persistence.py       # [EXISTING] SQLite scoped recovery audit & restart tests
│   ├── test_risk.py                       # [EXISTING] CRN scenario replay, CVaR & hard-window breach tests
│   ├── test_strong_baseline.py            # [EXISTING] Preregistration lock & leakage-isolation tests
│   ├── test_v1_1_benchmarks.py            # [EXISTING] V1.1 LP warm-start, 4-param VQC & contract tests
│   ├── test_v1_1_api.py                   # [NEW - M2] API tests for v1.1 LP-QAOA, 4-param VQC & artifacts
│   └── test_ortools_routes.py             # [NEW - M3] Offline OR-Tools candidate pruner & validator tests
│
├── tools/                                 # [EXISTING] Operational & diagnostic CLI utilities
│   ├── inspect_environment.py             # [EXISTING] Redacted OS/CPU/Python/package inventory script
│   ├── demo_cli.py                        # [NEW - M4] End-to-end terminal walkthrough & summary table CLI
│   └── sdk_probes/                        # [EXISTING] Standalone offline probes (ortools, qpiai_opt, quantum)
│
├── docs/                                  # [EXISTING] Architecture, planning & runtime specifications
│   ├── PRD.md                             # [NEW] Product requirements, personas, flows & scope
│   ├── ARCHITECTURE.md                    # [NEW] Component map, data flows, SQLite schema & Mermaid diagram
│   ├── TECH_STACK.md                      # [NEW] Pinned Python/SDK/API versions & excluded tech
│   ├── PROJECT_STRUCTURE.md               # [NEW] File tree, naming rules & layer import boundaries
│   ├── IMPLEMENTATION_PLAN.md             # [NEW] Prompt-sized milestones M0-M4, MVP cutoff & fallbacks
│   ├── API-RUNTIME.md                     # [EXISTING] Detailed /api/v1 endpoint & RBAC reference
│   ├── FINANCIAL.md                       # [EXISTING] Monetary scoring & dual-validation specification
│   ├── RECOVERY.md                        # [EXISTING] Locked-prefix static replay recovery specification
│   ├── RISK.md                            # [EXISTING] Empirical CRN scenario risk & CVaR specification
│   ├── SDK-INTERFACE-REQUIREMENTS.md      # [EXISTING] Proposed vendor SDK adapter contracts
│   ├── TERRA-V1-REPORT.md                 # [EXISTING] V1 acceptance evidence & refinement cycles
│   ├── V1-ACCEPTANCE.md                   # [EXISTING] Measurable V1 acceptance gates
│   └── V1-RUNBOOK.md                      # [EXISTING] Local environment & command runbook
│
├── bootstrap/                             # [EXISTING] Audited SDK requirement files & skill reviews
├── sources/                               # [EXISTING] Official QML-01 PDF extraction & legacy spec (read-only)
└── .runtime/                              # [GENERATED, GITIGNORED] Local SQLite DBs & pytest temp dirs
```

---

## 2. Naming Conventions

| Artifact Type | Convention | Concrete Examples in Repo |
| :--- | :--- | :--- |
| **Python Modules & Test Files** | `snake_case.py`; test files prefixed `test_` | `backend/qaoa_dispatch.py`, `backend/recovery_store.py`, `tests/test_v1_1_benchmarks.py` |
| **Public Functions** | Verb-first `snake_case`; private helpers prefixed `_` | `validate_scenario()`, `solve_exact()`, `predict_dispatch()`, `_canonical_scenario()` |
| **Classes, Dataclasses & Exceptions** | `PascalCase`; domain errors end in `Error` or `Exceeded` and expose `.code: str` + `.as_dict()` | `JobManager`, `RecoveryAuditStore`, `Qubo`, `HybridDispatchArtifact`, `InferenceError`, `SolveBudgetExceeded` |
| **Module Constants & Bounds** | `UPPER_SNAKE_CASE` declared at module top | `MAX_EXACT_ORDERS = 8`, `MAX_VERIFIED_QPIAI_VARIABLES = 8`, `CONTRACT_SCHEMA_VERSION` |
| **JSON Payload Keys & Error Codes** | `snake_case` keys; machine-readable `snake_case` error codes | `forecast_scenario`, `hard_window`, `cvar_alpha`, `"solver_budget_exceeded"`, `"capacity_reached"` |
| **Environment Variables** | `DEBUGLEAF_<SUBSYSTEM>_<NAME>` | `DEBUGLEAF_DB_PATH`, `DEBUGLEAF_AUTH_MODE`, `DEBUGLEAF_DISPATCHER_TOKEN` |
| **Documentation Files** | `UPPER-KEBAB-CASE.md` or `UPPER_SNAKE_CASE.md` in `docs/` | `docs/API-RUNTIME.md`, `docs/IMPLEMENTATION_PLAN.md` |

---

## 3. Rules for Where New Code Must Go

1. **Production Solvers, Validators & Adapters $\to$ `backend/`:**
   - Any solver, relaxation, VQC readout, or artifact validator exposed via `/api/v1/*` must live in `backend/` (not imported by `backend/` from `experiments*/`).
   - Keep `backend/` core modules (`domain.py`, `validator.py`, `qubo.py`, `qaoa.py`, `lp_relaxation.py`, `quantum_learning.py`, `artifact_contract.py`) **100% pure Python stdlib** so they always run without FastAPI, NumPy, or QpiAI installed.
2. **HTTP Request/Response Handling $\to$ `backend/api.py` (and `backend/auth.py`):**
   - Domain modules must never import `fastapi` or return `JSONResponse`. They accept plain dicts/primitives and return plain dicts/dataclasses or raise typed domain exceptions.
3. **Optional Vendor SDK Calls $\to$ Dedicated Leaf Adapters (`backend/qpiai_*.py`, `backend/ortools_routes.py`):**
   - Wrap third-party imports (`qpiai_opt`, `qpiai_quantum`, `ortools`) inside function-level `try ... except ImportError` blocks so importing `backend.api` succeeds even when `.venv-sdk` packages are absent.
4. **Research Benchmarks & Multi-Model Studies $\to$ `experiments_v1_1/` (or a new `experiments_<name>/` folder):**
   - Never overwrite locked JSON files in `experiments/`, `experiments_strong_baseline/`, or `experiments_matched_budget/`. Write new benchmark outputs to versioned files under `experiments_v1_1/artifacts/` or a new experiment directory.
5. **Tests $\to$ `tests/test_<feature>.py`:**
   - Every new backend module or endpoint requires a matching `tests/test_<feature>.py` discoverable by `pytest.ini` (`testpaths = tests`), using temporary SQLite paths via `tmp_path` (never mutating `.runtime/debugleaf.sqlite3` during tests).

---

## 4. Strict Layer Import & Dependency Rules

Imports must flow strictly top-down (no circular imports and **never** from `backend/` into `experiments*/` or `tests/`):

```text
Layer 5 (Presentation / CLI):   tools/*, experiments*/*
                                   │ (may import Layers 1–4)
                                   ▼
Layer 4 (HTTP & Auth):          backend/api.py, backend/auth.py
                                   │ (may import Layers 1–3; ONLY layer allowed to import fastapi)
                                   ▼
Layer 3 (Persistence/Orchestr.): backend/jobs.py, backend/recovery_store.py,
                                backend/qaoa_dispatch.py, backend/learning_inference.py
                                   │ (may import Layers 1–2)
                                   ▼
Layer 2 (Solvers & Adapters):   backend/financial.py, backend/risk.py, backend/recovery.py,
                                backend/quantum_learning.py, backend/qaoa.py, backend/lp_relaxation.py,
                                backend/qpiai_vqc.py, backend/qpiai_adapter.py, backend/ortools_routes.py
                                   │ (may import Layer 1; vendor SDKs lazy-imported inside adapters only)
                                   ▼
Layer 1 (Foundation & Trust):   backend/domain.py, backend/validator.py,
                                backend/qubo.py, backend/artifact_contract.py
```

### Critical Invariants to Enforce in Code Reviews
- **`backend/validator.py` independence:** `validator.py` may call `domain.validate_scenario`, but must **never** call `domain.generate_routes`, `domain._route_metrics`, or any solver/QUBO helper to compute arrival times, loads, or objectives.
- **No `backend/*` $\to$ `experiments*/*` imports:** When promoting V1.1 capabilities in Milestone `M1`, move the canonical pure-Python implementations into `backend/` (`backend/lp_relaxation.py`, `backend/artifact_contract.py`, `backend/qaoa.py`, `backend/quantum_learning.py`) and have `experiments_v1_1/` re-export or import from `backend/`.
- **Secret & Input Redaction:** Error payloads emitted by `backend/api.py` or stored in `jobs.error_json` / `recovery_audits.failure_json` must contain only static schema field paths and machine-readable codes—never raw user strings, headers, or environment variable values.

