# Implementation Plan — ArQ

> **Project Name:** **ArQ**  
> **Baseline State:** Core V1 backend (`backend/*`, 16 `/api/v1/*` endpoints, 153 tests) and V1.1 Quantum Optimization & 7-Way Comparator (`experiments_v1_1/*`, `tests/test_v1_1_benchmarks.py`) are **already implemented and verified**.  
> **Plan Scope:** Prompt-sized milestones (`M0`–`M4`) to converge `origin/main` + `master`, promote V1.1's high-feasibility LP Warm-Start QAOA (`p=1,2`), 4-parameter de-saturated VQC, and `debugleaf.hybrid_artifact.v1.1` contract into the live FastAPI backend, add optional OR-Tools medium-instance scaling, and ship a one-command demo harness.

---

## Milestone Overview & Dependency Order

```text
M0 (Env & Branch Convergence)
  └──► M1 (Promote V1.1 Core Modules into backend/)
         └──► M2 (Wire V1.1 Solvers & Artifact Contract into FastAPI + RBAC)  ◄── [MINIMUM VIABLE DEMO CUTOFF]
                ├──► M3 (Optional OR-Tools Top-K Route Pruner & 12-Var SDK Cap)
                └──► M4 (Interactive Demo CLI, Benchmark Summary & Final Audit)
```

---

## M0: Environment Provisioning & Branch Convergence

**Goal:** Provision reproducible `.venv-api` (and optional `.venv-sdk`) environments, fix the stale `.pth` path in `.venv`, and cleanly merge `origin/main` (`3cb42b4`, cooperative job cancellation) into `master` (`dcf59c6`, V1.1 quantum optimization).

### Tasks
- [ ] **M0.1 — Provision `.venv-api` and fix local path resolution:** Update `.venv/lib/python3.12/site-packages/debugleaf.pth` to the current repo root and create `.venv-api` via `uv venv --python 3.12 .venv-api` with `fastapi==0.162.0`, `pydantic==2.14.1`, `uvicorn==0.54.0`, `httpx==0.28.1`, and `pytest==9.2.0`.
  - *Files touched:* `.venv/lib/python3.12/site-packages/debugleaf.pth` (untracked/gitignored), `.venv-api/` (gitignored).
- [ ] **M0.2 — Merge `origin/main` cooperative cancellation into `master`:** Merge `origin/main` (`c457ecd`, `cd47028`, `3cb42b4`) into `master` so `backend/jobs.py` cooperative cancellation events and `tests/test_cancellation.py` coexist with `experiments_v1_1/` and `tests/test_v1_1_benchmarks.py`.
  - *Files touched:* `backend/jobs.py`, `tests/test_api.py`, `tests/test_cancellation.py`.
- [ ] **M0.3 — Run baseline regression & stress audit:** Execute full `pytest` and `tests/stress_api.py` in `.venv-api` without modifying any frozen JSON artifacts in `experiments*/`.
  - *Files touched:* None (verification only; verify `git status` shows zero modified frozen artifacts).

**Done When:**
```bash
./.venv-api/bin/python -m pytest -q && ./.venv-api/bin/python tests/stress_api.py
```
passes `100%` of collected tests (including `test_cancellation.py` and `test_v1_1_benchmarks.py`) with `0` failures.

---

## M1: Promote V1.1 Quantum & Contract Engines into `backend/`

**Goal:** Promote V1.1's pure-Python mathematical breakthroughs out of `experiments_v1_1/` into first-class `backend/` Layer 1–3 modules while keeping 100% backward compatibility with all existing V1 callers and re-exporting in `experiments_v1_1/` so no benchmark scripts break.

### Tasks
- [ ] **M1.1 — Promote Hybrid Artifact Contract (`backend/artifact_contract.py`):** Move `HybridDispatchArtifact`, `canonical_sha256`, `CONTRACT_SCHEMA_VERSION` (`"debugleaf.hybrid_artifact.v1.1"`), and `validate_hybrid_artifact` into `backend/artifact_contract.py`; re-export from `experiments_v1_1/artifact_contract.py`.
  - *Files touched:* `backend/artifact_contract.py` `[NEW]`, `experiments_v1_1/artifact_contract.py`.
- [ ] **M1.2 — Promote Primal-Dual LP Relaxation (`backend/lp_relaxation.py`):** Extract `lp_augmented_relaxation(qubo, steps=1200)` into `backend/lp_relaxation.py` (pure Python stdlib) and add `relaxation_method: Literal["box_v1", "lp_augmented"] = "box_v1"` support to `backend/qaoa.py` and `backend/qaoa_dispatch.py`.
  - *Files touched:* `backend/lp_relaxation.py` `[NEW]`, `backend/qaoa.py`, `backend/qaoa_dispatch.py`, `experiments_v1_1/quantum_optimization.py`.
- [ ] **M1.3 — Add Cooley-Tukey Fast Statevector & Depth `p=2` Support to `backend/qaoa.py`:** Integrate `fast_qaoa_state` ($O(n 2^n)$ bitwise butterfly gates) and optional `layers: int = 1` (`1` or `2`, with INTERP/coordinate pattern search when `layers=2`) into `backend/qaoa.py` and `backend/qaoa_dispatch.py`, preserving exact `< 1e-11` amplitude parity with V1 on `layers=1`.
  - *Files touched:* `backend/qaoa.py`, `backend/qaoa_dispatch.py`.
- [ ] **M1.4 — Add 4-Parameter Affine-Readout VQC Support (`backend/quantum_learning.py` & `backend/learning_inference.py`):** Extend `risk_probability` and `predict_dispatch` (plus `backend/qpiai_vqc.py`) to accept either legacy 2-angle `theta = [theta0, theta1]` (`sigmoid(-2 <Z0>)`) or V1.1 de-saturated 4-parameter `theta = [theta0, theta1, scale, bias]` (`sigmoid(scale * <Z0> + bias)`), recording readout architecture in `model_provenance`.
  - *Files touched:* `backend/quantum_learning.py`, `backend/qpiai_vqc.py`, `backend/learning_inference.py`, `tests/test_learning_inference.py`.

**Done When:**
```bash
./.venv-api/bin/python -m pytest tests/test_qubo.py tests/test_qaoa_dispatch.py tests/test_quantum_learning.py tests/test_learning_inference.py tests/test_v1_1_benchmarks.py -q
```
passes all existing V1 tests plus new unit assertions verifying `relaxation_method="lp_augmented"`, `layers=2`, 4-parameter `theta`, and `backend.artifact_contract.validate_hybrid_artifact`.

---

## M2: Expose V1.1 Capabilities on FastAPI & Async Jobs

**Goal:** Wire LP Warm-Start QAOA (`p=1,2`), 4-parameter VQC inference (including optional `labor_cost_per_duty_minute` financial accounting), and hybrid artifact validation into `/api/v1/*` and `backend/auth.py` with full negative/stress test coverage.

### Tasks
- [ ] **M2.1 — Extend Synchronous & Async QAOA Endpoints:** Allow optional `"relaxation": "box_v1" | "lp_augmented"` and `"layers": 1 | 2` on `POST /api/v1/scenarios/qaoa` and `POST /api/v1/jobs/qaoa` (persisting both fields in `jobs.settings_json` and `request_sha256`).
  - *Files touched:* `backend/api.py`, `backend/jobs.py`.
- [ ] **M2.2 — Enable 4-Param VQC & Optional Duty-Labor Term on `/api/v1/scenarios/vqc-plan` & `/financial-plan`:** Accept 2- or 4-element `theta` (`|theta_i| <= 20` for readout scale/bias) on `/api/v1/scenarios/vqc-plan`, and support optional `labor_cost_per_duty_minute: float` (default `0.0` when omitted) in `backend/financial.py` with independent `validate_financial_plan` recomputation.
  - *Files touched:* `backend/api.py`, `backend/financial.py`, `backend/learning_inference.py`.
- [ ] **M2.3 — Add `POST /api/v1/artifacts/validate` Endpoint & RBAC Policy:** Expose `backend/artifact_contract.validate_hybrid_artifact` at `POST /api/v1/artifacts/validate` (returning non-echoing structural/SHA-256 errors on tamper/mismatch) and authorize `researcher`, `dispatcher`, and `admin` roles in `backend/auth.py`.
  - *Files touched:* `backend/api.py`, `backend/auth.py`.
- [ ] **M2.4 — Add Dedicated V1.1 API & Persistence Test Suite (`tests/test_v1_1_api.py`):** Test HTTP `200`/`202`/`422` flows for `lp_augmented` QAOA jobs, 4-parameter VQC inference, hybrid artifact tamper rejection, RBAC role enforcement, and sentinel secret redaction.
  - *Files touched:* `tests/test_v1_1_api.py` `[NEW]`, `docs/API-RUNTIME.md`.

**Done When:**
```bash
./.venv-api/bin/python -m pytest -q && ./.venv-api/bin/python tests/stress_api.py
```
passes `100%` of tests including `tests/test_v1_1_api.py`, and submitting `experiments_qaoa/dispatch_fixture.json` with `relaxation="lp_augmented"` to `/api/v1/jobs/qaoa` succeeds with `origin == "qaoa_raw_sample"`, `fallback.used == False`, and `validation.valid == True`.

---

## === MINIMUM VIABLE DEMO CUTOFF (End of M2) ===

> **Cutoff Definition:** At the completion of **M2**, ArQ has a complete, crash-safe, independently validated hybrid quantum-classical backend running over HTTP (`127.0.0.1:8000`) with durable SQLite async jobs, cooperative cancellation, V1.1 de-saturated 4-parameter VQC inference, V1.1 LP Warm-Start QAOA (`60.1%` 8-qubit feasible shot rate, `0%` held-out fallback), caller-declared financial validation, CRN risk/CVaR replay, locked-prefix disruption recovery, and cross-team hybrid artifact validation. Milestones `M3` and `M4` are value-add extensions and demo polish.

---

## M3: Offline OR-Tools Candidate Pruner & 12-Variable QpiAI Adapter Scaling

**Goal:** Enable medium-scale scenarios (`9–15` orders) to route through the 12-qubit QAOA pipeline by pruning candidate routes offline with Google OR-Tools (`pywrapcp`) and raising the verified QpiAI local adapter cap from `8` to `12` variables.

### Tasks
- [ ] **M3.1 — Implement Optional OR-Tools Candidate Pruner (`backend/ortools_routes.py`):** Build `generate_top_k_routes(scenario, max_candidates=12, seed=42)` using lazy-imported `ortools.constraint_solver.pywrapcp` (diverse arc-cost perturbations + capacity/window checks) returning routes in the exact dictionary format of `backend.domain.generate_routes`, failing closed with `SDK_NOT_INSTALLED` if `ortools` is absent.
  - *Files touched:* `backend/ortools_routes.py` `[NEW]`, `fixtures/dispatch_medium_10orders.json` `[NEW]`.
- [ ] **M3.2 — Verify & Raise QpiAI Adapter Variable Cap to 12 (`backend/qpiai_adapter.py`):** Raise `MAX_VERIFIED_QPIAI_VARIABLES` from `8` to `12` ($2^{12} = 4,096$ exhaustive energy checks take `< 0.05s` in Python) and add an optional `lp_augmented` warm-start test on the 12-variable prospective instance in `.venv-sdk`.
  - *Files touched:* `backend/qpiai_adapter.py`, `tests/test_ortools_routes.py` `[NEW]`, `tests/test_qaoa_dispatch.py`.

**Done When:**
```bash
./.venv-sdk/bin/python -m pytest tests/test_ortools_routes.py tests/test_qpiai_vqc.py -q
```
passes with `100%` of OR-Tools-pruned QAOA dispatches validated independently by `backend.validator.validate_plan` on `fixtures/dispatch_medium_10orders.json`.

---

## M4: End-to-End Demo CLI & Final Documentation Sync

**Goal:** Provide a single deterministic CLI script (`tools/demo_cli.py`) that exercises every major ArQ flow (Classical Exact, 4-Param Decision VQC, LP Warm-Start QAOA, CRN Risk/CVaR, and Locked-Prefix Disruption Recovery) and prints a judge-ready comparison table with independent validator badges.

### Tasks
- [ ] **M4.1 — Build Self-Contained ArQ Demo CLI (`tools/demo_cli.py`):** Support both in-process mode (works in zero-install `.venv`) and `--base-url http://127.0.0.1:8000` HTTP mode (works against running Uvicorn), outputting formatted tables of vehicle routes, stop ETAs, USD costs, QAOA feasible-shot rates, CVaR exposure, recovery diffs, and validator pass/fail status.
  - *Files touched:* `tools/demo_cli.py` `[NEW]`.
- [ ] **M4.2 — Final Documentation & Evidence Audit:** Update `handover.md` and `README.md` with exact `demo_cli.py` commands, verify `git diff` leaves all frozen V1 study files and `_archive/` untouched, and run `tools/inspect_environment.py`.
  - *Files touched:* `tools/demo_cli.py`, `handover.md`, `README.md`.

**Done When:**
```bash
./.venv/bin/python tools/demo_cli.py && ./.venv-api/bin/python -m pytest -q
```
executes end-to-end in `< 5` seconds, prints `VALIDATOR: PASS` across every flow, and exits `0`.

---

## Risks & Fallback Plans

| Risk | Likelihood / Impact | Concrete Mitigation & Fallback Plan |
| :--- | :---: | :--- |
| **R-1: `qpiai-opt[quantum]` vs `qpiai-quantum` pip conflict (`0.3.0` vs `0.7.5`)** | Known / Medium | Already analyzed in `bootstrap/SDK-EVIDENCE.md`: use `bootstrap/requirements-sdk-strict.txt` (`0.3.0`) for strict extra parity or `bootstrap/requirements-sdk.txt` (`0.7.5` without `[quantum]` extra). Fallback: all core VQC and LP-QAOA paths run identically on the built-in pure-Python statevector engine (`execution_backend="reference"`). |
| **R-2: Statevector CPU blowup beyond 12–14 route variables** | High if unbounded / High | `api.py` and `qaoa_dispatch.py` hard-cap QAOA at `12` route variables (`4,096` states, `~0.5s` CPU for 100 grid evals using Cooley-Tukey `fast_qaoa_state`). Fallback: larger instances return `422 unsupported_capability` (or prune to top-12 candidates via `ortools_routes.py` in `M3`, with exact `solve_exact` fallback up to `100,000` work units). |
| **R-3: Zero feasible raw QAOA shots on tight `hard_window` scenarios** | Medium / Low | LP Warm-Start QAOA (`p=1,2`) already raised 8-qubit feasibility to `60.1%`, but when all finite shots violate a hard close, `qaoa_dispatch.py` never publishes a classically repaired sample as quantum—it invokes `solve_exact` and sets `"fallback": {"used": true, "reason": "no_domain_valid_raw_sample"}`. |
| **R-4: Sandbox / OS loopback socket restrictions blocking live HTTP tests** | Observed on Windows & Sandboxes / Low | Keep FastAPI route logic testable via direct handler calls (`tests/test_api.py`, `tests/stress_api.py`) so CI/sandbox verification passes without OS sockets, while running live `uvicorn` HTTP checks (`--base-url`) when outside sandbox. |
| **R-5: Accidental mutation of frozen V1 benchmark artifacts** | Low / High (violates `AGENTS.md`) | Never run `experiments/run_prospective_dispatch.py`, `experiments_strong_baseline/run_experiment.py`, or `experiments_matched_budget/run_experiment.py` against their locked output paths; `tests/test_strong_baseline.py` and `experiments_matched_budget/tests/` assert SHA-256 lock integrity automatically. |

