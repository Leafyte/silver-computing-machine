# Technology Stack — ArQ

> **Project Name:** **ArQ**  
> **Version Provenance:** Pinned from `bootstrap/requirements-sdk-strict.txt`, `bootstrap/requirements-sdk.txt`, `bootstrap/SDK-EVIDENCE.md`, `docs/V1-RUNBOOK.md`, and `pytest.ini`.

---

## 1. Core & Optional Technology Matrix

| Layer | Technology | Pinned Version | Environment | Why Chosen |
| :--- | :--- | :--- | :---: | :--- |
| **Language** | Python (CPython, 64-bit) | `3.12.x` (`3.12.0`–`3.12.14`) | All (`.venv*`) | Matches binary wheel support across `qpiai-opt`, `ortools`, `numba`, and `cvxpy` on macOS (`arm64`) and Windows (`x86_64`); avoids CPython `3.14` wheel gaps. |
| **Core Math, VQC & QAOA Engine** | Python Standard Library (`math`, `cmath`, `itertools`, `random`, `hashlib`, `tracemalloc`) | Built-in (`3.12`) | `.venv` / `.venv-api` | Zero external dependencies for `backend/domain.py`, `validator.py`, `quantum_learning.py`, `qubo.py`, `qaoa.py`, and `experiments_v1_1/*`; guarantees instant startup and deterministic statevector math on `4–12` qubits. |
| **HTTP API Framework** | FastAPI | `0.162.0` | `.venv-api` | Lightweight ASGI routing (`backend/api.py`), clean exception handlers, automatic OpenAPI `/docs` generation, and direct handler invocation in socket-restricted sandboxes. |
| **Schema / Data Validation** | Custom Whitelist Validators + Pydantic | `2.14.1` (`pydantic-core==2.50.1`) | `.venv-api` / `.venv-sdk` | FastAPI request body parsing paired with explicit non-echoing whitelist validators in `backend/api.py` and `backend/domain.py` so submitted secret/sentinel values are never echoed in 422 payloads. |
| **ASGI Server** | Uvicorn | `0.54.0` | `.venv-api` | Fast single-process loopback server (`127.0.0.1:8000`) aligned with ArQ's SQLite single-writer PID lease (`runtime_lock`). |
| **Database & Persistence** | SQLite 3 (`sqlite3`, WAL mode) | Built-in (`3.12`) | All | Zero-config ACID durability (`.runtime/debugleaf.sqlite3`) for async `jobs`, `recovery_audits`, and single-process `runtime_lock` without running an external DB daemon. |
| **Concurrency** | `threading` (`BoundedSemaphore`, `RLock`) + `concurrent.futures.ThreadPoolExecutor` | Built-in (`3.12`) | All | Deterministic 2-slot synchronous compute gate + 2-worker async job queue with cooperative cancellation and deadline suppression. |
| **Quantum Optimization SDK** | QpiAI Opt (`qpiai-opt`) | `0.0.9` | `.venv-sdk` | Official Quant-A-Maze 3.0 optimization SDK providing `Model`, `Binary`, `to_qubo_dict`, `QUBOProblem`, `WarmStart`, `QAOA`, and `build_circuit` (`backend/qpiai_adapter.py`). |
| **Quantum Circuit & Simulator SDK** | QpiAI Quantum (`qpiai-quantum`) | `0.3.0` (strict `[quantum]` extra) / `0.7.5` (standalone compatible) | `.venv-sdk` | Official Quant-A-Maze 3.0 quantum SDK providing `Circuit` and local `StatevectorSimulator` (`backend/qpiai_vqc.py`, `backend/qpiai_adapter.py`) with zero cloud credentials required. |
| **Classical Routing & MILP** | Google OR-Tools + HiGHS (`ortools`, `highspy`) | `ortools==9.15.6755`, `highspy==1.15.1` | `.venv-sdk` | 100% offline CP-SAT and vehicle-routing (`pywrapcp`) solvers verified via `tools/sdk_probes/probe_ortools.py` for scaling candidate generation beyond 8 orders (`M3`). |
| **Scientific Transitive Stack** | NumPy, SciPy, CVXPY, NetworkX, Numba | `numpy==2.5.4`, `scipy==1.18.1`, `cvxpy==1.9.3`, `networkx==3.7`, `numba==0.68.1` | `.venv-sdk` | Required transitive dependencies for `qpiai-quantum`'s multi-core `StatevectorSimulator` and `qpiai-opt`'s algebraic modeler. |
| **Package & Env Manager** | `uv` | Latest stable (`>=0.5`) | Host CLI | Fast, isolated `.venv-api` and `.venv-sdk` provisioning without mutating global Python or user `PATH`. |
| **Testing & Stress Harness** | `pytest` + `httpx` + `unittest` | `pytest==9.2.0` (`>=8.0.0`), `httpx==0.28.1` | `.venv-api` / `.venv` | Runs all 17 test modules under `tests/` (`pytest.ini`) plus bounded concurrency/recovery stress checks (`tests/stress_api.py`). Offline `.venv` also includes a stdlib `pytest` shim (`8.3.0` API subset) for zero-network runs. |
| **Deployment Target** | Localhost Loopback Process (`127.0.0.1`) | macOS `15.x` (`arm64`) / Windows `11` (`x86_64`) | Local CPU | Enforces local-first reproducibility, protects unauthenticated `local` mode from network exposure, and satisfies zero-cost CPU simulation rules. |

---

## 2. Isolated Virtual Environment Strategy

Because `qpiai-opt[quantum]==0.0.9` pins `qpiai-quantum==0.3.0` in its extra metadata while base `qpiai-opt==0.0.9` dynamically imports `qpiai-quantum==0.7.5` cleanly (documented in `bootstrap/SDK-EVIDENCE.md`), ArQ separates environments cleanly:

1. **`.venv` (Zero-Install Offline Core):** Pre-populated stdlib + lightweight `pytest` compatibility runner. Runs all 82 pure-Python domain, validator, VQC, QUBO, QAOA, financial, risk, recovery, and V1.1 benchmark tests in `< 1s`.
2. **`.venv-api` (FastAPI + Full Test Suite):** Contains `fastapi`, `pydantic`, `uvicorn`, `httpx`, and full `pytest` for running `backend.api:app` on `127.0.0.1:8000` and executing all 150+ unit, API, auth, cancellation, and persistence tests.
3. **`.venv-sdk` (Vendor Quantum/OR-Tools Runtime):** Provisioned from `bootstrap/requirements-sdk-strict.txt` (for exact `0.3.0` extra parity) or `bootstrap/requirements-sdk.txt` (for `0.7.5` compatibility) when exercising `execution_backend="qpiai_local"` or `backend/qpiai_adapter.py`.

---

## 3. Explicitly NOT Used (and Why)

| Excluded Technology | Why It Is Explicitly Not Used |
| :--- | :--- |
| **Qiskit / PennyLane / Cirq in `backend/`** | Challenge Track I (`QML-01`) scores QpiAI SDK integration (`qpiai-opt` + `qpiai-quantum`). Adding heavyweight third-party quantum frameworks to `backend/` increases dependency bloat without scoring rubric credit, whereas our custom pure-Python statevector + Cooley-Tukey simulators provide exact zero-dependency baselines and analytical gradient verification. |
| **PyTorch / JAX / TensorFlow** | VQC circuits here have `2–4` trainable parameters on `2` qubits, and QAOA circuits have `2–4` variational angles (`p=1,2`) on `4–12` qubits. Exact parameter-shift differentiation and spectral/INTERP coordinate pattern search are deterministic, faster on tiny CPU circuits, and require zero GPU/CUDA drivers. |
| **scikit-learn (`sklearn`) in Production Path** | Wheel downloads stalled during restricted V1 sprint verification (`TERRA-CHECKPOINT.md`), and non-linear baselines were implemented directly in pure Python (`Grouped-Bootstrap CART` in `experiments_strong_baseline/` and exact second-order `Newton GBDT / XGBoost` in `experiments_v1_1/decision_quality.py`) for 100% deterministic, zero-dependency execution. |
| **PostgreSQL / Redis / Celery** | Single-node SQLite in `WAL` mode + `ThreadPoolExecutor(max_workers=2)` satisfies durable async job polling, crash-recovery auditing, and bounded concurrency with zero background daemons or Docker requirements. |
| **QpiAI Cloud (`Client.solve`) / Indus-1 QPU (Default)** | `qpiai_opt.Client` requires online API key verification and QCloud QPU runs consume credits. Per `AGENTS.md`, paid/authenticated remote runs are prohibited without explicit owner authorization; all quantum paths default to local `StatevectorSimulator` or pure-Python statevector execution. |
| **JWT / OAuth2 / External KMS** | Out of scope for a localhost-bound (`127.0.0.1`) research & dispatch prototype; `backend/auth.py` implements constant-time `hmac.compare_digest` bearer tokens from process environment variables for local role separation (`dispatcher`, `researcher`, `admin`). |

