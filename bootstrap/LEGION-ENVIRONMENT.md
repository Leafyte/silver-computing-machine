# Legion Isolated Environment and Capability Smoke Verification

**Date:** 2026-10-28 (Asia/Kolkata)  
**Host / Device Label:** `Klein` (Lenovo Legion Pro 5 16AFR10)  
**Project Checkout Root:** `C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine-main`  
**Isolated Environment Path:** `C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine-main\.venv`  
**Execution Agent:** Antigravity  
**Authority:** Bounded reversible environment setup under `AGENTS.md` and `prompts/ANTIGRAVITY-LEGION.md`

---

## 1. Environment Details

| Field | Value |
|---|---|
| **Environment Type** | Project-local virtual environment (`venv`) |
| **Path** | `C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine-main\.venv` |
| **Creation Command** | `C:\Users\avkir\AppData\Local\Python\pythoncore-3.12-64\python.exe -m venv .venv` |
| **Venv Python Executable** | `.venv\Scripts\python.exe` |
| **Python Version** | Python 3.12.10 (64-bit AMD64, MSC v.1943) |
| **Pip Version** | pip 25.0.1 |
| **Lockfile Path** | `bootstrap/requirements-legion.lock` (53 packages, exact versions and SHA-256 hashes) |

---

## 2. Package Resolution and Skew Analysis

### Resolved Core Packages

| Package | Version | Wheel / Source Artifact | Purpose |
|---|---|---|---|
| `qpiai-opt[quantum]` | `0.0.9` | `qpiai_opt-0.0.9-cp312-cp312-win_amd64.whl` | Optimization modeller, QUBO/Ising converters, and gate-based orchestration |
| `qpiai-quantum` | `0.3.0` | `qpiai_quantum-0.3.0-py3-none-any.whl` | Quantum circuit representations and local statevector simulator |
| `ortools` | `9.15.6755` | `ortools-9.15.6755-cp312-cp312-win_amd64.whl` | Classical combinatorial and linear/CP-SAT solver baseline |

### Version Skew Investigation & Rationale

- **Published Skew:** The public PyPI repository hosts `qpiai-quantum 0.7.5` (released 2026-10-13) and `qpiai-opt 0.0.9` (released 2026-10-26).
- **Metadata Constraint:** In `qpiai-opt 0.0.9` distribution metadata, the optional `[quantum]` extra strictly specifies `qpiai-quantum==0.3.0` (`Requires-Dist: qpiai-quantum==0.3.0; extra == "quantum"`).
- **Resolution Behavior:** Pip dependency resolution strictly honors this constraint. Attempting to install `qpiai-opt[quantum]` alongside `qpiai-quantum==0.7.5` produces a dependency conflict (`ResolutionImpossible`).
- **Rationale:** The integration modules in `qpiai_opt.quantum` (`core.py`, `algorithms.py`, `_shared.py`) interface directly with `qpiai_quantum` (specifically `Circuit`, `QpiAIQuantumAuth`, `QubitHamiltonian`, `StatevectorSimulator`). To maintain documented integration compatibility and satisfy the prompt rule (*"Do not silently override the integration pin; only use a compatible documented pairing and record rationale"*), the official pairing declared by `qpiai-opt[quantum]==0.0.9` with `qpiai-quantum==0.3.0` was selected and installed.
- **Minimal Footprint:** No heavy optional frameworks (PyTorch, TensorFlow, JAX, CUDA runtimes, Jupyter notebook suites) were installed. The installed environment consists strictly of the 53 required wheels.

---

## 3. Stage C — Deterministic Smoke Verifications

All smoke checks were executed locally in the `.venv` environment without requiring, requesting, reading, or transmitting any credentials, tokens, or private keys.

### Smoke 1: Interpreter and Package Imports

- **Goal:** Verify that the isolated environment interpreter correctly loads all core modules and the integration subpackage.
- **Exact Command:**
  ```powershell
  & "C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine-main\.venv\Scripts\python.exe" -c "import sys, importlib.metadata, qpiai_quantum, qpiai_opt, qpiai_opt.quantum, ortools; print('Python:', sys.version); print('qpiai_quantum:', importlib.metadata.version('qpiai-quantum')); print('qpiai_opt:', importlib.metadata.version('qpiai-opt')); print('ortools:', importlib.metadata.version('ortools')); print('Imports successful!')"
  ```
- **Exit Status:** `0`
- **Output:**
  ```text
  Python: 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]
  qpiai_quantum: 0.3.0
  qpiai_opt: 0.0.9
  ortools: 9.15.6755
  Imports successful!
  ```
- **Classification:** **Proves import capability** of all required packages and the `qpiai_opt.quantum` integration module without namespace collisions or missing C extensions.

---

### Smoke 2: QpiAI Quantum Circuit Construction

- **Goal:** Verify that `qpiai_quantum.Circuit` can instantiate, accept quantum gates (H, CX), register operations, compute circuit depth/size, and emit standard representations.
- **Exact Command:**
  ```powershell
  & "C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine-main\.venv\Scripts\python.exe" -c "from qpiai_quantum import Circuit; qc = Circuit(2, name='bell_state'); qc.h(0); qc.cx(0, 1); print('Name:', qc.name); print('Qubits:', qc.num_qubits); print('Depth:', qc.depth()); print('Size:', qc.size()); print('QASM:\n' + qc.to_qasm())"
  ```
- **Exit Status:** `0`
- **Output:**
  ```text
  Name: bell_state
  Qubits: 2
  Depth: 2
  Size: 2
  QASM:
  OPENQASM 2.0;
  include "qelib1.inc";
  qreg q[2];
  creg c[2];
  h q[0];
  cx q[0], q[1];
  ```
- **Classification:** **Proves circuit construction and intermediate representation** in the installed `qpiai-quantum` SDK.

---

### Smoke 3: QpiAI Quantum Local Simulator Execution

- **Goal:** Verify that the built-in local statevector simulator executes quantum circuits deterministically without cloud authentication.
- **Exact Command:**
  ```powershell
  & "C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine-main\.venv\Scripts\python.exe" -c "from qpiai_quantum import Circuit; from qpiai_quantum.simulator import StatevectorSimulator; qc = Circuit(2, name='bell_state'); qc.h(0); qc.cx(0, 1); sim = StatevectorSimulator(); result = sim.run(qc); print('Simulator:', sim.name); print('Statevector:', result.statevector); print('Execution time:', round(result.execution_time, 4)); print('Message:', result.message)"
  ```
- **Exit Status:** `0`
- **Output:**
  ```text
  Simulator: QpiAI-QSV-Local
  Statevector: [(0.7071067811865475+0j), 0j, 0j, (0.7071067811865475+0j)]
  Execution time: 0.8258
  Message: Job executed successfully
  ```
- **Classification:** **Proves local simulator execution**. The simulation ran entirely locally via `QpiAI-QSV-Local`, returning the expected Bell state $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ with zero network requests or credential checks.

---

### Smoke 4: Opt Model / QUBO Construction, Credential-Free Feasibility Inspection, and QAOA Circuit Build

- **Goal:** Verify that `qpiai_opt` constructs optimization models, supports credential-free feasibility inspection and constraint violation analysis, converts models to QUBO format, and compiles gate-based QAOA circuits via `qpiai_opt.quantum.build_circuit`.
- **Exact Command:**
  ```powershell
  & "C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine-main\.venv\Scripts\python.exe" -c "import qpiai_opt.opt_modeller as om; from qpiai_opt.quantum import QAOA, build_circuit; from qpiai_opt.opt_solvers.qubo import QUBOProblem; model = om.Model('dispatch_toy'); x0 = om.Binary('x0'); x1 = om.Binary('x1'); model.add_variable(x0); model.add_variable(x1); model.minimize(2 * x0 + 3 * x1); model.add_constraint(x0 + x1 == 1, label='selection_constraint'); print('Model:', model.name); print('Variables:', [v.name for v in model.variables()]); print('Feasible (x0=1, x1=0):', model.is_feasible({'x0': 1, 'x1': 0})); print('Feasible (x0=1, x1=1):', model.is_feasible({'x0': 1, 'x1': 1})); print('Violations:', model.constraint_violations({'x0': 1, 'x1': 1})); qubo_dict, offset = model.to_qubo_dict(); print('QUBO dict:', qubo_dict); print('QUBO offset:', offset); qubo = QUBOProblem(num_variables=2, qubo_dict={(0, 0): -4.0, (0, 1): 3.0, (1, 0): 3.0, (1, 1): -3.0}, sense='MIN', offset=6.0); qaoa = QAOA(reps=1); qaoa_circuit = build_circuit(qubo, algorithm=qaoa, parameters=[0.2, 0.3]); print('QAOA Circuit Qubits:', qaoa_circuit.num_qubits); print('QAOA Circuit Depth:', qaoa_circuit.depth()); print('QAOA Circuit Size:', qaoa_circuit.size())"
  ```
- **Exit Status:** `0`
- **Output:**
  ```text
  Model: dispatch_toy
  Variables: ['x0', 'x1']
  Feasible (x0=1, x1=0): True
  Feasible (x0=1, x1=1): False
  Violations: {'selection_constraint': 1.0}
  QUBO dict: {(0, 0): -4.0, (0, 1): 6.0, (1, 0): 6.0, (1, 1): -3.0}
  QUBO offset: 6.0
  QAOA Circuit Qubits: 2
  QAOA Circuit Depth: 4
  QAOA Circuit Size: 6
  ```
- **Classification:** **Proves model construction, offline feasibility evaluation, QUBO matrix compilation, and QAOA circuit generation**. Does *not* invoke authenticated Opt cloud solving or QCloud execution.

---

### Smoke 5: OR-Tools Local Toy Optimization

- **Goal:** Verify that Google OR-Tools CP-SAT solver functions locally and deterministically.
- **Exact Command:**
  ```powershell
  & "C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine-main\.venv\Scripts\python.exe" -c "from ortools.sat.python import cp_model; model = cp_model.CpModel(); x = model.new_int_var(0, 10, 'x'); y = model.new_int_var(0, 10, 'y'); model.add(x + 2 * y <= 14); model.add(x - y <= 2); model.maximize(3 * x + 4 * y); solver = cp_model.CpSolver(); status = solver.solve(model); print('Status:', solver.status_name(status)); print('Optimal objective:', solver.objective_value); print(f'Solution: x={solver.value(x)}, y={solver.value(y)}')"
  ```
- **Exit Status:** `0`
- **Output:**
  ```text
  Status: OPTIMAL
  Optimal objective: 34.0
  Solution: x=6, y=4
  ```
- **Classification:** **Proves local toy optimization**. Solves a classical 2-variable integer problem to proven optimality in milliseconds without external calls.

---

## 4. Smoke Verification Summary Table

| Check ID | Target Component | Command Summary | Exit Code | Verified Outcome | What It Proves |
|---|---|---|---|---|---|
| **S1** | Package Imports | Import `qpiai_quantum`, `qpiai_opt`, `qpiai_opt.quantum`, `ortools` | 0 | All packages load cleanly | Environment import integrity |
| **S2** | Quantum Circuit | Construct Bell circuit in `qpiai_quantum.Circuit` | 0 | 2 qubits, depth 2, OpenQASM 2.0 output | Quantum circuit modeling |
| **S3** | Local Simulator | Execute Bell circuit on `QpiAI-QSV-Local` simulator | 0 | Statevector $(1/\sqrt{2})|00\rangle + (1/\sqrt{2})|11\rangle$ | Offline local simulation |
| **S4** | Opt Model / QUBO / QAOA | Formulate binary model, inspect feasibility, convert QUBO, build QAOA circuit | 0 | Feasibility verified, QUBO extracted, 2-qubit QAOA compiled | Model construction & gate compile |
| **S5** | Classical Solver | Solve 2-var integer model via OR-Tools CP-SAT | 0 | Status `OPTIMAL`, obj 34.0 ($x=6, y=4$) | Local classical toy solve |

---

## 5. Explicit Boundary and Scope Statement

In strict compliance with `AGENTS.md` and the bootstrap handoff instructions:
- **No algorithm implementation** (e.g., dispatch routing, risk models, QAOA parameter optimization) was performed.
- **No model training** or data fitting was executed.
- **No benchmark suite** or comparative performance run was performed.
- **No remote QCloud, QPU, or Opt API credentials** were accessed, requested, transmitted, or utilized.
- **No global/system changes**, administrative actions, driver updates, or registry modifications were made.
- **No project-level claims** of quantum advantage, novelty, or dispatch feasibility are made.
