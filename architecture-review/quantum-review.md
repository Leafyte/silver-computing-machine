# Quantum and optimization review

**Review date:** 2026-10-30 (Asia/Calcutta)<br>
**Checkout:** main, HEAD 2c7a4fb809c58a9ec02b6cf2d7b9a8a8bacb1c95; clean at review start. Sibling trainingv3 worktree untouched.<br>
**Scope:** QML sections I/J, quantum/solver capability matrix, exact QUBO math, independent validator coverage, preregistered refinements, QpiAI setup gates and handoff package. No implementation, training, cloud request or QPU request was run.<br>
**Usage:** get_usage_limits returned 9% weekly consumed (91% remaining) before review/check batches.

## Decision

Keep route optimization classical-first with the bounded local reference QAOA and QpiAI Opt model/circuit adapter as auditable experimental paths. Always validate a raw candidate through backend.validator.validate_plan; accept it as QAOA-selected only after independent validation. Preserve exact classical fallback and provenance. Do not deploy an ETA VQC, claim quantum advantage, or infer cloud/QPU execution from imports, compiled models, circuit construction or local simulation.

The India ETA 4-qubit residual VQCs and the active V1 dispatch decision-focused VQC are separate systems. RUN2 India predicts a residual for a whole-delivery-duration predictor; it does not decide routes, calculate dispatch risk, or create QUBOs. Its frozen operational fusion is alpha=0, so the operational model is GBDT-only. The active V1 circuits target synthetic dispatch decision scores and are not operationally calibrated either.

## Capability and evidence matrix

| Capability | Classification | Evidence and limits |
|---|---|---|
| Custom Python QAOA statevector | Implemented; local reference execution | backend/qaoa.py:3-4,53-90,132-200 implements bounded statevector, one QAOA layer, grid search and locally sampled bitstrings. Current focused tests passed. Not a QpiAI SDK or cloud call. |
| Exact classical dispatch | Implemented; local | backend/domain.py:159-200,227-257 builds capacity-feasible ordered routes and bounded exact-cover search. It returns infeasible/budget outcomes rather than publishing an unproved plan. Separate from QpiAI Opt. |
| Independent plan validation | Implemented; used by route QAOA adapter | backend/qaoa_dispatch.py:15-31,85-98,113-128 validates raw and repaired samples, accepts only raw samples passing QUBO and domain checks, and explicitly falls back to exact classical dispatch. backend/validator.py:16-116 recomputes coverage, timing, capacity and objective. |
| QUBO construction/energy | Implemented; exhaustive small-instance coverage | backend/qubo.py:37-96,98-117,131-160. Existing tests compare all 16 assignments in the four-variable fixture with direct penalty arithmetic and verify Ising energies. |
| QpiAI Opt model/QUBO compilation | Implemented in adapter; historical bounded execution evidence | backend/qpiai_adapter.py:19-75 constructs Model, compiles to QUBO, checks all assignments (cap 8 vars), then builds Opt warm-start QAOA. Adapter does not instantiate Client or call hosted solve. docs/TERRA-V1-REPORT.md and docs/V1-RUNBOOK.md record local SDK QUBO compilation equality over 256 assignments. |
| QpiAI Opt hosted solve | Absent / externally gated | SDK inspection says Client.solve needs online key verification; no authenticated Opt solve is evidenced. Model construction or local circuit execution is not an Opt solve. |
| QpiAI Quantum local statevector | Historically executed; not rerun in this review | RUN2 India report records 4-qubit local StatevectorSimulator parity against independent NumPy oracle, max Z error 6.67e-16 on eight validation rows, with no cloud/QPU use. V1 reports local SDK VQC parity and local Opt-generated warm-start circuit execution. |
| Authenticated QpiAI QCloud simulator | Not evidenced / external setup required | backend/qpiai_service.py:161-193 contains an optional cloud submission branch, but code presence is not successful invocation. Reports say no authenticated QCloud job. The bounded qpiai_adapter is local only. |
| Actual QPU | Not run; outside present authorization | No QPU evidence. AGENTS.md disallows paid jobs without specific owner authorization. |
| India ETA VQC in operation | Not active | RUN2_INDIA_QML_FINAL_TRAINING_REPORT.md: validation did not pass predeclared GBDT/Ridge gate; alpha=0. Original retrospective test was exposed, not pristine. |
| Quantum advantage | Absent | No evidence supports this claim. Exact tiny classical solver and classical controls remain mandatory. |

### QpiAI gates

1. **Offline reproducibility (available):** use pinned isolated .venv-sdk; record qpiai-opt/qpiai-quantum versions; run deterministic QUBO <=8 vars; compare SDK polynomial over all 2^n assignments; execute warm-start circuit with local simulator; decode preserved bit order; independently validate. This is local SDK/QAOA sampling, not hosted Opt.
2. **Hosted Opt (not configured/evidenced):** obtain owner-approved account/key and service-cost confirmation; inject key only through documented secret environment; invoke actual hosted solve on the frozen tiny problem. Record key presence only, SDK versions, request/job ID, status, assignment, wall/solver/queue times, and recomputed objective/feasibility. Never expose key. If key/license absent, mark blocked and retain local/exact route.
3. **QCloud simulator (not run):** with explicit authorization for authenticated non-QPU call, submit deterministic four-qubit parity circuit to a QpiAI cloud simulator; preserve job ID/backend/shots/submit, queue, compute and total times; compare with local oracle using predeclared sampling tolerance. Do not silently fall back and report cloud success. qpiai_service.py:189-193 includes raw exception text; sanitize before public exposure.
4. **QPU (not authorized):** separate explicit owner decision after queue/credit/cost limits are known. Simulator evidence is not a substitute.

## India ETA four-qubit VQC assessment

Source: experiments_india_vqc4/circuit.py; training/release evidence: RUN2_INDIA_QML_FINAL_TRAINING_REPORT.md. The circuit has 4 qubits and 5 blocks. Each block re-uploads all 4 prepared features with fixed RY(pi*x), then per-qubit trainable RX, forward ring CNOTs, trainable RY, reverse CNOTs, trainable RZ. Fixed readout applies RY(pi/7) then RX(pi/9) on each qubit; four Pauli-Z values feed four affine weights plus bias. The two rotations do not commute, avoiding a bare-Z readout gauge. Features must already be finite and in [0,1] (circuit.py:60-69); this is not raw feature engineering.

- **Parameter count:** baseline 5*4*3=60 quantum angles plus 5 classical readout values = 65 total. extended76 adds 4*4=16 RX angles after blocks 0-3: 76 quantum + 5 readout = 81 total. Source: circuit.py:30-37,123-153,171-178.
- **Gate totals:** report records 128 baseline / 144 extension: 20 uploads + 60 trainable single-qubit rotations + 40 CNOTs + 8 readout gates = 128; extension adds 16. Ordered logical depths 62/66 are sequence-derived, not transpiled hardware depth.
- **Encoding and expressivity:** repeated uploads make features available at each block, at added parameterization and optimization cost. The 76-angle version is an exact zero-extension of 60-angle at initialization. This establishes nested functions, not better generalization or quantum advantage.
- **Gradient/Jacobian:** historical numerical acceptance reports centered-difference checks, dense statevector/oracle agreement below 2e-12, and observable-Jacobian local rank 60/60 and 76/76 on 64 train-OOF rows at zero extension. Fitted-weight check reports rank 60/76 and nonzero projected extension directions. This argues against simple local duplicate columns at these evaluated points; it does not prove global identifiability, good conditioning or unique parameters. Singular-value condition number is not deployment acceptance evidence. New gradient/Jacobian checks were not run in this review.
- **Training/validation:** 29,592 courier-grouped train OOF; selection has 4,124 records/117 couriers; calibration has 1,865/58. Selection MAE/RMSE (minutes): GBDT 3.172583/3.932094; Ridge residual 3.172207/3.931903; baseline60 3.170207/3.932810; extended76 3.171809/3.933179. Gate required strictly better MAE and no worse RMSE than both classical references; neither circuit passed. Operational alpha remained 0. Test metrics (5,941 rows/176 groups) are retrospective and exposed: baseline60 3.147268/3.969599; extended76 3.144121/3.967810; GBDT 3.146740/3.968180. Do not tune or claim fresh-holdout performance from this split.
- **Classical control and calibration:** Ridge's 0.000377-minute selection MAE gain (~0.023 sec) is tiny and its median AE is slightly worse; experimental only. Alpha frozen at zero. MAE/RMSE and scenario risk do not establish calibrated uncertainty.
- **Runtime:** 4,176 updates, 76.61 sec summed epoch time / 76.85 sec original wall. Resource reproduction: 77.106 sec wall, 615.078 CPU sec over 8 threads, 860,368,896-byte peak working set. CPU chosen after tiny matched profile gave 16.90 ms/update CPU vs 56.19 ms/update CUDA; not production throughput. This review ran no inference performance benchmark.
- **Semantics:** target is Kaggle Time_taken(min) mapped to whole-delivery delivery_duration_min; VQC predicts its residual. It is not road-edge travel time or route-leg duration. Report notes order_to_pickup_min_imp is post-pickup and weather/traffic/rating/workload fields do not all have established timestamps. Do not promise a pre-pickup ETA from these features.

## Ranked refinements with predeclared protocols

All comparisons are exploratory unless a genuinely new untouched courier-group or temporal evaluation set is acquired and frozen before training/selection. Do not reuse original test. Match GBDT and Ridge on identical IDs, preprocessing, seed/scenarios and compute budget; report paired courier-group bootstrap, MAE/RMSE, distance/congestion/workload strata, wall time and memory. Keep operational GBDT unless a separate protocol passes.

| Rank | Hypothesis and motivation | Exact change and compute | Control, acceptance and stop |
|---|---|---|---|
| 1. Remove repeated uploads | Five data-upload blocks may add depth/unstable parameters without courier-group gain; full circuits failed operational gate. | Add upload_blocks setting; compare 1, 3, 5 uploads while preserving RX/RY/RZ motif and same four features. Three architectures x three fixed seeds, same 8/10-epoch caps. Historical two-model run took ~77 sec; budget ~6 min for nine candidates, then record actual runtime. | Compare with frozen GBDT/Ridge on new holdout. Accept only paired-group MAE CI below both and RMSE CI no worse than either, without material subgroup regression. Stop after three seeds if joint gate fails or compute bound reached. With no new holdout publish exploratory only; alpha=0. |
| 2. Readout and conditioning | Four Z observables plus affine head may be unnecessary/poorly conditioned; fitted local full rank does not imply robust optimization. | Compare four-Z head with one prespecified Z and two prespecified observables; retain circuit/training setup. Extend Jacobian report with singular values, rank tolerance and condition number. Three configurations x three seeds, same epoch cap. | Compare GBDT/Ridge plus classical linear/Ridge readout on identical rows and new holdout. Require same joint MAE/RMSE gate and finite-difference gradient checks. Stop on failed gradient check, materially worse conditioning or predictive gate failure. |
| 3. Residual shrinkage | Fitted correction may add noise even if some features correlate with residual; alpha=0 is correct policy now. | Prespecify alpha {0, 0.1, 0.25, 0.5, 1}; select only inside training/inner groups. Optional readout L2 is a separate registered experiment. No circuit change; inference/validation sweep after fit. | New untouched group evaluation against GBDT and Ridge; alpha>0 must be selected internally and show MAE improvement with RMSE noninferiority to both. Stop at alpha=0 if gate fails; never select from exposed test. |
| 4. Paired group uncertainty/strata | Sub-second differences may be noise or subgroup-specific; row averages hide courier dependence. | No model retraining: paired courier-cluster bootstrap on saved train/validation predictions and residual strata; likely <1 min, measure it. Do not read test labels. | Same keyed rows for VQC/GBDT/Ridge; MAE primary, RMSE guardrail, strata and multiplicity handling frozen first. Diagnostic only without new holdout. Stop if split/prediction hashes mismatch; no old test reopening. |

Priority is new untouched evaluation data, not more architecture search. If none exists, preserve research outputs and operational alpha=0.

## QUBO formulation, penalty scaling and validator coverage

For binary x_i, route cost c_i, required stop s and vehicle v, backend/qubo.py:37-96 constructs:

E(x) = sum_i c_i*x_i + Pc*sum_s(sum_{i: s in route_i} x_i - 1)^2 + Pv*sum_v sum_{i<j: vehicle_i=vehicle_j=v} x_i*x_j.

Expansion: offset Pc*|S|; linear coefficient c_i - Pc*m_i, where m_i is number of required stops in route i; quadratic coefficient 2*Pc for each shared-stop pair plus Pv for same-vehicle pair (terms add if both relations hold). This exact formula penalizes missing/duplicate coverage and multiple routes assigned to one vehicle. It does not encode capacity, windows, ordered timing, existing assignment/completed-stop locks, or approved plan version. Candidate facts must be enforced by generation and independently before commit.

Default penalty is 1 + sum_i c_i, with nonnegative costs required by builder. For this additive nonnegative objective it exceeds the maximum possible objective difference between subsets, so if a feasible cover exists, one violated constraint costs more than any possible objective improvement. Caller-supplied penalties can be arbitrarily smaller; the builder does not prove they suffice. Mathematical audit includes a concrete sweep: penalty 1.5 produced infeasible ground state while 29 restored feasible ground state for its fixture (tests/test_v1_1_benchmarks.py and experiments_v1_1/mathematical_verification.py). This bound does not establish appropriate scaling for additional financial terms or mixed units.

qubo_to_ising substitutes x_i=(1-z_i)/2 and retains the full constant. backend/qaoa.py:53-90 applies the complete Ising energy including offset in cost phase. Existing tests cover amplitude-level offset phase, analytical tilted-mixer expectation, exhaustive QUBO/Ising equivalence, SDK QUBO mapping for every assignment through 8 vars, variable-order mapping, and independent validation of decoded samples. In the route fixture, direct business cost is just the caller-supplied route cost.

Current dispatch cost does not yet equal challenge's validated transport + labor + financial service-risk exposure. V1 domain uses duration/late exposure, while docs/FINANCIAL.md and report identify currency as caller-defined/synthetic. qaoa_dispatch uses route objective as QUBO cost. Before adding terms, define units and independently recompute them in validator. Soft windows remain eligible with lateness exposure; explicit hard windows are filtered/rejected. Coverage/vehicle uniqueness are QUBO penalties then hard validation. Capacity/order are enforced upstream and rechecked. Assignment locks, completed-stop prefixes and approval/version are not represented in base QUBO; encode as candidate hard constraints or enforce outside the optimizer before publication.

## Checks run in this review

Command (main checkout):

    $env:PYTHONDONTWRITEBYTECODE='1'
    .\.venv\Scripts\python.exe -m pytest -q --basetemp=.runtime\pytest-quantum-audit tests/test_qubo.py tests/test_qaoa_dispatch.py

Observed: 15 passed in 0.13s. Executed deterministic QUBO/Ising, warm start, bit-order, sample decode/domain validator and fallback coverage. Adapter size-cap test did not submit cloud work. WindowsApps python alias failed to launch once; documented project venv executable worked. Existing QpiAI parity/cloud outcomes cited here are historical, not re-executed. No training or expensive benchmark.

## Worker packages

### Luna / optimization backend

- Inputs: AGENTS.md, docs/V1-ACCEPTANCE.md, docs/TERRA-V1-REPORT.md, docs/V1-RUNBOOK.md, this review and current scenario/validator/QUBO contracts.
- Own: backend/qubo.py, backend/qaoa_dispatch.py, backend/qpiai_adapter.py and focused tests/test_qubo.py/tests/test_qaoa_dispatch.py. Coordinate before touching domain.py/validator.py.
- Deliver: unit-bearing objective, bounded exact-vs-QAOA-vs-SDK-local comparisons, independently validated raw outputs, explicit fallback, actual execution/timing labels and preserved samples. No hosted call absent authorization.
- Verify: .venv\Scripts\python.exe -m pytest -q --basetemp=.runtime\pytest-quantum-audit tests/test_qubo.py tests/test_qaoa_dispatch.py; SDK-local only through documented offline .venv-sdk command.
- Effort/dependency: medium; depends on objective units and candidate/validator contract. Roll back only owned files.
- Done: exhaustive QUBO/offset match, exact tiny optimum/objective, every published sample independently valid, explicit failure/fallback, no unsupported cloud/QPU claims.

### Dell Codex / model integration

- Inputs: final India ETA training report, acceptance report, feature-availability audit, release manifest.
- Own: model loader and feature contracts assigned by coordinator; do not edit weights, original test split or unrelated API/UI.
- Deliver: configured-only SHA-verified CPU release loading, named/order-checked 59-feature prepared contract, research-only residual diagnostic, source/as-of/stage/version metadata, alpha=0 operational behavior. No raw feature engineering claim absent timestamp availability.
- Verify release hashes/reference parity without editing bundle, then bounded adapter/API and deployment latency/memory checks on authorized resources.
- Effort/dependency: medium; needs runtime and prepared schema agreement. Rollback: disable new endpoint/config and retain v1 fixture.
- Done: manifest/file/schema checked before trusted load, stable request errors, operational output exactly matches frozen GBDT.

### Mobile/Gemini handoff

No UI or mobile repository was inspected/mutated here. Consume coordinator's versioned backend contract; phones do not own credentials, assignment, approval or route mutation.

### Copy-paste prompt for optimization worker

> Work only in assigned backend optimization files. First read AGENTS.md, docs/V1-ACCEPTANCE.md, docs/TERRA-V1-REPORT.md, docs/V1-RUNBOOK.md and architecture-review/quantum-review.md. Compare exact classical optimum, existing classical fallback, custom local QAOA and QpiAI Opt-compiled warm-start local simulation on deterministic tiny instances. Preserve variable order, all 2^n compiled energy checks, full Ising offset, raw bits, repairs separately, independent validator results and execution/time labels. Treat capacity, time windows, assignment locks and approval as hard conditions outside penalty-only assumptions and recheck before publication. Do not call QpiAI Client, QCloud or QPU absent owner authorization. Run only bounded focused checks; report commands, outcomes, unsupported cases, measured times, rollback and owned files.



## Correction and evidence lineage (2026-10-30 follow-up)

The model-serving audit clarifies artifact integrity and runtime status:

- The published RUN2 release ZIP is intact: SHA-256 18d5d4859e53d3d59063e60cdcdd45cb6eff6757bba6474f024cdde91aa06b80 matches the final training report and all 11 inventoried members match their manifest entries.
- The unpacked release directory in the main working tree is not the same artifact: ten of eleven files match; only circuit.py differs (working copy 30,294 bytes / SHA-256 e347bade7e5a9081a08417aaa36fb8c390ea52bb02a6c44f9b9c7dfa9c0903d5 versus manifest 29,701 bytes / SHA-256 801468f544e25ae913c26e1f2131cf5f88b384b8495172da62be9051d5a96ba0). The release loader's fail-closed rejection is correct. Do not edit a manifest to bless the unpacked mismatch.
- Therefore historical training, circuit, Jacobian and QpiAI-local parity evidence applies to the frozen artifact/code whose hash is in the accepted report/archive, not automatically to the currently unpacked experiments_india_vqc4/circuit.py. Current source semantics were inspected, but this review did not verify its equivalence with the frozen trained release. Reproduce from a fresh administrator-controlled extraction of the verified ZIP if needed.
- Current .venv has NumPy 2.5.4 and no scikit-learn; release pins NumPy 1.26.4, scikit-learn 1.4.2 and joblib 1.6.0. No compatible RUN2 runtime was present at model audit. Neither joblib load nor prediction was executed. Thus the ZIP is integrity-verified, while deployment/model-serving readiness remains **NOT READY** pending fresh extraction, isolated exact dependencies, loader checks and reference parity.
- No release repair, dependency install or model deserialization was performed by this worker.

### Training and validation loss evidence

The checked-in historical artifacts/india_eta_vqc4_run2/training_curves.csv records the loss/selection metrics per epoch. Training objective was MSE in minutes squared; model-selection validation was reported as MAE/RMSE in minutes (there is no separate validation MSE in this log).

| Candidate | Epochs | Train MSE first -> last | Best train MSE | Best model-selection point | Best selection MAE / RMSE |
|---|---:|---:|---:|---:|---:|
| baseline60 | 8 | 15.134359 -> 15.119830 | 15.119830 at epoch 8 | epoch 3 by selection MAE | 3.170207 / 3.932810 min |
| extended76 | 10 | 15.113954 -> 15.096295 | 15.096295 at epoch 10 | epoch 5 by selection MAE | 3.171809 / 3.933179 min |

The selected early-stopping checkpoint is baseline60 epoch 3 and extended76 epoch 5; the lowest training MSE occurs later, which is not itself model selection. Validation MAE/RMSE fluctuate around the displayed bests; neither circuit met the RMSE guardrail against both frozen classical references. Do not interpret the later training MSE decrease as evidence of improved generalization.

### QML topic coverage and refinement protocol fields

The ranking table above states hypotheses, motivation, edits, compute bound, controls, protocol, acceptance and stop rules. To make expected information gain explicit per proposal:

| Refinement | Expected information gain |
|---|---|
| Upload-block ablation | Tests whether full repeated encoding depth contributes courier-group predictive value or only added circuit/parameter cost. |
| Readout/conditioning ablation | Separates quantum-feature contribution from classical-head complexity and identifies weak/ill-conditioned local directions. |
| Residual shrinkage | Determines whether VQC signal can be safely used at any predeclared scale or whether alpha=0 remains warranted because corrections add variance. |
| Paired group uncertainty/strata | Quantifies uncertainty of observed differences and locates error concentration without another fit or any exposed-test reopening. |

Additional circuit evidence: tests/test_india_vqc4_circuit.py checks Torch autograd against centered finite differences for every quantum angle and readout value on both variants using step 1e-5 radians and maximum absolute error tolerance 2e-5; the test also checks Torch/NumPy expectation parity at 2e-12. The historical suite result was 10 passed, 9.67 s, with local QpiAI simulator parity. These checks do not constitute a deployment-machine runtime benchmark or current unpacked-source parity because the current circuit file hash differs from the frozen acceptance/release hash.

Depth convention also needs precision: RUN2_INDIA_QML_FINAL_TRAINING_REPORT.md reports ordered sequence dependency depth 62/66 with CNOTs kept in listed order. experiments_india_vqc4/independent_numerical_acceptance.json reports ideal parallel logical DAG depths 42/46 by scheduling edge-disjoint CNOTs in two layers. These are distinct scheduling conventions, not a contradiction about hardware depth; neither is a transpiled or QPU measurement.

### Updated release recovery gate

Use the published ZIP (verified hash above), extract to a fresh immutable administrator-controlled directory, verify every member and loader pin again, then create/use an isolated Python 3.12 runtime with the release's exact dependency pins. Run release reference parity and ordinary integrity tests. Do not serve until runtime and readiness pass. No trained model / local circuit candidate becomes operational from this recovery; alpha=0 remains frozen, and any VQC comparison remains diagnostic/exploratory pending untouched data and the preregistered gate.



Refinement edit locations for implementation traceability (none were edited in this review):
- Upload-block ablation: experiments_india_vqc4/circuit.py configuration/forward path; experiments_india_vqc4/train.py frozen run configuration; add an isolated acceptance runner rather than mutating published checkpoints.
- Readout/conditioning: experiments_india_vqc4/circuit.py readout computation; tests/test_india_vqc4_circuit.py finite-difference/Jacobian tests; write new evidence JSON under a new run ID.
- Residual shrinkage: experiments_india_vqc4/train.py or a new separately versioned fusion-evaluation module; do not edit release fusion_policy.json or reselect from the exposed test.
- Group uncertainty/strata: add a read-only analysis script under experiments_india_vqc4/ and a new report artifact; it must consume only the already authorized train/model-selection/calibration prediction tables and validate key/split hashes before use.
