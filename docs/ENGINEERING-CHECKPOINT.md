# Engineering Checkpoint

**Date:** 2026-10-29  
**Checkout:** main at 9c00fa1 with RUN 1 work and concurrent engineering artifacts in the shared working tree.  
**External artifacts:** The pre-existing modified experiments/artifacts/quantum-learning-results.json was snapshotted and left unchanged. Other concurrently produced benchmark/data/notebook artifacts were preserved.

## Current RUN 1 result

The complete bounded run is .runtime/run1/20261029T083317Z_run1_v2_full/. It trained V1.1-compatible VQCs and classical baselines on synthetic Solomon/Dabia replay labels at n=25/50/100, with 60 predictive updates, 24 decision updates, and eight replay seeds. Native route generation avoided BKS. The train/validation/test split is geometry-grouped; C1 is the evaluation family, RC validation, and R/C2 training.

- **Technical acceptance: PASS.** Independent audit verified artifact hashes, whole-topology assignments, train-only scaling/severity, JSON reloading and hashes for VQC/logistic/Newton GBDT, all-model recomputed test metrics, finite-difference and isolated decision gradients, selected-model local QpiAI parity, all six source/size evaluation cells, and canonical native revalidation of all six dispatch plans.
- **Strict untouched-holdout gate: FAIL.** The v1 preliminary run opened C1 geometry before v2. V2 used fresh replay seeds fixed in its config and explicitly classifies its evaluation as retrospective. Do not describe these results as a pristine unseen-topology test.
- Inventory/native gates: 336 source-size records and 336 valid candidate pools; 336/336 source instances passed native checks; 168/168 supplied Dabia BKS solutions matched the canonical checker. One Dabia R101 n=100 pool was excluded from train/validation due duplicate route-only IDs; no v2 evaluation pools were excluded.
- V2 labels: 22,056 train; 11,920 validation; 6,408 evaluation rows (3,064 Solomon, 3,344 Dabia). All six source/size cells have test metrics.
- Selected decision model: [vqc_decision.json](../.runtime/run1/20261029T083317Z_run1_v2_full/models/vqc_decision.json), v1_1_base, SHA-256 1bb387d58b11579de145c185cd11ce15c707039917d0f12061a19fc3469287c8. Selected model parity against QpiAI Quantum local statevector: max absolute error 2.78e-16; no cloud or QPU jobs.
- Dispatch: six 8-variable QUBOs; five QpiAI local QAOA selections and one explicit exact candidate-pool fallback. All six full plans passed independent Solomon/Dabia native revalidation. Costs and regret are synthetic SCU; native Distance/Duration and BKS gaps remain separate.
- Synthetic classification: test Brier scores were 0.1959 predictive VQC, 0.1850 decision VQC, 0.2250 logistic, and 0.0349 Newton GBDT. The Newton GBDT led on these synthetic rows; no operational or real-traffic claim is made.
- Pipeline wall time before test evaluation: 1,832.8 seconds. Predictive VQC fit 337.1 seconds; selected decision VQC fit 171.4 seconds. Actual ablation/nonselected update counters were not persisted. No measured cache speedup is claimed.

See [RUN1_SOLOMON_DABIA_REPORT.md](../RUN1_SOLOMON_DABIA_REPORT.md) for metrics by source/size, same-replay SCU regret, route-overlay limits, refinement cycles, reproduction details, and artifact hashes.

## Verification commands

Combined RUN 1 training, native, acceptance, and optimization checks:

```powershell
$env:PYTHONPATH='.runtime/audit-deps'
& 'C:\Users\avkir\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest -q tests/test_run1_training.py tests/test_run1_native.py tests/test_run1_acceptance.py tests/test_run1_optimization_candidate.py
```

Result: **35 passed in 1.04s**. The pre-change repository selection recorded **24 passed, 11 subtests**.

Independent acceptance report:

```powershell
$env:PYTHONPATH='.runtime/audit-deps'
& 'C:\Users\avkir\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m experiments_run1.acceptance .runtime/run1/20261029T083317Z_run1_v2_full
```

It returns technical_status=PASS, overall status=FAIL because of the documented prior C1 exposure.

## Run sealing and next work

V1 and v2 have separate sentinels under .runtime/run1/: TEST-EVALUATION-OPENED.json and TEST-EVALUATION-OPENED-v2.json. The original v2 launch command is blocked from a second evaluation under that output root. A reproduction must use a new output root and retain the same retrospective-C1 disclosure; it is not an untouched-topology evaluation. A metadata-only parity identity correction and artifact-hash refresh were applied after v2 without rerunning training or test metrics.

The metadata repair has already been applied once and should not be rerun. Its historical command was:

```powershell
& 'C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine-main\.venv\Scripts\python.exe' .runtime\run1\repair_integrity_metadata.py .
```

To make a strict unseen-topology claim, acquire and audit additional native geometry before defining a new group split. Do not reuse C1 as an untouched holdout. Continue to disclose synthetic labels, the one train pool exclusion, and the candidate-pool limits.
