# RUN 1 Solomon / Dabia Report

## Status

RUN1-v2 completed the bounded training and qualified fresh-seed evaluation across Solomon and Dabia at 25, 50, and 100 customers. Independent technical acceptance is **PASS**. The strict untouched-holdout gate is **FAIL** because the shared C1 geometry had already been opened in the incomplete v1 evaluation. V2 clearly labels this as a retrospective fresh-seed evaluation; its results are useful for this bounded experiment but do not establish pristine, independent generalization to unseen geometry.

## Run and data

- Run directory: `.runtime/run1/20261029T083317Z_run1_v2_full/`.
- Frozen protocol: `RUN1-v2-safe-capped-all-source`; 60 predictive VQC updates, 24 decision VQC updates, 8 replay seeds (`202610290100`–`202610290107`), and 6 dispatch cases. Candidate generation and train/validation/test topology assignments were fixed before v2 test replay labels were opened.
- Split groups: R and C2 train; RC validation; C1 evaluation. R1/R2 and RC1/RC2 aliases stay within their merged geometry groups. The v2 evaluation used new replay seeds after v1's C1 exposure, and the manifest labels it retrospective.
- Native inventory gate: 336/336 source instances checked; all 168 supplied Dabia BKS solutions matched their canonical native evaluator costs exactly. The separate route-pool inventory reports 336 eligible pools and no invalid nominal baselines.
- Replay data: 22,056 train route samples (11,168 positive, 10,888 negative), 1,480 train decision scenarios; 11,920 validation samples (6,054 positive, 5,866 negative), 768 validation scenarios; 6,408 evaluation samples (2,743 positive, 3,665 negative). Evaluation coverage is 3,064 Solomon and 3,344 Dabia rows, with all six source/size cells present and zero evaluation exclusions.
- One train/validation pool was excluded: `dabia2013:R101:n100`, due duplicate route-only IDs (`route:42` and `route:14`) assigned to multiple vehicles. The agent audit recomputed the eight affected nominal route costs as finite and nonnegative, but RUN1-v2 did not use that pool. This one exclusion is recorded in `label_balance.json`.
- Dabia uncertainty uses positive per-leg additive travel overlays under native ATF domain caps computed before test labels. The cap file covers 54 C1 source/size entries (27 per source); Solomon requires no ATF cap. Hard window breaches remain outcomes when the native trace is finite. Targets and SCU costs are synthetic assumptions, not observed incidents or calibrated currency.

## Training and model selection

The two-input V1.1 feature contract was retained: minimum nominal soft-cutoff slack and native nominal route duration, each transformed by a scaler fit on train groups only. Replay features are recomputed from the native service-start trace used to generate labels. No outcome fields are supplied as model inputs. The conditional severity and decision-temperature estimates use train-only evidence; the temperature is the median positive predicted candidate-plan cost gap, `26.26923 SCU` across 1,480 train scenarios.

| Model | Selection / updates | Wall time | CPU time | Circuit evaluations |
|---|---:|---:|---:|---:|
| Predictive VQC, V1.1 base | 60 / 60 | 337.1 s | 336.6 s | 10,362,680 |
| Decision VQC, V1.1 base | 24 / 24 | 171.4 s | 171.1 s | 4,247,000 |
| Logistic regression | 300 steps | 20.0 s | 20.0 s | — |
| Newton GBDT | 40 estimators | 2.0 s | 2.0 s | — |

Measured train/validation pipeline time before test evaluation was 1,832.8 seconds. Ablation and nonselected-branch actual step/timing counters were not persisted, so their configured limits are not reported as completed updates. Validation selected the V1.1 base decision model (`9621.13065` decision loss vs `9621.44976` for the two-block alternative); test outcomes did not select the model.

The selected decision model is saved in [`models/vqc_decision.json`](.runtime/run1/20261029T083317Z_run1_v2_full/models/vqc_decision.json). It has SHA-256 `1bb387d58b11579de145c185cd11ce15c707039917d0f12061a19fc3469287c8`, scaler SHA-256 `3e37176c50943fc51ef28fa2745cb30dccb8e1af70c6cb7e8e3a8f0a4a5c7450`, and circuit variant `v1_1_base`. The two-qubit circuit uses two encoded features, one ansatz block, two trainable angles plus affine readout, four RY gates, two CNOT gates, and depth 4. Selected-model parity was checked on four validation rows using QpiAI Quantum 0.3.0's local `StatevectorSimulator`: maximum absolute error `2.78e-16`, zero remote calls, zero QPU jobs. The train-only decision-component gradient, excluding the BCE anchor, had L2 norm `2.14409`; finite-difference maximum absolute error was `4.46e-11`.

## Evaluation results

These are retrospective fresh-seed results on C1 geometry, not pristine held-out generalization. The same 6,408 labels and rows were used to score all models.

| Model | Brier ↓ | Log loss ↓ | ECE ↓ |
|---|---:|---:|---:|
| VQC predictive | 0.1959 | 0.5801 | 0.2775 |
| VQC decision-focused | 0.1850 | 0.5427 | 0.2465 |
| Logistic regression | 0.2250 | 0.6427 | 0.2331 |
| Newton GBDT | 0.0349 | 0.1149 | 0.0623 |

The Newton GBDT was the strongest predictive baseline on this synthetic evaluation. The decision-focused VQC improved Brier score and log loss relative to the predictive VQC; neither quantum model matched the Newton GBDT. Results are sensitive to the source-specific synthetic label rates: evaluation positive rates were 73.4% for Dabia and 9.4% for Solomon.

| Source / size | Rows | Positive labels | Decision VQC Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| Dabia n=25 | 632 | 589 | 0.2885 | 0.7706 | 0.5020 |
| Dabia n=50 | 928 | 784 | 0.3019 | 0.7997 | 0.4297 |
| Dabia n=100 | 1,784 | 1,083 | 0.2453 | 0.6780 | 0.3019 |
| Solomon n=25 | 632 | 31 | 0.0780 | 0.3073 | 0.2102 |
| Solomon n=50 | 928 | 90 | 0.0923 | 0.3355 | 0.1806 |
| Solomon n=100 | 1,504 | 166 | 0.1001 | 0.3548 | 0.1509 |

Same-replay decision regret is measured in synthetic cost units (SCU), against the minimum realized cost over the same native-feasible candidate plan pool for each replay. It is not an unlimited global optimum.

| Model | Mean same-replay candidate-pool regret (SCU) |
|---|---:|
| VQC predictive | 25.2150 |
| VQC decision-focused | 25.2824 |
| Logistic regression | 25.2552 |
| Newton GBDT | 21.7884 |

The six C101 dispatch cases each had 8 QUBO variables and passed a fresh canonical native solution check in this audit. Five selections used QpiAI local QAOA; one Dabia n=25 case used the explicitly reported exact candidate-pool fallback after local QpiAI execution returned no feasible selected sample. There were no remote or QPU jobs. Same-replay financial regrets by dispatch case were 0.00, 25.20, 5.18, 0.00, 287.70, and 203.78 SCU for Solomon n=25/50/100 then Dabia n=25/50/100. Dabia native Duration-to-BKS gaps are separate benchmark units: 18,291.63, 31,477.29, and 20,771.29 for n=25/50/100. Solomon uses native Distance; its BKS metrics are not merged with SCU regret.

## Independent acceptance

Acceptance command:

```powershell
$env:PYTHONPATH='.runtime/audit-deps'
& 'C:\Users\avkir\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m experiments_run1.acceptance .runtime/run1/20261029T083317Z_run1_v2_full
```

Independent technical acceptance: **PASS**. It verified artifact hashes, whole-topology splits, source/label provenance, train-only severity/scaler assignment, JSON reload and SHA matches for VQC, logistic, and Newton GBDT models, finite-difference gradients, isolated decision gradient, selected-model QpiAI parity, all six evaluation source/size cells, recomputed all four model metrics from sealed route samples and reloaded models, currency boundaries, and all six selected route plans through the canonical Solomon/Dabia native evaluators. The overall strict status remains **FAIL** only for `original_untouched_holdout`: the v1 run opened C1 geometry first. No test labels are represented as observed operational data.

Combined RUN1 acceptance/native/training/optimization tests:

```powershell
$env:PYTHONPATH='.runtime/audit-deps'
& 'C:\Users\avkir\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest -q tests/test_run1_training.py tests/test_run1_native.py tests/test_run1_acceptance.py tests/test_run1_optimization_candidate.py
```

Result: `35 passed in 1.04s`. The pre-change repository baseline selection had `24 passed, 11 subtests`.

## Refinement cycles and limits

1. **Clipped-logit derivative:** the derivative was aligned with the clipped logit actually used by inference, making the derivative zero in saturated tails. The final selected-model finite-difference check agrees within `4.46e-11`.
2. **Decision objective:** a diagnostic at temperature `0.25` produced an effectively zero decision-only gradient (`2.89e-33`). V2 instead fixes temperature from train-only positive predicted plan-cost gaps (`26.26923 SCU`) before validation selection; the final BCE-free decision-component gradient L2 is `2.14409`.
3. **Native replay/features:** an ATF trace returning `duration=None` is treated as unavailable/censored rather than a usable duration. Solomon candidate features are recomputed from the same native service-start trace as training labels. V2's native cap was frozen before test labels and all six C1 source/size evaluations completed with no test exclusions. One duplicate-route-ID Dabia R101 n=100 pool remains excluded from train/validation.

An optional route-feature cache candidate was checked against three cases for exact equivalence but was not applied to this run. No cache speedup is claimed. No real traffic, calibrated dollar costs, or operational business benefit was measured. Candidate-plan coverage and the single C1 geometry limit the decision results; different routes outside the bounded pool may perform differently.

## Reproduction and provenance

V2 was invoked through `run_full` (not the CLI), preserving the v1 sentinel and writing its own `TEST-EVALUATION-OPENED-v2.json`:

```powershell
$env:PYTHONPATH='.;.runtime\audit-deps'
C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine-main\.venv\Scripts\python.exe -u -c "from experiments_run1.run import run_full; print('RUN1_V2_OUTPUT=' + str(run_full(replay_count=8, train_steps=60, decision_steps=24, dispatch_count=6)), flush=True)"
```

That command is historical: the v2 sentinel now blocks another evaluation under the same output root. A statistical reproduction must use a new output root and the same manifest-qualified seeds/configuration; that remains a retrospective C1 replay, not a new unseen-geometry test. A metadata-only correction after the run added the selected decision model SHA/circuit variant and predictive-model derivative check to existing integrity evidence, then refreshed 25 artifact hashes and the run manifest. It did not retrain, regenerate test labels, or recalculate performance metrics. It has already been applied once; do not rerun the repair script. The historical command was:

```powershell
& 'C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine-main\.venv\Scripts\python.exe' .runtime\run1\repair_integrity_metadata.py .
```

Use the acceptance command above to verify the repaired files and hashes.

The v2 test record contains the frozen model digest `acef0e77af3b55069b9ce2dea3300d6107763ac13e8934a456f1876c51faef26`; run-config SHA-256 `990316874b90309fa147464e274700de246658bf589519e444a99daf0d541cde`; artifact manifest digest `aba6135d6617456831756eff5c7372b111f4d194cdae11431639fad86b4b61ab`; and frozen pool inventory SHA-256 `360d06895169b224ed10215c9fde3d66c55cf04e04d1268969dc21527f0d24e7`. `run_manifest.json` records source code hashes for dispatch, model, native, run, and train code; its `code_sha256` field is the authoritative source list for this run. The originally modified legacy result file `experiments/artifacts/quantum-learning-results.json` was preserved unchanged.

Key final file SHA-256 values:

| File | SHA-256 |
|---|---|
| `.runtime/run1/20261029T083317Z_run1_v2_full/run_manifest.json` | `5c716dbcdc21444ac0195619095d52cd8ff33099f394d666e7aeae1be264e463` |
| `.runtime/run1/20261029T083317Z_run1_v2_full/integrity.json` | `c2b8cbdc24a4986c69b5d71dba3914d888c2bfd1d633978902f5b8ee0b2f6111` |
| `.runtime/run1/20261029T083317Z_run1_v2_full/test_metrics.json` | `e3a4664a39a5756da02c12a84f97cc79e28a8fa0da6dc5e981441a0cf156fdee` |
| `.runtime/run1/20261029T083317Z_run1_v2_full/evaluations/dispatch.json` | `5966cb375215996c3dd4a9c30d954d33d5b76d7578885c56a716a4b8d0bdb2fd` |

The earlier v1 run `.runtime/run1/20261029T082608Z_run1_full/` remains preserved as preliminary evidence. It opened C1 but excluded all 27 Dabia C1 cases because of the missing ATF cap, so its Solomon-only evaluation metrics are superseded for result reporting and are not used as the v2 metrics.
