# Run 2 India QML Independent Acceptance

**Current status (2026-10-29):** The independently reconstructed ETA GBDT, v5 leakage-safe development predictions, trained two-qubit QML candidates, validation-role selection, local QpiAI checkpoint parity, pre-test release, and independent saved-prediction test-metric recomputation pass acceptance. The classical recovery is not the missing historical frozen/Gemini checkpoint. The alpha≈0.902 reupload candidate is preserved for comparison, while the operational release freezes the classical alpha=0 fallback because full-validation MAE/RMSE do not improve on GBDT. The exact pre-test manifest snapshot is preserved with its original byte hash.

## Scope and evidence rules

This report follows the attached Run 2 mission, repository `AGENTS.md`, and the repository-approved testing, statistical-analysis, experimental-design, PennyLane, debugging, and code-review skills. Work was confined to `trainingv3`; no edits were made to the original checkout, the route-risk model, the alternate Torch implementation, or the lead's training report. The acceptance worker did not train or select a model. Hashes and row/group checks are computed independently from the generated v5 manifest, checkpoint, development table, source schema, and split files.

Acceptance separates train/validation development evidence from sealed test evidence. Target-bearing development reads were checked with fastparquet's row filter and explicit partition assertions. The raw-source rechecks for fold statistics load only non-target feature columns and filter to original train rows before any statistics are recomputed. The default test suite is guarded to skip test prediction loading/scoring unless `RUN2_FINAL_TEST_ACCEPTANCE=1` is set; that opt-in remains unset.

## Frozen India input inventory

SHA-256 values are from the exact files in this checkout.

| Artifact | SHA-256 | Finding |
|---|---|---|
| `data/indian_delivery_gbdt.parquet` | `bd03b47c3d3e23050329cefc69d752aad607031cacdb9696151d237e3534a627` | Feature/target table; not itself a fitted checkpoint |
| `data/indian_delivery_vqc.parquet` | `a633677472e6e0e874c2797a66eb7b011e3a307285553f8fa00c6ee664df4b4e` | Contains target-derived normalized/SLA fields; only `vqc_x0_distance`, `vqc_x1_congestion` are QML features |
| `data/indian_delivery_clean.parquet` | `9034d24389c7162f0ea0e275e5b4cc9ec16f04bba75c0f239dbcef8535a1f7c1` | Clean source and raw feature columns |
| `data/feature_schema.json` | `a32f3c4491347be51c10a70d00d5bcc24dbb1d1db11a9a0105e40d809e8bed51` | Regression target `delivery_duration_min`, in minutes; 59 GBDT features |
| `data/split_manifest.json` | `10bee51eebac51bf68813a35a42b06adeb2e30588accdc582bbe9b0d8136c1bb` | Courier group split; 29,592 train / 5,989 validation / 5,941 test, plus 4,071 excluded |
| `data/preprocessing_artifacts.json` | `df20646f12b77cb9fb03bf831b3373e5f8c9f69d6e5f0c92c35bbd46c682f351` | Frozen preprocessing constants; not a fitted model or preprocessor checkpoint |
| `data/train.csv` | `d293b6fc7157aef30fb34301216a3432ec895d407ce2ea9edddf942b416cdde1` | Working bytes differ from recorded source hash because of CRLF/LF normalization; parsed cells and normalized bytes match parent blob |

The GBDT schema excludes target, ID, courier, and split columns. The VQC table contains `target_normalized`, `sla_threshold_min`, and `is_sla_breach_synthetic`; acceptance does not use these as inputs. The route-risk Newton GBDT is a Solomon/Dabia binary classifier and is excluded from India ETA evidence.

## Checkpoint search and recovery history

**Historical frozen India ETA checkpoint: not found.** Repository and sibling-workspace inventory included serialized `.joblib`, pickle, Torch, ONNX, CatBoost, NumPy, safetensors, msgpack, and custom JSON/model formats. No India duration predictor or saved classical predictions were found. The owner authorized one bounded India ETA recovery model and leakage-safe group cross-fitting; this recovered artifact must not be represented as the original frozen/Gemini model.

Recovery history is preserved in v5's `recovered_gbdt_manifest.json`:

- Initial v1 transiently materialized test target values due pandas row-group-only filtering. The runner stopped before fit, metric computation, score inspection, or selection. This is recorded as an evaluation-integrity incident; it is not described as “test labels never loaded.”
- v2 aborted before target values were read.
- v3 kept a fixed-configuration train-only final GBDT but its OOF imputation/one-hot statistics reused outer-train covariates. Its OOF evidence is superseded.
- v4 corrected OOF transforms to fit on each fold's train couriers and used one original-train-only QML feature mapping.
- v5 is the current immutable accepted development artifact set. It contains exactly train OOF + validation rows; no test targets or rows were used for this stage. The v3 one-time GBDT checkpoint is reused without further search/refit.

## Independent checks

### Source split and groups — PASS

The source-clean inspection loaded only ID, courier, split, and eligibility columns, not duration labels. It verified 45,593 total source rows; 41,522 eligible rows; unique eligible IDs; train/validation/test row counts of 29,592 / 5,989 / 5,941; and courier counts of 819 / 175 / 176. Each pair of partitions has zero courier overlap. Values are recorded in [`artifacts/run2_india/acceptance_data.json`](../artifacts/run2_india/acceptance_data.json).

### v5 recovered GBDT and development rows — PASS

`tests/test_run2_india_qml_acceptance.py` independently verifies the v5 manifest's checkpoint and source split/schema hashes, minute target, recovery authorization, train-only fit declaration, and no-test-use marker. In the pinned task-local sklearn runtime, acceptance hash-checks then reloads the trusted local joblib checkpoint. Predictions recomputed from the validation feature rows match the saved validation ETA values within `rtol=1e-12, atol=1e-10`.

The v5 development table contains 29,592 train OOF and 5,989 validation rows, with unique IDs, exact source split membership, finite minute values, and `residual_minutes = y_true_minutes - gbdt_pred_minutes`. No test rows appear. All five GroupKFold records are checked independently: fold fit/heldout courier sets are disjoint; each OOF courier appears in exactly one heldout fold; row counts and canonical row-ID hashes match; and all source train couriers are covered.

For each fold, acceptance rereads only target-free original-train raw feature columns and recomputes the exact numeric medians, traffic mode, and fit category sets in the manifest. The recorded fold `fit_groups` equal the estimator fit courier list, and the fold preprocessing settings hashes are present. The single QML scaler's file hash, original-train-only declaration, and 29,592 fit-row count are verified. The validation early-stop and fusion-calibration roles are courier-disjoint, cover all validation couriers/rows, and match independently recomputed canonical row-ID hashes.

Machine-readable v5 hashes and counts are in [`artifacts/run2_india/acceptance_v5.json`](../artifacts/run2_india/acceptance_v5.json).

### QML numerical fixtures and local QpiAI execution — PASS on fixtures

Deterministic synthetic 32-row fixtures are used only for circuit correctness, not as ETA training/validation evidence. Both one-block and genuine reupload circuits pass batched-vs-row expectation parity with error 0. The circuit uses two qubits; statevector index `|q1 q0>` with q0 least-significant; parity `Z(q0), Z(q1)`; trainable real readout weights and bias. The one-block gate sequence is `RY(pi*x0) q0`, `RY(pi*x1) q1`, `CX(0,1)`, `RY(theta0) q0`, `RY(theta1) q1`, `CX(1,0)`. Reupload repeats both same features before the second trainable rotation pair.

Parameter-shift gradients match centered finite differences at `h=1e-5`: maximum absolute errors are `2.85e-12` (one block) and `1.23e-11` (reupload); angle gradients are nonzero. Actual QpiAI Quantum 0.3.0 local StatevectorSimulator calls on eight fixture inputs match the NumPy circuit expectations within `2.22e-16` and `7.77e-16`. A direct fixed-circuit SDK statevector also matched an independent little-endian matrix reference with max amplitude error 0, state norm 1, completed local simulator result, and zero credits. Detailed values are in [`artifacts/run2_india/acceptance_quantum.json`](../artifacts/run2_india/acceptance_quantum.json).

### Trained QML candidates — PASS; operational freeze pending

Both trained NPZ checkpoints reload reproducibly. Acceptance recomputed each checkpoint's early-stop-role residual RMSE in raw minutes from the saved `vqc_early_stop` rows, and verified the selected architecture is the candidate with lower early-stop RMSE. The separately frozen `validation_role_protocol.json` is hash-checked against the recovery/development inputs and marks that it was written before QML fitting. Alpha and gate metrics were independently recomputed only on the courier-disjoint `fusion_selection` rows.

The selected validation candidate is genuine reupload, alpha `0.9015612266`, with a pre-circuit `vqc_x1_congestion <= 0.55` gate. Acceptance recomputes the mask from that feature, reloads the model, and instruments `predict_minutes`: circuit calls match the mask count exactly. It makes 3,188 calls (53.23%) on full validation; the one-block candidate selected alpha 0 and makes zero calls. Candidate full-validation results are reporting-only. Independent metrics (MAE / RMSE / median absolute error, minutes) are GBDT `3.23399 / 4.03016 / 2.82963`, reupload hybrid `3.23500 / 4.03035 / 2.82404`, and Ridge control `3.23448 / 4.02985 / 2.83014`. The quantum candidate slightly regresses GBDT MAE and RMSE on full validation. The frozen release records candidate D separately from deployed fallback E with alpha 0 and gate disabled.

Actual QpiAI 0.3.0 local `StatevectorSimulator` execution on the reupload checkpoint and eight validation feature rows matches circuit expectations within `2.64e-16` and predicted residual minutes within `8.33e-17`. This is a local simulator result, not cloud or QPU execution. The acceptance suite also independently checks both saved candidates against QpiAI locally.

### Sealed test — saved-output acceptance PASS

The exact pre-test snapshot [`release_manifest_pretest.json`](../artifacts/run2_india/luna_hgbr_recovery_20261029T131809Z_v5/release_manifest_pretest.json) hashes to `e8c8f679761018036f1ebe3eff3251ea44572fab7c766057625b7302b53ff189`, matching both evaluation-start records. Its test timestamp, metrics, and prediction path were null. The post-evaluation manifest hashes to `cf0d6d0fc1758369b053dedf88938844e123ca347b102570efe736ae6fa665da`; the saved row-prediction table hashes to `ee92ecaa747db935e4a9349ac9337f65b4a6f1991b252fd737945310c2e19a85`; and the metric record hashes to `e54d0154538b25aaed474b26e6b39866ac26043e366aed404d560537bb0d70e1`.

There were **two execution attempts and one completed evaluation**. Attempt 1 read test data but stopped on a post-prediction bookkeeping `KeyError` before metrics or prediction rows were persisted. Its preserved record says no score was inspected or used for selection. The lead fixed only the mapping bug, retained the same frozen manifest/configuration, and resumed under attempt 2 using the same pre-test manifest hash. Acceptance verified the attempt records, hashes, freeze timestamp, validation-only selection, operational alpha 0, and separately preserved candidate. It independently recomputed metrics from the saved table and checked all 5,941 unique IDs and 176 courier groups against the test partition. Acceptance ran no inference, fitting, or selection.

| Test comparison | MAE (min) | RMSE (min) | Median absolute error (min) |
|---|---:|---:|---:|
| GBDT baseline (A) | 3.258762 | 4.114282 | 2.776486 |
| GBDT + Ridge (B) | 3.259587 | 4.114413 | 2.766414 |
| Reupload hybrid candidate (C) | 3.262014 | 4.117239 | 2.772520 |
| Selective reupload candidate (D) | 3.260359 | 4.114836 | 2.774438 |
| Operational alpha=0 release (E) | 3.258762 | 4.114282 | 2.776486 |

The operational release exactly matches the GBDT predictions. The quantum candidate remains available for comparison and did not improve test MAE or RMSE. Test outcomes did not change the frozen fallback.

## Executed acceptance commands

Task-local Python 3.12 / scikit-learn 1.4.2 / NumPy 1.26.4 / pandas 2.2.3 runtime (Parquet and trusted local joblib checks):

```powershell
& '.runtime\run2\venv\Scripts\python.exe' -m pytest tests/test_run2_india_qml_acceptance.py -q -rs
```

Result after final saved predictions: **22 passed, 4 skipped, 1 sklearn feature-name warning** with `RUN2_FINAL_TEST_ACCEPTANCE=1`. The independent saved-test metric recompute ran here. Four QpiAI-only checks skip because this runtime uses NumPy 1.26; the exact pytest output lists them.

Original NumPy 2.5 / QpiAI Quantum 0.3.0 runtime (SDK parity plus split and fold integrity; sklearn checkpoint reload tests skip because sklearn is not installed there):

```powershell
$env:PYTHONPATH='C:\Users\avkir\Downloads\silver-computing-machine-main\trainingv3\.runtime\run2\site-packages'
& 'C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine-main\.venv\Scripts\python.exe' -m pytest tests/test_run2_india_qml_acceptance.py -q -rs
```

Result after final saved predictions: **22 passed, 4 skipped, 9 warnings** (seven pandas fastparquet deprecation notices and two QpiAI SDK import deprecations), with the test guard unset. Trained-checkpoint QpiAI statevector parity and local fixtures ran. The two sklearn reload tests skip because sklearn is unavailable here; the post-evaluation manifest gate and guarded test-recompute gate also skip. The independent recompute ran in the task-local runtime above. Capability-specific QpiAI skips in the task runtime avoid importing a NumPy 2 SDK into the NumPy 1.26/scikit-learn environment.
