# RUN2 India independent acceptance and handoff

**Overall RUN2 status: PARTIAL.** Worker A's recovered ETA GBDT and OOF artifacts passed the independent pretest audit. The final residual VQC training, fusion freeze, sealed test evaluation, and complete portable release are not complete. The current user gate pauses VQC training and test evaluation; this report does not treat an unrun stage as a pass.

## Independent result: classical recovery

The bounded recovery is a separate `sklearn.ensemble.HistGradientBoostingRegressor` with squared-error objective, fitted on the original train partition. The estimator is a reconstruction, not the missing historical ETA checkpoint. Selection compared three bounded HGB configurations on the frozen courier-group `model_selection` subset and selected `hgb_medium` (160 iterations, 31 leaf nodes, `min_samples_leaf=25`, learning rate 0.06, L2=1.0). It reports 59 frozen features and target `delivery_duration_min` in minutes.

I independently verified the original data digests, source provenance, split identities, courier isolation, validation role assignment, five OOF fold courier sets, artifact hashes, validation metrics, and checkpoint reload predictions. The checker read test IDs/group/split metadata to confirm partition membership, but did not read test duration labels or calculate any test metrics.

| Check | Independently recomputed result |
|---|---|
| Eligible source rows | 41,522: 29,592 train / 5,989 validation / 5,941 test |
| Courier groups | 819 train / 175 validation / 176 test; all partitions disjoint |
| OOF coverage | 29,592 unique original-train IDs; every record assigned once to a courier-held-out fold |
| OOF HGB metrics | MAE 3.11392 min; RMSE 3.88911 min; median absolute error 2.69822 min |
| Validation role assignment | 117 courier groups / 4,124 rows for model selection; 58 groups / 1,865 rows for fusion calibration |
| Validation HGB metrics | Selection: MAE 3.17258, RMSE 3.93209, median AE 2.79392 min. Calibration: MAE 3.07707, RMSE 3.84757, median AE 2.64365 min |
| Validation reload parity | Recomputed predictions from the SHA-verified checkpoint had maximum absolute difference 0.0 min from saved predictions |
| Test outcomes | Not read or scored by this independent acceptance run |

The training-median validation baseline was MAE 7.53698 min and RMSE 9.37213 min across all validation rows. These are development results and do not estimate sealed-test performance.

## Independent classical residual-control audit

The separate Ridge comparison control was independently recomputed from the frozen 29,592 training-only OOF residual targets (`y_true_min - gbdt_pred_min`) and four candidate features joined by `(delivery_record_id, courier_id, split_partition)`. The input Parquet hashes match the previously accepted GBDT predictions; validation control rows match all 5,989 original validation keys. The 117 model-selection couriers / 4,124 rows alone select among alpha `{0.01, 1, 100}`. The 58 fusion-calibration couriers / 1,865 rows did not select alpha. No test labels were opened.

Alpha `0.01` independently minimizes selection MAE. Its selection metrics are MAE `3.172207`, RMSE `3.931903`, median absolute error `2.796163` minutes, compared with GBDT-only MAE `3.172583`, RMSE `3.932094`, median absolute error `2.793923`. The MAE reduction is only about `0.000377` minutes and median error is slightly worse. The calibration metrics are diagnostics only; there the control MAE `3.077200` is slightly worse than GBDT-only `3.077069`. This is a small experimental classical control, not evidence of a meaningful improvement and not a VQC result. Its checkpoint explicitly remains inactive.

The independent calculation matched the saved coefficients, alpha trials, all saved validation row IDs and predictions, with maximum prediction and correction differences both `0.0` minutes. Evidence is in [`acceptance_residual_control.json`](artifacts/run2_india/acceptance_residual_control.json), SHA-256 `B479B529B90CE87EA89EE319DD476E104EDF5CF06AE58F2D63B32187A41C74FE`.

## Provenance and frozen validation protocol

The actual worktree CSV byte SHA is `d293b6fc7157aef30fb34301216a3432ec895d407ce2ea9edddf942b416cdde1`. Git reports `core.autocrlf=true` and `i/lf w/crlf`; CRLF-to-LF normalization matches both the committed `data/train.csv` blob at the source commit and the expected SHA `4baa699e1c09d286f06790010a8c0b284ed3376cbde7e346e19d0ded16a240e3`. This difference is checkout line-ending conversion, not changed CSV contents. The GBDT, VQC, and clean Parquet digests match their frozen values.

The validation courier split is frozen by sorting unique courier IDs by the raw SHA-256 digest of `b'india-eta-val-role-v1|20261029|' + courier_id.encode('utf-8')`, then assigning the first `floor(N/3)` groups to fusion calibration and the rest to model selection. This is label-blind. The pretest activation rule requires selective VQC calibration MAE to be strictly below both the frozen GBDT and the small classical residual control, with RMSE no worse than either; otherwise the operational VQC scale is zero. These calibration comparisons are policy-selection evidence, not an unbiased estimate. The sealed test is reserved for one evaluation after all models and policy are frozen.

## Verified artifacts

- Recovered GBDT checkpoint: [`india_eta_hgb_regressor.joblib`](artifacts/india_eta_recovery/gbdt/india_eta_hgb_regressor.joblib), SHA-256 `f1f94fbcc932f3519aaafcff7dc99d65b7951c0240730e54c320259152bbebdf`.
- Train OOF records: [`oof_predictions.parquet`](artifacts/india_eta_recovery/oof_predictions.parquet), 29,592 rows, SHA-256 `78a5461c89387e8630421008b92dfffcc84864e9e25bea7476eefa2c8d48032f`.
- Validation records: [`validation_predictions.parquet`](artifacts/india_eta_recovery/validation_predictions.parquet), 5,989 rows, SHA-256 `3d9485c513e6a997516baea1a4a6c497c6021240f9cbf64d524037e9e8753f1a`.
- Courier role map: [`validation_role_assignment.json`](artifacts/india_eta_recovery/validation_role_assignment.json), SHA-256 `c13704aa421e4d05138640c60c73dc734afece75610c1aaf43b2c884524640ef`; canonical courier-map SHA-256 `631dbb800d8759858ccbed05345d95549940782fa8c81390e9a4affeb0ba6425`.
- Recovery manifest: [`recovery_manifest.json`](artifacts/india_eta_recovery/recovery_manifest.json).
- Independent pretest audit: [`acceptance_classical.json`](artifacts/run2_india/acceptance_classical.json), SHA-256 `A9660E4AEB8C33703525333069F64E66A0F380ADEC5CA04904D7B3A57F104605`.
- Frozen role and activation protocol: [`validation_role_protocol.json`](artifacts/run2_india/validation_role_protocol.json), SHA-256 `FD65D9C9F0404EA4F223177ABF57D79ABC65A79D6A405F02EE3FCA76C60053F0`.
- Experimental Ridge residual-control checkpoint: [`residual_ridge.json`](artifacts/india_eta_residual_control_v1/residual_ridge.json), SHA-256 `5203fe9844a9721cfed1f26411a32bb0ebd872cafa8f64e661c73c2afc8189e2`.
- Ridge validation output: [`validation_predictions.parquet`](artifacts/india_eta_residual_control_v1/validation_predictions.parquet), SHA-256 `3a0b20e1c59c7a71d986e574309acc1f13cf854c6b73bbd0bba639928cc57b69`.

### Classical-only portable release

The independent CPU bundle is [`artifacts/india_eta_classical_release_v1/`](artifacts/india_eta_classical_release_v1/), with archive [`india_eta_classical_release_v1.zip`](artifacts/india_eta_classical_release_v1.zip). The archive SHA-256 is `f110fd17a52b742e92c3320ab6b88abc6cd1efa4ac463212eb8a40da54efdfee`; the adjacent `.sha256` file matches it. The bundle contains the exact frozen GBDT checkpoint, ordered 59-feature schema, preprocessing parameters, inference loader, pinned requirements, README, release manifest, and a three-row synthetic reference fixture. It contains no raw dataset, OOF records, or validation/test records.

I independently compared the bundle's model bytes to the verified GBDT checkpoint; checked every manifest entry's path, byte count, and SHA-256; required the feature names/order to match the source schema exactly; checked archive member names for traversal and symlinks; extracted the ZIP into a fresh workspace temporary directory; and verified the extracted loader's reference predictions. The official loader and an independent `joblib` reload both reproduced all three expected minute predictions exactly (maximum absolute error `0.0`). Tampering with the model was rejected before `joblib.load`; three manifest traversal forms were rejected. The independent evidence is in [`acceptance_classical_release.json`](artifacts/run2_india/acceptance_classical_release.json), SHA-256 `78010a86985976d5d6e6a53fda23f63a2c69e11d7fa3dfcd5bf5dc3d2a372625`.

This is a complete classical-only CPU inference bundle, not the full RUN2 hybrid release. The preprocessing JSON is included, but raw-record feature engineering is not; callers must provide the 59 numeric inputs in the specified order. The independent CPU residual circuit helper remains in [`release.py`](experiments_india_qml/release.py); its synthetic 2-feature and 4-feature fixtures matched QpiAI's local statevector predictions to `1e-12`, but no trained residual VQC is packaged.

## Commands and test evidence

Independent audit command, run from `trainingv3/` with the isolated data runtime:

```powershell
@'
from experiments_india_qml.independent_acceptance import write_classical_acceptance
print(write_classical_acceptance('.'))
'@ | & '.runtime\run2\venv\Scripts\python.exe' -
```

Focused audit test:

```powershell
& '.runtime\run2\venv\Scripts\python.exe' -m pytest -q tests\test_run2_independent_acceptance_helpers.py -k frozen_classical_recovery
```

Result: **1 passed**. The synthetic helper and portable circuit tests previously completed with **9 passed** under the base project environment, including actual local QpiAI statevector parity for both circuit shapes. A later combined run under `.runtime/run2/venv` terminated in a Windows access violation while importing QpiAI's CVXPY dependency; that run is not counted as passing. No cloud request or QPU job was submitted.

Independent release command, run from `trainingv3/`:

```powershell
& '.runtime\run2\venv\Scripts\python.exe' -m pytest -q tests\test_india_eta_independent_release_acceptance.py
```

Result: **1 passed**. The audit checked the directory and ZIP, extracted and reloaded the archive in a fresh temporary directory, reproduced synthetic reference outputs through both the release loader and independent model reload, and verified tamper/traversal rejection. It read no real delivery rows and did not open the sealed test.

Independent residual-control audit test:

```powershell
& '.runtime\run2\venv\Scripts\python.exe' -m pytest -q tests\test_india_eta_independent_residual_control.py
```

Result: **1 passed**. It recomputed all alpha trials from train OOF residuals, checked model-selection-only choice and courier-group role separation, and matched all saved validation predictions without opening test labels.

## Incomplete stages and limits

- The user has paused full VQC training and test evaluation under the current quantum gate. There is no frozen trained residual VQC, no frozen operational fusion policy, and no test metric table for the five requested pipelines.
- The classical-only CPU bundle is complete and independently reload-verified. The full hybrid release remains unfinished because no VQC checkpoint, fusion policy, or integrated multi-pipeline loader is available.
- Worker B's Torch/CUDA runtime install was in progress when the gate was applied. No QML training result is claimed.
- The base runtime does not include scikit-learn or Parquet engines; the independent classical audit ran in the isolated `.runtime/run2/venv`. Deployment requires declared compatible CPU dependencies and a fresh reload check on the target Dell environment.
- The full combined RUN2 suite has not completed in a single environment due the QpiAI/CVXPY access violation. The narrowly scoped independent classical acceptance passed.

