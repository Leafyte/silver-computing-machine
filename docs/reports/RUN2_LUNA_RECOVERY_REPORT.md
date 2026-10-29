# RUN2 India ETA/QML Recovery Report

Date: 2026-10-29. Worktree: `trainingv3`, branch `trainingv3`. This is a local, uncommitted recovery experiment. The original checkout, route-risk model, and protected source data were not modified.

## Result

The historical frozen India ETA checkpoint and its saved predictions were not found. With the user's explicit one-time recovery authorization, this run fit a bounded replacement GBDT, generated courier-grouped train OOF residual targets, trained two two-qubit residual candidates, selected candidate settings on disjoint validation courier groups, and performed the sealed test evaluation after freezing the release.

The recovered GBDT is an experiment-specific replacement; it is not the historical, Gemini, or Dell model. On the sealed test, the GBDT fallback achieved MAE **3.258762**, RMSE **4.114282**, and median absolute error **2.776486** minutes. Neither the ridge residual control nor either quantum candidate improved the frozen fallback. The operational release therefore remains **alpha = 0**, which returns the recovered GBDT prediction.

## Frozen release and outputs

- Final post-evaluation manifest: [`release_manifest.json`](artifacts/run2_india/luna_hgbr_recovery_20261029T131809Z_v5/release_manifest.json), SHA-256 `cf0d6d0fc1758369b053dedf88938844e123ca347b102570efe736ae6fa665da`.
- Pre-test freeze payload SHA-256: `e8c8f679761018036f1ebe3eff3251ea44572fab7c766057625b7302b53ff189`. Frozen at `2026-10-29T13:40:38.212385+00:00` (19:10:38 IST); evaluation completed at `2026-10-29T13:43:33.364149+00:00` (19:13:33 IST).
- Exact pre-test frozen bytes: [`release_manifest_pretest.json`](artifacts/run2_india/luna_hgbr_recovery_20261029T131809Z_v5/release_manifest_pretest.json), SHA-256 `e8c8f679761018036f1ebe3eff3251ea44572fab7c766057625b7302b53ff189`; independently matched to the pre-read sentinel and acceptance snapshot.
- Test metrics and trace: [`final_test_metrics.json`](artifacts/run2_india/luna_hgbr_recovery_20261029T131809Z_v5/final_test_metrics.json), SHA-256 `e54d0154538b25aaed474b26e6b39866ac26043e366aed404d560537bb0d70e1`.
- Test row predictions (under ignored `.runtime`): [`final_test_predictions.parquet`](.runtime/run2/luna_hgbr_recovery_20261029T131809Z_v5/final_test_predictions.parquet), SHA-256 `ee92ecaa747db935e4a9349ac9337f65b4a6f1991b252fd737945310c2e19a85`. Contains 5,941 unique test IDs, minute targets/predictions and residual outputs.
- Portable inference bundle: [`portable_bundle`](artifacts/run2_india/luna_hgbr_recovery_20261029T131809Z_v5/portable_bundle). It includes the baseline checkpoint, both quantum checkpoints, exact feature order and scaler metadata, preprocessing/schema metadata, a loader, and an eight-row synthetic fixture with expected outputs; it contains no training or test dataset. A copy to a temporary directory loaded and passed all three mode parity checks.
- Start-to-finish final evaluator command (resume flag is required to preserve the failed attempt record):

  ```powershell
  & '.runtime\run2\venv\Scripts\python.exe' -m experiments_india_qml.portable_release_v5 --resume-failed-evaluation
  ```

## Sealed test comparison

All metrics are in minutes and use the same 5,941 test rows. MAE, RMSE, and median absolute error are reported in that order.

| Candidate | Frozen use | MAE | RMSE | Median AE |
|---|---|---:|---:|---:|
| A. Recovered GBDT | Baseline | 3.258762 | 4.114282 | 2.776486 |
| B. GBDT + ridge residual control | Control | 3.259587 | 4.114413 | 2.766414 |
| C. GBDT + reupload Q residual, ungated | Candidate comparison | 3.262014 | 4.117239 | 2.772521 |
| D. GBDT + reupload Q residual, congestion gate `<= 0.55` | Candidate comparison | 3.260359 | 4.114836 | 2.774438 |
| E. Operational release, alpha = 0 | Deployment fallback | 3.258762 | 4.114282 | 2.776486 |

The validation-selected reupload candidate used alpha `0.901561` and `vqc_x1_congestion <= 0.55`; the policy would invoke QML for 3,243/5,941 test rows (54.59%). Candidate D still underperformed A, so the frozen deployment remains E. The one-block candidate selected alpha `0` on validation and made no hybrid correction. The ridge control's residual-only MAE/RMSE/median AE were 3.259611/4.114422/2.768007; the selected quantum residual-only values were 3.262496/4.117725/2.769733 ungated and 3.260617/4.114995/2.770074 gated. These residual metrics compare to `test_target - recovered_gbdt_eta`, and are distinct from end-to-end ETA metrics.

Independent acceptance verified 5,941 unique test IDs across 176 couriers and recomputed all listed prediction metrics from the saved Parquet. Prediction rows match the metrics record, and the alpha-zero output is exactly equal to the GBDT output.

For test diagnostics, both candidate residual outputs were computed on all rows and the gated reupload path was also evaluated on its selected mask. This used 15,125 row-wise local statevector example evaluations in the scoring run. The 54.59% figure is the candidate deployment policy's expected invocation rate; it is not the total diagnostic work performed by the scorer. The operational alpha-zero policy invokes zero quantum rows.

## Recovered classical baseline and crossfit

The one-time classical recovery uses `HistGradientBoostingRegressor` with fixed configuration: 100 iterations, learning rate 0.08, 15 leaf nodes, minimum leaf size 30, L2 regularization 1, 255 bins, early stopping disabled, random state 42. No hyperparameter search was run. The final full-original-train checkpoint from the v3 recovery is reused, SHA-256 `da709331715ad3fd312e8708f17d0c86a135bec4e1bfaa1d185545db766c351a`; it is packaged byte-for-byte in the portable bundle.

V5 regenerated train predictions with five `GroupKFold` folds over the original train couriers. Each fold's medians, modes, and category presence are fit using only that fold's training couriers. The fold manifest exposes exact fit/heldout courier lists, row-ID hashes, preprocessing values and hashes, and checkpoint hashes. The VQC scaler is a single two-feature map fit on original-train covariates only and shared by OOF, validation, and inference. The development parquet contains only train OOF plus validation (29,592 + 5,989 rows), no test rows. Train OOF MAE/RMSE/median AE: **3.213865/4.012015/2.804714**. Validation: **3.233990/4.030158/2.829631**.

Validation couriers were split into 3,117 early-stop/architecture rows and 2,872 fusion-selection rows, with disjoint courier IDs; seed 20261029. Architecture selection uses the early-stop role. Ridge alpha and quantum alpha/gate use only fusion-selection rows. The full-validation metrics are report-only; they support the conservative operational alpha-zero decision before the test was opened.

## Quantum candidate training and checks

Both candidates are NumPy differentiable two-qubit statevector regressors fit to minute residuals on a deterministic 4,000-row subset of the 29,592 leakage-safe OOF rows (seed 20261029), with validation early stopping in raw minutes. One-block: 17 updates, best update 2, best residual RMSE 4.042078 minutes, 0.0185 s wall time, checkpoint SHA-256 `023a94860617ba86db2b6905bd04028639a831e33e6611713276d701423ed317`. Genuine reupload: 42 updates, best update 27, best residual RMSE 4.035329 minutes, 0.1307 s wall time, checkpoint SHA-256 `423aa60e2b2d1c129ce756665b83714957c3fd7cc348bbb34640ac4c429a6382`.

The trained reupload checkpoint was independently reloaded and compared with the QpiAI 0.3.0 local statevector SDK: 8 examples, maximum expectation error `2.64e-16`, maximum residual-minute error `8.33e-17`. Independent acceptance also verified parameter-shift gradients against finite differences, reload prediction parity, calibration alpha/gate recomputation, raw-minute early stopping, and masked gate behavior. These are local simulators; no paid QPU or cloud job was used.

The device profile found an RTX 5060, but Numba CUDA was unavailable (`cuda.is_available() == false`), no PyTorch was installed, and no GPU kernel ran. Training/profiling therefore used CPU NumPy. This report makes no CPU-vs-GPU speed claim.

## Test-attempt trace

The frozen pre-test release passed acceptance before the authorized final scoring attempts. Earlier, recovery v1 transiently materialized target-bearing test rows due a parquet row-group-only filter and aborted at a partition assertion before fit, metric computation, score inspection, or selection; no outcomes were inspected or used. The first authorized final evaluator attempt also read the test rows and produced in-memory predictions, then aborted on a bookkeeping `KeyError` (`'reupload'`) before metrics or predictions were persisted. No outcomes were inspected or used for selection. Evidence for the final scoring attempt is preserved in [`sealed_test_attempt1_aborted.json`](artifacts/run2_india/luna_hgbr_recovery_20261029T131809Z_v5/sealed_test_attempt1_aborted.json) and `.runtime/run2/luna_hgbr_recovery_20261029T131809Z_v5/sealed_test_evaluation_started.json`.

The root authorized a code-only recovery guarded by the recorded failure. The corrected attempt reused the exact frozen release manifest and checkpoint/scaler/alpha/gate hashes, made no model fit or selection changes, and persisted the completed metrics and prediction artifact above. It is reported as two executions: one aborted attempt and one completed code-only recovery. Neither attempt's test outcomes influenced selection or deployment; operational alpha remains the pre-test alpha-zero fallback.

## Data provenance and source integrity

- Original frozen ETA checkpoint/predictions were absent after targeted search across the worktree, original checkout, nested checkout, ignored runtime and relevant serialized JSON/NPZ/NPY/model formats. This model is the explicitly authorized recovery.
- Working `data/train.csv` SHA-256: `d293b6fc7157aef30fb34301216a3432ec895d407ce2ea9edddf942b416cdde1`. The split manifest's parent-source SHA-256 `4baa699e1c09d286f06790010a8c0b284ed3376cbde7e346e19d0ded16a240e3` matches the Git blob at base `40bd7fa`; normalizing the working file's CRLF line endings to LF yields the committed bytes. Parsed 45,594 CSV rows (including header) and all cells match. No data file was edited to reconcile the hashes.
- Other source hashes: clean Parquet `9034d24389c7162f0ea0e275e5b4cc9ec16f04bba75c0f239dbcef8535a1f7c1`; GBDT view `bd03b47c3d3e23050329cefc69d752aad607031cacdb9696151d237e3534a627`; VQC view `a633677472e6e0e874c2797a66eb7b011e3a307285553f8fa00c6ee664df4b4e`; feature schema `a32f3c4491347be51c10a70d00d5bcc24dbb1d1db11a9a0105e40d809e8bed51`; split `10bee51eebac51bf68813a35a42b06adeb2e30588accdc582bbe9b0d8136c1bb`; preprocessing metadata `df20646f12b77cb9fb03bf831b3373e5f8c9f69d6e5f0c92c35bbd46c682f351`.
- Partitions: 29,592 train, 5,989 validation, 5,941 sealed test, and 4,071 excluded. Courier groups: 819/175/176 train/validation/test with zero overlap according to the parent split manifest.
- Earlier implementation attempts v1/v2 were aborted before fit. V3's recovered checkpoint was retained; its OOF preprocessing was superseded by v5 fold-local regeneration. V4 aborted before any fold model fit due a nullable-category comparison error. An initial residual-training pass that used a separately resampled validation role assignment was superseded; the final candidate checkpoints were refit against the v5 frozen role groups. No test rows were used in those development fits.

## Runtime and verification

The task-local runtime is under ignored `.runtime/run2/venv`: Python 3.12, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.4.2, fastparquet 2024.11.0, pytest available through the local environment. The original QpiAI environment uses NumPy 2.5.4/pandas 3.1.0/QpiAI 0.3.0. No global package install was made. Final post-evaluation suites reported **22 passed, 4 skipped** in the task runtime and **22 passed, 4 skipped** in the original SDK runtime; these counts are per environment and are not combined. The skips are runtime/scope gates.

Portable command for an already transformed matrix is documented in [`portable_bundle/README.md`](artifacts/run2_india/luna_hgbr_recovery_20261029T131809Z_v5/portable_bundle/README.md); the API takes the exact 59-column GBDT feature order plus `[vqc_x0_distance, vqc_x1_congestion]` scaled by the included train-only mapping. The bundle does not include the source dataset or a raw-record cleaning pipeline.

The separately mentioned route-risk optimization was not launched; no route-risk model training or tuning was performed.
