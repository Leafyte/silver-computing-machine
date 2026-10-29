# India ETA GBDT recovery report

Status: **trained**. This is a bounded reconstruction from the committed feature view, not a recovered historical checkpoint.

## Data and leakage controls

- Canonical Git source CSV SHA-256: `4baa699e1c09d286f06790010a8c0b284ed3376cbde7e346e19d0ded16a240e3` (verified after CRLF-to-LF normalization; worktree bytes SHA-256: `d293b6fc7157aef30fb34301216a3432ec895d407ce2ea9edddf942b416cdde1`).
- GBDT Parquet SHA-256: `bd03b47c3d3e23050329cefc69d752aad607031cacdb9696151d237e3534a627`; VQC Parquet SHA-256: `a633677472e6e0e874c2797a66eb7b011e3a307285553f8fa00c6ee664df4b4e`.
    - Input rows: 29,592 train / 5,989 validation; target `delivery_duration_min` in minutes; 59 numeric features.
- Model inputs exclude stable ID, courier ID, split partition, and target. VQC features were keyed by ID, courier, and split and are retained only in aligned prediction files.
- Training OOF uses five `GroupKFold` folds by courier. Validation courier groups are deterministically separated into model-selection and fusion-calibration roles; calibration rows do not choose the GBDT configuration.
- Sealed test partition was not included in the DataBundle, scored, or serialized. The committed GBDT Parquet is one row group, and this environment's fastparquet reader materializes the table before applying the split filter; its test rows are therefore transiently decoded, then dropped before model arrays are formed. No test target was used for fitting, model selection, calibration, metrics, or saved outputs. This is not a claim of byte-level non-read.

## Estimator and bounded selection

Estimator: `sklearn.ensemble.HistGradientBoostingRegressor`, objective `squared_error`, configuration `{"l2_regularization": 1.0, "learning_rate": 0.06, "max_iter": 160, "max_leaf_nodes": 31, "min_samples_leaf": 25, "name": "hgb_medium"}`. Selected by minimum MAE on the courier-isolated `model_selection` subset among three bounded configurations. CPU training time: 4.23 s total across all fits; final full-training refit 0.56 s.

## Validation results (minutes)

| Partition | Model | MAE | RMSE | Median absolute error |
|---|---|---:|---:|---:|
| All validation (descriptive) | Training-median baseline | 7.5370 | 9.3721 | 7.0000 |
| All validation (descriptive) | Frozen HGB regressor | 3.1428 | 3.9060 | 2.7425 |
| Model-selection groups | Training-median baseline | 7.5628 | 9.3994 | 7.0000 |
| Model-selection groups | Frozen HGB regressor | 3.1726 | 3.9321 | 2.7939 |
| Fusion-calibration groups | Training-median baseline | 7.4799 | 9.3116 | 7.0000 |
| Fusion-calibration groups | Frozen HGB regressor | 3.0771 | 3.8476 | 2.6436 |

## Complexity and inference

- Fitted boosting iterations: 160; total trees: 160; total leaf nodes: 4960; input features: 59.
- Median single-row inference latency over 31 routine prediction calls: 756.10 microseconds per row on `AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD`.

## Artifacts

- Model: `artifacts\india_eta_recovery\gbdt\india_eta_hgb_regressor.joblib` (SHA-256 `f1f94fbcc932f3519aaafcff7dc99d65b7951c0240730e54c320259152bbebdf`).
- Feature schema: `artifacts\india_eta_recovery\gbdt\feature_schema.json`.
- Train OOF: `artifacts\india_eta_recovery\oof_predictions.parquet` (29,592 rows).
- Validation predictions: `artifacts\india_eta_recovery\validation_predictions.parquet` (5,989 rows).
- Fold provenance: `artifacts\india_eta_recovery\oof_fold_manifest.json`; per-fold courier lists and hashes are recorded.
- Reproducibility manifest: `artifacts/india_eta_recovery/recovery_manifest.json`.

The model remains a separate continuous ETA regressor and does not replace the Solomon/Dabia route-risk classifier.

## Portable inference release

- Bundle directory: `artifacts/india_eta_classical_release_v1/`.
- ZIP: `artifacts/india_eta_classical_release_v1.zip`.
- External SHA-256 file: `artifacts/india_eta_classical_release_v1.zip.sha256`.
- ZIP SHA-256: `f110fd17a52b742e92c3320ab6b88abc6cd1efa4ac463212eb8a40da54efdfee`.
- The bundled `model.joblib` retains the frozen checkpoint SHA-256 above. The release includes the 59-feature schema, preprocessing parameter JSON, pinned `numpy==1.26.4`, `scikit-learn==1.4.2`, and `joblib==1.6.0` requirements, a hash-checked CPU loader, and three synthetic reference rows with exact expected predictions. It contains no raw dataset, OOF, or validation prediction files.
- The model accepts the already-engineered 59 numeric features in the recorded order. The preprocessing parameters are included, but full raw-record cleaning/geospatial/one-hot feature-engineering code is not; the portable boundary is documented in the bundle README.
- Directory and ZIP-extracted verification command: `& '.runtime\run2\venv\Scripts\python.exe' -m pytest tests\test_india_eta_classical_release.py -q` — **3 passed** in 8.10 s. This checks exact same-machine reference outputs, archive SHA, every manifest file hash/size, traversal rejection, and tampered-model rejection before deserialization.
- Direct Dell/CPU reference command after installing `requirements.txt`: `python predictor.py --verify-reference` from the extracted bundle.

## Execution and verification

- Training command: `& '.runtime\run2\venv\Scripts\python.exe' -u -m experiments_india_eta.recovery --root .` — exit code 0; completed bounded selection, five group-held-out OOF fits, and final full-train refit.
- Targeted tests: `& '.runtime\run2\venv\Scripts\python.exe' -m pytest tests\test_india_eta_recovery.py -q` — **3 passed** in 2.26 s.
- The tests independently reloaded the checkpoint, checked exact saved validation predictions, verified all artifact hashes, and checked the OOF courier-group and validation-role assignments.
