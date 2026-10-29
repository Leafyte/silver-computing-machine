# India ETA classical residual control

This is a small comparison control trained on courier-held-out GBDT OOF residuals. It is experimental and is not operationally activated.

## Protocol

- Training target: `y_true_min - gbdt_pred_min` from 29,592 training-only OOF records; features are the four committed VQC candidate inputs in their stored [0,1] range.
- Stable join keys are unique in both frozen prediction artifacts and follow the upstream one-to-one source join contract: `delivery_record_id, courier_id, split_partition`.
- Ridge alpha candidates: `[0.01, 1.0, 100.0]`; centered features/target from OOF training only, unpenalized intercept; no feature scaling.
- Alpha selection used only the 4,124 model-selection validation rows from 117 couriers. The 1,865 fusion-calibration rows from 58 couriers did not choose alpha.
- Test records were not loaded, scored, used for choice, or written to outputs.
- Base checkpoint, OOF, validation, split, source and role-map hashes are recorded in `residual_ridge.json`.

## Model-selection validation metrics (minutes)

| Candidate | Alpha | MAE | RMSE | Median absolute error |
|---|---:|---:|---:|---:|
| GBDT only | — | 3.172583 | 3.932094 | 2.793923 |
| Ridge residual control | 0.01 | 3.172207 | 3.931903 | 2.796163 |
| Ridge residual control | 1 | 3.172207 | 3.931903 | 2.796156 |
| Ridge residual control | 100 | 3.172248 | 3.931910 | 2.795910 |
| Selected alpha 0.01 | 0.01 | 3.172207 | 3.931903 | 2.796163 |

The selected control's model-selection MAE improvement over GBDT-only is 0.000377 minutes (about 0.023 seconds); its median absolute error is 0.002240 minutes worse. This small difference is comparison evidence only and does not justify activating the control.

## Held-out calibration diagnostics (choice not made here)

Selected alpha 0.01: MAE 3.077200, RMSE 3.847371, median absolute error 2.654300 minutes. These rows remain reserved for later calibration/fusion review; no parameter was selected from them.

## Files

- Coefficients/intercept checkpoint: `artifacts\india_eta_residual_control_v1\residual_ridge.json` (SHA-256 `5203fe9844a9721cfed1f26411a32bb0ebd872cafa8f64e661c73c2afc8189e2`).
- Validation comparison predictions: `artifacts\india_eta_residual_control_v1\validation_predictions.parquet` (5,989 rows; SHA-256 `3a0b20e1c59c7a71d986e574309acc1f13cf854c6b73bbd0bba639928cc57b69`).
- Recovery manifest: `artifacts/india_eta_residual_control_v1/recovery_manifest.json`.

No quantum code, QPU, test data, retraining of the GBDT, or operational fusion choice was used.

## Execution and verification

- Training command: `& '.runtime\run2\venv\Scripts\python.exe' -u -m experiments_india_eta.residual_control --root .` — exit code 0.
- Tests: `& '.runtime\run2\venv\Scripts\python.exe' -m pytest tests\test_india_eta_residual_control.py -q` — **2 passed** in 0.74 s.
