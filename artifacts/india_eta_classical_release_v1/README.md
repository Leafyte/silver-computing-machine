# India ETA classical CPU release

This is an inference-only bundle for the already-frozen `HistGradientBoostingRegressor` ETA reconstruction. Packaging does not retrain the model. It predicts `delivery_duration_min` in minutes from 59 preprocessed numeric features.

## Load and predict

Requires Python 3.12 and the exact dependencies in `requirements.txt`. Verify the external ZIP digest before extraction:

```powershell
Get-FileHash ./india_eta_classical_release_v1.zip -Algorithm SHA256
python predictor.py --input-json ./input.json
```

The input JSON has exactly this shape, with the full ordered feature-name list and one or more rows keyed by those feature names:

```json
{"feature_names": ["...59 names in feature_schema.json order..."], "rows": [{"feature_a": 1.0, "...": 0.0}]}
```

The prediction loader checks hard-coded SHA-256 pins for the checkpoint, feature schema, preprocessing artifact, and synthetic reference fixture before it calls `joblib.load`. It then checks the recorded scikit-learn, NumPy, and joblib versions and validates that every row has exactly the 59 expected finite numeric features.

## Preprocessing boundary

`preprocessing_artifacts.json` contains the committed imputation and scaling parameters. The full raw-record cleaning, geospatial feature engineering, categorical one-hot encoding, and split pipeline is not included. The release contains no raw CSV, Parquet dataset, OOF file, or validation prediction file. A Dell deployment must provide already-engineered features matching `feature_schema.json`; this bundle does not accept raw delivery records.

## Reference check

`reference_fixture.json` contains three synthetic feature rows created for this release and their exact minute-valued outputs. They are not dataset records and are not accuracy evidence. Verify them with:

```powershell
python predictor.py --verify-reference
```

## Provenance and validation

- Frozen model SHA-256: `f1f94fbcc932f3519aaafcff7dc99d65b7951c0240730e54c320259152bbebdf`.
- Estimator: `sklearn.ensemble.HistGradientBoostingRegressor`, squared-error objective, `{"l2_regularization": 1.0, "learning_rate": 0.06, "max_iter": 160, "max_leaf_nodes": 31, "min_samples_leaf": 25, "name": "hgb_medium"}`.
- Validation-only MAE/RMSE/median absolute error: 3.142840 / 3.905970 / 2.742487 minutes. Sealed test results are not included.
- `release_manifest.json` inventories every bundle file and records source/split hashes, dependency versions, input order, preprocessing availability, and reference outputs.
