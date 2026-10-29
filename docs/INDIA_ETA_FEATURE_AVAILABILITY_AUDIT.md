# India ETA feature availability audit

This is a source-code audit for the pending architecture review. It does not approve a VQC architecture, change preprocessing, train a model, or open the sealed test for scoring.

## Verified input contract

The committed `data/feature_schema.json` lists 59 GBDT predictors: 9 numeric, 7 binary, and 43 one-hot columns. The training implementation uses that explicit feature order. `ID`, `Delivery_person_ID`, `split_partition`, and `delivery_duration_min` are metadata/target, not predictors. The derived `target_normalized` and `is_sla_breach_synthetic` must also stay out of a residual predictor.

`tools/preprocess_indian_delivery.py:72` extracts `delivery_duration_min` from the dataset's `Time_taken(min)` field. The feature schema describes it as pickup/departure-to-drop-off duration. The preprocessing code does not independently establish that temporal interpretation from a recorded drop-off timestamp. Release consumers should distinguish the schema's declared target from independently verified operational event semantics.

## Prediction-time requirements

| Predictor group | Count | Source and availability requirement |
|---|---:|---|
| Haversine distance | 1 | Restaurant and delivery coordinates; straight-line distance, not road travel time. |
| Calendar and departure/rush indicators | 6 | `Order_Date` and actual `Time_Order_picked`. The latter is available at pickup; it is not an observed input for an earlier order-time forecast. |
| Courier age and rating | 2 | Dataset courier attributes. Their historical as-of timestamps are not supplied, so pre-delivery availability of the rating is an assumption to verify operationally. |
| Preparation duration and missing flag | 2 | Actual pickup time minus order time, with midnight adjustment. Available after pickup; using it before pickup would expose a future event. |
| Vehicle condition and vehicle type | 5 | Known assigned vehicle attributes. |
| Multiple deliveries | 1 | Requires a workload snapshot known at prediction time; the dataset does not establish when its count was observed. |
| Festival flag | 1 | Calendar/event information known at prediction time. |
| Traffic and weather fields | 12 | Snapshot/forecast values must be available at prediction time. Dataset values have no independent observation timestamps. |
| Order type | 4 | Known order attributes. |
| City type and city code | 25 | Geographic/context attributes; city code is derived from the courier-ID prefix, while the unique courier ID itself is excluded. |

These counts sum to 59. The audit supports a development model over the committed prepared rows. It does not establish that every feature can be supplied by the current two-feature backend fixture API.

## Four VQC candidates: existing preprocessing equations

All four quantities are clipped to `[0, 1]` by the committed preprocessing code:

- `vqc_x0_distance = (haversine_distance_km - train_distance_min) / train_distance_span`.
- `vqc_x1_congestion = 0.60 * traffic_ordinal / 3 + 0.25 * is_rush_hour + 0.15 * is_adverse_weather`.
- `vqc_x2_multidelivery = multiple_deliveries_imp / 3`.
- `vqc_x3_preptime = (order_to_pickup_min_imp - 5) / 10`.

The distance bounds, imputation constants, and encoded categories are fitted on the original training partition (`tools/preprocess_indian_delivery.py:239`, `:254`, `:327`). The committed feature table is reused for group-safe cross-fitting; it is conditional on this already-frozen, training-wide preprocessing, rather than newly fold-fitted preprocessing. Outer OOF regressors exclude held-out couriers and their targets.

## Architecture and integration boundaries

The committed data schema recommends two features for two qubits and describes the four-input option as a four-qubit extension. The prepared RUN2 candidate instead reuploads four inputs through two qubits in two stages. This is a separately versioned proposal, not the same circuit contract and not an approved final architecture. Opus review must resolve the mapping, gate order, observables, parameterization, and sensitivity evidence before expensive VQC fitting.

The current `origin/main` India ETA fixture API exposes distance and traffic only and loads a fixed linear JSON model. It cannot silently substitute the 59-feature trained regressor or four-input residual. A later integration must either supply the complete declared feature contract at the correct prediction time or explicitly version the model/API boundary. The classical portable release should identify itself as a development ETA regressor and must not relabel whole-delivery predictions as individual road-arc times.

## Preserved safeguards

- Original train/validation/test courier assignments remain unchanged.
- Validation model-selection and fusion-calibration couriers are separately assigned without labels.
- OOF and validation predictions join the feature views by stable delivery ID and split, not row position.
- Residual targets are observed duration minus training-OOF GBDT prediction; they are minute-valued corrections, not route-risk probabilities.
- Residual scale must be learned from training OOF residuals, not from test targets or the global target range.
- No new training, sealed-test metrics, raw-dataset redistribution, or backend replacement was performed by this audit.

## Verified recovery and preflight state (2026-10-29)

The earlier note about absent GBDT artifacts described the pre-recovery audit snapshot. Worker A has since produced `artifacts/india_eta_recovery/recovery_manifest.json`, the frozen HistGradientBoostingRegressor, 29,592 courier-grouped OOF predictions, 5,989 validation predictions, fold provenance, and validation role assignments. The manifest binds target `delivery_duration_min` in minutes and the ordered 59-feature GBDT view. The current preflight verifies the checkpoint and all declared artifact hashes without deserializing the model; it reports `CLASSICAL_READY` while keeping `quantum_training_authorized=false` pending external architecture review.

The available source CSV remains CRLF on disk. Its raw SHA-256 is `d293b6fc7157aef30fb34301216a3432ec895d407ce2ea9edddf942b416cdde1`; CRLF-to-LF normalization yields the split-manifest SHA-256 `4baa699e1c09d286f06790010a8c0b284ed3376cbde7e346e19d0ded16a240e3`, matching the bytes returned by `git show HEAD:data/train.csv` (Git blob SHA-1 `785d8d773a68eaa01850c9ad5c454ca7e48c5218`). The preflight preserves and records both identities; it does not rewrite source bytes.

The preflight tests exercise the local CRLF proof, altered-content rejection, missing and tampered artifact rejection, traversal protection, manifest-bound explicit CLI paths, and the held quantum flag. Their result is reported in the RUN2 handoff. This source/schema audit does not approve the candidate four-feature circuit, authorize VQC fitting, or open the sealed test.
