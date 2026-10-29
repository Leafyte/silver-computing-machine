# Model serving audit and proposed inference contract

**Scope:** RUN2 India ETA release, feature semantics, current fixture boundary, and the additive inference contract. This is an audit/proposal, not an implementation. All additions are outside the Git repository. Audit date: 2026-10-30 (Asia/Calcutta).

## Decision

Keep `/api/v1/eta/*` untouched as the two-feature synthetic linear `travel_time` fixture. Add a separate `/api/v2/eta/predictions` research/development contract for exactly 59 prepared named numeric features. The v2 request must carry an explicit post-pickup or historical prepared stage, the frozen feature-schema hash, per-feature source/time evidence, and an as-of timestamp. It is not a raw-record feature engineer.

The published RUN2 ZIP is intact: its SHA-256 matches the report-published `18d5d4859e53d3d59063e60cdcdd45cb6eff6757bba6474f024cdde91aa06b80`, all 11 inventoried artifact members match their manifest sizes/hashes, and it contains the pinned release loader/manifest. The unpacked working-tree directory has a divergent `circuit.py`, so the loader correctly rejects that copy before `joblib.load`. Use the verified ZIP as the deployment source and extract it into a fresh administrator-controlled immutable directory; do not repair this copy by editing hashes. The current project `.venv` still cannot load the release: scikit-learn is missing and NumPy 2.5.4 differs from pinned 1.26.4. No compatible RUN2 virtual environment was present in `.runtime/run2` at audit time. Until the exact pinned dependencies are provisioned in an isolated runtime and parity is executed there, startup readiness must remain unavailable. Do not bypass checks or serve a numerically similar fallback.

Even after that package blocker is resolved, the model is only a development prediction candidate. The frozen operational fusion is `alpha=0` (GBDT only); VQC outputs remain diagnostic and must never influence a dispatch decision. No fresh prospective, operationally timed validation exists here. The target is the dataset's `delivery_duration_min` / `Time_taken(min)`, not proven road-leg time or a fleet ETA with independently verified pickup/drop-off event boundaries.

## Repository evidence and classification

| Claim | Classification | Evidence / boundary |
|---|---|---|
| Existing fixture API is a hash-checked, JSON-only, two-feature linear model with `distance_km` and `traffic_index`; it reports `prediction_target: travel_time`, minute units, timestamps, and explicitly marks the missing duration-QML path as not executed/fallback. | Verified implemented | `backend/india_eta/prediction.py:18-20, 188-256`; retain its contract for v1. It is not compatible with the 59-feature HGBR. |
| A standalone CPU predictor, HGBR checkpoint, VQC JSON candidates, Ridge control, feature schema, preprocessing constants, policy and synthetic-only parity fixture are packaged in the published archive. | Archive integrity verified; runtime execution not repeated | The ZIP SHA matches `RUN2_INDIA_QML_FINAL_TRAINING_REPORT.md:101-107` and all 11 manifest members match; the unpacked repo directory has one divergent circuit file and must not be used. |
| Loader validates a pinned manifest, exact inventory, symlink/path safety, dependency versions, all artifact sizes and hashes, 59 unique ordered features, and only then calls `joblib.load`. | Verified in code | `predictor.py:30-44, 168-218`. The security ordering is correct for a trusted release and should be preserved. The deployment wrapper must bind the path to administrator configuration, not request input. |
| Current frozen selection policy is GBDT-only (`alpha=0`). VQC 60/76 outputs are candidates; policy requires validation selection gates and test partition is retrospective/exposed. | Verified in published report and package policy | `fusion_policy.json`; `RUN2_INDIA_QML_FINAL_TRAINING_REPORT.md:3-9, 58-83, 99-107`. Do not change alpha or re-open the old test split. |
| The 59-column inference release consumes an already prepared matrix. It is not a raw-feature preparation service. | Verified | `predictor.py:68-91, 130-166`; `docs/INDIA_ETA_FEATURE_AVAILABILITY_AUDIT.md:5-26, 39-43`. |
| Feature-aware FastAPI endpoint for the 59-column model exists. | Absent | `backend/api.py` mounts existing v1 ETA routes/controllers; no v2 prepared-feature prediction endpoint is present. Additive proposal below. |
| Production feature preparation and availability verification exists. | Absent | Audit notes several source timestamps are unknown; actual pickup and preparation time are post-pickup. Earlier order-time output needs a separately trained earlier-stage model and its own feature contract. |
| Model inference is performance-characterized for production. | Not established | Report has a single batch timing and model-training resource reproduction, not repeated cold/warm p95/p99 serving data. See benchmark plan below. |

### Release integrity check performed in this audit

Read-only checks ran before any model load. The published ZIP SHA-256 is `18d5d4859e53d3d59063e60cdcdd45cb6eff6757bba6474f024cdde91aa06b80`, matching the final report, and each of its 11 manifest-inventoried members matches byte count and SHA-256, including the expected circuit bytes/hash. In the unpacked repository directory, ten of eleven members match: `circuit.py` is **30,294 bytes**, actual SHA-256 `e347bade7e5a9081a08417aaa36fb8c390ea52bb02a6c44f9b9c7dfa9c0903d5`; the tracked manifest declares **29,701 bytes**, expected SHA-256 `801468f544e25ae913c26e1f2131cf5f88b384b8495172da62be9051d5a96ba0`. The manifest itself matches the loader-pinned SHA-256 `1a0a2052b655a4306189ed0fe0d5f99be29b0379b0e322cf31aa514eb9200c5b`. The GBDT artifact hash matches in both archive and directory, but was **not deserialized**.

Local Python inventory: CPython 3.12.10, NumPy 2.5.4, joblib 1.6.0, FastAPI 0.162.0, Pydantic 2.14.1; scikit-learn is not installed. Release pins are NumPy 1.26.4, scikit-learn 1.4.2, joblib 1.6.0. No `.runtime/run2/venv` or `.venv-sdk` interpreter was present; the report's training runtime (Python 3.12.10/Torch) is not proof this repository venv can load the inference release.

**Executed here:** manifest/member hash and size verification; Python package inventory; JSON parsing and schema/fixture contract checks. **Not executed here:** release loader, joblib deserialization, GBDT prediction, endpoint, batch parity, model timing. No model was deserialized and no training was performed.

## Exact 59-feature contract

The feature names and order in `model-prepared-feature-request.example.json` match the committed `data/feature_schema.json` `gbdt_view.features` set: 59 of 59 names. The names (ordered per release) are:

```text
haversine_distance_km, departure_hour, day_of_week, is_weekend, is_rush_hour, is_lunch_rush, is_dinner_rush, delivery_person_age_imp, delivery_person_ratings_imp, order_to_pickup_min_imp, order_to_pickup_missing_flag, vehicle_condition, multiple_deliveries_imp, is_festival, traffic_density_ordinal_imp, is_adverse_weather, weather_clean_imp_Cloudy, weather_clean_imp_Fog, weather_clean_imp_Sandstorms, weather_clean_imp_Stormy, weather_clean_imp_Sunny, weather_clean_imp_Windy, traffic_density_clean_High, traffic_density_clean_Jam, traffic_density_clean_Low, traffic_density_clean_Medium, type_of_order_clean_Buffet, type_of_order_clean_Drinks, type_of_order_clean_Meal, type_of_order_clean_Snack, type_of_vehicle_clean_bicycle, type_of_vehicle_clean_electric_scooter, type_of_vehicle_clean_motorcycle, type_of_vehicle_clean_scooter, city_type_clean_imp_Metropolitian, city_type_clean_imp_Semi-Urban, city_type_clean_imp_Urban, city_code_AGR, city_code_ALH, city_code_AURG, city_code_BANG, city_code_BHP, city_code_CHEN, city_code_COIMB, city_code_DEH, city_code_GOA, city_code_HYD, city_code_INDO, city_code_JAP, city_code_KNP, city_code_KOC, city_code_KOL, city_code_LUDH, city_code_MUM, city_code_MYS, city_code_PUNE, city_code_RANCHI, city_code_SUR, city_code_VAD
```

The schema excludes IDs, courier IDs, split labels and target; the audit states the schema has 9 numeric, 7 binary and 43 one-hot columns. A valid request requires all 59 numeric values exactly once (no null, bool, NaN/Infinity, missing or extra key), matching schema ID/version/hash, an ISO-8601 timezone-aware `prediction_as_of`, and an evidence record for each feature with source field and `observed_at <= prediction_as_of`. The service must reject future-dated evidence and unknown feature contracts. Pydantic v2 can implement the JSON Schema with `ConfigDict(extra="forbid", strict=True)`, bounded request bytes, finite-number validation and a model-level cross-validator for key equality and timestamp ordering.

**Availability gate:** `order_to_pickup_min_imp` and `order_to_pickup_missing_flag`, and pickup-time calendar/departure features based on actual `Time_Order_picked`, are not knowable at order time. The feature audit also does not establish timestamp lineage for every weather, traffic, rating and workload datum. Therefore the current contract only permits `post_pickup` or explicitly historical precomputed research rows; a pre-pickup forecast must use a separately trained model whose feature set is available at order time. Do not fill future pickup facts with dataset medians and call the resulting forecast operational.

## Proposed P0 modules and lifecycle

These are additions/modifications to implement later; none is claimed implemented here.

| Location | Responsibility |
|---|---|
| `backend/india_eta/model_release.py` (add) | Administrator-only configured release root and expected identity/hash; resolve and constrain paths beneath the trusted root; exact manifest/member/schema/dependency checks; reject symlinks, traversal, extra files and version mismatch; verify hashes **before** trusted local joblib deserialization; load once under a lock and expose immutable identity. Never accept a client model path or serialized bytes. |
| `backend/india_eta/feature_contract.py` (add) | Frozen ordered names/schema hash; strict finite 59-feature mapping; per-feature source/availability/timestamp validation; schema compatibility. Separate from raw preparation. |
| `backend/india_eta/real_prediction.py` (add) | Small inference service over cached trusted predictor; operational-policy call uses GBDT alpha=0 only. Optional diagnostic branch explicitly computes named baseline60/extended76 comparisons and marks them `not_for_dispatch`. Validate every numeric output. No silent numerical/model fallback. |
| `backend/india_eta/schemas_v2.py` (add) | Strict versioned request, response, feature provenance, uncertainty, model identity and safe error schemas. |
| `backend/api.py` (modify additively) | `POST /api/v2/eta/predictions`, readiness status in health, stable 422/503/500 errors; preserve existing `/api/v1/eta/*`. Body-size limit/auth middleware applies. |
| `backend/india_eta/...` deployment settings (add/configure) | Pin configured release path, manifest SHA, expected format, versions and explicit mode. Disable model at startup if any trust/dependency/parity preflight fails; do not silently switch v1 fixture to GBDT. |

Startup states: (1) `MODEL_DISABLED` by explicit default; (2) `MODEL_LOADING`; (3) `MODEL_READY_DEVELOPMENT`; or (4) `MODEL_UNAVAILABLE` with stable reason code. A startup/readiness failure gives health/readiness 503 and prediction 503. A well-formed request with missing/extra features or invalid timestamps gets 422 and does not impact readiness. A non-finite or invalid predictor output is `inference_failure` 500, is logged without raw row values, and returns no prediction. Do not report fallback success. Existing v1 is an independent fixture route and must only be used when the caller explicitly requests its old contract.

The release's general `IndiaETARelease.predict()` computes both quantum residual candidates before returning its `operational_min`, even at alpha zero (`predictor.py:130-166`). Keep the user-facing operational call isolated to GBDT-only; execute candidate diagnostics only when explicitly requested on a separate development/diagnostics path. This avoids making the frozen operational baseline depend on experimental-circuit code and prevents a diagnostic exception from silently changing dispatch output.

### Target, metadata and response policy

The response carries `prediction_target=delivery_duration_min`, `duration_unit=minute`, `target_semantics="dataset_delivery_duration_min; operational event start/end not independently verified"`, `prediction_as_of`, `observation_stage`, feature source and availability through time, model version, pinned release-manifest SHA, feature schema SHA, `prediction_status=development_prediction`, `operational_policy.alpha=0`, and `diagnostics.not_for_dispatch=true`. Do not call the output a road edge time, leg time, arrival time or verified whole-fleet ETA without separate evidence and conversion logic.

Uncertainty must be typed and provenance-bearing:

- `source_derived`: only a documented, source-split-derived residual interval/spread with a method and evidence ID; label it uncalibrated unless coverage was measured.
- `calibrated`: lower/upper interval and nominal coverage are allowed only after a frozen calibration procedure on a separate calibration cohort, reporting groupwise coverage and limitations.
- `unavailable`: null bounds and coverage with a reason. This is the correct current response. Never derive confidence intervals from VQC candidate spread or use VQC disagreement as calibrated uncertainty.

## Proposed FastAPI contract and examples

The full JSON Schema / OpenAPI fragment is in `model-openapi-inference-v2.proposed.json`; the complete 59-feature valid request example is in `model-prepared-feature-request.example.json`. These are proposals, not live routes. Pydantic implementation should return a stable envelope: `{"error":{"code":"...","message":"...","details":[{"field":"...","message":"..."}]}}`.

| Outcome | Status | Code / behavior |
|---|---:|---|
| Valid prepared feature request | 200 | GBDT prediction, alpha=0, development status, model/hash/provenance metadata, uncertainty unavailable. |
| Optional diagnostic comparison | 200 | Same GBDT output plus VQC candidates in an explicitly requested diagnostic section, `not_for_dispatch=true`; alpha stays 0. Values in OpenAPI example are from the bundle's published synthetic fixture, not execution in this audit. |
| Missing/extra/wrong-schema/non-finite feature, bad stage/timestamp | 422 | `invalid_feature_schema`, `feature_schema_mismatch`, `feature_observed_after_prediction_as_of`, or `unsupported_prediction_stage`; field-level safe details. |
| Release hash/manifest/dependency/readiness mismatch | 503 | `model_integrity_error`, `model_dependency_mismatch`, or `model_unavailable`; no synthetic/v1/numerical fallback. |
| Unexpected predictor exception/nonfinite output after valid loaded release | 500 | `inference_failure`; generic sanitized message. |

## Release security and reload policy

1. Release path and pinned expected digest are admin configuration only. Canonicalize path once at startup, disallow traversal/symlinks, constrain it to a configured immutable release root, verify regular files and exact inventory.
2. Parse bounded JSON manifest, validate format/version/feature count and expected manifest SHA. Verify every declared file size and SHA and dependency versions before `joblib.load`. Never load user-supplied pickle/joblib. No configurable hash from a caller.
3. Cross-check schema order/count/hash against the expected contract and estimator identity/dimension after deserialization. Run the package's synthetic reference parity check on the successfully verified object before READY; parity is packaging validation, not a quality evaluation.
4. Cache one verified model object per process. Thread-safe initial load. Do not reload on requests. An explicit administrator reload constructs/validates a new object off to the side, then atomically swaps it only if all gates pass; preserve prior ready object if a separately approved hot reload fails, while emitting a readiness/reload event. Never auto-download or auto-select newest release.
5. Current audit hash mismatch means no safe load now. Repair requires a new release identity/manifest pin and traceable sign-off, not editing the manifest hash in place. Require exact supported pins in an isolated runtime: NumPy 1.26.4, scikit-learn 1.4.2, joblib 1.6.0.
6. Bound JSON request body and batch size, authenticate through existing backend middleware/role policy, redact feature rows and secrets from logs, record request ID, model/release hash, schema hash, status, latency and safe error code. Preserve `/api/v1` request behavior.

## Performance plan (targets, not results)

Use CPU only first, with CPU thread count configured explicitly to avoid oversubscription. Measure on the actual deployment machine after integrity/readiness is fixed. Keep prediction outputs pinned to the synthetic release reference fixture and verify numeric parity before timing.

| Measurement | Procedure | Prototype acceptance target (not measured) |
|---|---|---|
| Cold start | Fresh process; verify+load+reference parity, repeat 5 fresh processes; record load separately from parity. | p95 under 5 seconds; failure always leaves model unready. |
| Warm single-row | 1 warmup then at least 1,000 serial predictions, perf-counter timing. | GBDT-only p95 under 100 ms, p99 under 200 ms. |
| Batch | batch 1/16/64/256 (body-limit permitting), at least 100 batches each; report per-row and request latency. | batch 64 p95 under 250 ms; preserve exact row count/order. |
| Concurrency | 1/2/4/8 simultaneous requests, each 500 warm requests; GBDT-only and diagnostics measured separately. | p95 under 500 ms at concurrency 4; no queue unbounded growth or prediction mismatch. |
| Memory | Capture process peak RSS / working set at cold load, warm steady state, batch 64, diagnostics. | Stay under 1 GiB prototype budget, including Python overhead; report observed peak. |
| Reload | 10 valid reloads and one tampered/incompatible release; track pause, errors, identity swap. | Valid swap only after parity; invalid release never replaces ready model. |
| Reference parity | Every release build/runtime and after reload compare all 5 published synthetic reference outputs with declared tolerance (e.g. 1e-6 minutes). | 3 rows pass; this verifies packaging, not accuracy. |

Existing report evidence: one batch measurement reports GBDT-only/alpha-zero at 0.011717 s and VQC residual calls at 0.067975 s/0.078538 s; it excludes parsing/metrics and is not a latency distribution (`RUN2_INDIA_QML_FINAL_TRAINING_REPORT.md:87-97`). Training reproduction peak working set was about 860 MB, not inference memory (`:50`). Neither result satisfies the serving benchmark above.

## Implementation acceptance gates

1. Startup fails closed against the current package with a stable release-integrity status; no call reaches deserialization. The package is repaired/re-pinned independently before enabling it.
2. Verified package and exact dependency environment pass synthetic fixture parity with matching output shape/finite values. Tests assert hash verification precedes deserialization using a spy or tamper fixture; do not unpickle unverified files.
3. Strict schema tests cover each missing key, extra key, bool, null, NaN/Infinity, reordered declared schema, stale schema hash, malformed timestamp, future timestamp, and pre-pickup attempt. 59-feature positive fixture passes parser and feature evidence cross-check.
4. Output target, units, as-of, stage, source, availability, release/model/schema hashes are present. Operational alpha stays exactly 0; candidate outputs never affect `predicted_duration_minutes`.
5. Model-ready failure returns 503; malformed request returns 422; predictor runtime/numeric failure returns sanitized 500. No synthetic fallback or silent identity change.
6. Bounded CPU benchmarks and concurrent/reload tests report distributions and memory. All targets are targets until those measurements exist.
7. Separate prospective feature availability and held-out validation plan is approved before claiming operationally verified output. Pre-pickup forecast requires a separately trained model with only order-time available features.

## Audit artifacts and verification

- `model-prepared-feature-request.example.json`: generated from the release's explicitly synthetic `reference_fixture.json` first row; contains 59 finite numeric values and 59 time/source provenance rows. Not operational data.
- `model-openapi-inference-v2.proposed.json`: OpenAPI 3.1 proposed endpoint, complete 59-field strict request schema, response/error schemas and examples. Expected values in successful response example are published fixture expectations, not a prediction executed during this audit.
- Read-only checks completed: published ZIP hash; archive and directory member hashes/byte counts; manifest pin; package inventory; feature-name ordered equality with `data/feature_schema.json`; fixture JSON shape/types/timestamps; OpenAPI JSON parse and route response statuses.
