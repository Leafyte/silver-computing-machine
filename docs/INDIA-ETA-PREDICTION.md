# India ETA prediction and uncertainty slice

**Scope:** classical travel-time fixture inference, weighted synthetic SLA arithmetic, and route-arc compatibility checks. This slice does not train/refit a model, fit residuals, claim calibration, invoke a quantum circuit, mutate the dispatch optimizer, or add API routes.

## Contracts

Public Python functions are `backend.india_eta.prediction.predict_eta(payload, registry=None)`, `backend.india_eta.uncertainty.evaluate_sla(payload)`, and `backend.india_eta.compatibility.check_compatibility(payload)`. Each accepts a plain JSON-like `dict` and returns a JSON-like `dict`. Invalid inputs raise `EtaError(code, message, status=422)` with stable non-echoing messages; a missing/integrity-failed release reports a stable 503 model error.

### Prediction request v1

Required keys are `schema_version="india_eta.prediction.request.v1"`, `request_id`, timezone-aware ISO `prediction_time`, `source_domain="india_food_delivery"`, `geography="IN"`, `origin`, `destination`, `vehicle_type`, `features`, `model_id`, and `model_version`. Optional keys are `feature_schema_hash` and boolean `qml_requested`.

`origin` and `destination` contain an identifier and may contain both `latitude` and `longitude`. `features` is exactly `distance_km`, `traffic_index`, and `observed_at`. Distance is finite positive km; traffic index is finite in [0, 1]. `observed_at` contains one timezone-aware timestamp for each feature, both at or before `prediction_time`. Vehicle/model/domain values must be supported by the release. Unknown keys, boolean-as-number values, non-finite values, naive/future timestamps, unsupported features/schema, invalid coordinates, and unknown model versions fail closed.

The frozen feature contract is [prediction_feature_schema.json](../fixtures/india_eta/prediction_feature_schema.json). `feature_schema_hash` is SHA-256 over its canonical JSON encoding (`sort_keys=true`, separators `(',', ':')`, UTF-8, no ASCII escaping): `sha256:25c0fe95ab8d1cb722a5c7f8c3b9d059c09004ddfee1ebd7bfd5529930838563`. This replaces an earlier descriptive label that was not a digest. Model ID `india-eta-linear-fixture` and version `1.0.0` are retained because this local deterministic fixture release has not been published; the feature contract now has a real digest and is pinned in the release JSON.

The result reports duration in minutes and arrival time, target, model/feature/data/split provenance, execution metadata, and QML/fallback flags. No model training or split provenance is present in the hand-authored release (`data_hash`, `split_hash`, and training hash are null). The deterministic release computes `5 + 3.2 * distance_km + 12 * traffic_index`; fixture values 5 km and 0.75 yield 30 minutes. This is example arithmetic only.

Model loading uses JSON at a fixed package fixture path, checks its canonical SHA-256 against a source-pinned constant, validates the exact schema and parameters, and never imports code, unpickles, follows a request path, or fetches a URL. For frozen releases, the server may build an allowlist with `load_model_registry(root, allowlist)`. The server-provided allowlist maps `model_id@model_version` to `{relative_path, sha256}`; the loader accepts only relative `.json` files that resolve inside the configured model root, caps the file at 128 KiB, checks the independently configured SHA-256, and returns an immutable registry snapshot. `predict_eta` accepts entries `{release, sha256}` keyed by the requested model identity. Request payloads never contain paths or digest authority.

Every release uses the same exact JSON linear schema and the fixed feature schema. `model_kind` is `fixture` or `frozen_classical`. Frozen releases must carry `status="validated_source_domain"` plus nonempty source, data hash, split hash, and training-data hash in their hash-pinned provenance. Response source and execution labels are derived from the trusted release kind; provenance hashes/status are copied from that release. These labels describe the configured release and do not establish external evaluation beyond its recorded provenance.

If `qml_requested` is true, output still has `qml_executed=false`, no hybrid/quantum residual, `execution_type="classical_fixture"`, `fallback_used=true`, and `fallback_reason="duration_quantum_checkpoint_missing"`. The current route-risk VQC does not predict duration and is not used here.

### SLA request v1

Required keys are `schema_version="india_eta.sla.request.v1"`, a prediction response, timezone-aware `deadline`, finite nonnegative `late_penalty_per_minute`, and alphabetic `currency`. Optional `scenarios` and `scenario_source` default to unavailable. A synthetic request must explicitly set `scenario_source="synthetic_fixture"` and provide 1-100 `{residual_minutes, weight}` rows; residuals are finite, scenario duration cannot be negative, weights are positive and sum to one. Unknown source labels, including caller claims of calibrated/validated residuals, are rejected.

A separately frozen uncertainty release can be selected by optional `uncertainty_id` and `uncertainty_version`, passed to `evaluate_sla(payload, registry=None)`. The server injects `uncertainty_id@uncertainty_version -> {artifact, sha256}`, with the expected digest kept in trusted server configuration. The JSON artifact binds model ID/version, feature-schema hash, source domain, geography, target, residual scenarios, conditional support ranges, status, and provenance. A validated artifact requires source ID, data hash, calibration/evaluation split hashes, sample count, calibration method, and calibration-artifact hash. Evidence hashes and residual weights are validated; artifact and embedded content hash must both match the server-pinned digest. Caller-provided scenario arrays cannot select the trusted path or self-assert its status.

Conditional support uses finite inclusive `{minimum, maximum}` ranges for supported context fields such as `traffic_index` and `distance_km`; values outside the frozen ranges return `status="out_of_domain"` with null risk metrics. A trusted release marked out-of-domain also returns null metrics. Missing registry/release returns unavailable nulls. Identity/schema/domain/target mismatches fail with a stable `uncertainty_mismatch` error.

The prediction timestamp, arrival, and duration are cross-checked before applying residuals. For each scenario, residual minutes shift the travel-only arrival; positive lateness is measured against the deadline. Expected penalty is penalty-rate times weighted expected excess-late-minutes, not probability times rate. Scenario outputs are labeled `synthetic_fixture` / `synthetic_arithmetic_only`; no calibration claim is made. `interval_coverage` returns descriptive empirical coverage only and cannot certify calibration.

Oracle: prediction at 10:00 plus 20 travel minutes, equal synthetic residuals -5, 0, +10 minutes, and deadline 10:25 gives arrivals 10:15, 10:20, 10:30 and lateness 0, 0, 5. Therefore late probability = 1/3, expected late minutes = 5/3, and at 6 currency units/minute expected penalty = 10. This is synthetic arithmetic, not measured ETA quality.

### Compatibility request v1

Exact keys: `schema_version="india_eta.compatibility.request.v1"`, `source_domain`, `geography`, `source_units`, `prediction_target`, `road_routing_provenance`, `vehicle_type`, `features`, `time_semantics`, `service_semantics`, positive finite `horizon_minutes`, and `uncertainty_status`.

The checker returns `supported`, `scenario_only`, or `rejected`, a list of stable reason codes, and an explicit `road_arc_mapping_allowed` flag. Whole-delivery duration cannot be mapped to an individual road arc; minutes, India, supported vehicles/features, timezone-aware service-start semantics, and verified route-duration provenance are required for an operational road arc. Straight-line distance is not road travel time. An explicit `source_domain="synthetic_overlay"` plus `road_routing_provenance="synthetic_fixture"` is accepted as `scenario_only`, never as a road arc; optional positive finite `duration_minutes` is echoed as `scenario_duration_minutes` for review. An India-domain request cannot use synthetic provenance for an operational road arc. Synthetic overlays or unavailable residual evidence are scenario-only. A caller-supplied `validated_source_domain` label does not configure trusted evidence and cannot upgrade the result to operational support. Unsupported geography, vehicle, features, target, units, clocks, service semantics, or road provenance are rejected.

## Files and checks

- `backend/india_eta/`: stable safe errors, prediction, SLA arithmetic, compatibility logic.
- `fixtures/india_eta/prediction_model.json`: tiny fixture release; request/result JSON show the default flow.
- `tests/india_eta/`: classical/fallback, hashing, strict schema, temporal validation, weighted arithmetic, and compatibility boundary checks.

Executed refinement record (all commands used the repository `.venv` Python from the isolated implementation worktree):

| Cycle | Failure hypothesis and evidence | Material edit | Retest and result | Remaining concern |
|---|---|---|---|---|
| 1 | Non-finite horizon could cross the compatibility boundary. First test run showed `horizon_minutes=NaN` did not raise (`17 passed, 1 failed, 8 subtests`). | Added explicit finite-number validation before positivity check. | `python -m pytest tests/india_eta -q --basetemp=.runtime/pytest-india-eta-prediction-cycle2`: 20 passed, 8 subtests. | More type and semantic edge cases remain covered by later cycles. |
| 2 | A caller could label uncertainty `validated_source_domain` and reach an operational decision despite no trusted residual registry. The initial branch treated that label as sufficient; the boundary is caller-controlled. | Made the checker keep all uncertainty states scenario-only until a trusted registry is actually configured; added a regression case for the self-asserted label. | Same retest: 20 passed, 8 subtests; the claim remains scenario-only. | A future server-owned evidence registry needs its own verification contract before any upgrade. |
| 3 | SLA residuals shift the prediction arrival, so a response whose arrival disagrees with its stated duration could create arbitrary late risk. | Added prediction-time output and require duration/arrival consistency before risk arithmetic; added a tampered-arrival test. | Same retest: 20 passed, 8 subtests; mismatch returns `invalid_prediction`. | Synthetic residual arithmetic does not imply coverage or calibration. |

The follow-up registry refinement added regression cases for a separately allowlisted frozen model, server path/hash validation, a trusted frozen residual release, caller spoof rejection, metadata mismatch, conditional out-of-domain behavior, and held-out interval-coverage reporting.

Final owned test command from the implementation worktree:

```powershell
C:\ommd3076\Proejcts\nmit-hackathon\.venv\Scripts\python.exe -m pytest tests/india_eta/test_prediction.py tests/india_eta/test_uncertainty.py tests/india_eta/test_compatibility.py -q --basetemp=.runtime/pytest-india-eta-refinement-final
```

Result: **27 passed, 10 subtests passed**. An earlier refinement run failed because Windows sandbox temporary paths defaulted outside the writable checkout; test temporary model releases were moved under `.runtime`, and the server loader now translates path errors to safe ETA errors. The prior broad directory run had two failures outside this slice (optimization/projection tests); those files remain unchanged. No route/API, optimizer, training, dataset, auth, quantum, cloud, or GPU work was executed in this slice.
