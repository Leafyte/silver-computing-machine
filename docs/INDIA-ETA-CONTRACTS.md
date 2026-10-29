# India ETA integration contracts

Implemented 2026-10-29, Asia/Kolkata. API contract version is `/api/v1`; this subsystem supports explicitly simulated dispatch state. Model inference and uncertainty evidence have separate versioned schemas. The planning document is historical; this file describes the implemented surface.

## Running locally

From the isolated implementation worktree:

```powershell
C:\ommd3076\Proejcts\nmit-hackathon\.venv\Scripts\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Prediction/uncertainty work in the existing local development mode. Initial plan registration and approval require explicitly configured token-role authentication. The existing settings are `DEBUGLEAF_AUTH_MODE=token`, `DEBUGLEAF_DISPATCHER_TOKEN`, `DEBUGLEAF_RESEARCHER_TOKEN`, and `DEBUGLEAF_ADMIN_TOKEN`. Set actual credentials privately in the process environment; never pass them in JSON, persist them in fixture files or expose them in logs. This prototype does not implement tenant/driver identity or production security.

The credential-free demonstration uses publicly labeled non-secret test sentinels, binds loopback, creates its own database and stops its server:

```powershell
C:\ommd3076\Proejcts\nmit-hackathon\.venv\Scripts\python.exe tools/india_eta_http_smoke.py
```

`create_app` accepts server-only `eta_db_path`, `eta_registry`, `eta_uncertainty_registry`, `eta_controller`, and `eta_job_manager` injection. Default SQLite path is `.runtime/india_eta.sqlite3`; `DEBUGLEAF_ETA_DB_PATH` overrides it. Existing jobs/recovery tables and endpoints remain separate.

## Implemented route map

| Method/path under `/api/v1/eta` | Contract and role |
| --- | --- |
| POST `/predictions` | `india_eta.prediction.request.v1` to `.response.v1`; local mode or configured researcher/dispatcher/admin. |
| POST `/sla` | `india_eta.sla.request.v1` to `.response.v1`; same roles. |
| POST `/compatibility` | `india_eta.compatibility.request.v1` to `.response.v1`; same roles. |
| POST `/plans` | Register an independently validated initial approved plan, dispatcher/admin token required even in local mode. |
| GET `/plans/{plan_id}` | Active approved version, source inputs, costs and proposal summaries. |
| POST `/events` | Resolve optional ETA overlay, admit bounded durable event job, HTTP 202; dispatcher/admin in token mode. |
| GET `/events/{event_id}` | Job state, result, safe error, input hash, times/settings. |
| POST `/events/{event_id}/cancel` | Suppress late output and invalidate pending proposals; dispatcher/admin in token mode. |
| GET `/proposals/{proposal_id}` | Wrapper with status/base version and nested `proposal`; context hash is `proposal.context_hash`. |
| POST `/proposals/{proposal_id}/approve` | Explicit dispatcher/admin token required; version/context/expiry/job-state revalidation and atomic publication. |
| GET `/manifests/{vehicle_id}?version=N` | Latest or historical **vehicle manifest revision**; JSON `plan_version` is its parent plan version. |

In token mode researchers can read ETA resources but cannot mutate operational state. No resource-ownership/tenant filtering is claimed. Unknown ETA routes are denied by the role policy. Bad requests/errors use `{error:{code,message}}` without submitted values or arbitrary exception messages; framework schema errors are sanitized. Statuses include 401/403 authorization, 404 missing resource, 409 version/state conflict, 422 invalid data, 429 admission capacity, 501 unsupported live capability and 503 runtime/storage budget failure. Async application errors are durable terminal job errors.

## Prediction and frozen model release

Request example: `fixtures/india_eta/prediction_request.json`. Required fields: schema version, request ID, timezone-aware prediction time, source domain `india_food_delivery`, geography `IN`, origin/destination objects, vehicle type, features, model ID/version. Optional `feature_schema_hash` must match; optional `qml_requested` is a strict boolean. Features are `distance_km`, `traffic_index`, and per-feature `observed_at`; future features are rejected. Units/ranges/types are frozen in `fixtures/india_eta/prediction_feature_schema.json`. Origin/destination allow an ID and optional paired latitude/longitude. Unknown fields and unsupported vehicles/domains fail safely.

Current safe model format is JSON linear travel-time inference over the two declared features. The fixed hand-authored fixture is `india-eta-linear-fixture@1.0.0`. Its release and feature schema have actual canonical SHA-256 digests. A server can configure other frozen releases of this supported format using `load_model_registry(root, allowlist)`; allowlist entries are `{relative_path, sha256}`. Root confinement, file-size, schema, identity and externally pinned content hashes are checked. Request JSON cannot name files, URLs, Python classes or loaders. No pickle/joblib or dynamically executable model format is accepted.

Response separates classical duration/arrival, unavailable quantum residual/hybrid result, source/model/schema/data provenance, execution type, measured runtime and fallback. A missing duration-QML checkpoint always leaves `qml_executed=false`; requesting QML never converts the route-risk VQC into a duration predictor. Fixture data/split hashes are unavailable and remain null. A trusted frozen classical release must supply its required training/data/split provenance; it is not automatically a verified real-world model merely because JSON loading succeeds.

## Uncertainty/SLA

SLA request requires schema version, a versioned prediction response, a timezone-aware supplied deadline, nonnegative late penalty per minute and currency. Without scenarios/trusted release, output status is `unavailable`, with null probability, expected late minutes and penalty.

Synthetic arithmetic uses `scenario_source=synthetic_fixture` and 1–100 `{residual_minutes,weight}` rows. Weights must be positive and sum to one; negative resulting durations are rejected. A caller cannot claim calibrated status. Trusted residual selection instead uses `uncertainty_id` and `uncertainty_version`; the server registry supplies `{artifact,sha256}` with pinned model/schema/domain/target identity, condition support and calibration/split/sample evidence. Mismatch fails safely; missing support returns `out_of_domain` with null metrics. The server owner is responsible for the truth of separately supplied calibration evidence. Tests use synthetic evidence fixtures; no real India calibration was performed.

Compute probability from the weighted late indicator and expected late minutes from weighted positive excess. Penalty equals rate times expected late minutes, not rate times probability. The independent oracle is 10:00 + 20 minutes with residuals [-5,0,10], equal weights, deadline 10:25, rate 6: probability 1/3; expected late minutes 5/3; penalty 10. Food-delivery observed duration does not establish a real contractual deadline.

## Source adapter and event input

Base event schema requires `event_id`, `plan_id`, `plan_version`, `model_version`, `as_of`, `observed_state`, `scenario`, `weighted_scenarios`, and `event_type`. Optional terms are `currency`, `operating_rate`, and `mode`. Event types: `eta_update`, `delay`, `vehicle_status`, `route_change`. Model/plan versions must match the approved state. Event IDs are idempotency keys; changed content conflicts. Timestamps must advance the context.

The event API additionally accepts optional `eta_update`:

```json
{
  "schema_version": "india_eta.synthetic_update.v1",
  "arcs": [{
    "origin": "D",
    "destination": "A",
    "prediction_request": {"see": "the complete prediction request in fixtures/india_eta/http_propose.json"},
    "residual_minutes_by_scenario": {"joint1": 0}
  }]
}
```

The abbreviated prediction above is explanatory, not a valid request. `fixtures/india_eta/http_propose.json` contains the complete reproducible registration/event/result. The API recomputes each prediction server-side, checks locations, prediction timestamp and active model version, calls the explicit synthetic domain gate, and derives nominal/joint arc durations from that prediction plus a declared residual for **every** frozen joint world. This adapter requires explicit `mode=synthetic_scenario`, `source_label=synthetic_overlay` and target `travel_time`. It never certifies a real road arc. Deterministic model/input hashes and compatibility evidence are persisted in `weighted_scenarios[].eta_source_provenance`; runtime fluctuations are excluded so exact duplicate requests remain idempotent.

Whole-delivery targets, incompatible geography/vehicle/features/time semantics, unlabeled matrices and straight-line-as-road transfers are rejected. Solomon/Dabia retain source matrices; real India-domain transfer remains unsupported without a separately evidenced source adapter. Synthetic overlays are explicitly experimental, not operational calibration.

Each weighted travel world is `{id,weight,travel,source_label,prediction_target}` with a complete nonnegative finite minute matrix, unique ID and positive normalized weight. The ETA overlay checks these before durable admission and retains the bounded original prediction request for independent replay. Worlds are joint scenarios; no independence or sum-of-marginal-quantiles assumption is introduced. Existing `scenario` fields are depot, vehicles, orders and travel; order windows/service/penalty/hard-window semantics remain unchanged.

Each observed vehicle row is `{vehicle_id,position,position_type,elapsed_minutes,available_at,completed_prefix:[{order_id,completed_at}]}`. Plan-clock minutes start remaining projection at max(elapsed,available) from the supplied location. Current activation requires `position_type=simulated`; `actual`/live state is rejected. Completed prefixes/timestamps and clocks cannot regress; incomplete work cannot be skipped.

## Registration, decision and approval

Initial registration requires `plan_id`, `version=1`, `model_version`, `as_of`, source scenario/worlds and a full domain plan. Optional policy terms: currency, rate, minimum net benefit (`switching_cost_threshold`), explicit switching cost, cooldown, hysteresis margin and proposal expiry. Supplied validation/cost fields, if present, must equal independent recomputation. Defaults and exact value validation are in `controller.py`; no client principal/role is accepted in a payload.

When a trusted server release matches the initial model version, registration binds its model ID, version and content hash. Optional API-only `eta_model_id` chooses among multiple allowlisted identities sharing a version; otherwise ambiguous selection fails. Client-supplied `eta_model_identity` is rejected. The server-derived identity survives approval/restart, and every ETA overlay must match it; matching the version alone is insufficient. Raw scenario-only plans may have no ETA identity and cannot accept model-derived overlays until explicitly registered with one.

The controller computes updated remaining exposure before invoking optimization; only material changes request a bounded reoptimization. KEEP means no material actionable benefit. ALERT covers risk without a justified supported proposal, including hard-window risk even at zero monetary penalty. PROPOSE_REPLAN stores a validated beneficial candidate while active assignments remain unchanged. Quantum and exact selection use the identical complete candidate pool and source-recomputed costs. Raw samples, repairs, fallback and validators remain separate. No quantum superiority is claimed.

Approval body is exactly `{base_version,event_id,model_version,context_hash}`. One SQLite `BEGIN IMMEDIATE` transaction checks authorization context, current base/event/model, job success, expiry/cancellation, independent static feasibility, completed prefixes, scenario hard-window outcomes, freshly recomputed same-world keep/candidate costs, switching cost and hysteresis. It atomically writes a new active plan, audit and versioned manifests. Removed vehicles receive empty revoked manifests. Race losers and stale approvals return conflicts. Browser-local approval metadata has no authority here.

## Bounds and persistence

The ETA job layer admits at most two running workers, defaults to a ten-second publication deadline, caps input at 256 KB, result at 1 MB and retained jobs at 256. ASGI middleware enforces the ETA body cap before JSON parsing, including chunked bodies; oversize requests return 413. It rejects further new work at the history cap while retaining idempotency/audit evidence. Python threads are not forcibly killed: cancelled/timed-out work retains its bounded slot until it exits, and late publication is suppressed. Restart marks unfinished jobs `restart_interrupted` and invalidates pending proposals; successful records survive. This is a single-runtime local process model, not a multi-host queue.

Tables are namespaced `eta_*`; job metadata records runtime/settings/input hashes and the result preserves solver/model provenance. Candidate generation/exact search retain existing limits, with local QAOA at <=8 variables and explicit exact fallback above that bound; no favorable candidate truncation occurs. Protected preexisting math/validators/training are unchanged. See `reports/INDIA_ETA_INTEGRATION_REPORT.md` and raw evidence for exact executed checks and resource measurements.
