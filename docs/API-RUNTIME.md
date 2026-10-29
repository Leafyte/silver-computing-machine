# V1 local API and job runtime

The backend exposes a versioned FastAPI API. Start locally with
`.venv\\Scripts\\python.exe -m uvicorn backend.api:app --host 0.0.0.0 --port 8000`.
The synchronous decision path is classical exact enumeration over the tiny
scenario domain; responses label this `local_classical`. No remote execution,
QPU call, or automatic fallback is implied.

## Scenario contract

All `POST /api/v1/scenarios/*` and `POST /api/v1/jobs` requests accept a JSON
scenario with `depot`, `vehicles` (`id`, `capacity`), `orders` (`id`, `demand`,
`window: [open, close]`, `service`, `penalty`, optional boolean `hard_window`),
and a complete nested `travel` duration map. `hard_window` defaults to false;
omitting it preserves the existing soft-deadline behavior. Order IDs are travel locations. See
[`../fixtures/dispatch_tiny.json`](../fixtures/dispatch_tiny.json). The API
accepts only these known fields and rejects unknown scenario, vehicle, or
order fields with stable, non-echoing field errors. Invalid requests return HTTP 422 with stable
`invalid_scenario` and field paths, without echoing submitted values.

## Endpoints

| Method and path | Behavior |
| --- | --- |
| `GET /api/v1/health` | Local process health and API version. |
| `POST /api/v1/scenarios/validate` | Validate scenario fields. |
| `POST /api/v1/scenarios/submit` | Submit an asynchronous local plan job (same contract as `/jobs`). |
| `POST /api/v1/scenarios/routes` | Enumerate bounded capacity-feasible ordered routes. |
| `POST /api/v1/scenarios/plan` | Solve exactly, then independently validate. HTTP 409 means infeasible. |
| `POST /api/v1/scenarios/financial-plan` | Select an exact-cover plan using explicit `currency`, `operating_cost_per_travel_minute`, and `penalty_rate_unit`; recompute and independently validate monetary totals. |
| `POST /api/v1/scenarios/qaoa` | Optional warm-start QAOA attempt; any exact classical fallback is explicit. |
| `POST /api/v1/scenarios/vqc-plan` | Single-request frozen VQC inference from explicit `forecast_scenario`, `nominal_scenario`, `theta` (two finite angles, absolute value at most 10), and `model_version`; output is independently validated against forecast input. Optional `execution_backend` is `reference` by default or `qpiai_local`; optional `financial_terms` requests money-valued scoring. |
| `POST /api/v1/recovery/plan` | Recover from an independently validated original plan while preserving specified locked route prefixes; static depot replay only. |
| `GET /api/v1/recovery/audits/latest` | Read latest attempt and latest successful recovery; failed attempts do not change latest success. |
| `GET /api/v1/recovery/audits/{audit_id}` | Read one durable recovery attempt by ID. |
| `POST /api/v1/risk/evaluate` | Score one independently validated fixed plan across 1–100 explicit travel-matrix scenarios; optional empirical CVaR. |
| `POST /api/v1/plans/validate` | Accept `{scenario, plan}` and recompute feasibility and objective. |
| `POST /api/v1/jobs` | Submit the same exact plan computation as a durable async job; HTTP 202. |
| `POST /api/v1/jobs/qaoa` | Submit local statevector QAOA with exact classical fallback; requires `scenario`, `seed`, `shots`, `warm_start`, and `angle_grid`; HTTP 202. |
| `GET /api/v1/jobs/{id}` | Poll state, terminal result/error, and execution metadata. |
| `POST /api/v1/jobs/{id}/cancel` | Cancel queued work or suppress publication for running work. |

Stable errors include `invalid_scenario`, `infeasible`, `solver_budget_exceeded`,
`validation_failure`, `solver_failure`, `capacity_reached`, `job_submission_failed`, and
`job_not_found`. The synchronous exact-plan endpoint returns HTTP 503 with
`solver_budget_exceeded` when it cannot prove an optimum or infeasibility within
its work budget. Async exact jobs become `failed` with the same error code and
publish no incumbent plan. This differs from `infeasible`, which is returned
only after exhaustive search proves no valid exact cover. Unexpected solver exceptions are not copied into API errors,
logs, or the job database. Job results contain the decoded plan and the
independent validation report. The solver output and validator output remain
separate fields.

Financial planning requires all of `scenario`, `currency`,
`operating_cost_per_travel_minute`, and `penalty_rate_unit`; there is no
implicit currency or cost rate. The accepted unit is
`currency_per_late_minute`. Currency is checked for uppercase three-letter
syntax only, not against a registry. The response keeps the domain proxy plan
separate from `financial_objective`, and includes both proxy and independent
financial validator reports. Financial requests are bounded to 6 orders, 8
vehicles, a conservative 3,000-route upper bound, and 5,000 deterministic
search work units. HTTP 409 denotes proven infeasibility; HTTP 503 denotes
budget exhaustion and publishes no partial plan. This is caller-defined
accounting, not a claim of calibrated or financially realistic rates.
Hard-window orders are filtered during candidate generation and checked again
by the independent validator; if no exact cover meets every hard close, the
financial endpoint returns proven infeasibility without publishing a plan.

VQC inference optionally accepts the same exact `financial_terms` object with
`currency`, `operating_cost_per_travel_minute`, and
`penalty_rate_unit`. Invalid or incomplete terms return
`invalid_financial_terms`. Financial mode reports a one-minute expected-lateness
exposure assumption and separate source-recomputed forecast financial totals;
the response warns that caller-supplied parameters are not known to be trained
or calibrated for the supplied rates. If inference fails in financial mode,
the API does not replace the monetary result with a proxy-only classical
fallback. Omit the field for the prior proxy-only behavior.
The VQC inference module currently has a stricter frozen-model order schema
and rejects `hard_window` as unsupported; it does not strip the constraint.
Use the bounded classical, financial, QAOA, recovery, or risk paths for
hard-window inputs until that VQC contract is extended and independently
verified.

Live localhost integration was exercised by the coordinator with `.venv-sdk`
Uvicorn on `127.0.0.1:18377`, then stopped. The financial endpoint used an
explicit USD rate of 2 and the declared penalty unit; it returned a feasible
plan, `financial_validation.valid: true`, and recomputed total cost 20. The
same explicit terms on `qpiai_local` VQC inference returned a valid financial
report and forecast total 20 with the requested backend. The smoke did not
inspect the predicted estimate total; none is claimed from that check.

## Persistence and resource bounds

SQLite defaults to `.runtime/debugleaf.sqlite3`; override with
`DEBUGLEAF_DB_PATH`. The `jobs` table stores job ID, status, canonical scenario,
result, stable error, submitted/start/complete times, deadline, solver label,
execution type, canonical-input SHA-256, settings, and Python/FastAPI runtime
versions. Request hashes cover canonical scenario plus solver settings. Exact
solver seed is marked not applicable; QAOA stores the caller-supplied seed,
shots, warm-start flag, angle grid, and requested local backend.
SQLite WAL mode is enabled. On process start, stale
`queued`, `running`, and `cancel_requested` records become `failed` with
`restart_interrupted`; in-memory exact search is not resumed.

`JobManager(max_workers=N)` admits at most N queued/running jobs and rejects
further submissions with `capacity_reached`. The exact domain is capped at
eight orders, 16 vehicles, and 10,000 capacity-feasible ordered route
candidates across all vehicles. Route generation prunes over-capacity prefixes
before extending permutations and preserves stable candidate order. Exact-cover search then counts
each recursive node and candidate-route visit against a deterministic budget
of 100,000 work units. Exceeding it raises a budget error even if a feasible
incumbent exists; no partial result is published. The bound limits combinatorial
search but is not a hard wall-clock deadline. Scenarios above the structural
route-pool bound are rejected before route generation. Delivery close times
are soft SLA deadlines by default: early service waits until opening, and late
service contributes penalty exposure. Set `hard_window: true` to require the
service start to be at or before the close. A service start exactly at close
is feasible; a later start excludes that route. The order flag must be a JSON
boolean. API canonicalization preserves the flag when supplied and omits it
when absent so legacy input representation remains unchanged. The independent
plan validator also checks the hard close, including for caller-supplied plans.
Risk replay returns `validation_failure` without publishing risk metrics if a
realized travel matrix makes a hard-window plan infeasible; it does not silently
skip the case. Recovery and QAOA use the same generation and validator rules;
no feasible hard-window exact cover yields a stable infeasibility or explicit
fallback result. Job timeout defaults to 30 seconds. Timeout/cancellation
prevents publishing late output, but Python cannot safely terminate a running
thread: it retains its bounded worker slot until the tiny exact computation
returns. `close(wait=False)` is suitable for process teardown; already-running
workers may finish without publishing if cancelled/timed out.

Synchronous CPU-heavy endpoints (`routes`, exact plan, QAOA, VQC inference) share a
two-slot nonblocking compute gate and return HTTP 429 when busy. QAOA accepts at
most 12 route variables, 512 shots, and angle grid 10. VQC
inference accepts at most 3 orders, 4 vehicles, and 20 route candidates; it
uses the caller's frozen `theta` and `model_version` and does not retrain. On
unsupported inference size or validation failure, a feasible independently
validated exact plan is returned with `fallback.used: true` and a reason.
Ordinary `/scenarios/plan` does not load or depend on a fitted model. The API
does not rerun the prospective held-out study.

The QAOA endpoint uses the repository's dependency-free Python statevector
reference implementation, not QpiAI QCloud, a QPU, or authenticated service
execution. VQC inference likewise uses the repository's local two-qubit exact
statevector implementation. The `model_version` and angle vector are caller
supplied and not registry-verified; inference reports the unverified training
origin and lacks calibration metadata. Use separate experiment artifacts for
frozen study evidence.

Recovery accepts `{original_scenario, original_plan, updated_scenario,
locked_prefixes, change_penalty?}`. Prefixes must be proven against the
validated original route and are checked again against the updated plan. It
supports updated travel and vehicle removals, but does not have live vehicle
positions or clocks; its state basis is depot replay. Infeasible locked work
returns a stable conflict instead of silently dropping a commitment.

Risk scoring accepts `{forecast_scenario, nominal_scenario, plan,
realized_scenarios, cvar_alpha?}` at `/api/v1/risk/evaluate`. The fixed plan
is validated under the forecast scenario, and every supplied alternative
travel matrix is replayed and independently validated. Malformed cases fail
the request rather than being skipped. This is an equal-weight finite-sample
estimate with no population calibration or live-state claim; see
[`RISK.md`](RISK.md) for the estimator and CVaR definition. Risk evaluation
shares the two-slot synchronous compute gate.

Before publishing success, the endpoint independently validates the recovered
plan and appends one audit record to `recovery_audits` in the shared SQLite
database. Records include canonical validated original/updated scenarios, the
validated original plan, proven requested prefixes, penalty, status, result or
stable failure, UUID, UTC attempt-start and completion timestamps, and SHA-256
hashes for source inputs and outcome. Unknown fields and invalid
scenarios/plans/prefixes are rejected before storage. A scope hash identifies
the original scenario and original plan. Successful-plan pointers are kept per
scope, so a later attempt for a different baseline cannot be returned as the
last valid plan for this one. `/audits/latest` follows the latest attempt's
scope; pass `scope_sha256` to select a scope explicitly. A failed attempt is
retained and leaves that scope's latest success intact. Lookup is read-only.
Audit writes share the job database's single-process PID lease;
the unauthenticated API remains localhost-only.

`qpiai_local` requires the isolated QpiAI SDK runtime. If that capability is
missing or execution fails, the API returns a stable error and does not silently
switch to the reference simulator or classical solver. The default `reference`
path remains independent of the optional SDK. Caller-supplied model parameters
and version are not registry-verified or evidence of training provenance.

Direct CPU-heavy endpoints admit two computations and durable jobs admit two
worker computations, for at most four bounded solver computations at once.
A job retains its worker slot after timeout/cancellation until the thread exits.
`/api/v1/jobs/qaoa` requires all settings explicitly: signed 32-bit seed, 1–512
shots, boolean `warm_start`, and angle grid 2–10. The request is rejected if
route generation produces more than 12 variables. Its worker stores the full
QAOA result, including raw samples and decoded decision, only after independently
revalidating the selected plan. A successful decision labels `execution_type`
as `local_simulator` for an accepted raw sample or `classical` when exact
fallback supplied the decision; `optimization.execution_type` separately
records whether the local simulator ran. Fallback reason and raw QAOA samples
remain separate. Infeasible, budget-exhausted, failed, timed-out, cancelled,
and restart-interrupted jobs remain distinct; late work cannot publish after
timeout or cancellation. Jobs share the same worker admission limit and SQLite
restart policy as exact jobs. No cloud/QPU path is part of this endpoint.

The service is unauthenticated and intended for localhost development only.
Keep it bound to `127.0.0.1`; do not expose it to a network or production users
without authentication, authorization, TLS, and deployment hardening. Run one
worker process per SQLite database. A live PID lease prevents another same-host
manager from recovering active records; cross-process shared-database serving
is unsupported.

## Optional token authorization prototype

The default `DEBUGLEAF_AUTH_MODE=local` preserves the unauthenticated local
prototype behavior and does not read role-token environment variables. To opt
into bearer-token mode, set `DEBUGLEAF_AUTH_MODE=token` and provide three
distinct, nonempty environment variables:
`DEBUGLEAF_DISPATCHER_TOKEN`, `DEBUGLEAF_RESEARCHER_TOKEN`, and
`DEBUGLEAF_ADMIN_TOKEN`. Incomplete, duplicate, or unknown-mode configuration
fails closed: every request, including health, gets HTTP 503
`auth_misconfigured`. With valid token configuration, `GET /api/v1/health` is
public liveness; all other requests require `Authorization: Bearer <token>`.
Missing/invalid credentials return generic 401 errors; a recognized token with
an unsupported role/path returns 403. Responses never echo credentials.

Role policy is deliberately limited to implemented endpoints. Researchers can
validate, route, plan, run QAOA/VQC and risk scoring, use exact/QAOA jobs, poll
and cancel jobs, and read recovery audits. Dispatchers have those capabilities
plus financial planning and recovery planning. Admin can access the explicit
implemented route set; future/unknown paths are denied until added to policy.
There is no driver endpoint or driver role. The hook uses constant-time byte
comparison and only stores tokens in process memory; it has no token rotation,
expiry, identity claims, audit attribution, rate limiting, TLS, or production
secret-manager integration. It is an optional localhost prototype, not
production authentication.

Coordinator localhost smoke on `127.0.0.1:18379` used fresh test-only
sentinels supplied to the child process and did not print token values: missing
token returned 401, invalid token returned 401, dispatcher plan returned 200
with independent validation true, researcher financial planning returned 403,
and admin access to an unknown route returned 403. The temporary server was
stopped. Focused policy/configuration tests use injected sentinel values and do
not read actual credentials.

An infeasible plan is terminal `infeasible`; validation failures and solver
errors are terminal `failed` with stable codes. Quantum-informed fallbacks are
explicit and report their reason. Clients should treat `succeeded` as usable only when its
included validation report has `valid: true`.

## Test command

Run the focused API, persistence, and domain-specific financial suite with
`.venv\\Scripts\\python.exe -m pytest --basetemp=.runtime\\pytest-api tests/test_api.py tests/test_auth.py tests/test_domain.py tests/test_recovery_persistence.py tests/test_financial.py -q`.
The sandbox blocks Windows loopback socketpairs, so these tests invoke FastAPI
route handlers directly instead of claiming HTTP transport integration. They
check route behavior, field validation, result validation, async polling, cancel,
timeout, capacity, restart recovery, and unknown IDs. Passing tests establish
local code behavior only; they do not demonstrate external deployment or
quantum execution.

## Bounded stress check

Run `.venv\\Scripts\\python.exe tests/stress_api.py` for a direct-handler and
SQLite worker stress pass. It uses the three-order fixture, two worker slots,
thread barriers (not load loops), a 50 ms injected timeout, temporary SQLite
files under `.runtime`, and no external calls. On 2026-10-28 the recorded run
passed in 0.779 seconds: two concurrent exact jobs submitted; three extra
submissions rejected across saturation, cancellation, and timeout; one cancel;
one timeout; one interrupted-job recovery; one invalid payload; one secret
sentinel redaction; maximum two active solver threads. Memory was not measured.
The harness cleans its temporary SQLite directory.

If localhost sockets are available, start the service as above and add
`--base-url http://127.0.0.1:8000` to run eight concurrent validation requests,
two job submissions, cancellation, polling, and sentinel redaction over HTTP.
The current sandbox blocks the Windows event-loop socketpair, so its stress
result above is explicitly direct-handler/SQLite evidence. A separate bounded
localhost HTTP smoke was run by the coordinator for health, feasible classical
plan validation, VQC inference validation, and async job success; it was not a
concurrency stress run.
