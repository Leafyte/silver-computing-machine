# Legion and phone demo implementation plan

Prepared 30 October 2026, Asia/Calcutta. Scope: integrate a separately maintained frontend with the existing local backend and SDK. This is a source-inspected plan; no phone, server, SDK execution, or model benchmark was run in preparing it. All new interfaces below are proposals until implemented and tested.

## Delivery decision

Build one vertical flow first: pair a phone as a driver, load its assigned route, report congestion, start a durable planning job, validate a candidate, publish an approved version, and show that version on the phone and dispatcher screen.

For this two-hour window, use a fixed small road scenario with explicitly simulated vehicle progress and congestion. Phones send real actions to the Legion; road conditions and vehicle movement remain simulated unless a separately tested live-state adapter is added. Start with two vehicles, four deliveries, and two or three prevalidated road corridors. Keep the input below the existing exact-search and quantum limits.

Default “Plan route” means choose the fastest feasible candidate for the driver's remaining assigned stops under current server-side travel estimates. It does not promise a global road-network optimum. Congestion automatically starts evaluation. A server policy may automatically publish a validated route change for the same vehicle in this demo; fleet reassignment requires dispatcher approval. If auto-publication is unfinished, show the proposal and require one dispatcher approval. Never describe proposal generation as route publication.

Spend the window on integration and recovery. Keep the frozen GBDT policy. Put VQC diagnostics behind an explicit action after the primary flow works. Larger retraining, quantum superiority claims, live traffic integration, and public deployment are outside this window.

## What is present and what remains

These findings apply to the local checkout, not the separate frontend repository, whose URL is still needed.

| Capability | Source-inspected state | Required work or proof |
| --- | --- | --- |
| FastAPI and scenario planning | `backend/api.py`: validation, routes, classical/financial plans, VQC/QAOA paths, durable jobs | Run bounded smoke against the selected checkout; preserve API shapes |
| ETA event lifecycle | `backend/india_eta/api.py`: events, polling, cancellation, proposals, approval, manifests | Connect driver actions through a narrow server adapter |
| Approval safety | Controller/store document version/context/expiry checks, independent validation, atomic publication, revoked manifests | Reuse application services; test races and automatic policy through the same checks |
| Driver identity | `backend/auth.py` has dispatcher, researcher, admin; explicitly no driver endpoints or ownership | Add per-device scoped driver sessions and ownership checks |
| Phone application | Local `frontend/` uses React/Vite/Leaflet; API client has old scenario endpoints and no authorization header | Inspect actual separate repo; add session, ETA/driver clients, mobile screens |
| Hosting | Local Vite configuration binds loopback and proxies `/api`; backend has no CORS middleware | Build frontend; serve same origin through local HTTPS gateway |
| Road geometry | No verified provider-to-driver routing integration established by this audit | Versioned route catalog and valid GeoJSON; provider adapter later |
| Congestion | Existing event types are `eta_update`, `delay`, `vehicle_status`, `route_change` | Translate congestion into a supported event; add impact location, severity, TTL and deduplication |
| Live state | ETA controller rejects actual state; source adapter requires synthetic mode | Demo simulation adapter now; verified live-state projection later |
| ETA inference | Fixture API loads safe two-feature linear JSON model | Separate versioned adapter for the trained 59-feature release |
| Trained release | RUN2 report describes CPU GBDT and 4Q diagnostic bundles; operational fusion is alpha=0 | Verify hashes, dependencies, reference parity and latency before use |
| Local SDK | Adapters and SDK evidence exist; `qpiai_service.py` includes local/cloud branches | Explicit local execution mode, no automatic remote calls; fresh numerical smoke |
| Updates | Versioned server manifests exist | Poll only scoped driver status; reconnect and acknowledgement UI |
| Security | Token-role prototype; no resource ownership for drivers | Driver boundary, TLS, rate caps, sanitized logs, static-file confinement |

The primary checkout currently has existing edits in `tests/test_qpiai_service.py` and `tools/qpiai_service_cli.py`. Preserve and review those edits before changing the SDK path. The training worktree currently reports branch `feature/india-vqc4-run2`. Recheck status at implementation start; other work may continue.

## Runtime architecture

```text
Phone A / Phone B / dispatcher browser
             | HTTPS, same origin
             v
Legion gateway: static frontend build + /api reverse proxy
             | loopback HTTP only
             v
FastAPI: scoped session/auth -> driver API -> demo application service
                                           |             |
                                  route catalog     event coordinator
                                           |             |
                                      candidate costs / ETA context
                                                   |
                           classical solver + optional bounded local SDK
                                                   |
                               independent validator -> approval transaction
                                                   |
                         SQLite: events, proposals, plan versions, manifests
                                                   |
                         driver status polling + dispatcher status polling
```

Keep a single backend process for the prototype job manager and SQLite writer. Do not start multiple Uvicorn workers or use hot reload during the demo. SQLite and artifacts stay on Legion local disk. The gateway serves only the compiled frontend directory; never serve the repository root, `.env`, `.git`, data, or model directories.

Frontend GitHub repo and backend GitHub repo remain independent. Runtime integration is HTTP plus a versioned schema, not copying UI source into the backend. Record both commit hashes in the demo release manifest. Export OpenAPI from the selected backend and generate or check TypeScript interfaces in the frontend; runtime validation remains server-side. Serve relative `/api/...` URLs so a phone never tries to call its own `localhost`.

Prefer application-service calls within the backend over loopback HTTP to its own API. The service principal for automatic demo approval is created server-side, narrowly scoped, and recorded in the audit. Never put a dispatcher token in the phone or grant a driver the dispatcher role.

Suggested additions, avoiding a rewrite of existing math:

```text
backend/demo/api.py              narrow phone endpoints and response mapping
backend/demo/auth.py             session lookup, scope and expiry
backend/demo/service.py          plan requests, congestion admission, publication policy
backend/demo/routes.py          fixed road-route candidates and segment costs
backend/demo/store.py           sessions, reports, route state, acknowledgements
backend/demo/schemas.py         strict versioned phone contracts
fixtures/demo/                  tiny synthetic scenario + checked route geometry
tools/demo_preflight.py         no-secret environment and readiness checks
tools/demo_start.ps1            explicit launch paths and resource limits
tools/demo_reset.py             stopped-service reset of demo namespace only
```

Use one coherent state transition for route geometry and manifest version. If the ETA controller cannot represent a road alternative because the stop order is unchanged, store a separate versioned driver-route record referencing its parent manifest. Commit the route publication and its audit together; do not pretend that changing a Leaflet polyline changes a backend plan. Fleet reassignment must still use the existing plan approval transaction.

## Required inspection before implementation

1. Obtain the frontend URL, branch and commit. Inspect its actual API layer, token storage, map order conventions, mobile layouts, mock data, build scripts and environment settings. The local frontend is evidence of a possible implementation, not proof of that repo's state.
2. Select one backend checkout/commit and preserve existing changes. Run current ETA/controller tests and local HTTP smoke in an isolated demo database before adding code.
3. Inventory the Legion's OS, Python executable/version, installed SDK/ML versions, free disk/RAM and available CPU. Verify the model bundle against pinned hashes. Do not assume GPU/CUDA; use CPU.
4. Test two phones reaching a static page on the chosen network immediately. Record network isolation/VPN issues and whether the host's private IP is stable. If the current machine is not the Legion, make transfer and execution on Legion an explicit gate.
5. Choose transport within ten minutes: trusted local HTTPS on each phone preferred. Confirm certificate name/IP match and trust on both devices. A browser warning bypass is insufficient evidence that protected browser APIs work.
6. Fix demo geography, vehicle profile, source and licensing of road geometry, stop IDs, directionality, pickup/service times, capacity and hard windows. Freeze route-catalog version and scenario seed.
7. Confirm the intended product behavior: fixed assigned-stop reroute can be automatic; reassignment stays dispatcher-controlled. Use this default if no further preference is supplied.

## Phone API contract to implement

All paths in this table are proposed additions under `/api/v1/demo`. Return sanitized `{error:{code,message},request_id}` on failure. Never accept identity, role, penalty coefficients, trusted scenario matrices, or model paths from the phone.

| Method and path | Input / output | Authorization and behavior |
| --- | --- | --- |
| POST `/sessions` | One-time pairing code -> opaque expiring session, driver/vehicle label | Atomic redemption; short expiry, low attempt limit; no public driver-ID selector |
| DELETE `/sessions/current` | Revoke current session | Own session only |
| GET `/me` | Assigned driver, vehicle, plan, capabilities | Server-derived identity |
| GET `/state` | Current route, manifest revision, parent plan version, pending event, freshness, last acknowledgement | Own assigned resources; route includes GeoJSON, stop IDs, seconds/metres, source, status |
| POST `/route-requests` | `request_id`, `base_route_version`, `manifest_revision` -> job ID | Own remaining assigned stops; server supplies destination/state |
| POST `/reports` | `event_id`, base versions, `kind=congestion`, severity enum, optional bounded note/position | Own active corridor; server validates/map-matches report; returns 202 |
| GET `/jobs/{id}` | Durable job state and optional result/version | Own job only; unknown and foreign IDs do not leak resources |
| POST `/acknowledgements` | Displayed route/manifest version and client receipt time | Own vehicle; audit acknowledgement, never equate this with physical completion |

Proposed report envelope:

```json
{
  "event_id": "client-generated-uuid",
  "base_route_version": 3,
  "manifest_revision": 2,
  "kind": "congestion",
  "severity": "heavy",
  "position": null,
  "note": "Congestion ahead"
}
```

Position, when supplied, has named latitude/longitude, accuracy metres, and timezone-aware observed time. Backend admission time is authoritative for ordering. For the fixed demo without GPS, resolve the report to the server's current simulated segment. Label that segment as simulated. Treat free text as display-only escaped text; no LLM or command interpretation is needed.

Persist a hash with each event/request ID: same ID and content returns the same job; changed content returns 409. Do not retry POST with a new ID after a lost response. Validate before enqueueing, return 202 quickly, and surface capacity at admission. Poll jobs every second while active, then poll state every two seconds; slow down hidden/offline tabs and stop on logout. These intervals are proposed defaults, not measured limits.

## Normal planning flow

1. Driver taps Plan route. Button shows pending state and suppresses duplicate taps locally.
2. Server resolves the session to a vehicle and checks the active manifest, current simulated location/time, completed prefix and any in-progress locked leg.
3. Select remaining stops; an empty set returns `no_remaining_stops`, not a solver error.
4. Load candidate road geometries and segment base durations for those exact endpoints/profile. One-way constraints and reachability must already be verified. A straight line is not a drivable alternative.
5. Score candidates with current nonexpired congestion overlays. Initial selection minimizes remaining travel time subject to feasibility, with a deterministic tie-break by route ID. Display the objective and candidate count.
6. Validate endpoints, continuity, remaining stop coverage, completed work, capacity and time windows. Reject a candidate outside the prepared catalog or resource budget.
7. Recheck the base versions at publication. Save the new driver-route revision, source/cost snapshot and audit; notify through polling. Keep the previous validated version until this transaction succeeds.
8. Phone replaces route/ETA together, shows version and source, and sends acknowledgement. Arrival predictions use explicit server time and units.

For multi-stop changes, the existing dispatch planner chooses assignments/stop order; the road adapter supplies leg travel times and geometry. Road routing, fleet optimization, ETA prediction and map rendering are separate responsibilities.

## Congestion flow and automatic publication

1. A Congestion button always works. Optional push-to-talk recognizes a narrow command, displays what was heard, and submits the same structured report when recognized. Do not use always-on recording. Negated, uncertain or unrecognized text must not create a report; offer the button.
2. Validate session, route scope, report timestamp and segment. A driver cannot mark unrelated routes congested. Store a driver report as unverified input, not an authoritative traffic feed.
3. The fixed demo translates severity into a declared synthetic segment delay. Suggested starting values: moderate +3 minutes, heavy +8 minutes, expires after five minutes. Freeze these values in configuration and show the simulation label.
4. Apply delay only to affected directed segments. Repeated same-source reports refresh one overlay; they do not multiply delay. Opposite direction and completed segments are unaffected. Expiry/removal is a separate update that can trigger evaluation.
5. Serialize processing per active plan; coalesce bursts and retain report IDs. Recompute the keep-route cost under the updated world before comparing alternatives. Avoid adding delay twice through both base durations and an ETA overlay.
6. Use controller outcomes KEEP, ALERT and PROPOSE_REPLAN where applicable. Suggested demo switching rule: require at least two minutes and ten percent reduction in remaining travel time, after switching cost, with a 30-second cooldown. These are configurable product defaults requiring a test, not learned thresholds.
7. Auto-publication is permitted only for this configured simulated demo, same vehicle, unchanged remaining order set, intact completed/in-progress locks, fresh state and passing validator. Recheck versions, event success, model/context identity and expiry inside the publication transaction.
8. Fleet reassignment, uncertain state, stale reports or infeasibility creates an alert/proposal for dispatcher review. Conflicting newer updates supersede old proposals. A stale worker never publishes over a newer route.
9. Distinguish accepted report, job running, candidate found, route published and phone acknowledged. Show KEEP reasons when no change helps. If the active route becomes invalid, show an explicit stop/dispatcher alert rather than calling the retained route safe.

Persist the publication decision with event ID, reporting principal, policy version, before/after routes, same-world costs, validator result, fallback reason and execution type. Automatic policy must use the same safety conditions as manual approval.

## Maps, routing and offline behavior

For the demo, prepare a small licensed route catalog with actual road polylines, segment IDs and travel-time inputs. Store GeoJSON coordinates in longitude/latitude order; convert explicitly for Leaflet's latitude/longitude APIs. Validate seconds versus minutes and metres versus kilometres at the adapter boundary. Store routing profile, source date, route-catalog hash and snapped endpoints.

OSRM is a possible later adapter: its Route service provides geometry and alternatives, and Table provides travel durations. Alternative routes are not guaranteed and do not automatically avoid an arbitrary newly reported incident. For incident-aware network routing, add an edge-weight-aware local graph/provider or verify a prepared unaffected corridor. Do not assume an alternatives response supplies it.

Do not spend this window downloading and preprocessing a large road network. Use a live provider only if already available and tested; bound connect/read times and use server-side fixed host configuration. No browser-supplied provider URLs. Cache by endpoints/profile/catalog version; overlay state has a separate revision in the score cache key.

Basemap tiles can need internet even when all computation is local. Use lawful locally available assets or show route lines and stop markers against a plain background when tiles fail. Do not bulk-download public tiles without checking policy. Keep attribution. Prepared route replay is labeled as replay; an offline cached route is not a fresh solve.

## Models and local SDK

The feature audit identifies 59 trained GBDT predictors and prediction-time constraints. The current fixture API exposes two features. Delivery-duration labels do not establish road-arc durations, and repeated per-leg use would double count whole-delivery effects. Do not plug the trained bundle into road costs by renaming fields.

Use road-provider/catalog times for navigation. If adding a model panel, pass a complete allowlisted 59-feature delivery context through frozen preprocessing, show its target as delivery duration, and reject unsupported/missing context. Missing actual pickup time prevents a pickup-dependent prediction before pickup. Never supply future observations or silently fill unavailable required features with zero.

The RUN2 final report records operational alpha=0 because neither 4Q candidate passed the validation deployment gate. Its retrospective extended76 result cannot be used to choose a replacement after viewing test outcomes. Keep the production prediction policy frozen. Quantum residual, route-risk VQC, and QAOA dispatch are distinct computations with distinct units and claims.

Model adapter requirements: trusted server-only artifact path; independently pinned inventory/hash; compatible NumPy/sklearn/joblib versions; reference parity; finite/bounded outputs; named feature order; load once; runtime/version metadata; bounded failure. Existing fixture loader refuses executable model formats. A trained joblib release therefore needs a separate trusted deployment adapter or isolated local sidecar; never weaken the fixture loader to accept uploads or arbitrary paths. Hash verification depends on a separately trusted manifest.

For local SDK proof, execute a tiny deterministic circuit with pinned parameters and compare a numerical observable against the existing independent oracle. Record package version, qubits, seed, measured execution time and `execution_type=local_simulator`. An import or animation does not establish circuit execution. SDK calls remain on Legion, and no keys reach the frontend.

`backend/qpiai_service.py` contains credential loading, cloud connection diagnostics, identity metadata and raw exception strings. It must not be exposed as a public diagnostics endpoint. Local-only startup must skip cloud login/status and remote execution even when credentials exist. Verify this with network calls patched to fail. Paid jobs remain disabled.

Useful performance work after the main flow passes: preload the frozen model, avoid diagnostic bundle calls that calculate both unused VQC candidates under alpha=0, deduplicate requests, cache immutable geometry and bound solver concurrency. Measure cold start and repeated warm end-to-end timings before/after. If this work becomes a performance implementation task, apply the project's benchmark/optimization workflow before claiming improvement.

## Network and security work

Preferred topology: dedicated private hotspot/LAN, trusted local HTTPS gateway on Legion, API bound to 127.0.0.1:8000. Expose only the gateway port to the intended private subnet. Do not disable Windows Firewall or expose a Vite development server publicly. Confirm network changes with the owner when they require administrator settings under project rules.

Fallback within the timebox: private-LAN HTTP for synthetic data and temporary demo sessions, with GPS/voice treated as unavailable unless tested. This does not provide encrypted transport and must be labeled a restricted demo configuration. Keep manual position/segment and Congestion button available. A public tunnel is a separate exposure decision, not an automatic workaround for Wi-Fi isolation.

| Threat | Required control | Acceptance check |
| --- | --- | --- |
| Driver spoofing | One-use expiring pairing code, unpredictable per-device session; server binds driver/vehicle | Invalid/reused code denied, logout/expiry revokes access |
| Cross-driver access | Ownership on every state/job/report/ack path | Phone A cannot read or mutate Phone B by changing IDs |
| Privilege escalation | Driver API accepts no role/actor/model path; dispatcher token server-only | Driver cannot call approval, admin or research job endpoints |
| Tampering/replay | Strict schema/ranges, idempotency hash, freshness, base versions | Changed duplicate and stale publication rejected |
| Repudiation | Persist admitted subject, request/event IDs and decision audit | Each published revision traces to an authorized action |
| Information disclosure | Minimal phone payload, sanitized errors, no secrets/raw exception or account identity | Sentinel token absent from responses/logs/static bundle |
| Resource exhaustion | Body limits, per-session/IP pairing/report limits, queue cap, bounded solver | Burst produces stable 429; UI stays responsive |
| Browser abuse | Same-origin requests, allowed Host/Origin, escaped text, restrictive asset policy | Foreign-origin mutation denied; note renders as text |
| Artifact/provider abuse | Pinned local release, fixed provider host, bounded file/response size | Tampered model denied before deserialization; URL injection rejected |

Session proposal: memory-held opaque bearer token over HTTPS, expiring at demo end; refresh/reload requires pairing again. If cookie sessions are chosen instead, implement Secure/HttpOnly/SameSite plus CSRF checks and change the client from its current `credentials: omit` behavior. Choose one method consistently. Keep pairing tokens out of URLs, screenshots, logs and frontend environment variables. Vite public variables must contain no secrets.

Disable external access to API docs, arbitrary scenario execution and diagnostics at the gateway unless the dispatcher needs them. Use a narrow route allowlist and enforce auth in the application too. Authentication failures must not reveal whether another driver's resource exists. SQLite statements use parameter binding; cap note lengths and reject NaN/infinite coordinates.

## Failure behavior to implement and exercise

| Condition | Expected outcome |
| --- | --- |
| Double tap or voice callback repeated | One admitted report/job by idempotency key |
| Response lost after admission | Retry same ID returns same job; no duplicate publication |
| Two phones report simultaneously | Serialize/rebase against latest context; no lost event or conflicting publication |
| New report arrives during solve | Old result superseded or revalidated; never overwrites newer context |
| No improvement | KEEP with current version and reason |
| No feasible route / blocked only corridor | ALERT; dispatcher action; no invented safe route |
| Completed or active leg would change | Reject candidate or defer to next safe decision point |
| Position denied/inaccurate/stale/off-route | Manual simulated segment; reject unsupported live inference |
| Microphone denied/unsupported/unrecognized | Visible button; no report from uncertain speech |
| App backgrounded / phone locks | Show stale on resume, fetch state before actions, do not promise background tracking |
| Network drops | Retain visibly stale last route; pending report marked unsent/unknown; reconcile before resend |
| Session expires | Re-pair; no cached sensitive state for another user |
| Provider fails / no road snap | Use explicitly labeled prepared catalog if valid; otherwise unavailable |
| Model/SDK fails or exceeds deadline | Preserve valid plan, explicit fallback; never silently mark quantum success |
| Solver timeout/cancel | Terminal job; suppress late publication; bounded worker retains slot until exit |
| Restart during work | Mark interrupted; preserve committed versions; invalidate pending proposals |
| Crash after commit before response | Same ID retrieves committed result and version |
| SQLite locked/full/corrupt | Safe unavailable response; no partial publication; operator restart/recovery path |
| Reports never expire or oscillate | TTL, cooldown, same-source coalescing, switching threshold |
| Timezone or device clock differs | Server ordering and UTC storage, explicit local display and units |
| Revoked vehicle assignment | Empty/revoked manifest displayed; stale route cannot remain active |
| Unsupported instance size | Stable bounded error, no silent candidate truncation |

An acknowledged manifest proves receipt, not driver compliance. The demo uses stationary testers; actual road deployment needs separate operational safety and driver interaction design.

## Two-hour implementation schedule

Use T+0 at implementation start. The requested two-hour delivery budget includes inspection, integration and rehearsal; do not spend it all building. Work sequentially if one implementer is available. Frontend/backend task labels below describe ownership boundaries, not evidence of delegated work.

| Elapsed | Work and deliverable | Gate / scope cut |
| --- | --- | --- |
| 0–10 min | Pin checkouts, inspect separate frontend, run narrow baseline smoke, test both phones and transport | If HTTPS trust fails, select labeled restricted manual-input mode now |
| 10–25 min | Seed scenario/catalog and sessions; scoped `/me` and `/state`; serve frontend build same origin | Two phones must show different server-assigned vehicles before more features |
| 25–45 min | Implement normal route request, geometry response, deterministic selection and validation | Phone displays genuine server-selected route and version |
| 45–70 min | Congestion report, idempotent durable job, segment overlay/TTL, KEEP/ALERT/proposal | One report yields a traceable evaluated result |
| 70–85 min | Atomic same-vehicle publication policy or dispatcher approval; state polling and acknowledgement | Both views agree on version; stale result cannot publish |
| 85–100 min | Ownership, rate limits, restart/duplicate/offline tests; local SDK numerical smoke | Fix failures before optional work |
| 100–110 min | Optional tested voice; model diagnostic or measured hot-path improvement only if primary gates pass | Otherwise retain button and frozen baseline |
| 110–120 min | Restart full stack, rehearse twice on two phones, record release manifest and startup commands | Freeze code; keep reproducible fallback scenario |

At T+45, if normal planning fails, stop model/voice work. At T+70, if event approval integration is incomplete, use explicit dispatcher approval through existing guarded services. At T+100, freeze additions. If route geometry cannot be connected to actual backend selection, cut the claim of road rerouting and demo the validated simulated stop-plan flow honestly.

## Acceptance and handoff

Target budgets, to measure on Legion: report admission within one second; classical fixed-scenario plan within five seconds; job deadline ten seconds; two-phone update within three seconds of publication; at most two planning workers. Report actual observed durations and sample counts. Warm the SDK before the presentation and record cold-start separately.

Required checks:

1. Existing targeted ETA/controller/approval tests and HTTP smoke still pass using a disposable DB; no broad retraining is run.
2. Two independent sessions load distinct assignments; cross-driver access and dispatcher escalation fail.
3. Normal planning returns checked road geometry, units, source and version; map follows the backend output.
4. Heavy synthetic congestion on the chosen corridor produces an independently validated alternate; an unaffected corridor stays unchanged.
5. Small congestion returns KEEP; all alternatives blocked returns ALERT; completed/in-progress work remains locked.
6. Duplicate submission, stale approval, simultaneous update, job timeout and crash/restart preserve committed state and idempotency.
7. Phone goes offline then reconnects; pending state reconciles and acknowledges latest revision without replaying a duplicate action.
8. Button flow works with microphone/GPS denied; voice is included only after passing on both actual devices.
9. Local SDK executes a numerical circuit check with remote calls disabled; UI reports the execution type accurately.
10. Sentinel redaction, static directory confinement, queue limits and model tamper checks pass; no raw SDK diagnostics are exposed.

Presentation sequence: pair both phones; show original assignments and plan version; tap Plan route; report heavy congestion; show event running then KEEP/ALERT/proposal; publish automatically within the demo policy or approve as dispatcher; show changed corridor, before/after simulated durations and shared version; acknowledge on the driver phone; disconnect/reconnect; show the preserved version. Finish with the separate local SDK evidence panel if implemented.

Deliver a release record with frontend/backend commits, scenario/catalog/model hashes, Python/package versions, execution type, start commands, private gateway URL, tested phone/browser versions, actual test outcomes, measured timings, known cuts and rollback instructions. Reset only the dedicated demo namespace while stopped; preserve previous audit DBs. Verify restart from the recorded commands before calling the demo ready.

## Follow-on development after the demo

Implement verified live vehicle-state ingestion and remaining-state projection; real road graph costs and incident matching; traffic provenance/confidence aggregation; correct route navigation profiles; trusted model feature acquisition at prediction time; geographic/domain evaluation and uncertainty calibration; durable hard-cancellable workers; dispatcher/driver identity and ownership across all APIs; scalable routing for larger instances; driver acknowledgement/escalation policy; deployment backup, retention and observability. Train or select improved models using a fresh predefined development protocol, then evaluate on a genuinely untouched future/group holdout. Do not retune against the already exposed RUN2 test partition.

## Evidence and external references

Local evidence: `silver-computing-machine/backend/api.py`, `backend/auth.py`, `backend/india_eta/{api,controller,source_adapter}.py`, `backend/qpiai_service.py`, `docs/INDIA-ETA-CONTRACTS.md`, `docs/INDIA-ETA-CONTROLLER.md`, `docs/INDIA_ETA_FEATURE_AVAILABILITY_AUDIT.md`, `bootstrap/SDK-EVIDENCE.md`, `frontend/src/api/{client,endpoints}.ts`, `frontend/vite.config.ts`; `trainingv3/RUN2_INDIA_QML_FINAL_TRAINING_REPORT.md`. Reported historical test results were not rerun during this planning task.

Official references checked 30 October 2026:

- [MDN geolocation](https://developer.mozilla.org/en-US/docs/Web/API/Geolocation/getCurrentPosition): secure context and user permission requirements.
- [MDN speech recognition](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition): limited availability; some implementations use a server recognition service, so local backend execution does not establish offline voice.
- [OSRM HTTP API](https://project-osrm.org/docs/v26.4.0/http): route/table services, profiles, geometry and coordinate contract.
- [FastAPI HTTPS](https://fastapi.tiangolo.com/deployment/https/) and [proxy deployment](https://fastapi.tiangolo.com/advanced/behind-a-proxy/): TLS termination and proxy boundaries.

Applied project-local skills: architecture-patterns for adapter boundaries; stride-analysis-patterns for threats and checks; no-ai-slop as an editorial pass on this plan. No implementation, training, remote jobs or delegated agents were started by this planning task.
