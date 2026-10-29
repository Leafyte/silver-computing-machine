# India ETA to SLA and dispatch: Luna implementation plan

**Status:** planning only, 2026-10-29 Asia/Kolkata. No integration implementation, training, calibration experiment or application test was run for this plan. Owner instruction: Codex plans/reviews; Luna implements/tests when execution is authorized. The attached brief's START NOW does not override JUST PLAN.

## Scope and ownership

Deliver the classical-first versioned ETA adapter, uncertainty/SLA translation, source/domain checks, bounded event-to-proposal workflow, backend contracts, versioned examples and `reports/INDIA_ETA_INTEGRATION_REPORT.md`. Start against labeled fixtures without waiting for Codex 1. Preserve current backend behavior and existing optimization mathematics.

Planning branch: `codex/india-eta-integration-plan`, fast-forwarded from `origin/main` at `ccf2d42`. Reviewed API skills were committed as `bca392c` and pushed before plan authoring. Another agent subsequently switched the shared checkout to `feat/ui`; planning finished in `.runtime/india-eta-planning`, a separate worktree on the planning branch. Future Luna implementation should create `codex/india-eta-integration` from the published planning commit in an isolated worktree. Never copy the current dirty checkout wholesale.

- Luna owns new `backend/india_eta/`, `tests/india_eta/`, `fixtures/india_eta/`, `docs/INDIA-ETA-CONTRACTS.md`, and the final root report. Use a small package with contracts, prediction, uncertainty, domain compatibility, route projection, coefficient adapter, event policy and durable store responsibilities; combine files when simpler.
- Codex owns independent review, acceptance evidence and coordination of narrow edits to `backend/api.py`/`backend/auth.py`. Luna may prepare these bounded mounting/policy edits after checking current ownership; do not overwrite concurrent changes.
- Codex 1 owns India training and the frozen model release. Gemini owns Newton GBDT refinement. Luna must not train/refit either model, consume GPU, change dataset preparation, touch RUN1-v2 artifacts, mutate `experiments_run2/` or alter `experiments_v1_1/`.
- Read `AGENTS.md`, README, V1 acceptance, this plan and reviewed local skills. You are not alone in this codebase; preserve all other work. `_archive/` remains inert. No commits/pushes during future execution unless explicitly assigned; this turn's GitHub sync covers planning/skills only.

## Inspected contracts and gaps

These are source-inspected facts, not fresh runtime results. Record line pointers again at execution because other agents may change files.

| Boundary | Current code | Consequence |
| --- | --- | --- |
| APIs | `backend/api.py:create_app`; actual routes in `docs/API-RUNTIME.md` | Reuse scenario validation/planning, financial plan, risk replay, plan validation, recovery and jobs where contracts fit. No existing ETA prediction endpoint was found. |
| Route clock | `backend/domain.py:_route_metrics`; `backend/validator.py:validate_plan` | Travel is a complete duration matrix in minutes. Clock starts at depot time zero; arrival, waiting, service start, departure and return are separate. Lateness is service start minus close, floored at zero. Preserve this definition or explicitly version a different contract event. |
| Frozen VQC | `backend/learning_inference.py:predict_dispatch` and `_model_provenance` | Takes forecast/nominal scenarios, two-angle theta and model version; predicts route risk using a two-qubit circuit. It does not consume India features or predict duration/residual minutes. Do not treat its risk output as a duration correction. |
| Exposure estimate | `backend/learning_inference.py:PENALTY_HORIZON_MINUTES`, `_financial_outputs` | Existing estimated exposure uses a stated one-minute risk surrogate. New ETA expected-lateness calculations need their own explicit contract; preserve the legacy surrogate and label both. Source currently supports hard-window flags despite older contradictory runtime-guide wording. |
| Hybrid artifact | `experiments_v1_1/artifact_contract.py:HybridDispatchArtifact`, `validate_hybrid_artifact` | Schema is `debugleaf.hybrid_artifact.v1.1`; exact key `scenario_and_seed_Provenance`; currency-specific coefficient keys end in `_usd`. Do not silently rename, relabel INR as USD or fabricate calibration/split fields. Build a separate ETA envelope and an explicit compatible export adapter. |
| Optimization | `backend/qubo.py:build_route_qubo`; `backend/qaoa.py:continuous_relaxation`, `run_qaoa`; `backend/qaoa_dispatch.py:solve_qaoa_dispatch`; `experiments_v1_1/quantum_optimization.py:lp_augmented_relaxation` | QUBO candidate input is `{id, vehicle, stops, cost}`. Existing dispatch wrapper derives cost from its own domain objective. Core continuous relaxation is projected gradient descent on the box-constrained QUBO, not LP. The separate V1.1 augmented LP routine must be preserved and independently checked when used. Changing a response field cannot inject new coefficients. A new bounded adapter must call unchanged primitives and preserve exact candidate-to-route mapping. |
| Validation/fallback | `backend/validator.py`, `backend/financial.py:validate_financial_plan`, QAOA dispatch raw-sample checks | Validate decoded plans from source inputs; validate uncertainty-derived expected costs separately. Preserve raw samples, repaired plans, selected plan, validators and explicit classical fallback. Existing financial validator alone does not validate an expected stochastic coefficient. |
| Recovery | `backend/recovery.py:recover_dispatch`; `backend/recovery_store.py:RecoveryAuditStore` | Supports travel/vehicle-removal changes with locked prefixes; static depot replay. It has no live position/clock, active assignment transaction or completed-stop state. Locking completed stops and replaying from depot is insufficient for live remaining-route ETA. |
| Jobs/auth | `backend/jobs.py`, `backend/auth.py` | Durable exact/QAOA jobs, bounded workers, cancellation/publication suppression and restart-interrupted status exist. Auth is local by default with optional token roles; no driver identity. Do not claim production authentication. |
| Approval | `frontend/src/lib/storage.ts:ApprovalRecord`, `approvalsStore` | Approval is browser-local metadata. It is not authoritative backend plan activation or a driver manifest update. |
| Other domains | Active backend and `experiments_v1_1` inspected/search-matched | No India frozen model, RUN1-v2 artifact, or named Solomon/Dabia evaluator was located there. Do not claim they exist or restore archive code. Source owner must supply exact frozen paths/contracts for those integrations. |

## Frozen ETA and uncertainty interface

First deliver `docs/INDIA-ETA-CONTRACTS.md` plus JSON request/result fixtures. These are proposals to freeze before coding, not endpoints that already exist.

Request fields: schema version, request/event ID, source/domain ID, origin/destination identifiers and coordinates where supported, timezone-aware prediction timestamp, vehicle type, supported conditions with observation/as-of timestamps, model ID/version, feature-schema hash and optional trusted release hash. Weather/traffic can be absent; never infer observed context from a fixture or future values. Reject unknown fields, booleans masquerading as numbers, NaN/infinity, ambiguous timestamps, incompatible schema, invalid coordinates and unsupported vehicles.

Model release interface: immutable model/feature versions, source and artifact hashes, supported geography/vehicle/domain, prediction target (travel-only versus total delivery time), feature availability, units, training/split hashes or explicit unavailable status, trusted loader/inference callable and calibration artifact version. Accept only trusted allowlisted release paths. No arbitrary pickle/code/path/URL loading from API input. Do not copy private dataset rows into fixtures.

Response fields: schema version; predicted duration in minutes and predicted arrival timestamp; classical prediction; optional quantum residual and hybrid prediction separately; selected prediction source; uncertainty status/type; residual or scenario weights and provenance where available; prediction target; model/schema/data hashes; QML requested/executed flags; execution type/backend; measured runtime; fallback used/reason; safe error/status. Missing checkpoint must still return a valid classical result with `qml_executed=false`, hybrid unavailable and explicit fallback reason. Schema mismatch cannot be hidden by silently applying the wrong model.

Use deterministic labeled classical fixtures initially. A stub quantum success is labeled simulated test metadata, never actual quantum execution. A future real quantum residual is added only if its frozen contract has a duration target, supported features and demonstrated execution metadata. The current route-risk VQC is not that checkpoint.

Uncertainty states: `unavailable`, `synthetic_fixture`, `validated_source_domain`, `out_of_domain`. Null probability/interval fields carry reasons; do not use zero for unavailable risk. An interval or point prediction alone does not determine a lateness probability or expected excess. Consume a separately frozen residual distribution/conditional scenario artifact from its owner; no model/residual fitting in this integration scope. Synthetic residual fixtures prove arithmetic only.

For real calibration evidence, consume independent calibration/validation/test provenance, sample counts, condition support, coverage at declared levels and the predeclared acceptance rule. Freeze thresholds before opening evaluation outcomes. No tuning on test, no claim that nominal coverage is conditional coverage. Brier-calibrated route probabilities from RUN2 are not an ETA residual distribution.

## Scenario risk, route timing and domain gates

Externally supplied or explicitly synthetic deadlines include source/type, event definition, timezone, currency and per-late-minute penalty. Food-delivery observed durations do not establish contractual deadlines or real SLA labels.

For each weighted scenario s and stop i: compute arrival, service start and departure from remaining state and ordered legs; retain waiting and service separately. Let L be max(0, service-start minus deadline) under the existing contract definition. Estimate P(late) as the weighted indicator average and expected late minutes as the weighted L average. Expected penalty is rate times expected late minutes; report transport and switching costs separately. Validate weights, units and nonnegative physically valid durations. Do not silently clip invalid tails or double-count service/waiting in a total-delivery-time model.

Tiny oracle: prediction at 10:00, travel-only point estimate 20 minutes, synthetic residuals [-5, 0, 10] with equal weights, no service/waiting, deadline 10:25. Arrivals are 10:15, 10:20 and 10:30. Lateness is [0, 0, 5], so late probability is 1/3, expected late minutes 5/3 and expected penalty at 6 currency units/minute is 10. Multiplying probability by rate gives 2 and is wrong for this contract. These are synthetic arithmetic examples, not measured calibration.

Compatibility rules inspect prediction target, source units, road-routing provenance, geography, vehicle, feature coverage, clock and service semantics, horizon and uncertainty support. Return a reasoned supported/scenario-only/rejected decision. Straight-line distance may be an explicitly declared model feature; it is never substituted for verified road travel duration. A whole-delivery-duration model must not populate travel arcs without an evidenced decomposition.

Solomon/Dabia keep their own original duration matrices and source-specific adapter. Without compatible India support, run only clearly synthetic traffic/scenario overlays with unchanged source IDs and explicit assumptions; do not report an India-calibrated benchmark. For sums across stops, use a frozen joint scenario set preserving shared traffic/delay dependence, or label an explicit hypothetical independence assumption. Never add marginal quantiles or multiply independent on-time probabilities silently.

## Optimization and independent checks

New coefficient adapter enumerates bounded feasible route candidates from source-specific inputs, computes nonnegative costs in one declared currency, and passes them to unchanged `build_route_qubo` and warm-start/QAOA primitives. Use the existing V1.1 augmented LP path for the LP warm-start requirement after inspecting its full invocation and convergence evidence; record its tolerance/budget and do not call the core box relaxation LP. Keep candidate order, IDs, variable order, offsets, coverage/vehicle penalties and warm-start vector provenance intact. Use a tiny <=8-variable fixture initially; broader SDK limits remain separate capability gates. Classical exact selection uses the same candidates/scenarios/costs for a fair fallback.

Validate deterministic route feasibility against original source constraints; soft expected penalties never certify a hard deadline. Preserve hard-window failures per scenario and report empirical hard violation frequency separately; no implied chance-constraint guarantee. Independently recompute expected financial coefficients from immutable scenario inputs using a separate validation path, not cached solver coefficients, and enumerate every tiny bitstring against direct objective/constraint calculations. Keep expected and nominal/realized financial totals separate. If route-state support cannot meet the original validator's depot assumptions, retain static scenario-only mode and report live activation unsupported rather than altering the validator to pass.

## Event, proposal and approval state machine

Input includes event ID, active plan ID/version/hash, as-of time, observed completed-stop prefix and timestamps, vehicle position/current stop or a declared simulated position, and context/model versions. Reject stale/out-of-order events or inconsistent prefixes. Without current-state evidence, evaluate a named simulated scenario only.

1. Load the last independently validated plan and freeze completed work. Recompute remaining stop forecasts and exposure using the same future scenarios for keep/candidate comparisons.
2. Apply predeclared material risk/cost thresholds. `KEEP` means no actionable material change. `ALERT` means material risk exists but no supported, feasible, beneficial proposal can be offered, or a capability/uncertainty gap requires attention.
3. Only material changes enter bounded reoptimization. Validate raw decoded and any repaired candidate independently. Preserve failures and explicit classical fallback without replacing the active plan.
4. Compute net benefit = expected keep-plan cost minus expected candidate cost minus explicit switching cost, all under the updated event and one currency. Proposal requires net benefit above a frozen minimum and supported feasibility. Example: keep 120, candidate 100, switching 25 gives -5 and `ALERT`, not a proposal.
5. `PROPOSE_REPLAN` creates an immutable pending candidate with source hashes, validator evidence, expiry and base plan version; it never changes assignments.
6. Approval requires an explicit dispatcher/admin action. In one transaction, verify role, expiry, current input/context hash and base version; revalidate; atomically update active version, approval audit and versioned per-vehicle manifest. Reject stale, cancelled, invalid and competing approvals. An initial active plan must also be explicitly registered/approved from a validated result.
7. Persist duplicate-event/proposal detection, cooldown, minimum benefit and a reverse-plan hysteresis rule. Record suppression reasons. Restart must not clear these controls or autoapprove pending work.

New durable state belongs in namespaced ETA SQLite tables/store, preserving existing jobs/recovery tables. Manifests include vehicle ID, plan version/hash, ordered remaining stops, ETAs, completed-stop references and approval provenance; retain prior versions for audit. No driver authentication or real vehicle dispatch is claimed. Current browser approval records cannot authorize this transaction.

## Execution sequence and acceptance

| Step | Luna deliverable | Required evidence |
| --- | --- | --- |
| 0. Freeze boundaries | Refresh Git/contracts; inventory CPU/Python/dependencies; protected-path hashes; resolve source owners; freeze contracts, fixture IDs, budgets and route map | Existing-vs-new capability map and baseline status. No need to wait for model release. |
| 1. Classical ETA | Strict schemas, trusted fixture provider, provenance, timestamp/unit conversion, safe missing-checkpoint path | Classical result without SDK/GPU; mismatch/invalid/unsupported cases; immutable inputs. |
| 2. Risk/domain | Weighted synthetic residual oracle, source adapters, compatibility gates, conditional-uncertainty unavailable behavior | Hand calculations above, coverage-test harness with source labels, dependency/correlation and hard/soft tests. |
| 3. Coefficients | Unchanged QUBO/LP/QAOA primitives behind new adapter; independently checked same-cost classical fallback | Exhaustive tiny coefficient/Ising regression; raw/repair/decision/validation separation; infeasible output rejected. |
| 4. Acting | Remaining-state simulation, policy, durable proposals/approval/manifests and restart behavior | KEEP/ALERT/PROPOSE_REPLAN fixtures; no-benefit, stale/duplicate, cooldown, completed-work and concurrency tests. |
| 5. API integration | Narrow FastAPI mounting and explicit auth policy; route/schema snapshots; bounded event job lifecycle | Existing API regression; new malformed/error/redaction checks; timeout/cancel/restart; loopback smoke if startup works. |
| 6. Review/report | Independent acceptance, three substantive cycles, report and reproducible JSON fixtures | Exact commands/outcomes, source labels, dependency gaps and protected-file hashes; no inferred completion. |

Do not invent existing endpoints. Freeze an additive route map during step 0 from these missing capabilities: ETA prediction, initial plan registration, event submission/status/cancellation, pending proposal approval and versioned manifest retrieval. Reuse `/api/v1/plans/validate`, `/api/v1/risk/evaluate`, `/api/v1/recovery/plan` and existing jobs only where actual payload/solver/state semantics fit. Existing job manager is not a generic ETA executor: add a bounded ETA job adapter with persisted IDs/status/deadline and stable errors if extension is needed; do not pretend `/jobs` already accepts ETA events. Every new write capability needs explicit role policy and optimistic concurrency. Default local mode must not silently enable active-assignment approval; require the configured token-role mode for approval and test with non-secret sentinels.

Planned tests under `tests/india_eta/`: prediction, uncertainty, compatibility, coefficients, route projection, events, approval/persistence, and API contracts. Mandatory cases include classical-only; absent quantum checkpoint; schema mismatch; real-versus-mocked inference metadata; units/timezones/midnight; residual coverage harness and expected positive lateness; synthetic deadlines; unsupported vehicle/domain; joint scenarios; hard-window failure; infeasible/raw/repaired QAOA output; classical fallback; unchanged/no-benefit route; completed stops; stale/invalid/unauthorized approval; simultaneous approvals; manifest versioning; duplicate/oscillating events; timeout/cancel/restart and redacted sentinels.

Commands are targets, NOT RUN in this turn:

```powershell
.venv\Scripts\python.exe -m pytest tests/india_eta -q --basetemp=.runtime/pytest-india-eta
.venv\Scripts\python.exe -m pytest tests/test_api.py tests/test_auth.py tests/test_domain.py tests/test_financial.py tests/test_recovery.py tests/test_recovery_persistence.py tests/test_risk.py tests/test_qubo.py tests/test_qaoa_dispatch.py tests/test_learning_inference.py -q --basetemp=.runtime/pytest-india-regression
```

Use actual installed runtime capabilities; SDK-specific checks run separately in `.venv-sdk` only if needed. New tests use isolated temporary databases. Do not run training scripts, overwrite frozen experiments or rerun other agents' held-out studies. Report any unrelated existing failures separately.

Bounded stress proposal: at most 2 workers, 20 invalid requests, 10 tiny event jobs at concurrency 2, one forced timeout, one cancel, one process restart and one simultaneous-approval race. <=100 joint scenarios/request, <=3 orders/4 vehicles for projection fixtures, <=8 route variables for circuit integration. Record actual wall/CPU/memory and enforce request/worker budgets; no unbounded retries. Freeze these ceilings before execution and reduce them if existing service bounds are tighter.

Three required substantive cycles, each failure hypothesis -> observed counterexample -> material edit -> retest -> artifact: (1) duration/time/target or penalty semantics; (2) domain/uncertainty/dependence and coefficient validation; (3) stale approval/restart/completed-state correctness. These are planned hypotheses; none is a completed cycle. Do not manufacture a bug or relabel documentation work to satisfy this gate.

## Release handoff and stop conditions

Codex 1 need only supply a frozen model/interface manifest and independent residual/calibration evidence when ready. Missing release evidence leaves uncertainty unavailable and classical fixtures runnable. Missing Solomon/Dabia paths leave those source-specific integrations NOT RUN. No retraining, post-hoc scenario selection, fabricated metrics or quantum advantage claim.

Before each bounded batch/dispatch use `get_usage_limits`; stop new work and interrupt active workers at >=50% weekly consumed, or if unavailable, and prepare a safe handoff. Latest planning read was 2% weekly consumed; this is a point-in-time reading, not global enforcement. This planning turn starts no Terra/Luna implementation window. At future start, inspect the persisted Terra deadline; never reset/extend an existing expired window. If the existing five-hour window governs execution, record actual start/deadline and wrap up 15 minutes before it. A separate new five-hour window requires owner authorization.

Final report lists paths, changed/protected hashes, commands/counts/outcomes, fixtures versus real observations, classical versus actual quantum/mocked execution, raw outputs/repairs/validation, source and calibration provenance, measured resources, failures, missing artifacts and remaining gates. No calibrated India risk, feasible live dispatch, production auth, remote/QPU execution or scientific superiority claim without direct evidence. Future delegation requires an actual Luna handle/start record; no agent was dispatched by this plan.
