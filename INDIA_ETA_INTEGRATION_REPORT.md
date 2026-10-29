# India ETA integration: executed implementation report

**PASS for the authorized bounded local fixture workflow.** Final current-tree evidence: **107 focused tests + 10 subtests**, **120 existing regressions + 18 subtests**, and a real localhost application demonstration. These are local/synthetic results, not measured India performance, calibration or live fleet deployment.

## Location, ownership and usage correction

- Branch: `codex/india-eta-integration`, base `819913e` from the published planning branch.
- Worktree: `C:\ommd3076\Proejcts\nmit-hackathon\.runtime\india-eta-implementation`.
- The final acceptance run was captured before publication, while implementation files were uncommitted. This report preserves that execution snapshot; the owner has since authorized publishing it from this branch. The source, fixtures and acceptance evidence are included in the pull request.
- Luna handles `/root/eta_prediction`, `/root/eta_optimizer`, `/root/eta_controller` implemented their slices and are terminal/completed, verified through agent status. Codex coordinated, independently reviewed code, implemented the API/job/source adapter, and ran final acceptance.
- Worker start `2026-10-29T11:13:37Z` (16:43:37 Asia/Kolkata); integration deadline `16:13:37Z`, wrap-up `15:58:37Z`. The new integration finished before its bound; no earlier Terra timer was reset. See `execution_start.json`.
- **Correction to the interim report:** the workers had confused the five-hour 50–51% reading with the weekly allowance. The resumed reliable reading showed primary 56% / 300 minutes and secondary **9% / 10,080 minutes**. Last pre-final validation reading: primary 70%, secondary **11% weekly consumed / 89% remaining**. Work resumed within the actual weekly reserve; no global usage enforcement is claimed.

## Implemented mechanism

The additive `backend/india_eta/` package provides:

1. Strict versioned India input, timezone/unit checking, JSON-only hash-pinned classical release loading, duration/arrival/provenance and explicit missing-QML fallback. The safe format supports a two-feature linear travel-time release. Default inference is clearly synthetic; unsupported trained checkpoint formats need a reviewed adapter.
2. Weighted residual SLA arithmetic and separately trusted frozen residual releases with identity/schema/domain/condition/evidence checks. Missing uncertainty returns null risk fields. No caller-declared calibrated label is accepted.
3. A real ETA-to-dispatch source adapter: the event API computes predictions server-side, checks a persisted model ID/version/hash and time/location/target semantics, validates all joint worlds, and derives explicitly synthetic travel durations. It retains the bounded prediction request, hashes and compatibility evidence for replay. Whole-delivery predictions are not treated as road durations.
4. Source-recomputed remaining costs passed to unchanged candidate generation, QUBO, V1.1 augmented LP warm start and Python statevector QAOA. Quantum/exact selection uses the identical complete candidate pool; candidates are not truncated to fit a circuit. Full stop coverage preserves completed-only vehicles. Raw samples, repairs, fallback and independent static/stochastic checks remain separate.
5. Persistent KEEP/ALERT/PROPOSE_REPLAN processing from explicit simulated vehicle position/time, immutable completed prefixes, material-change checks, cooldown and reverse-plan hysteresis. No automatic approval occurs.
6. Authorized approval under one SQLite `BEGIN IMMEDIATE` transaction: current base/model/event/context/job/expiry/cancellation checks, independent feasibility and fresh same-world keep/candidate benefit minus switching cost. Active version, audit and manifests publish atomically. Removed vehicles receive versioned empty revoked manifests; historical revisions survive restart.
7. Mounted versioned FastAPI endpoints, bounded durable event jobs, pre-JSON body limits, stable safe errors, timeout/cancel publication suppression, restart recovery, idempotency and a runtime lease. Metadata includes Python/dependency versions, implementation hash, settings, source inputs, request hash and timestamps.

Exact schemas/routes/runnable examples: `docs/INDIA-ETA-CONTRACTS.md`. Mechanism/code/numerical/cycle details: `docs/INDIA-ETA-PREDICTION.md`, `docs/INDIA-ETA-OPTIMIZATION.md`, `docs/INDIA-ETA-CONTROLLER.md`.

Only preexisting `backend/api.py` (mounting/injection/body cap) and `backend/auth.py` (ETA role policy) changed. **All 54 protected tracked files matched initial SHA-256**; see `protected-file-check.json`. Core mathematics, circuits, validators and frozen experiments are unchanged. Other agents' untracked training/`experiments_run2` work was never copied into this isolated checkout or modified. No training, GPU/QPU/credits, runtime-package/global/PATH/CUDA changes, remote execution or deployment occurred.

## Requirement-by-requirement acceptance

| Requirement | Direct evidence | Outcome |
| --- | --- | --- |
| India-specific ETA and trusted frozen classical loader | Prediction/schema/registry/path/hash tests; real HTTP duration 20 with fixture provenance | PASS supported format/fixture; real checkpoint not supplied |
| Uncertainty with supplied deadlines | Trusted residual identity/provenance/condition tests; unavailable/null API path | PASS interface; real India calibration NOT RUN |
| Independent uncertainty oracle | Separate numerical tests and real HTTP: probability 1/3, late minutes 5/3, penalty 10 | PASS synthetic arithmetic |
| Correct compatible versioned costs enter optimizer | Server ETA event to source gate to weighted costs/proposal/approval; exact/QAOA same-pool and LP/QUBO/Ising checks | PASS local simulator |
| Invalid assignments cannot publish | Independent static/state validation; fully nominally feasible completed-prefix tamper rejection; hard-window and false-benefit tests | PASS simulated domain |
| KEEP, ALERT, PROPOSE_REPLAN | Complete versioned fixtures and actual Uvicorn HTTP responses | PASS |
| Explicit safe approval and atomic manifests | Configured role guard, race/stale/cancel/expiry rejection, completed snapshots, revoked vehicles/history/restart | PASS local transactional prototype |
| Strict API, safe errors, bounded robustness | Real ASGI body/chunked 413, invalid/schema/sentinel tests; job capacity/deadline/lease/history tests | PASS bounded checks |
| Cancellation, timeout, restart, duplicates | Real-ASGI delayed event cancellation/timeout; restart-interrupted tests; real server restart and duplicate result | PASS; threads are not forcibly killed |
| Required focused/existing regressions | Saved final pytest results below | PASS 227 tests and 28 subtests |
| Paths, commands, resources, reviewable implementation diff | This report, consolidated contracts, raw fixtures/evidence, and the pull request diff | PASS; acceptance artifacts are a pre-publication snapshot |

Unsupported actual live state is rejected, and unsupported real domain transfers are rejected or explicitly scenario-only. No real vehicle activation, production authorization, calibrated India SLA outcomes, QPU execution or quantum superiority is claimed.

## Exact commands and final results

Run in the implementation worktree. `PYTHON` below is the absolute existing runtime `C:\ommd3076\Proejcts\nmit-hackathon\.venv\Scripts\python.exe`.

```powershell
PYTHON -m pytest tests/india_eta -q --basetemp=.runtime/pytest-india-final-reviewed
```

**107 passed, 10 subtests passed, 1 warning in 5.22s.** Saved `evidence/india_eta/focused-tests.txt`. This includes final schema/model hashes, full-prefix encoding, zero-penalty hard-window ALERT, model binding, replayable source adapter, body limits and actual ASGI cancel/timeout behavior.

```powershell
PYTHON -m pytest tests/test_api.py tests/test_auth.py tests/test_domain.py tests/test_financial.py tests/test_recovery.py tests/test_recovery_persistence.py tests/test_risk.py tests/test_qubo.py tests/test_qaoa_dispatch.py tests/test_learning_inference.py -q --basetemp=.runtime/pytest-india-regression-reviewed
```

**120 passed, 18 subtests passed in 5.39s.** Saved `evidence/india_eta/regression-tests.txt`. This is the required existing regression set, not a claim that unrelated experiments ran. Pre-change baseline was the same 120/18 in 4.58s.

```powershell
PYTHON tools/india_eta_http_smoke.py
```

**PASS**, actual loopback Uvicorn. Started `2026-10-29T11:46:21.024719Z`; **69 HTTP requests, 3.4917685s wall, 0.28125s driver CPU**. The server was stopped after demonstration/restart. Raw output: `evidence/india_eta/http-smoke.json`; complete input/result fixtures: `fixtures/india_eta/http_keep.json`, `http_alert.json`, `http_propose.json`.

- All three action states executed. The event API recomputed ETA and propagated the prediction through explicit scenario compatibility, joint travel worlds, cost coefficients, proposal and approval. Recorded model SHA is `f644ed71adeecc4d8f2399910a087f57db6e5774ea40fe6f8534e2fa017a036a`.
- Approval advanced plan 1 to 2 and ordered manifest B,A; repeat approval returned 409. Manifest and exact duplicate-event result survived a real process restart.
- **20 invalid requests at concurrency 2:** all 422.
- **10 tiny job submissions at concurrency 2:** 8 admitted with 202 and all succeeded, 2 explicitly rejected with 429. No retry manufactured the admission count; scheduling affects these counts.
- Server CPU and memory/RSS were unavailable, not fabricated. Bounds: 2 workers, 100 joint worlds, <=8 quantum variables, bounded existing route/exact search, 256 KB input, 1 MB result, 256 retained jobs by default. Cancelled/timed-out work retains a worker slot until computation exits.

Observed environment: CPython **3.12.14**, 64-bit Windows 11 build 26300; FastAPI **0.162.0**, Pydantic **2.14.1**, pytest **9.2.0**, httpx **0.28.1**, Uvicorn **0.54.0**. CIM CPU/memory inventory was denied. No new runtime packages were installed. The final warning is Starlette's httpx TestClient deprecation. Sandbox TestClient loop/socket initialization stalled; real ASGI/localhost checks used approved unsandboxed execution. Loopback-only access was retained.

## Numerical mechanism and substantive refinement cycles

The hand-authored model is `5 + 3.2*distance_km + 12*traffic_index`. At 1.875 km and 0.75 it returns **20 minutes**. Prediction at 10:00 arrives 10:20. Residuals [-5,0,10], equal weights, deadline 10:25 give late minutes [0,0,5]. At rate 6, expected penalty is `6*(5/3)=10`, while late probability is `1/3`. The API and independent tests reproduce this; it is not measured business performance.

| Cycle | Failure hypothesis/evidence | Material change | Relevant retest |
| --- | --- | --- | --- |
| Remaining-state timing | API initially returned remaining_state_invalid when new travel differed from saved timing; depot pricing could include history | Refresh independently validated nominal sequence, preserve historical rows, price suffix from explicit position/clock | Projection/controller/API regressions pass |
| Completed-only coverage | Review found remaining-only QUBO coverage could omit completed work when another vehicle served future orders | Full candidate stop IDs and all-order coverage, with future-only cost and unchanged core math | Completed-only/all-completed and same-pool regressions pass |
| Approval/state integrity | Nominally valid route could move completed work; saved benefit could be forged; removed vehicle could retain work | Recompute prefix/cost/net benefit inside transaction; race/job/version guards; immutable times and revocation/history | Feasible-tamper, false-benefit, race, expiry, completed/revoked tests pass |
| Hard risk without penalty | Zero-penalty hard-window breach could appear cost-neutral | Include violation frequency in ALERT classification | Independent zero-penalty hard-window regression passes |
| Source/provenance integrity | Review found version-only identity, one-way feature hash and late world validation insufficient | Persist server-derived model ID/version/hash; replayable request; complete-world validation before admission; pre-parse cap | Alternate-model/replay/weights/matrix/chunked-body regressions pass |

Earlier actual failures also included missing worktree `.runtime` setup, an incorrect job INSERT placeholder count, and cleanup of an event rejected before reservation. These were fixed and retested. Intermediate failed logs are retained and superseded by final outputs. No held-out model tuning or scenario selection occurred.

## Remaining limits and handoff

- Real India checkpoints/calibration releases were not supplied. Safe JSON linear releases and trusted residual interfaces are tested with fixtures; other model formats and real calibration performance remain future integration work.
- Existing route-risk VQC is preserved and is not a duration predictor. Missing duration-QML returns an explicit classical fallback. New quantum optimization evidence is genuine **local Python statevector** execution, not QpiAI authenticated/cloud/QPU work or advantage.
- Real Solomon/Dabia-to-India transfer needs evidenced source/target/road/geography/vehicle/clock compatibility. Current overlays are experimental; dataset observations are not contract SLA outcomes.
- Actual/live state is rejected. No driver authentication/client/acknowledgement, tenant isolation, production security or external fleet dispatch exists. Researchers can read ETA resources by ID in the current prototype role policy.
- The retained-history cap rejects new jobs rather than deleting idempotency/audit records. Preserve records before starting a new local database. Event processing has one runtime owner and is not a distributed queue.
- Trusted registry owners must supply honest, independently reviewed training/calibration evidence. Hash checks and imports do not establish scientific validity.

Within the expressly authorized fixture/local boundaries, all implementation and acceptance items are complete. The captured evidence records the isolated worktree before publication; the separate owner-authorized PR carries the implementation. No training or frozen experiment rerun is implied.
