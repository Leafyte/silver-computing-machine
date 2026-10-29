# India ETA optimizer and remaining-route projection

**Status:** Implemented locally in this worktree; focused checks executed on 2026-10-29 (Asia/Kolkata). This is a synthetic/local optimization slice. It does not establish India ETA calibration, live dispatch feasibility, QPU use, or quantum advantage.

## Export contract

`backend/india_eta/optimization.py` exports:

- `optimize_dispatch(scenario, weighted_scenarios, *, currency="XTS", operating_rate=1.0, locked_prefixes=None, vehicle_states=None, quantum=True, seed=7)`
- `evaluate_plan(scenario, plan, weighted_scenarios, *, currency="XTS", operating_rate=1.0)`

`weighted_scenarios` is a list of at most 100 `{id, weight, travel, source_label, prediction_target}` records. Each `travel` is a complete nonnegative finite duration matrix over the source depot and every order; IDs must be unique and positive weights must sum to one. `source_label` is required and `prediction_target` must be `travel_time`. A suitable synthetic label is `synthetic_overlay`; source-specific duration matrices should retain their source identity. Whole-delivery targets such as `total_delivery_time` are rejected as travel arcs. Scenarios are treated as joint traffic worlds, not as independently sampled stop marginals. The optimizer does not claim or infer India provenance from the label.

The stable result schema is `debugleaf.india_eta.optimization.v1`:

| Field | Meaning |
|---|---|
| `plan` | Selected routes in the existing domain shape. |
| `validation` | `backend.validator.validate_plan` recomputation from original scenario constraints and deterministic source travel. |
| `expected_cost` | Independently recomputed weighted transport plus lateness penalty totals, with travel, transport, penalty and total kept separate. |
| `remaining_cost_validation` | When `vehicle_states` is supplied, independently recomputed future cost from explicit current vehicle positions and clocks; completed-prefix travel and lateness are excluded. |
| `state_validation` | Confirms every selected vehicle has a consistent state and prefix. |
| `coefficients` | `debugleaf.india_eta.coefficients.v1`, candidate IDs/order/costs, each scenario cost, currency/rate, QUBO coefficients/offset/penalties, completed-prefix constraints and exhaustive audit. |
| `quantum` | Local statevector QAOA metadata, raw counts/samples, raw decoded plans and validation, repairs and repaired validation kept in separate arrays, plus the complete V1.1 LP vector and mapping when QAOA is within its limit. |
| `classical_reference` | Exact route selection over the same full candidate pool and expected candidate costs. It is never counted as a quantum sample. |
| `fallback` | Explicit `used` flag and reason. The entire candidate pool is kept; no favorable subset is selected to fit a circuit. |
| `execution_type` | `local_python_statevector_simulator`, `local_classical_exact_candidate_selection`, or `classical_trivial`. |

With more than eight route variables, QAOA is marked `unsupported_candidate_pool_size`; the deterministic classical exact-cover selector remains the path. Candidate enumeration uses the existing bounded domain generator (up to eight source orders and 10,000 route candidates). The exact-cover selector has a 100,000-node cap and raises a stable `solver_budget_exceeded` message rather than publishing an unproven incumbent. Small route candidate pools are exhaustively checked in QUBO and Ising form (through 12 variables; at most 4,096 bitstrings).

`locked_prefixes` maps vehicle IDs to completed stop IDs in route order. Optional `vehicle_states` is a list of `{vehicle_id, position, position_type, elapsed_minutes, available_at, completed_prefix}` records. If both state and prefix arguments are supplied, their completed stops must agree. When states are present, candidates are limited to vehicles with state evidence. The route generator is filtered so a completed stop cannot move to another vehicle or out of order. The QUBO exact-cover requirement includes **all source orders**, including completed stops; prefix filtering forces those stops to remain on their original vehicle. This retains a completed-only route as a valid candidate when another vehicle is cheaper for future work. Each candidate's coefficient begins at that vehicle's supplied position and `max(elapsed_minutes, available_at)`, then skips historical completed legs and penalties; the independent state-aware repricing repeats this computation from source matrices. Thus exact selection, LP warm start and QAOA use the same future-cost coefficients. Full-plan `validate_plan` still separately checks deterministic source constraints from depot/time zero. Scenario hard-window violations from the current state are reported as empirical frequencies and never represented as certified chance constraints.

## Independent objective and warm-start path

For route `r`, scenario `s`, order `i`, travel rate `q`, source penalty `p_i`, service duration `u_i`, and close time `c_i`:

`arrival_i = prior_departure + travel_s(previous_location, i)`

`service_start_i = max(arrival_i, open_i)`

`lateness_i = max(0, service_start_i - close_i)`

`route_cost_s = q * route_travel_minutes_s + sum_i(p_i * lateness_i)`

`expected_cost = sum_s(weight_s * route_cost_s)`

The coefficient construction and independent plan evaluator each traverse raw scenario travel and order fields. The evaluator does not trust cached coefficients, QUBO cost or submitted ETA values. Deterministic hard-window feasibility is checked by `validate_plan`; weighted hard-window violation frequency is reported separately. Neither expected cost nor violation frequency certifies a chance constraint.

Tiny arithmetic example (synthetic): one vehicle, one order, depot-to-order and return travel are both 2 minutes, no lateness, and `operating_rate=1 XTS/minute`. The route cost is `1*(2+2)+0=4 XTS`. The route-selection QUBO has one variable and default coverage penalty `1+4=5`: offset `5`, linear coefficient `4-5=-1`; energy is 5 for `x=0` and 4 for `x=1`. The Ising form has offset `4.5` and field `h=0.5` under `z=1-2x`, preserving energies 5 and 4. A single-candidate set-partitioning LP has relaxation vector `[1.0]`; the vector is recorded and passed to unchanged `run_qaoa`, whose warm-start initialization clamps only if outside `[0.05,0.95]`. This example shows coefficient arithmetic, not business economics.

The warm-start function comes from `experiments_v1_1/quantum_optimization.py::lp_augmented_relaxation`, not `backend.qaoa.continuous_relaxation` (which is the older box-projected QUBO relaxation). The implementation passes up to 5,000 iterations and records method, steps, complete candidate-order vector and mapping. No external SDK, QpiAI Opt service or QPU is invoked by this adapter. The existing Python statevector implementation is labeled local simulation only.

## Remaining-route state projection

`backend/india_eta/projection.py` exports:

`project_remaining(scenario, plan, vehicle_states, weighted_scenarios, *, currency="XTS", operating_rate=1.0)`

`vehicle_states` is a list containing one state for each assigned vehicle:

```json
{
  "vehicle_id": "V1",
  "position": "A",
  "position_type": "actual",
  "elapsed_minutes": 25,
  "available_at": 27,
  "completed_prefix": [{"order_id": "A", "completed_at": 23}]
}
```

Position must be a location in the source matrix or depot; `position_type` is explicitly `actual` or `simulated`. Clock values are nonnegative minutes on the same plan clock. Forecasting begins at `max(elapsed_minutes, available_at)` from the supplied position. Completed order IDs and their timestamps must match a nondecreasing prefix of the saved route; rows are carried forward with `immutable=true` and are not replayed. Because event travel may differ from the last saved nominal matrix, projection rebuilds and validates the current nominal route metrics from the approved order sequence before projecting. It does not use those refreshed depot-based metrics as the future live clock: the suffix starts from supplied position/time. Completed rows are carried from the saved plan. A missing vehicle state returns `unsupported_missing_vehicle_state` with `valid=false`. No depot replay is passed off as live state.

Projection output labels per-vehicle state mode, carries immutable completed rows, computes future arrivals/service starts/wait/service/departures separately for each joint scenario, and reports weighted remaining travel, penalty and empirical hard-window violation frequency. It also returns `valid`, `projection_mode`, and an `evaluation.expected_cost.amount` adapter for the bounded proposal controller. The legacy `validate_plan` is intentionally not applied to these projected clock/position values: it assumes a depot and time zero. `prefix_validation` checks input prefix consistency, while the original plan must pass `validate_plan` before projection. Currency/rate terms can be provided to keep before/after candidate costs in common units.

## Refinement evidence

| Cycle | Failure hypothesis and evidence | Material change | Retest and result | Remaining concern |
|---|---|---|---|---|
| 1 — prefix-constrained selection | A completed-only route could enter QUBO with zero remaining stops, while an extended route also contained the same prefix; focused tests exposed invalid zero-stop candidates and duplicate vehicle selection. | Preserve completed stops in each vehicle-locked candidate and enforce coverage over every original order, so the exact-cover solver cannot drop historical work when another vehicle takes the remaining route. | `python -m pytest tests/india_eta/test_optimization.py::test_locked_prefix_stays_with_vehicle_and_does_not_enter_remaining_qubo -q` — PASS (included in final rerun). | Prefix ETAs are not live-stated by this optimizer; use the separate state projection for them. |
| 2 — solver objective could be trusted | A solver-selected candidate cost could differ from raw-matrix weighted costs, and QUBO/Ising offsets can preserve plausible rankings while changing absolute energy. | Independently recompute every scenario route cost; include full candidate/scenario costs and exhaustively compare direct business penalty, QUBO polynomial and Ising energy on every bounded bitstring. Record the V1.1 LP vector separately from the old box relaxation. | `python -m pytest tests/india_eta/test_optimization.py::test_optimizer_records_qubo_isings_audit_and_separates_raw_from_repairs tests/india_eta/test_optimization.py::test_scenario_cost_is_recomputed_and_expected_lateness_uses_minutes tests/india_eta/test_optimization.py::test_small_candidate_pool_records_complete_v11_lp_vector -q` — PASS (included in final 14-pass suite). | The cost coefficients are caller-input arithmetic, not calibrated India financial values. |
| 3 — historical travel could distort a replan | If a replan used depot/time-zero cost for an active vehicle, completed travel and lateness could affect a remaining-route choice, and an unrelated whole-delivery model could be mistaken for an arc predictor. | Added optional explicit `vehicle_states`; route coefficients and independent stochastic validation now restart at each vehicle's position and `max(elapsed_minutes,available_at)`, skip completed-stop costs, and require caller `source_label` plus `prediction_target='travel_time'`. Reject whole-delivery targets and unlabeled matrices. | `python -m pytest tests/india_eta/test_optimization.py::test_state_aware_cost_skips_completed_history_and_reprices_from_position tests/india_eta/test_optimization.py::test_whole_delivery_prediction_cannot_be_used_as_a_travel_arc tests/india_eta/test_optimization.py::test_unlabeled_scenario_matrices_are_rejected -q` — PASS (included in final 14-pass suite). | Complete source and location compatibility gates still belong to source-specific adapters; labels do not prove provenance. |
| 4 — route projection could conceal hard-window exposure | A future route may be feasible under nominal travel and violate a hard close under a delayed joint scenario. | Carry per-stop scenario hard-window flags and compute weighted empirical hard-window violation frequency separately from expected soft penalty. Return controller-required `valid`, mode and expected-cost fields; use caller currency/rate for common-unit comparisons. | `python -m pytest tests/india_eta/test_projection.py::test_projection_starts_at_observed_position_and_keeps_completed_prefix tests/india_eta/test_projection.py::test_projection_reports_hard_window_violation_frequency_across_joint_scenarios -q` — PASS (included in final 12-pass suite). | Frequency is scenario-specific only and has no calibration or chance-constraint guarantee. |
| 5 — stale saved ETAs could block every event | The saved plan’s nominal stop metrics differ from the event-updated travel matrix, so direct validation against new travel would reject an otherwise valid route sequence before projection. | Rebuild nominal route timing from the approved ordered route IDs and current event source matrix, validate the rebuilt full plan, preserve completed stop snapshots from the original saved plan, and forecast only the suffix from supplied live/simulated position and clock. | `python -m pytest tests/india_eta/test_projection.py::test_projection_refreshes_stale_nominal_metrics_but_preserves_completed_snapshot -q` — PASS (included in final 12-pass suite). | Rebuilt future route metrics are checked from current source, but completed historical rows remain the saved snapshot plus caller completion timestamps. |
| 6 — physical position could skip uncompleted work | A supplied current position farther along the planned route could cause the projector to silently omit stops lacking completed timestamps. | Require an order-location position to be the first remaining stop or latest completed stop; reject positions that skip uncompleted route rows. Optimizer candidate filtering applies the same route-position rule. | `python -m pytest tests/india_eta/test_projection.py::test_projection_rejects_position_that_skips_an_uncompleted_route_stop -q` — PASS (included in final 14-pass suite). | Positions between order locations require a richer road-network state contract and are currently unsupported. |
| 7 — a cheaper vehicle could erase completed work | When V1 had completed A but could cheaply extend to B, route candidates priced only for uncovered stops could select V2:B and omit A from the final plan. | Keep full order IDs in QUBO candidates and restore exact coverage constraints over all source orders. A completed-only V1:A route can now be selected alongside V2:B; historical cost is still excluded from remaining-cost coefficients. | `python -m pytest tests/india_eta/test_optimization.py::test_completed_prefix_remains_covered_when_other_vehicle_takes_remaining_work tests/india_eta/test_optimization.py::test_locked_prefix_stays_with_vehicle_and_does_not_enter_remaining_qubo -q` — **2 passed in 0.14s**. | QUBO includes completed-stop variables to preserve exact-cover bookkeeping; these variables are vehicle-locked and their future-cost coefficient excludes historical travel/penalties. |
| 8 — completed routes could retain historical cost or lose their snapshots | A route with every stop completed still needs its return leg, but must not reprice historical deliveries or erase immutable stop records. | Keep all-stop route candidates in exact coverage, start the candidate cost at the last observed location/clock, and preserve completed snapshots during projection. Added exact-reference/local-QAOA same-pool cost checks and a no-truncation candidate-count assertion. | `python -m pytest tests/india_eta/test_optimization.py tests/india_eta/test_projection.py -q` — **17 passed in 0.45s**. | The return-to-depot leg is included as future work; intermediate road positions remain unsupported unless they map to source matrix locations. |

## Executed evidence and limits

- Focused command before the completed-prefix coverage correction: `C:\ommd3076\Proejcts\nmit-hackathon\.venv\Scripts\python.exe -m pytest tests/india_eta/test_optimization.py tests/india_eta/test_projection.py -q` — **14 passed in 0.52s**.
- Critical regression command after that correction: `C:\ommd3076\Proejcts\nmit-hackathon\.venv\Scripts\python.exe -m pytest tests/india_eta/test_optimization.py::test_completed_prefix_remains_covered_when_other_vehicle_takes_remaining_work tests/india_eta/test_optimization.py::test_locked_prefix_stays_with_vehicle_and_does_not_enter_remaining_qubo -q` — **2 passed in 0.14s**.
- Final owned-suite command: `C:\ommd3076\Proejcts\nmit-hackathon\.venv\Scripts\python.exe -m pytest tests/india_eta/test_optimization.py tests/india_eta/test_projection.py -q` — **17 passed in 0.45s** after the all-order exact-cover, same-pool cost, all-completed optimizer/projection and no-truncation regressions.
- The tests execute a real, tiny, local Python statevector QAOA path as well as classical-only paths. No GPU, QPU, paid credits, external API, ETA training or model fitting was used.
- SDK references checked from project evidence: QpiAI Quantum `0.7.5` and QpiAI Opt `0.0.9`, pinned public repository refs recorded in `bootstrap/SDK-EVIDENCE.md` as of 2026-10-28 Asia/Kolkata. Those SDK facts do not imply adapter invocation here.
- Source requirement: `sources/problem-extract.txt` states exact vehicle-to-route delivery, ordered stops/ETA risk, cost/exposure, and evaluation of SLA, coverage, cost and solver quality. Acceptance targets and limitations are in `docs/V1-ACCEPTANCE.md`; route and objective interfaces were source-inspected in `backend/domain.py`, `backend/qubo.py`, `backend/qaoa.py`, `backend/validator.py`, and the V1.1 optimizer.
- Not measured here: CPU/memory profile beyond test wall time; calibrated SLA probability; contractual India penalties; road-routing support; real vehicle state provenance; production concurrency/persistence; authenticated service integration; cloud/QPU runtime. No quantum advantage is claimed.
