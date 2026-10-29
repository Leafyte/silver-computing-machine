# DebugLeaf / Q-VANTIS

ArQ is a research prototype for the Quant-A-Maze 3.0 QML-01 challenge: delay-risk-aware fleet dispatch for contract logistics. The active problem statement and rubric are extracted from the official source under `sources/`; the older team-submitted challenge PDF is historical context.

## Current stage

The owner has opened a bounded V1 implementation stage. The goal is a full backend centered on a decision-focused variational quantum circuit (VQC), with a classical warm start and warm-start QAOA as supporting approaches. V1 acceptance gates are in [docs/V1-ACCEPTANCE.md](docs/V1-ACCEPTANCE.md). These are targets, not claims that implementation, tests, or results already exist.

The new Terra coordinator owns the 5-hour build window. Its start time is when that agent actually begins; its goal was submitted directly in the new Terra chat. Codex usage limits, stage boundaries, preservation, and reporting rules are in [AGENTS.md](AGENTS.md).

## Current verified workspace state

The earlier preflight text above is a snapshot of the workspace before implementation. The current checkout now contains a bounded local backend prototype, decision-focused VQC and QAOA slices, API/job and risk/recovery/financial paths, and recorded synthetic experiments. Current evidence and limitations are maintained in [docs/TERRA-V1-REPORT.md](docs/TERRA-V1-REPORT.md); reproducible local commands and bounds are in [docs/V1-RUNBOOK.md](docs/V1-RUNBOOK.md). The overall result remains partial and is not a production or live-fleet deployment.

Start with:

1. [AGENTS.md](AGENTS.md) for current stage, safety, evidence, timebox, and usage rules.
2. [docs/V1-ACCEPTANCE.md](docs/V1-ACCEPTANCE.md) for measurable targets and reporting requirements.
3. [bootstrap/SDK-EVIDENCE.md](bootstrap/SDK-EVIDENCE.md) for inspected SDK/package facts and unresolved capabilities.
4. [bootstrap/ARCHITECTURE-READINESS.md](bootstrap/ARCHITECTURE-READINESS.md) for earlier proposals; treat its bootstrap-stage gates as historical where they conflict with the owner-opened V1 stage.
5. [sources/q-vantis-build-spec.md](sources/q-vantis-build-spec.md) for the verbatim legacy specification, retained as reference only.

## Evidence boundaries

The challenge expects an operationally feasible dispatch plan with vehicle assignments, ordered stops, ETAs/SLA-risk estimates, financial exposure, and evaluation of SLA violations, coverage, cost, calibration, solver quality, and runtime. See `sources/problem-extract.txt` and the source PDF for the extracted requirements. SDK package inspection is recorded in `bootstrap/SDK-EVIDENCE.md`; it is not proof of authenticated service calls, QPU use, or end-to-end integration.

No quantum advantage, novelty, calibrated risk, live-fleet dispatch, or operational performance claim is made by this README. Tiny plans have passed the independent validator in local tests and HTTP checks; this is prototype evidence only. The default local API mode has no authentication. An optional token-role mode passed a sentinel localhost smoke, but it is a prototype and has not had production security review. Keep local services on loopback. Results must identify whether they came from classical execution, a local simulator, a mock, cloud simulation, or QPU, and include only metrics actually measured. See [the runtime guide](docs/API-RUNTIME.md) for its role policy and [the acceptance report](docs/TERRA-V1-REPORT.md) for evidence and limits.
