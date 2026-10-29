/** One function per confirmed backend route (backend/api.py). */
import { request } from "./client";
import type {
  AuditRecord,
  FinancialPlanResponse,
  FinancialTerms,
  HealthResponse,
  JobRecord,
  LatestAudits,
  Plan,
  PlanResponse,
  PlanValidateResponse,
  QaoaResponse,
  QaoaSettings,
  RecoveryRequest,
  RecoveryResponse,
  RiskRequest,
  RiskResponse,
  RoutesResponse,
  Scenario,
  ScenarioValidateResponse,
  VqcRequest,
  VqcResponse,
} from "./types";

const V1 = "/api/v1";

export const api = {
  health: (signal?: AbortSignal) => request<HealthResponse>("GET", `${V1}/health`, undefined, signal),

  /** Auth-mode probe: an empty body returns 422 in local mode, 401 in token mode, 503 if misconfigured. */
  authProbe: () => request<unknown>("POST", `${V1}/scenarios/validate`, {}),

  validateScenario: (scenario: Scenario) => request<ScenarioValidateResponse>("POST", `${V1}/scenarios/validate`, scenario),
  routes: (scenario: Scenario) => request<RoutesResponse>("POST", `${V1}/scenarios/routes`, scenario),
  plan: (scenario: Scenario) => request<PlanResponse>("POST", `${V1}/scenarios/plan`, scenario),
  financialPlan: (scenario: Scenario, terms: FinancialTerms) =>
    request<FinancialPlanResponse>("POST", `${V1}/scenarios/financial-plan`, { scenario, ...terms }),
  qaoa: (scenario: Scenario, settings: QaoaSettings) => request<QaoaResponse>("POST", `${V1}/scenarios/qaoa`, { scenario, ...settings }),
  vqcPlan: (body: VqcRequest) => request<VqcResponse>("POST", `${V1}/scenarios/vqc-plan`, body),
  validatePlan: (scenario: Scenario, plan: Plan) => request<PlanValidateResponse>("POST", `${V1}/plans/validate`, { scenario, plan }),

  submitJob: (scenario: Scenario) => request<JobRecord>("POST", `${V1}/jobs`, scenario),
  submitQaoaJob: (scenario: Scenario, settings: QaoaSettings) => request<JobRecord>("POST", `${V1}/jobs/qaoa`, { scenario, ...settings }),
  getJob: (id: string, signal?: AbortSignal) => request<JobRecord>("GET", `${V1}/jobs/${encodeURIComponent(id)}`, undefined, signal),
  cancelJob: (id: string) => request<JobRecord>("POST", `${V1}/jobs/${encodeURIComponent(id)}/cancel`),

  evaluateRisk: (body: RiskRequest) => request<RiskResponse>("POST", `${V1}/risk/evaluate`, body),
  recover: (body: RecoveryRequest) => request<RecoveryResponse>("POST", `${V1}/recovery/plan`, body),
  latestAudits: (scope?: string) =>
    request<LatestAudits>("GET", `${V1}/recovery/audits/latest${scope ? `?scope_sha256=${encodeURIComponent(scope)}` : ""}`),
  getAudit: (id: string) => request<AuditRecord>("GET", `${V1}/recovery/audits/${encodeURIComponent(id)}`),
};

