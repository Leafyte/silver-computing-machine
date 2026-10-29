/**
 * Interpreting stored run records. Every extractor reads only fields the
 * backend actually returns; missing values come back as undefined so the UI
 * can show "not returned" instead of inventing numbers.
 */
import type {
  FinancialPlanResponse,
  JobRecord,
  Plan,
  PlanResponse,
  QaoaRawResult,
  QaoaResponse,
  RecoveryResponse,
  RiskResponse,
  ValidationReport,
  VqcResponse,
} from "../api/types";
import { TERMINAL_JOB_STATUSES } from "../api/types";
import type { MethodId, RunRecord } from "./storage";

export const METHOD_LABEL: Record<MethodId, string> = {
  exact: "Classical exact",
  exact_job: "Classical exact (job)",
  financial: "Financial exact",
  vqc: "VQC risk-aware",
  qaoa: "QAOA warm-start",
  qaoa_job: "QAOA (job)",
  risk: "Risk replay",
  recovery: "Recovery",
};

export type PlanOrigin = "solver" | "qaoa_raw_sample" | "classical_exact_fallback" | "recovery" | "empty_instance";

export interface ExtractedPlan {
  plan?: Plan;
  validation?: ValidationReport | null;
  origin?: PlanOrigin;
  fallback?: { used: boolean; reason: string | null };
  executionType?: string;
}

function jobInner(run: RunRecord): JobRecord | undefined {
  return run.source === "job" ? (run.result as JobRecord | undefined) : undefined;
}

function fromQaoa(r: QaoaResponse | undefined): ExtractedPlan {
  if (!r || !r.decision) return { fallback: r?.fallback, executionType: r?.execution_type ?? undefined };
  return {
    plan: r.decision.plan,
    validation: r.decision.validation,
    origin: r.decision.origin,
    fallback: r.fallback,
    executionType: r.execution_type ?? undefined,
  };
}

export function extractPlan(run: RunRecord): ExtractedPlan {
  switch (run.method) {
    case "exact": {
      const r = run.result as PlanResponse | undefined;
      return r ? { plan: r.plan, validation: r.validation, origin: "solver", executionType: r.execution_type } : {};
    }
    case "financial": {
      const r = run.result as FinancialPlanResponse | undefined;
      return r ? { plan: r.plan, validation: r.proxy_validation, origin: "solver", executionType: r.execution_type } : {};
    }
    case "vqc": {
      const r = run.result as VqcResponse | undefined;
      if (!r) return {};
      return {
        plan: r.plan,
        validation: r.validation,
        origin: r.fallback?.used ? "classical_exact_fallback" : "solver",
        fallback: r.fallback ? { used: r.fallback.used, reason: r.fallback.reason } : { used: false, reason: null },
        executionType: r.execution_type,
      };
    }
    case "qaoa":
      return fromQaoa(run.result as QaoaResponse | undefined);
    case "exact_job": {
      const job = jobInner(run);
      if (!job?.result) return { executionType: job?.metadata.execution_type };
      if (job.status === "infeasible") return { plan: job.result as Plan, validation: null, origin: "solver", executionType: job.metadata.execution_type };
      const r = job.result as PlanResponse;
      return { plan: r.plan, validation: r.validation, origin: "solver", executionType: job.metadata.execution_type };
    }
    case "qaoa_job": {
      const job = jobInner(run);
      return fromQaoa((job?.result as QaoaResponse | undefined) ?? undefined);
    }
    case "recovery": {
      const r = run.result as RecoveryResponse | undefined;
      return r ? { plan: r.recovered_plan, validation: r.validation, origin: "recovery", executionType: r.execution_type } : {};
    }
    default:
      return {};
  }
}

export function isValidated(run: RunRecord): boolean {
  return extractPlan(run).validation?.valid === true && extractPlan(run).plan?.status === "feasible";
}

export function runStatus(run: RunRecord): string {
  if (run.source === "job") return (run.result as JobRecord | undefined)?.status ?? run.status;
  return run.status;
}

export function isTerminal(status: string): boolean {
  return (TERMINAL_JOB_STATUSES as string[]).includes(status) || !["queued", "running", "cancel_requested"].includes(status);
}

export function isActiveJob(run: RunRecord): boolean {
  return run.source === "job" && ["queued", "running", "cancel_requested"].includes(runStatus(run));
}

/** Backend-measured runtime for jobs (started→completed); client round trip for sync calls. */
export function runtimeOf(run: RunRecord): { ms: number; basis: "backend timestamps" | "client round-trip" } | undefined {
  const job = jobInner(run);
  if (job) {
    if (job.started_at && job.completed_at) {
      const ms = Date.parse(job.completed_at) - Date.parse(job.started_at);
      if (Number.isFinite(ms) && ms >= 0) return { ms, basis: "backend timestamps" };
    }
    return undefined;
  }
  return run.roundTripMs !== undefined ? { ms: run.roundTripMs, basis: "client round-trip" } : undefined;
}

export function objectiveOf(run: RunRecord): number | undefined {
  const { plan } = extractPlan(run);
  return plan?.status === "feasible" ? plan.objective : undefined;
}

export function qaoaRaw(run: RunRecord): QaoaRawResult | undefined {
  if (run.method === "qaoa") return (run.result as QaoaResponse | undefined)?.optimization?.raw_result ?? undefined;
  if (run.method === "qaoa_job") return ((jobInner(run)?.result as QaoaResponse | undefined)?.optimization?.raw_result) ?? undefined;
  return undefined;
}

export function riskOf(run: RunRecord): RiskResponse | undefined {
  return run.method === "risk" ? (run.result as RiskResponse | undefined) : undefined;
}

export function failureReason(run: RunRecord): string | undefined {
  const job = jobInner(run);
  if (job?.error) return `${job.error.code}: ${job.error.message}`;
  if (run.error) return `${run.error.code ?? "error"}: ${run.error.message}`;
  const ex = extractPlan(run);
  if (ex.fallback?.used) return `fallback: ${ex.fallback.reason ?? "reason not returned"}`;
  return undefined;
}

export function executionLabel(type?: string): string {
  if (!type) return "Not returned";
  if (type.includes("statevector") || type === "local_simulator" || type.includes("qaoa")) return "Local simulation";
  if (type.includes("classical") || type === "classical") return "Classical (local)";
  if (type.includes("replay")) return "Local scenario replay";
  return type;
}

