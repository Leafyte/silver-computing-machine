/**
 * Thin fetch wrapper. Every network failure is normalised into ApiError so
 * pages can render offline / timeout / http states uniformly. Request bodies
 * and error payloads are never logged.
 */
import { apiBaseUrl, apiTimeoutMs } from "../config/env";
import type { ErrorEnvelope, FieldError } from "./types";

export type ApiErrorKind = "offline" | "timeout" | "http" | "parse";

export class ApiError extends Error {
  kind: ApiErrorKind;
  status?: number;
  code?: string;
  details: FieldError[];
  body?: unknown;

  constructor(kind: ApiErrorKind, message: string, opts: { status?: number; code?: string; details?: FieldError[]; body?: unknown } = {}) {
    super(message);
    this.kind = kind;
    this.status = opts.status;
    this.code = opts.code;
    this.details = opts.details ?? [];
    this.body = opts.body;
  }
}

export interface ApiResult<T> {
  data: T;
  status: number;
  /** Client-measured round trip in ms (not backend compute time). */
  roundTripMs: number;
}

export async function request<T>(method: "GET" | "POST", path: string, body?: unknown, signal?: AbortSignal): Promise<ApiResult<T>> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort("timeout"), apiTimeoutMs());
  const onAbort = () => controller.abort("cancelled");
  signal?.addEventListener("abort", onAbort);
  const started = performance.now();
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl()}${path}`, {
      method,
      headers: body === undefined ? { Accept: "application/json" } : { "Content-Type": "application/json", Accept: "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
      credentials: "omit",
    });
  } catch (err) {
    if (controller.signal.aborted && controller.signal.reason === "timeout") {
      throw new ApiError("timeout", `Request timed out after ${Math.round(apiTimeoutMs() / 1000)}s`);
    }
    if (controller.signal.aborted) throw new ApiError("offline", "Request cancelled");
    throw new ApiError("offline", "Backend unreachable");
  } finally {
    window.clearTimeout(timeout);
    signal?.removeEventListener("abort", onAbort);
  }
  const roundTripMs = performance.now() - started;
  let parsed: unknown = null;
  const text = await response.text();
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      throw new ApiError("parse", "Backend returned a non-JSON response", { status: response.status });
    }
  }
  if (!response.ok) {
    const env = parsed as Partial<ErrorEnvelope> | null;
    const code = env?.error?.code;
    const message = env?.error?.message ?? `HTTP ${response.status}`;
    throw new ApiError("http", message, { status: response.status, code, details: env?.error?.details, body: parsed });
  }
  return { data: parsed as T, status: response.status, roundTripMs };
}

/** Human-readable label for an error, without echoing request content. */
export function describeError(err: unknown): { title: string; detail: string; tone: "red" | "amber" | "grey" } {
  if (err instanceof ApiError) {
    if (err.kind === "offline") return { title: "Backend unreachable", detail: "Check that the API is running and the base URL in Settings is correct.", tone: "grey" };
    if (err.kind === "timeout") return { title: "Timed out", detail: err.message, tone: "amber" };
    if (err.kind === "parse") return { title: "Unexpected response", detail: err.message, tone: "red" };
    const map: Record<string, string> = {
      infeasible: "Infeasible",
      invalid_scenario: "Invalid scenario",
      invalid_request: "Invalid request",
      solver_budget_exceeded: "Solver budget exceeded",
      capacity_reached: "Backend busy",
      unsupported_capability: "Unsupported",
      validation_failure: "Validation failure",
      authentication_required: "Authentication required",
      authentication_failed: "Authentication failed",
      authorization_denied: "Not authorised",
      auth_misconfigured: "Backend auth misconfigured",
      recovery_infeasible: "Recovery infeasible",
      locked_prefix_infeasible: "Locked stops conflict",
      locked_work_vehicle_unavailable: "Locked vehicle unavailable",
    };
    const title = (err.code && map[err.code]) || (err.code ?? `HTTP ${err.status}`);
    const tone = err.status === 409 || err.status === 429 || err.status === 503 ? "amber" : "red";
    return { title, detail: err.message, tone };
  }
  return { title: "Unexpected error", detail: "The request could not be completed.", tone: "red" };
}

