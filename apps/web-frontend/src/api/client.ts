import {
  PlanRun,
  SolverJob,
  OperationalEvent,
  IntelligenceEvidence,
  SystemHealth,
} from '../types/api';
import {
  BENGALURU_SCENARIO,
  SOLOMON_SCENARIO,
  INTELLIGENCE_EVIDENCE,
  DEFAULT_HEALTH,
} from '../data/mockScenarios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const ROUTE_COLORS = ['#2F5BEA', '#1E5A44', '#D97706', '#8B5CF6', '#DC2626'];

type BackendJob = {
  id: string;
  status: string;
  scenario: DispatchScenario;
  result: Record<string, any> | null;
  error: { code: string; message: string } | null;
  submitted_at: string;
  started_at: string | null;
  completed_at: string | null;
  metadata: {
    execution_type: string;
    solver: string;
    request_sha256: string;
  };
};

export type DispatchScenario = {
  depot: string;
  vehicles: { id: string; capacity: number }[];
  orders: { id: string; demand: number; window: [number, number]; service: number; penalty: number; hard_window?: boolean }[];
  travel: Record<string, Record<string, number>>;
};

export type ScenarioValidation = { valid: boolean; errors: string[]; execution_type: string };
export type VqcPlanResult = {
  status: string;
  execution_type: string;
  model?: { execution_backend?: string; execution_backend_versions?: Record<string, string> };
  plan?: Record<string, any>;
  validation?: { valid: boolean; feasible: boolean; errors: string[] };
  route_predictions?: Array<Record<string, any>>;
  estimated_plan_cost?: number;
  estimated_exposure?: number;
  error?: { code: string; message: string };
};

class ApiError extends Error {
  constructor(readonly status: number, readonly code: string, message: string, readonly retryAfter?: number) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
      signal: init?.signal ?? AbortSignal.timeout(10_000),
    });
  } catch {
    throw new ApiError(0, 'backend_unreachable', `Can't reach backend at ${API_BASE_URL}`);
  }

  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const error = body?.error;
    const retryAfter = Number(response.headers.get('Retry-After')) || undefined;
    throw new ApiError(
      response.status,
      typeof error?.code === 'string' ? error.code : 'backend_error',
      typeof error?.message === 'string' ? error.message : `Backend request failed (${response.status})`,
      retryAfter,
    );
  }
  return body as T;
}

function isTerminal(status: string): boolean {
  return ['succeeded', 'failed', 'cancelled', 'infeasible', 'timed_out'].includes(status);
}

function elapsedSeconds(job: BackendJob): number {
  const start = Date.parse(job.started_at ?? job.submitted_at);
  const end = Date.parse(job.completed_at ?? new Date().toISOString());
  return Number.isFinite(start) && Number.isFinite(end) ? Math.max(0, Math.round((end - start) / 100) / 10) : 0;
}

function toSolverJob(job: BackendJob): SolverJob {
  const solver = job.metadata?.solver ?? 'local solver';
  const isQaoa = solver.toLowerCase().includes('qaoa');
  const status = (['queued', 'running', 'succeeded', 'failed', 'cancelled', 'infeasible', 'timed_out'].includes(job.status)
    ? job.status
    : job.status === 'cancel_requested' ? 'running' : 'failed') as SolverJob['status'];
  return {
    id: job.id,
    scenario_id: job.metadata?.request_sha256?.slice(0, 12) ?? 'submitted-scenario',
    status,
    progress_pct: isTerminal(status) ? 100 : null,
    elapsed_seconds: elapsedSeconds(job),
    engine: solver,
    provenance_badge: isQaoa ? 'QAOA (warm start)' : 'Exact (enumeration)',
    result_plan_id: status === 'succeeded' ? job.id : null,
    error_message: job.error?.message ?? null,
    submitted_at: job.submitted_at,
  };
}

function timeLabel(value: number): string {
  return `T+${Number(value.toFixed(2))} units`;
}

function toPlanRun(job: BackendJob): PlanRun | null {
  const scenario = job.scenario;
  const result = job.result;
  const rawPlan = result?.plan ?? result?.decision?.plan;
  const validation = result?.validation ?? result?.decision?.validation;
  if (job.status !== 'succeeded' || !scenario || !rawPlan || !Array.isArray(rawPlan.routes)) return null;

  const orderById = new Map(scenario.orders.map((order) => [order.id, order]));
  const vehicles = scenario.vehicles.map((vehicle) => {
    const route = rawPlan.routes.find((item: any) => item.vehicle_id === vehicle.id);
    const used = (route?.order_ids ?? []).reduce((sum: number, orderId: string) => sum + (orderById.get(orderId)?.demand ?? 0), 0);
    return {
      id: vehicle.id,
      callsign: vehicle.id,
      driver_name: 'Unassigned',
      phone: '',
      status: 'idle' as const,
      vehicle_type: 'Capacity units only',
      battery_or_fuel_pct: 0,
      capacity_kg: vehicle.capacity,
      used_capacity_kg: used,
      route_id: `route-${vehicle.id}`,
      last_ping_iso: job.submitted_at,
      eta_deviation_minutes: 0,
      assigned_stops_count: route?.order_ids?.length ?? 0,
      completed_stops_count: 0,
      telemetry_available: false,
    };
  });

  const routes = rawPlan.routes.map((route: any, routeIndex: number) => {
    const stops = (route.stops ?? []).map((metric: any, index: number) => {
      const order = orderById.get(metric.order_id);
      const x = index + 1;
      const y = routeIndex + 1;
      return {
        id: metric.order_id,
        route_id: `route-${route.vehicle_id}`,
        vehicle_id: route.vehicle_id,
        sequence_idx: index + 1,
        customer_name: metric.order_id,
        address: 'Location not supplied',
        window_start: order ? timeLabel(order.window[0]) : 'Unavailable',
        window_end: order ? timeLabel(order.window[1]) : 'Unavailable',
        eta_min: timeLabel(metric.service_start ?? metric.arrival ?? 0),
        eta_max: timeLabel(metric.service_start ?? metric.arrival ?? 0),
        predicted_late_prob_min: null,
        predicted_late_prob_max: null,
        penalty_exposure_inr: Number(metric.penalty_exposure ?? 0),
        penalty_exposure_usd: 0,
        penalty_exposure_label: `${Number(metric.penalty_exposure ?? 0).toFixed(2)} objective units`,
        demand_kg: Number(order?.demand ?? 0),
        demand_label: `${Number(order?.demand ?? 0)} capacity units`,
        is_locked: false,
        is_completed: false,
        status: 'pending' as const,
        x,
        y,
        notes: 'Sequence sketch only; this scenario has no coordinates or ETA risk model.',
      };
    });
    return {
      id: `route-${route.vehicle_id}`,
      vehicle_id: route.vehicle_id,
      color_token: ROUTE_COLORS[routeIndex % ROUTE_COLORS.length],
      total_distance_km: 0,
      total_duration_minutes: Number(route.return_time ?? 0),
      distance_label: 'Distance unavailable',
      duration_label: `${timeLabel(Number(route.return_time ?? 0))} return`,
      stops,
      geometry_coordinates: [[0, 0], ...stops.map((stop: any) => [stop.x, stop.y] as [number, number])],
    };
  });

  const totalPenalty = Number(rawPlan.lateness_exposure ?? 0);
  const totalObjective = Number(rawPlan.objective ?? 0);
  return {
    id: job.id,
    scenario_id: job.metadata?.request_sha256?.slice(0, 12) ?? job.id,
    scenario_name: `Backend run · ${scenario.orders.length} orders`,
    coordinate_system: 'cartesian_xy',
    status: 'draft',
    engine: job.metadata?.solver ?? 'exact_enumeration',
    provenance_badge: job.metadata?.solver?.toLowerCase().includes('qaoa') ? 'QAOA (warm start)' : 'Exact (enumeration)',
    created_at: job.submitted_at,
    total_vehicles: vehicles.length,
    total_stops: scenario.orders.length,
    completed_stops: 0,
    total_sla_exposure_inr: 0,
    assumed_rate_label: 'Synthetic objective units · no currency calibration',
    total_exposure_label: `${totalPenalty.toFixed(2)} objective units`,
    objective_cost: `${totalObjective.toFixed(2)} objective units`,
    validator_result: {
      passed: validation?.valid === true,
      constraints_checked: ['vehicle capacity', 'order coverage', 'route feasibility'],
      violations: Array.isArray(validation?.errors) ? validation.errors : [],
      evaluated_at: job.completed_at ?? job.submitted_at,
    },
    vehicles,
    routes,
    data_source: 'backend',
    cost_unit_label: 'objective units',
    capacity_unit_label: 'capacity units',
    coordinate_caption: 'Sequence sketch · source scenario has no GPS coordinates',
    backend_payload: { scenario: scenario as unknown as Record<string, unknown>, plan: rawPlan as Record<string, unknown> },
  };
}

function previewRuns(): PlanRun[] {
  return [BENGALURU_SCENARIO, SOLOMON_SCENARIO].map((plan) => ({ ...plan, data_source: 'preview' }));
}

function asOperationalEvent(job: SolverJob): OperationalEvent {
  const failed = ['failed', 'infeasible', 'timed_out'].includes(job.status);
  return {
    id: `job-${job.id}`,
    timestamp: job.submitted_at,
    severity: failed ? 'warning' : job.status === 'succeeded' ? 'success' : 'info',
    category: 'job_lifecycle',
    title: `Solver job ${job.status}`,
    description: `${job.engine} · ${job.id}${job.error_message ? ` · ${job.error_message}` : ''}`,
    actor: 'Backend job service',
    related_id: job.id,
  };
}

export const apiClient = {
  getBaseUrl(): string {
    return API_BASE_URL;
  },

  async getExampleScenario(): Promise<DispatchScenario> {
    const response = await fetch('/dispatch_tiny.json', { signal: AbortSignal.timeout(5_000) });
    if (!response.ok) throw new ApiError(response.status, 'scenario_unavailable', 'Bundled example scenario is unavailable');
    return await response.json() as DispatchScenario;
  },

  validateScenario(scenario: DispatchScenario): Promise<ScenarioValidation> {
    return request('/api/v1/scenarios/validate', { method: 'POST', body: JSON.stringify(scenario) });
  },

  async getHealth(): Promise<SystemHealth> {
    try {
      const data = await request<{ status: string; api_version?: string; capacity?: { max_slots: number; active_slots: number; available_slots: number } | null }>('/api/v1/health');
      const jobs = await this.getJobs();
      const active = jobs.filter((job) => job.status === 'queued' || job.status === 'running');
      const maxSlots = data.capacity?.max_slots ?? 2;
      return {
        status: data.status === 'ok' ? 'healthy' : 'degraded',
        backend_url: API_BASE_URL,
        version: data.api_version ?? 'v1',
        timestamp: new Date().toISOString(),
        capacity: { max_slots: maxSlots, active_slots: data.capacity?.active_slots ?? active.length, available_slots: data.capacity?.available_slots ?? Math.max(0, maxSlots - active.length), active_job_ids: active.map((job) => job.id) },
        engines_available: ['exact_enumeration', 'warm_start_qaoa_local'],
      };
    } catch {
      return { ...DEFAULT_HEALTH, status: 'offline', backend_url: API_BASE_URL, capacity: null, engines_available: [] };
    }
  },

  async getRuns(): Promise<PlanRun[]> {
    try {
      const jobs = await request<BackendJob[]>('/api/v1/jobs?limit=50');
      const liveRuns = jobs.map(toPlanRun).filter((plan): plan is PlanRun => plan !== null);
      return liveRuns.length ? liveRuns : previewRuns();
    } catch {
      return previewRuns();
    }
  },

  async getRun(id: string): Promise<PlanRun | null> {
    try {
      const job = await request<BackendJob>(`/api/v1/jobs/${encodeURIComponent(id)}`);
      return toPlanRun(job);
    } catch {
      return previewRuns().find((plan) => plan.id === id) ?? null;
    }
  },

  async submitSolve(params: { scenario_id: string; scenario: DispatchScenario; engine: string; shots?: number; angle_grid?: number }): Promise<{ job_id: string } | { error: string; retry_after?: number }> {
    try {
      const path = params.engine === 'qaoa' ? '/api/v1/jobs/qaoa' : '/api/v1/jobs';
      const body = params.engine === 'qaoa'
        ? { scenario: params.scenario, seed: 7, shots: params.shots ?? 128, warm_start: true, angle_grid: params.angle_grid ?? 4 }
        : params.scenario;
      const result = await request<{ id: string }>(path, { method: 'POST', body: JSON.stringify(body), signal: AbortSignal.timeout(30_000) });
      return { job_id: result.id };
    } catch (error) {
      if (error instanceof ApiError) return { error: error.code, retry_after: error.retryAfter };
      return { error: 'scenario_unavailable' };
    }
  },

  async runVqc(scenario: DispatchScenario): Promise<VqcPlanResult | { error: string }> {
    try {
      return await request<VqcPlanResult>('/api/v1/scenarios/vqc-plan', {
        method: 'POST',
        body: JSON.stringify({
          forecast_scenario: scenario,
          nominal_scenario: scenario,
          theta: [0.31, -0.47],
          model_version: 'dashboard-caller-supplied-v1',
          execution_backend: 'qpiai_local',
        }),
        signal: AbortSignal.timeout(30_000),
      });
    } catch (error) {
      return { error: error instanceof ApiError ? error.code : 'vqc_request_failed' };
    }
  },

  evaluateBaselineRisk(plan: PlanRun): Promise<Record<string, any>> {
    const context = plan.backend_payload;
    if (!context) throw new ApiError(422, 'backend_context_unavailable', 'Risk replay requires a backend-produced plan.');
    const scenario = context.scenario as unknown as DispatchScenario;
    return request('/api/v1/risk/evaluate', {
      method: 'POST',
      body: JSON.stringify({
        forecast_scenario: scenario,
        nominal_scenario: scenario,
        plan: context.plan,
        realized_scenarios: [{ scenario_id: 'source-baseline', travel: scenario.travel }],
        cvar_alpha: 0.95,
      }),
      signal: AbortSignal.timeout(30_000),
    });
  },

  registerSyntheticManifest(plan: PlanRun, dispatcherToken: string): Promise<Record<string, any>> {
    const context = plan.backend_payload;
    if (!context || !plan.validator_result.passed) {
      throw new ApiError(422, 'plan_not_publishable', 'Only a backend plan that passed independent validation can be activated.');
    }
    const scenario = context.scenario as unknown as DispatchScenario;
    return request('/api/v1/eta/plans', {
      method: 'POST',
      headers: { Authorization: `Bearer ${dispatcherToken}` },
      body: JSON.stringify({
        plan_id: `web-${Date.now().toString(36)}`,
        version: 1,
        model_version: '1.0.0',
        as_of: new Date().toISOString(),
        scenario,
        weighted_scenarios: [{ id: 'source-baseline', weight: 1, travel: scenario.travel, source_label: 'synthetic_overlay', prediction_target: 'travel_time' }],
        plan: context.plan,
        mode: 'synthetic_scenario',
        currency: 'XTS',
        operating_rate: 1,
      }),
      signal: AbortSignal.timeout(30_000),
    });
  },

  async getJobs(): Promise<SolverJob[]> {
    try {
      const jobs = await request<BackendJob[]>('/api/v1/jobs?limit=50');
      return jobs.map(toSolverJob);
    } catch {
      return [];
    }
  },

  async cancelJob(id: string): Promise<boolean> {
    try {
      const result = await request<{ status: string }>(`/api/v1/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' });
      return result.status === 'cancelled';
    } catch {
      return false;
    }
  },

  async executeRecoveryReplan(params: {
    plan: PlanRun;
    vehicle_id: string;
    delayed_stop_id: string;
    delay_units: number;
    locked_stop_ids: string[];
  }): Promise<{ success: boolean; audit_log: string; objective_delta_units: number; reordered_count: number; recovered_plan: PlanRun | null }> {
    const context = params.plan.backend_payload;
    if (!context || !Number.isFinite(params.delay_units) || params.delay_units <= 0) {
      return { success: false, audit_log: 'Recovery requires a validated backend run and a positive scenario delay.', objective_delta_units: 0, reordered_count: 0, recovered_plan: null };
    }
    const scenario = JSON.parse(JSON.stringify(context.scenario)) as DispatchScenario;
    const originalPlan = JSON.parse(JSON.stringify(context.plan)) as Record<string, any>;
    const route = originalPlan.routes?.find((item: any) => item.vehicle_id === params.vehicle_id);
    const stopIndex = route?.order_ids?.indexOf(params.delayed_stop_id) ?? -1;
    if (stopIndex < 0) {
      return { success: false, audit_log: 'The selected stop is not in this backend plan.', objective_delta_units: 0, reordered_count: 0, recovered_plan: null };
    }
    const from = stopIndex === 0 ? scenario.depot : route.order_ids[stopIndex - 1];
    if (!scenario.travel[from]?.[params.delayed_stop_id] && scenario.travel[from]?.[params.delayed_stop_id] !== 0) {
      return { success: false, audit_log: 'The source scenario has no travel time for this leg.', objective_delta_units: 0, reordered_count: 0, recovered_plan: null };
    }
    const updatedScenario = JSON.parse(JSON.stringify(scenario)) as DispatchScenario;
    updatedScenario.travel[from][params.delayed_stop_id] += params.delay_units;
    try {
      const response = await request<Record<string, any>>('/api/v1/recovery/plan', {
        method: 'POST',
        body: JSON.stringify({
          original_scenario: scenario,
          original_plan: originalPlan,
          updated_scenario: updatedScenario,
          locked_prefixes: { [params.vehicle_id]: params.locked_stop_ids },
          change_penalty: 0,
        }),
        signal: AbortSignal.timeout(30_000),
      });
      const nextRawPlan = response.recovered_plan;
      const previousObjective = Number(originalPlan.objective ?? 0);
      const nextObjective = Number(nextRawPlan?.objective ?? previousObjective);
      const originalSequences = new Map((originalPlan.routes ?? []).map((item: any) => [item.vehicle_id, (item.order_ids ?? []).join('\u0000')]));
      const reorderedCount = (nextRawPlan?.routes ?? []).filter((item: any) =>
        originalSequences.get(item.vehicle_id) !== (item.order_ids ?? []).join('\u0000'),
      ).length;
      const recoveryJob: BackendJob = {
        id: `recovery-${response.audit?.audit_id ?? Date.now()}`,
        status: 'succeeded',
        scenario: updatedScenario,
        result: { plan: nextRawPlan, validation: response.validation },
        error: null,
        submitted_at: response.audit?.started_at ?? new Date().toISOString(),
        started_at: response.audit?.started_at ?? null,
        completed_at: response.audit?.completed_at ?? new Date().toISOString(),
        metadata: { execution_type: 'local_classical_recovery', solver: 'classical_recovery', request_sha256: response.audit?.scope_sha256 ?? '' },
      };
      return {
        success: true,
        audit_log: `Recovery recorded as ${response.audit?.audit_id ?? 'audit unavailable'}`,
        objective_delta_units: Number((nextObjective - previousObjective).toFixed(4)),
        reordered_count: reorderedCount,
        recovered_plan: toPlanRun(recoveryJob),
      };
    } catch (error) {
      return { success: false, audit_log: error instanceof ApiError ? error.message : 'Recovery request failed.', objective_delta_units: 0, reordered_count: 0, recovered_plan: null };
    }
  },

  async getIntelligenceEvidence(): Promise<IntelligenceEvidence> {
    return INTELLIGENCE_EVIDENCE;
  },

  async getOperationalEvents(): Promise<OperationalEvent[]> {
    return (await this.getJobs()).map(asOperationalEvent);
  },

  async getDriverReportEvents(dispatcherToken?: string): Promise<OperationalEvent[]> {
    const headers = dispatcherToken ? { Authorization: `Bearer ${dispatcherToken}` } : undefined;
    const reports = await request<Array<Record<string, any>>>('/api/v1/driver/reports?limit=200', { headers });
    return reports.map((report) => ({
      id: `driver-${report.event_id}`,
      timestamp: report.created_at,
      severity: report.kind === 'delay_report' ? 'warning' : report.kind === 'stop_completed' ? 'success' : 'info',
      category: report.kind === 'delay_report' ? 'delay_alert' : 'status_change',
      title: report.kind === 'delay_report' ? 'Driver reported a delay' : report.kind === 'stop_arrived' ? 'Driver arrived at stop' : 'Driver completed stop',
      description: `${report.vehicle_id} · ${report.stop_id ?? 'route'}${report.severity ? ` · ${report.severity}` : ''}${report.note ? ` · ${report.note}` : ''}`,
      actor: `Driver ${report.driver_id}`,
      related_id: report.event_id,
    } as OperationalEvent));
  },
};
