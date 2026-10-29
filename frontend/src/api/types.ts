/**
 * Typed request/response models hand-written from the ArQ backend source
 * (backend/api.py, domain.py, validator.py, jobs.py, qaoa.py, qaoa_dispatch.py,
 * learning_inference.py, financial.py, risk.py, recovery.py, recovery_store.py).
 * Field names match the backend exactly; the backend rejects unknown fields.
 */

// ---------- Scenario (domain.py / api.py _canonical_scenario) ----------
export interface Vehicle {
  id: string;
  capacity: number;
}

export interface Order {
  id: string;
  demand: number;
  window: [number, number];
  service: number;
  penalty: number;
  hard_window?: boolean;
}

export type TravelMatrix = Record<string, Record<string, number>>;

export interface Scenario {
  depot: string;
  vehicles: Vehicle[];
  orders: Order[];
  travel: TravelMatrix;
}

// ---------- Plan (domain.py _route_metrics / solve_exact) ----------
export interface Stop {
  order_id: string;
  arrival: number;
  service_start: number;
  departure: number;
  lateness: number;
  penalty_exposure: number;
}

export interface Route {
  vehicle_id: string;
  order_ids: string[];
  stops: Stop[];
  return_time: number;
  travel_cost: number;
  lateness_exposure: number;
  objective: number;
}

export interface Plan {
  status: "feasible" | "infeasible";
  routes: Route[];
  travel_cost: number;
  lateness_exposure: number;
  objective: number;
  reason?: string;
}

// ---------- Validator (validator.py) ----------
export interface ValidationReport {
  valid: boolean;
  feasible: boolean;
  errors: string[];
  recomputed: {
    travel_cost: number;
    lateness_exposure: number;
    objective: number;
    routes: Array<Omit<Route, "stops"> & { stops: Stop[] }>;
  } | null;
}

// ---------- Error envelope ----------
export interface FieldError {
  field: string;
  message: string;
}

export interface ErrorEnvelope {
  error: {
    code: string;
    message: string;
    details?: FieldError[];
    audit?: AuditRef;
  };
  plan?: Plan;
  execution_type?: string;
}

// ---------- Endpoint responses ----------
export interface HealthResponse {
  status: string;
  api_version: string;
}

export interface ScenarioValidateResponse {
  valid: boolean;
  errors: unknown[];
  execution_type: string;
}

export interface RoutesResponse {
  routes: Route[];
  count: number;
  execution_type: string;
}

export interface PlanResponse {
  plan: Plan;
  validation: ValidationReport;
  execution_type: string;
}

export interface PlanValidateResponse {
  valid: boolean;
  feasible: boolean;
  errors: FieldError[];
  recomputed: ValidationReport["recomputed"];
}

// financial.py
export interface FinancialTerms {
  currency: string;
  operating_cost_per_travel_minute: number;
  penalty_rate_unit: "currency_per_late_minute";
}

export interface FinancialObjective {
  currency: string;
  operating_cost_per_travel_minute: number;
  penalty_rate_unit: string;
  travel_minutes: number;
  transport_cost: number;
  late_penalty_cost: number;
  total_cost: number;
}

export interface FinancialValidation {
  valid: boolean;
  errors: string[];
  proxy_validation: ValidationReport;
  recomputed_financial: FinancialObjective | null;
}

export interface FinancialPlanResponse {
  status: "feasible";
  execution_type: string;
  plan: Plan;
  proxy_validation: ValidationReport;
  financial_objective: FinancialObjective;
  financial_validation: FinancialValidation;
  assumptions?: string[];
  [key: string]: unknown;
}

// qubo.py validate_selection
export interface SelectionValidation {
  feasible: boolean;
  selected_route_ids: string[];
  coverage_counts: Record<string, number>;
  duplicate_vehicles: string[];
  business_cost: number;
  coverage_ok: boolean;
  vehicle_ok: boolean;
}

// qaoa.py raw sample rows (+ qaoa_dispatch _attach_independent_checks)
export interface QaoaSampleRow {
  bits: string;
  energy: number;
  validation: SelectionValidation;
  repair: {
    bits: string;
    edits: Array<{ index: number; from: number; to: number; reason: string }>;
    validation: SelectionValidation;
    independent_plan_validation?: ValidationReport;
  } | null;
  decoded?: { bitstring: string; selected_route_ids: string[] };
  independent_plan_validation?: ValidationReport;
}

export interface QaoaRawResult {
  mode: "warm_start" | "standard";
  backend: string;
  probabilities: number[];
  angles: { gamma: number[]; beta: number[] };
  expected_qubo_energy: number;
  raw_counts: Record<string, number>;
  raw_samples: QaoaSampleRow[];
  feasible_samples: number;
  shots: number;
  repair_edits: unknown[];
  relaxation: { method: string; vector: number[] | null; clamping: string };
}

export interface QaoaResponse {
  status: "success" | "infeasible" | "validation_failed" | "invalid_input";
  execution_type: string | null;
  decision: {
    origin: "qaoa_raw_sample" | "classical_exact_fallback" | "empty_instance";
    plan: Plan;
    validation: ValidationReport | null;
  } | null;
  fallback: { used: boolean; reason: string | null };
  optimization?: {
    candidate_count?: number;
    variable_order?: string[];
    execution_type?: string;
    qaoa_failure?: Record<string, unknown> | null;
    selected_raw_bits?: string | null;
    raw_result?: QaoaRawResult | null;
    qubo?: Record<string, unknown>;
    skipped?: string;
  };
  errors?: string[];
}

export interface QaoaSettings {
  seed: number;
  shots: number;
  warm_start: boolean;
  angle_grid: number;
}

// learning_inference.py
export type VqcBackend = "reference" | "qpiai_local";

export interface VqcRequest {
  forecast_scenario: Scenario;
  nominal_scenario: Scenario;
  theta: [number, number];
  model_version: string;
  execution_backend?: VqcBackend;
  financial_terms?: FinancialTerms;
}

export interface VqcRoutePrediction {
  route_id: string;
  features: number[];
  risk_probability: number;
  estimated_exposure: number;
  forecast_travel_cost: number;
  estimated_transport_cost?: number;
  estimated_route_cost?: number;
}

export interface VqcResponse {
  status: "completed" | "infeasible" | "feasible";
  execution_type: string;
  model?: {
    version: string;
    parameter_vector_sha256: string;
    architecture: string;
    qubits: number;
    execution: string;
    execution_backend: string;
    execution_backend_versions: Record<string, string>;
    training_provenance: string;
    calibration: string;
    [key: string]: unknown;
  };
  plan: Plan;
  estimated_plan_cost?: number;
  estimated_exposure?: number;
  route_predictions?: VqcRoutePrediction[];
  validation: ValidationReport | null;
  fallback?: { used: boolean; reason: string };
  vqc_error_code?: string;
  financial_estimate?: Record<string, unknown>;
  forecast_financial_objective?: FinancialObjective;
  financial_validation?: FinancialValidation;
  model_suitability_warning?: string;
}

// jobs.py get()
export type JobStatus =
  | "queued"
  | "running"
  | "cancel_requested"
  | "succeeded"
  | "failed"
  | "infeasible"
  | "cancelled"
  | "timed_out";

export const TERMINAL_JOB_STATUSES: JobStatus[] = ["succeeded", "failed", "infeasible", "cancelled", "timed_out"];

export interface JobRecord {
  id: string;
  status: JobStatus;
  result: PlanResponse | QaoaResponse | Plan | null;
  error: { code: string; message: string } | null;
  submitted_at: string;
  started_at: string | null;
  completed_at: string | null;
  metadata: {
    execution_type: string;
    solver: string;
    settings: Record<string, unknown> | null;
    request_sha256: string;
    runtime: Record<string, string> | null;
  };
}

// risk.py
export interface RealizedScenario {
  scenario_id: string;
  travel: TravelMatrix;
}

export interface RiskRequest {
  forecast_scenario: Scenario;
  nominal_scenario: Scenario;
  plan: Plan;
  realized_scenarios: RealizedScenario[];
  cvar_alpha?: number;
}

export interface RiskScenarioResult {
  scenario_id: string;
  travel_matrix_sha256: string;
  late_event: boolean;
  feasible: boolean;
  hard_window_violation: boolean;
  travel_cost: number;
  penalty_exposure: number;
  proxy_objective: number;
  routes: Route[];
  validation: ValidationReport;
}

export interface RiskResponse {
  status: "completed";
  execution_type: string;
  estimator: {
    method: string;
    scenario_count: number;
    weight_per_scenario: number;
    confidence_intervals: string;
  };
  summary: {
    empirical_probability_any_late: number;
    empirical_probability_hard_window_violation: number;
    empirical_late_stop_frequency: number;
    per_order_late_frequency: Record<string, number>;
    expected_travel_cost: number;
    expected_penalty_exposure: number;
    expected_proxy_objective: number;
  };
  scenarios: RiskScenarioResult[];
  provenance: Record<string, string>;
  cvar?: {
    confidence_level: number;
    tail_fraction: number;
    sample_count: number;
    system_wide_penalty_exposure: number;
    system_wide_proxy_objective: number;
    definition: string;
  };
}

// recovery.py / api.py
export interface RecoveryRequest {
  original_scenario: Scenario;
  original_plan: Plan;
  updated_scenario: Scenario;
  locked_prefixes: Record<string, string[]>;
  change_penalty?: number;
}

export interface AuditRef {
  audit_id: string;
  scope_sha256: string;
  started_at: string;
  completed_at: string;
  hashes: Record<string, string>;
}

export interface RecoveryResponse {
  status: "recovered";
  execution_type: string;
  original_plan: Plan;
  recovered_plan: Plan;
  validation: ValidationReport;
  static_original_baseline:
    | { available: true; plan: Plan; validation: ValidationReport; comparison_basis: string }
    | { available: false; reason: string };
  comparison: {
    original_objective: number;
    updated_objective: number;
    objective_delta: number;
    original_exposure: number;
    updated_exposure: number;
    reassigned_order_count: number;
    change_penalty: number;
    selection_cost_including_change_penalty: number;
    route_sequences_changed: boolean;
    search_nodes: number;
  };
  state_basis: string;
  audit: AuditRef;
}

export interface AuditRecord {
  audit_id: string;
  status: "recovered" | "failed";
  started_at: string;
  completed_at: string;
  scope_sha256: string;
  hashes: Record<string, string>;
  original_scenario: Scenario;
  updated_scenario: Scenario;
  original_plan: Plan;
  locked_prefixes: Record<string, string[]>;
  change_penalty: number;
  result: Omit<RecoveryResponse, "audit"> | null;
  failure: { code: string; message: string } | null;
}

export interface LatestAudits {
  latest_attempt: AuditRecord | null;
  latest_success: AuditRecord | null;
}

