// arQ Core Domain Types & API Contracts

export type CoordinateSystem = 'lat_lon' | 'cartesian_xy';

export type VehicleStatus = 'active' | 'idle' | 'at_risk' | 'delayed' | 'maintenance';

export type StopStatus = 'pending' | 'en_route' | 'arrived' | 'completed' | 'at_risk' | 'failed';

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted' | 'infeasible' | 'timed_out';

export type ProvenanceBadge =
  | 'Validated'
  | 'Validator failed'
  | 'Exact (enumeration)'
  | 'Exact (financial cover)'
  | 'QpiAI Opt · local solver'
  | 'QAOA (warm start)'
  | 'QCloud'
  | 'VQC · local statevector'
  | 'GBDT + Platt (classical)'
  | 'Fallback'
  | 'Classical re-plan'
  | 'Simulated · held-out run'
  | 'Preview';

export interface Vehicle {
  id: string;
  callsign: string;
  driver_name: string;
  phone: string;
  status: VehicleStatus;
  vehicle_type: string; // e.g. "Tata Ace EV (1.2T)" | "Mahindra Bolero Maxi (1.7T)"
  battery_or_fuel_pct: number;
  capacity_kg: number;
  used_capacity_kg: number;
  route_id: string;
  current_lat?: number;
  current_lng?: number;
  current_x?: number; // for synthetic Solomon grids
  current_y?: number;
  heading_deg?: number;
  speed_kmh?: number;
  last_ping_iso: string;
  eta_deviation_minutes: number;
  assigned_stops_count: number;
  completed_stops_count: number;
  telemetry_available?: boolean;
}

export interface Stop {
  id: string;
  route_id: string;
  vehicle_id: string;
  sequence_idx: number;
  customer_name: string;
  address: string;
  window_start: string; // "10:30 IST"
  window_end: string;   // "11:15 IST"
  eta_min: string;      // "10:35 IST"
  eta_max: string;      // "10:50 IST"
  predicted_late_prob_min: number | null; // null when backend risk scoring was not run
  predicted_late_prob_max: number | null;
  penalty_exposure_inr: number;    // converted from financial.py at ₹83/$
  penalty_exposure_usd: number;
  demand_kg: number;
  is_locked: boolean;    // recovery re-plan locked prefix
  is_completed: boolean;
  status: StopStatus;
  lat?: number;
  lng?: number;
  x?: number; // for Solomon x/y
  y?: number;
  notes?: string;
  penalty_exposure_label?: string;
  demand_label?: string;
}

export interface Route {
  id: string;
  vehicle_id: string;
  color_token: string; // e.g. "var(--map-route-1)" or "#2F5BEA"
  total_distance_km: number;
  total_duration_minutes: number;
  distance_label?: string;
  duration_label?: string;
  stops: Stop[];
  // [lng, lat] pairs for MapLibre or [x, y] for SVG Solomon
  geometry_coordinates: [number, number][];
}

export interface ValidatorResult {
  passed: boolean;
  constraints_checked: string[];
  violations: string[];
  evaluated_at: string;
}

export interface PlanRun {
  id: string;
  scenario_id: string;
  scenario_name: string;
  coordinate_system: CoordinateSystem;
  status: 'active' | 'draft' | 'archived';
  engine: string;
  provenance_badge: ProvenanceBadge;
  created_at: string;
  total_vehicles: number;
  total_stops: number;
  completed_stops: number;
  total_sla_exposure_inr: number;
  assumed_rate_label: string;
  objective_cost?: string;
  validator_result: ValidatorResult;
  vehicles: Vehicle[];
  routes: Route[];
  bounds?: [[number, number], [number, number]]; // [[minLng, minLat], [maxLng, maxLat]]
  data_source?: 'backend' | 'preview';
  cost_unit_label?: string;
  total_exposure_label?: string;
  capacity_unit_label?: string;
  coordinate_caption?: string;
  backend_payload?: { scenario: Record<string, unknown>; plan: Record<string, any> };
}

export interface SolverJob {
  id: string;
  scenario_id: string;
  status: JobStatus;
  progress_pct: number | null;
  elapsed_seconds: number;
  engine: string;
  provenance_badge: ProvenanceBadge;
  result_plan_id: string | null;
  error_message: string | null;
  submitted_at: string;
}

export interface OperationalEvent {
  id: string;
  timestamp: string;
  severity: 'info' | 'warning' | 'critical' | 'success';
  category: 'assignment' | 'status_change' | 'delay_alert' | 'plan_publish' | 'job_lifecycle' | 'recovery';
  title: string;
  description: string;
  actor: string;
  related_id?: string;
}

export interface CalibrationPoint {
  predicted_prob: number;
  observed_freq: number;
}

export interface EngineBenchmark {
  engine: string;
  provenance: ProvenanceBadge;
  gap_to_optimum_pct: number;
  feasibility_pct: number;
  runtime_seconds: number;
  objective_cost: string;
}

export interface IntelligenceEvidence {
  model_name: string;
  brier_score: number;
  brier_reduction_pct: number;
  regret_reduction_pct: number;
  user_claim_verified: boolean;
  user_claim_note: string;
  calibration_points: CalibrationPoint[];
  benchmarks: EngineBenchmark[];
  quantum_circuit_summary: {
    type: string;
    qubits: number;
    backend_mode: string;
    execution_badge: ProvenanceBadge;
  };
  disclaimer: string;
}

export interface SystemCapacity {
  max_slots: number;
  active_slots: number;
  available_slots: number;
  active_job_ids: string[];
}

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'offline';
  backend_url: string;
  version: string;
  timestamp: string;
  capacity: SystemCapacity | null;
  engines_available: string[];
}
