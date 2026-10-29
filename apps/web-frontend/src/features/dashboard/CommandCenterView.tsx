import React, { useState } from 'react';
import { PlanRun } from '../../types/api';
import { RouteMap } from '../../components/map/RouteMap';
import { StopDrawer } from '../../components/stops/StopDrawer';
import { DataBadge } from '../../components/ui/DataBadge';
import {
  Truck,
  CheckCircle2,
  AlertTriangle,
  Clock,
  ShieldCheck,
  Lock,
  RotateCcw,
  Boxes
} from 'lucide-react';

interface CommandCenterViewProps {
  plan: PlanRun;
  selectedVehicleId?: string | null;
  selectedStopId?: string | null;
  onSelectVehicle?: (vehicleId: string) => void;
  onSelectStop?: (stopId: string) => void;
  searchQuery?: string;
  onTriggerReplan: (stopId: string, vehicleId: string) => void;
}

export const CommandCenterView: React.FC<CommandCenterViewProps> = ({
  plan,
  selectedVehicleId,
  selectedStopId,
  onSelectVehicle,
  onSelectStop,
  searchQuery = '',
  onTriggerReplan
}) => {
  const currentVehicleId = selectedVehicleId || plan.vehicles[0]?.id || null;
  const currentStopId = selectedStopId || null;

  const [stopFilter, setStopFilter] = useState<'all' | 'pending' | 'at_risk' | 'completed'>('all');
  const [reRouteToastDismissed, setReRouteToastDismissed] = useState<boolean>(false);

  const handleSelectVehicle = (vId: string) => {
    if (onSelectVehicle) onSelectVehicle(vId);
  };

  const handleSelectStop = (sId: string) => {
    if (onSelectStop) onSelectStop(sId);
  };

  // Active vehicle and route
  const activeVehicle = plan.vehicles.find((v) => v.id === currentVehicleId) || plan.vehicles[0];
  const activeRoute = plan.routes.find((r) => r.vehicle_id === activeVehicle?.id);
  const selectedStop = activeRoute?.stops.find((s) => s.id === currentStopId) || null;

  // Key operational totals
  const totalStops = plan.total_stops;
  const completedStops = plan.completed_stops;
  const progressPct = Math.round((completedStops / totalStops) * 100);

  // Find first at-risk stop for re-route advisory prompt (Warehouse & RouteIQ pattern)
  const candidateDelayStop = activeRoute?.stops.find((s) => !s.is_completed && (
    plan.data_source === 'backend' || (s.predicted_late_prob_max !== null && s.predicted_late_prob_max > 0.3) || s.status === 'at_risk'
  ));

  // Filter stops by searchQuery and stopFilter tab
  const displayedStops = (activeRoute?.stops || []).filter((s) => {
    // 1. Text Search Filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      const match =
        s.customer_name.toLowerCase().includes(query) ||
        s.address.toLowerCase().includes(query) ||
        s.sequence_idx.toString().includes(query);
      if (!match) return false;
    }

    // 2. Tab Filter
    if (stopFilter === 'completed') return s.is_completed;
    if (stopFilter === 'at_risk') return !s.is_completed && ((s.predicted_late_prob_max || 0) > 0.25 || s.status === 'at_risk');
    if (stopFilter === 'pending') return !s.is_completed;
    return true;
  });

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden select-none">
      {/* Top Operational Header & Atlas 3-Card KPI Overview */}
      <div className="px-6 py-4 border-b border-[#EAE6DF] bg-white/60 flex flex-col gap-4">
        {/* Scenario Header Bar */}
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-extrabold text-[#141F1A] tracking-tight">
              {plan.scenario_name}
            </h1>
            <div className="flex items-center space-x-2 mt-1 text-xs text-[#4D5C54]">
              <DataBadge badge={plan.provenance_badge} size="sm" />
              <span>·</span>
              <span className="tabular-nums font-medium">Created: {plan.created_at.split('T')[1].slice(0, 5)} IST</span>
              <span>·</span>
              <span className="flex items-center text-[#1E5A44] font-medium gap-1">
                <ShieldCheck className="w-3.5 h-3.5" />
                {plan.data_source === 'backend'
                  ? plan.validator_result.passed ? 'Backend validator passed' : 'Backend validation failed'
                  : 'Preview validator'}
              </span>
              {plan.data_source === 'backend' && plan.coordinate_caption && <span className="text-[#7D8D85]">· {plan.coordinate_caption}</span>}
            </div>
          </div>

          <div className="flex items-center space-x-2 text-xs">
            <span className="text-[11px] font-mono text-[#7D8D85]">
              Objective: {plan.data_source === 'backend' ? (plan.objective_cost ?? 'Unavailable') : plan.objective_cost ? (plan.objective_cost.startsWith('₹') ? plan.objective_cost : `₹${plan.objective_cost}`) : `₹${plan.total_sla_exposure_inr.toLocaleString('en-IN')}`}
            </span>
          </div>
        </div>

        {/* Atlas 3-Card KPI Ribbon (ref_01_atlas_dashboard.png) */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Card 1: Fleet Efficiency Donut (Atlas Ref 1) */}
          <div className="bg-white p-4 rounded-2xl border border-[#EAE6DF] shadow-xs flex items-center justify-between">
            <div className="flex items-center space-x-4">
              {/* Segmented Donut Chart */}
              <div className="relative w-14 h-14 shrink-0">
                <svg viewBox="0 0 36 36" className="w-full h-full -rotate-90">
                  <circle cx="18" cy="18" r="14" fill="none" stroke="#F5F3ED" strokeWidth="3.5" />
                  <circle
                    cx="18"
                    cy="18"
                    r="14"
                    fill="none"
                    stroke="#1E5A44"
                    strokeWidth="3.5"
                    strokeDasharray="67, 100"
                    strokeLinecap="round"
                  />
                  <circle
                    cx="18"
                    cy="18"
                    r="14"
                    fill="none"
                    stroke="#D97706"
                    strokeWidth="3.5"
                    strokeDasharray="20, 100"
                    strokeDashoffset="-67"
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                  <span className="text-xs font-black text-[#141F1A] tabular-nums leading-none">
                    {plan.vehicles.length}
                  </span>
                  <span className="text-[8px] uppercase tracking-wider text-[#7D8D85] font-semibold mt-0.5">
                    Fleet
                  </span>
                </div>
              </div>

              <div>
                <div className="text-[11px] font-bold uppercase tracking-wider text-[#7D8D85]">
                  Fleet In-Field
                </div>
                <div className="text-lg font-black text-[#141F1A] tabular-nums mt-0.5">
                  71% <span className="text-[11px] font-semibold text-[#1E5A44] ml-1">↗ +4%</span>
                </div>
                <div className="flex items-center space-x-2 text-[10px] text-[#4D5C54] mt-0.5 font-medium">
                  <span className="flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#1E5A44]" />
                    <span>on the move: {plan.vehicles.filter((v) => v.status === 'active').length}</span>
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#D97706]" />
                    <span>idle: {plan.vehicles.filter((v) => v.status !== 'active').length}</span>
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Card 2: Delivery Status & Performance Ring (Atlas Ref 1) */}
          <div className="bg-white p-4 rounded-2xl border border-[#EAE6DF] shadow-xs flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="relative w-14 h-14 shrink-0">
                <svg viewBox="0 0 36 36" className="w-full h-full -rotate-90">
                  <circle cx="18" cy="18" r="14" fill="none" stroke="#F5F3ED" strokeWidth="3.5" />
                  <circle
                    cx="18"
                    cy="18"
                    r="14"
                    fill="none"
                    stroke="#2563EB"
                    strokeWidth="3.5"
                    strokeDasharray={`${progressPct}, 100`}
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <CheckCircle2 className="w-5 h-5 text-[#2563EB]" />
                </div>
              </div>

              <div>
                <div className="text-[11px] font-bold uppercase tracking-wider text-[#7D8D85]">
                  Delivery Status
                </div>
                <div className="text-lg font-black text-[#141F1A] tabular-nums mt-0.5">
                  89% <span className="text-xs font-normal text-[#4D5C54]">Within Time Limit</span>
                </div>
                <div className="text-[10px] text-[#7D8D85] mt-0.5 tabular-nums">
                  {completedStops} of {totalStops} stops completed ({progressPct}%)
                </div>
              </div>
            </div>

            <DataBadge badge="Simulated · held-out run" size="sm" />
          </div>

          {/* Card 3: Vehicles on the Road & SLA Financial Exposure (Atlas Ref 1) */}
          <div className="bg-white p-4 rounded-2xl border border-[#EAE6DF] shadow-xs flex items-center justify-between">
            <div>
              <div className="text-[11px] font-bold uppercase tracking-wider text-[#7D8D85]">
                SLA Exposure
              </div>
              <div className="text-xl font-black text-[#DC2626] tabular-nums mt-0.5">
                {plan.data_source === 'backend' ? (plan.total_exposure_label ?? 'Unavailable') : `₹${plan.total_sla_exposure_inr.toLocaleString('en-IN')}`}
                <span className="text-[10px] font-normal text-[#4D5C54] ml-1.5">
                  ({plan.data_source === 'backend' ? plan.cost_unit_label : plan.assumed_rate_label.split(' ')[2]})
                </span>
              </div>
              <div className="text-[10px] text-[#7D8D85] mt-0.5">
                {plan.vehicles.length} Active Vehicles on the Road
              </div>
            </div>

            <div className="w-12 h-12 rounded-2xl bg-[#F5F3ED] border border-[#E5E0D8] flex items-center justify-center text-[#141F1A]">
              <Truck className="w-6 h-6 text-[#1E5A44]" />
            </div>
          </div>
        </div>
      </div>

      {/* Main Command Surface: Map (Left/Center) + Operations Rail (Right) */}
      <div className="flex-1 flex overflow-hidden p-4 gap-4">
        {/* Geospatial Map Canvas Column */}
        <div className="flex-1 relative flex flex-col h-full rounded-3xl overflow-hidden shadow-sm border border-black/5">
          {/* Tactical Delay & Re-Routing Toast (Warehouse Ref 4 & RouteIQ Ref 5) */}
          {candidateDelayStop && !reRouteToastDismissed && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 w-[92%] max-w-xl glass-panel-strong p-3.5 rounded-2xl shadow-xl border border-white flex items-center justify-between gap-3 text-xs animate-in slide-in-from-top-2">
              <div className="flex items-center space-x-2.5 min-w-0">
                <div className="w-7 h-7 rounded-full bg-[#FDF4EB] flex items-center justify-center shrink-0 border border-[#FBD6B0]">
                  <AlertTriangle className="w-4 h-4 text-[#D78A45]" />
                </div>
                <div className="truncate">
                  <div className="font-bold text-[#15211D] flex items-center gap-1.5">
                    <span>{plan.data_source === 'backend' ? 'Scenario travel delay (+1 unit)' : 'Corridor Congestion (+20 min)'}</span>
                    <span className="text-[10px] font-mono text-[#D78A45] bg-[#FDF4EB] px-1.5 py-0.2 rounded">
                      {plan.data_source === 'backend' ? 'Synthetic input' : 'Silk Board / Hosur Rd'}
                    </span>
                  </div>
                  <div className="text-[11px] text-[#55534F] truncate">
                    {plan.data_source === 'backend' ? `Re-plan after a delay at ${candidateDelayStop.customer_name}?` : `Save 14 min & recover ₹8,300 exposure on ${candidateDelayStop.customer_name} via ORR bypass?`}
                  </div>
                </div>
              </div>

              <div className="flex items-center space-x-2 shrink-0">
                <button
                  onClick={() => onTriggerReplan(candidateDelayStop.id, activeVehicle.id)}
                  className="px-3 py-1.5 bg-[#205A46] text-white rounded-xl font-semibold hover:bg-[#17231F] transition-all flex items-center gap-1 shadow-sm interactive-tap cursor-pointer"
                >
                  <RotateCcw className="w-3 h-3" />
                  <span>Re-Plan Route</span>
                </button>
                <button
                  onClick={() => setReRouteToastDismissed(true)}
                  className="px-2.5 py-1.5 text-[#77736D] hover:text-[#15211D] hover:bg-black/5 rounded-xl transition-colors cursor-pointer"
                >
                  Dismiss
                </button>
              </div>
            </div>
          )}

          <RouteMap
            plan={plan}
            selectedVehicleId={currentVehicleId}
            selectedStopId={currentStopId}
            onSelectVehicle={handleSelectVehicle}
            onSelectStop={handleSelectStop}
            className="flex-1 w-full h-full"
          />

          {/* Bottom Hub Distribution Strip (Warehouse Ref 4) */}
          <div className="absolute bottom-4 right-4 z-10 glass-panel p-2.5 px-3.5 rounded-2xl border border-white/70 shadow-md text-xs flex items-center space-x-4 hidden lg:flex">
            <div className="flex items-center space-x-1.5 text-[11px] font-bold text-[#15211D]">
              <Boxes className="w-3.5 h-3.5 text-[#205A46]" />
              <span>Depot Fulfillment Stage:</span>
            </div>
            <div className="flex items-center space-x-3 text-[10px] tabular-nums">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-[#205A46]" />
                <span className="font-semibold text-[#15211D]">Picked: 48%</span>
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-[#2F5BEA]" />
                <span className="font-semibold text-[#15211D]">Packed: 36%</span>
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-[#D78A45]" />
                <span className="font-semibold text-[#15211D]">Staged: 16%</span>
              </span>
            </div>
          </div>
        </div>

        {/* Right Operations Panel */}
        <div className="w-96 flex flex-col gap-3 shrink-0 h-full overflow-hidden">
          {/* Selected Stop Drawer if active */}
          {selectedStop ? (
            <StopDrawer
              stop={selectedStop}
              vehicle={activeVehicle}
              onClose={() => {
                if (onSelectStop) onSelectStop('');
              }}
              onTriggerReplan={onTriggerReplan}
            />
          ) : (
            /* Selected Vehicle Itinerary & Ordered Stops */
            <div className="flex-1 bg-white rounded-3xl p-4 flex flex-col overflow-hidden border border-[#EAE6DF] shadow-xs text-xs">
              {/* Vehicle Header Card (Tipax & Vanta Ride Ref) */}
              <div className="pb-3 border-b border-[#EAE6DF]">
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-3">
                    {/* Truck Silhouette Glyph */}
                    <div className="w-10 h-10 rounded-2xl bg-[#141F1A] flex items-center justify-center text-[#A8DBC9] shadow-xs">
                      <Truck className="w-5 h-5" />
                    </div>
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="font-extrabold text-sm text-[#141F1A]">
                          {activeVehicle.callsign}
                        </span>
                        <span
                          className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                            activeVehicle.status === 'active'
                              ? 'bg-[#E8F5EE] text-[#1E5A44]'
                              : activeVehicle.status === 'at_risk'
                              ? 'bg-[#FEF3C7] text-[#D97706]'
                              : 'bg-[#FEE2E2] text-[#DC2626]'
                          }`}
                        >
                          {activeVehicle.status.toUpperCase()}
                        </span>
                      </div>
                      <div className="text-[11px] text-[#4D5C54] mt-0.5">
                        {activeVehicle.driver_name} · {activeVehicle.vehicle_type}
                      </div>
                    </div>
                  </div>

                  {/* Battery & Payload */}
                  <div className="text-right">
                    <div className="font-extrabold text-xs text-[#1E5A44] tabular-nums">
                      {activeVehicle.telemetry_available === false ? 'Battery —' : `${activeVehicle.battery_or_fuel_pct}% Batt`}
                    </div>
                    <div className="text-[10px] text-[#7D8D85] tabular-nums">
                      {activeVehicle.used_capacity_kg}/{activeVehicle.capacity_kg} {plan.capacity_unit_label ?? 'kg'}
                    </div>
                  </div>
                </div>

                {/* Driver Action Quick Buttons (Ref 6 Delivery Card) */}
                {activeVehicle.telemetry_available === false ? (
                  <div className="mt-3 px-3 py-2 bg-[#F5F3ED] rounded-xl text-[11px] text-[#7D8D85]">
                    Driver identity and live telemetry were not included with this solver input.
                  </div>
                ) : <div className="mt-3 px-3 py-2 bg-[#F5F3ED] rounded-xl text-[11px] text-[#7D8D85]">
                  Driver communications are preview controls and are not connected to a messaging service.
                </div>}
              </div>

              {/* Filter Tabs for Stops (Ref 1 Atlas Table Tabs) */}
              <div className="py-2.5 flex items-center justify-between text-[#4D5C54] font-semibold text-[11px] border-b border-[#EAE6DF]">
                <div className="flex items-center space-x-1 bg-[#F5F3ED] p-0.5 rounded-full border border-[#E5E0D8]">
                  {(['all', 'pending', 'at_risk', 'completed'] as const).map((tab) => (
                    <button
                      key={tab}
                      onClick={() => setStopFilter(tab)}
                      className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold capitalize transition-all cursor-pointer ${
                        stopFilter === tab
                          ? 'bg-[#141F1A] text-white shadow-xs'
                          : 'text-[#4D5C54] hover:text-[#141F1A]'
                      }`}
                    >
                      {tab.replace('_', ' ')}
                    </button>
                  ))}
                </div>
                {searchQuery && (
                  <span className="text-[#1E5A44] font-medium text-[10px]">
                    Filtered by: "{searchQuery}"
                  </span>
                )}
                {!searchQuery && (
                  <span className="text-[10px] font-normal text-[#7D8D85]">
                    {displayedStops.length} Stops
                  </span>
                )}
              </div>

              {/* Ordered Stops List (Vanta Ride & Fieldnotes Style) */}
              <div className="flex-1 overflow-y-auto space-y-2 pr-1 mt-2">
                {displayedStops.map((stop) => {
                  const isLate = (stop.predicted_late_prob_max || 0) > 0.4 || stop.status === 'at_risk';
                  const isSelected = currentStopId === stop.id;

                  return (
                    <div
                      key={stop.id}
                      onClick={() => handleSelectStop(stop.id)}
                      className={`p-3 rounded-2xl border transition-all cursor-pointer interactive-tap ${
                        isSelected
                          ? 'bg-[#141F1A] text-white border-transparent shadow-md'
                          : stop.is_completed
                          ? 'bg-white border-[#EAE6DF] opacity-85 hover:bg-[#F5F3ED]/40'
                          : isLate
                          ? 'bg-[#FEF3C7]/60 border-[#FDE68A] hover:bg-[#FEF3C7]'
                          : 'bg-[#F5F3ED]/60 border-[#E5E0D8] hover:bg-[#F5F3ED]'
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-center space-x-2.5">
                          <span
                            className={`w-6 h-6 rounded-full flex items-center justify-center font-bold text-[10px] tabular-nums ${
                              isSelected
                                ? 'bg-[#A8DBC9] text-[#141F1A]'
                                : stop.is_completed
                                ? 'bg-[#1E5A44] text-white'
                                : 'bg-[#E5E0D8] text-[#141F1A]'
                            }`}
                          >
                            {stop.sequence_idx}
                          </span>
                          <span
                            className={`font-semibold text-xs truncate max-w-[170px] ${
                              isSelected ? 'text-white' : 'text-[#141F1A]'
                            }`}
                          >
                            {stop.customer_name}
                          </span>
                        </div>

                        {/* Status Icon */}
                        {stop.is_locked ? (
                          <Lock className={`w-3.5 h-3.5 ${isSelected ? 'text-[#A8DBC9]' : 'text-[#4D5C54]'}`} />
                        ) : stop.is_completed ? (
                          <CheckCircle2 className="w-3.5 h-3.5 text-[#1E5A44]" />
                        ) : isLate ? (
                          <AlertTriangle className="w-3.5 h-3.5 text-[#DC2626]" />
                        ) : null}
                      </div>

                      <div
                        className={`mt-1.5 flex items-center justify-between text-[11px] tabular-nums ${
                          isSelected ? 'text-white/80' : 'text-[#4D5C54]'
                        }`}
                      >
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3 opacity-70" />
                          <span>{stop.window_start} - {stop.window_end}</span>
                        </span>
                        <span className="font-semibold">
                          ETA {stop.eta_min}{plan.data_source === 'backend' ? ' · solver estimate' : ' (simulated)'}
                        </span>
                      </div>

                      {/* SLA Penalty Exposure Pill */}
                      {stop.penalty_exposure_inr > 0 && (
                        <div
                          className={`mt-1.5 pt-1.5 border-t flex items-center justify-between text-[10px] tabular-nums ${
                            isSelected
                              ? 'border-white/10 text-[#A8DBC9]'
                              : 'border-black/5 text-[#DC2626]'
                          }`}
                        >
                          <span>Exposure: {stop.penalty_exposure_label ?? `₹${stop.penalty_exposure_inr.toLocaleString('en-IN')}`}</span>
                          <span className="font-mono">P(late): {stop.predicted_late_prob_max === null ? 'Unavailable' : `${Math.round(stop.predicted_late_prob_max * 100)}%`}</span>
                        </div>
                      )}
                    </div>
                  );
                })}

                {displayedStops.length === 0 && (
                  <div className="text-center py-8 text-[#77736D]">
                    <div className="font-semibold text-xs">No stops match this filter</div>
                    <div className="text-[10px] mt-0.5">Switch tab or clear search query.</div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
