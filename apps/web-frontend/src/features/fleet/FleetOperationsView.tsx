import React, { useState, useMemo } from 'react';
import { PlanRun, Vehicle } from '../../types/api';
import {
  Truck,
  Phone,
  Search,
  ChevronRight,
  MapPin,
  X
} from 'lucide-react';

interface FleetOperationsViewProps {
  plan: PlanRun;
  selectedVehicleId?: string | null;
  searchQuery?: string;
  onSelectVehicleForMap?: (vehicleId: string) => void;
}

export const FleetOperationsView: React.FC<FleetOperationsViewProps> = ({
  plan,
  selectedVehicleId,
  searchQuery = '',
  onSelectVehicleForMap
}) => {
  const [localSearch, setLocalSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | 'active' | 'at_risk' | 'delayed'>('all');
  const [selectedVehicle, setSelectedVehicle] = useState<Vehicle | null>(() => {
    if (selectedVehicleId) {
      return plan.vehicles.find((v) => v.id === selectedVehicleId) || null;
    }
    return null;
  });

  const effectiveSearch = searchQuery || localSearch;

  // Filter vehicles
  const filteredVehicles = useMemo(() => {
    return plan.vehicles.filter((v) => {
      const matchSearch =
        v.callsign.toLowerCase().includes(effectiveSearch.toLowerCase()) ||
        v.driver_name.toLowerCase().includes(effectiveSearch.toLowerCase()) ||
        v.vehicle_type.toLowerCase().includes(effectiveSearch.toLowerCase());
      const matchStatus = statusFilter === 'all' || v.status === statusFilter;
      return matchSearch && matchStatus;
    });
  }, [plan.vehicles, effectiveSearch, statusFilter]);

  return (
    <div className="flex-1 flex overflow-hidden p-6 gap-6 select-none">
      {/* Left/Main Data Grid */}
      <div className="flex-1 flex flex-col bg-white rounded-3xl p-6 border border-[#EAE6DF] shadow-xs overflow-hidden">
        {/* Top Header & Search Bar */}
        <div className="flex flex-wrap items-center justify-between pb-4 border-b border-[#EAE6DF] gap-4">
          <div>
            <h2 className="text-xl font-extrabold text-[#141F1A] tracking-tight">
              Fleet Operations & Telemetry
            </h2>
            <p className="text-xs text-[#7D8D85] mt-0.5">
              Monitoring {plan.vehicles.length} assigned units across {plan.scenario_name}
            </p>
          </div>

          {/* Search & Status Filters (Atlas Pill Style) */}
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-[#7D8D85] absolute left-3.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search callsign or driver…"
                value={effectiveSearch}
                onChange={(e) => setLocalSearch(e.target.value)}
                className="pl-9 pr-3.5 py-1.5 text-xs bg-[#F5F3ED]/80 border border-[#E5E0D8] rounded-full text-[#141F1A] placeholder-[#7D8D85] focus:outline-none focus:bg-white focus:ring-2 focus:ring-[#1E5A44]/20 focus:border-[#1E5A44] transition-all"
              />
            </div>

            <div className="flex items-center space-x-1 bg-[#F5F3ED] p-1 rounded-full border border-[#E5E0D8] text-xs">
              {(['all', 'active', 'at_risk', 'delayed'] as const).map((st) => (
                <button
                  key={st}
                  onClick={() => setStatusFilter(st)}
                  className={`px-3 py-1 rounded-full font-semibold transition-all cursor-pointer capitalize text-[11px] ${
                    statusFilter === st
                      ? 'bg-[#141F1A] text-white shadow-xs'
                      : 'text-[#4D5C54] hover:text-[#141F1A]'
                  }`}
                >
                  {st.replace('_', ' ')}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Atlas Metric Strip */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 my-4">
          <div className="p-4 bg-[#F5F3ED]/60 border border-[#E5E0D8] rounded-2xl">
            <span className="text-[10px] uppercase font-bold tracking-wider text-[#7D8D85] block">Assigned Fleet</span>
            <div className="text-2xl font-black text-[#141F1A] tabular-nums mt-1">{plan.vehicles.length} Units</div>
            <span className="text-[10px] text-[#1E5A44] font-semibold">{plan.data_source === 'backend' ? 'Planned assignments · telemetry unavailable' : '100% active dispatch'}</span>
          </div>
          <div className="p-4 bg-[#F5F3ED]/60 border border-[#E5E0D8] rounded-2xl">
            <span className="text-[10px] uppercase font-bold tracking-wider text-[#7D8D85] block">Payload Utilization</span>
            <div className="text-2xl font-black text-[#141F1A] tabular-nums mt-1">
              {Math.round((plan.vehicles.reduce((acc, v) => acc + v.used_capacity_kg, 0) / plan.vehicles.reduce((acc, v) => acc + v.capacity_kg, 1)) * 100)}%
            </div>
            <span className="text-[10px] text-[#4D5C54] font-medium tabular-nums">{plan.vehicles.reduce((acc, v) => acc + v.used_capacity_kg, 0)} / {plan.vehicles.reduce((acc, v) => acc + v.capacity_kg, 0)} {plan.capacity_unit_label ?? 'kg'}</span>
          </div>
          <div className="p-4 bg-[#F5F3ED]/60 border border-[#E5E0D8] rounded-2xl">
            <span className="text-[10px] uppercase font-bold tracking-wider text-[#7D8D85] block">Fleet Stop Progress</span>
            <div className="text-2xl font-black text-[#1E5A44] tabular-nums mt-1">
              {plan.completed_stops} / {plan.total_stops}
            </div>
            <span className="text-[10px] text-[#4D5C54] font-medium tabular-nums">{Math.round((plan.completed_stops / plan.total_stops) * 100)}% complete</span>
          </div>
          <div className="p-4 bg-[#F5F3ED]/60 border border-[#E5E0D8] rounded-2xl">
            <span className="text-[10px] uppercase font-bold tracking-wider text-[#7D8D85] block">Impending Delays</span>
            <div className="text-2xl font-black text-[#DC2626] tabular-nums mt-1">
              {plan.vehicles.filter(v => v.status === 'at_risk' || v.eta_deviation_minutes > 5).length} Units
            </div>
            <span className="text-[10px] text-[#DC2626] font-semibold">{plan.data_source === 'backend' ? 'Live delay data unavailable' : 'SLA exposure buffer active'}</span>
          </div>
        </div>

        {/* Dense Table View */}
        <div className="flex-1 overflow-y-auto mt-2">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-[#EAE6DF] text-[#7D8D85] text-[10px] uppercase font-bold tracking-wider">
                <th className="pb-3 font-bold">Vehicle Callsign</th>
                <th className="pb-3 font-bold">Driver Name</th>
                <th className="pb-3 font-bold">Status</th>
                <th className="pb-3 font-bold">Capacity Used</th>
                <th className="pb-3 font-bold">Stop Progress</th>
                <th className="pb-3 font-bold">ETA Deviation</th>
                <th className="pb-3 font-bold">Freshness</th>
                <th className="pb-3 text-right font-bold">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#EAE6DF]">
              {filteredVehicles.map((vehicle) => {
                const route = plan.routes.find((r) => r.id === vehicle.route_id);
                const isSelected = selectedVehicle?.id === vehicle.id;
                const capacityPct = Math.round((vehicle.used_capacity_kg / vehicle.capacity_kg) * 100);

                return (
                  <tr
                    key={vehicle.id}
                    onClick={() => setSelectedVehicle(vehicle)}
                    className={`hover:bg-[#F5F3ED]/50 transition-colors cursor-pointer ${
                      isSelected ? 'bg-[#F5F3ED]/70 font-medium' : ''
                    }`}
                  >
                    {/* Callsign & Type */}
                    <td className="py-4 pr-3">
                      <div className="flex items-center space-x-2.5">
                        <span
                          className="w-2.5 h-2.5 rounded-full"
                          style={{ backgroundColor: route?.color_token || '#2563EB' }}
                        />
                        <div>
                          <div className="font-extrabold text-[#141F1A]">{vehicle.callsign}</div>
                          <div className="text-[11px] text-[#7D8D85]">{vehicle.vehicle_type}</div>
                        </div>
                      </div>
                    </td>

                    {/* Driver */}
                    <td className="py-4 pr-3">
                      <div className="font-semibold text-[#141F1A]">{vehicle.driver_name}</div>
                      <div className="text-[10px] text-[#7D8D85] tabular-nums">{vehicle.phone || 'Not supplied'}</div>
                    </td>

                    {/* Status Chip */}
                    <td className="py-4 pr-3">
                      <span
                        className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-bold ${
                          vehicle.status === 'active'
                            ? 'bg-[#E8F5EE] text-[#1E5A44]'
                            : vehicle.status === 'at_risk'
                            ? 'bg-[#FEF3C7] text-[#D97706]'
                            : 'bg-[#FEE2E2] text-[#DC2626]'
                        }`}
                      >
                        {vehicle.telemetry_available === false ? 'PLANNED' : vehicle.status.toUpperCase()}
                      </span>
                    </td>

                    {/* Capacity */}
                    <td className="py-4 pr-3 tabular-nums">
                      <div className="font-semibold text-[#141F1A]">
                        {vehicle.used_capacity_kg} / {vehicle.capacity_kg} {plan.capacity_unit_label ?? 'kg'}
                      </div>
                      <div className="w-24 h-1.5 bg-[#F5F3ED] border border-[#E5E0D8] rounded-full mt-1.5 overflow-hidden">
                        <div
                          className="h-full bg-[#1E5A44] rounded-full"
                          style={{ width: `${capacityPct}%` }}
                        />
                      </div>
                    </td>

                    {/* Progress */}
                    <td className="py-4 pr-3 tabular-nums font-semibold text-[#141F1A]">
                      {vehicle.completed_stops_count} / {vehicle.assigned_stops_count} stops
                    </td>

                    {/* ETA Deviation */}
                    <td className="py-4 pr-3 tabular-nums">
                      {vehicle.telemetry_available === false ? <span className="text-[#7D8D85]">Not reported</span> : vehicle.eta_deviation_minutes > 0 ? (
                        <span className="font-bold text-[#DC2626]">
                          +{vehicle.eta_deviation_minutes} min late
                        </span>
                      ) : vehicle.eta_deviation_minutes < 0 ? (
                        <span className="font-bold text-[#1E5A44]">
                          {Math.abs(vehicle.eta_deviation_minutes)} min early
                        </span>
                      ) : (
                        <span className="text-[#1E5A44] font-medium">On-Schedule</span>
                      )}
                    </td>

                    {/* Freshness */}
                    <td className="py-4 pr-3 tabular-nums text-[#7D8D85] text-[11px] font-mono">
                      {vehicle.telemetry_available === false ? 'No GPS ping' : vehicle.last_ping_iso ? vehicle.last_ping_iso.split('T')[1].slice(0, 8) + ' IST' : '—'}
                    </td>

                    {/* Action */}
                    <td className="py-4 text-right">
                      <ChevronRight className="w-4 h-4 text-[#7D8D85] inline-block" />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Right Drawer: Vehicle Deep Detail */}
      {selectedVehicle && (
        <div className="w-96 bg-white rounded-3xl p-6 border border-[#EAE6DF] shadow-xl flex flex-col justify-between text-xs">
          <div>
            <div className="flex items-start justify-between pb-3 border-b border-[#EAE6DF]">
              <div>
                <span className="text-[10px] uppercase font-bold text-[#7D8D85]">
                  Vehicle Telemetry Dossier
                </span>
                <h3 className="text-base font-extrabold text-[#141F1A]">
                  {selectedVehicle.callsign}
                </h3>
                <p className="text-[#7D8D85]">{selectedVehicle.vehicle_type}</p>
              </div>
              <button
                onClick={() => setSelectedVehicle(null)}
                className="p-1.5 rounded-xl hover:bg-black/5 text-[#7D8D85]"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="mt-4 space-y-3">
              {/* Driver Contact */}
              <div className="p-3.5 bg-[#F5F3ED]/70 border border-[#E5E0D8] rounded-2xl">
                <div className="text-[10px] uppercase font-bold text-[#7D8D85]">Assigned Driver</div>
                <div className="font-bold text-sm text-[#141F1A] mt-0.5">
                  {selectedVehicle.driver_name}
                </div>
                <div className="text-[#1E5A44] font-semibold mt-1 flex items-center gap-1.5 tabular-nums">
                  <Phone className="w-3.5 h-3.5" />
                  {selectedVehicle.phone || 'No contact supplied'}
                </div>
              </div>

              {/* Real-time Telemetry */}
              <div className="grid grid-cols-2 gap-2.5">
                <div className="p-3 bg-[#F5F3ED]/70 border border-[#E5E0D8] rounded-2xl">
                  <div className="text-[10px] uppercase font-bold text-[#7D8D85]">Battery / Fuel</div>
                  <div className="text-sm font-extrabold text-[#1E5A44] mt-0.5 tabular-nums">
                    {selectedVehicle.telemetry_available === false ? '—' : `${selectedVehicle.battery_or_fuel_pct}%`}
                  </div>
                </div>

                <div className="p-3 bg-[#F5F3ED]/70 border border-[#E5E0D8] rounded-2xl">
                  <div className="text-[10px] uppercase font-bold text-[#7D8D85]">Speed</div>
                  <div className="text-sm font-extrabold text-[#141F1A] mt-0.5 tabular-nums">
                    {selectedVehicle.telemetry_available === false ? '—' : `${selectedVehicle.speed_kmh ?? '—'} km/h`}
                  </div>
                </div>
              </div>

              {/* Location Coordinates */}
              <div className="p-3 bg-[#F5F3ED]/70 border border-[#E5E0D8] rounded-2xl">
                <div className="text-[10px] uppercase font-bold text-[#7D8D85] flex items-center gap-1">
                  <MapPin className="w-3 h-3 text-[#1E5A44]" />
                  Last Known Position
                </div>
                <div className="font-mono text-xs text-[#141F1A] mt-1 tabular-nums">
                  {selectedVehicle.telemetry_available === false ? 'Location unavailable · no GPS data in solver input' : selectedVehicle.current_lat
                    ? `${selectedVehicle.current_lat.toFixed(4)}°N, ${selectedVehicle.current_lng?.toFixed(4)}°E`
                    : `Grid X: ${selectedVehicle.current_x}, Y: ${selectedVehicle.current_y}`}
                </div>
              </div>
            </div>
          </div>

          {/* Action button */}
          <div className="pt-4 border-t border-[#EAE6DF]">
            <button
              onClick={() => {
                if (onSelectVehicleForMap) onSelectVehicleForMap(selectedVehicle.id);
              }}
              className="w-full py-3 bg-[#141F1A] hover:bg-[#1E5A44] text-white rounded-full font-bold transition-all flex items-center justify-center space-x-2 cursor-pointer shadow-xs"
            >
              <Truck className="w-4 h-4" />
              <span>Locate on Command Center Map</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
