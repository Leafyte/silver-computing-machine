import React from 'react';
import { Stop, Vehicle } from '../../types/api';
import {
  X,
  Clock,
  AlertTriangle,
  Lock,
  Unlock,
  MapPin,
  RefreshCw,
  Package
} from 'lucide-react';

interface StopDrawerProps {
  stop: Stop | null;
  vehicle: Vehicle | null;
  onClose: () => void;
  onTriggerReplan?: (stopId: string, vehicleId: string) => void;
}

export const StopDrawer: React.FC<StopDrawerProps> = ({
  stop,
  vehicle,
  onClose,
  onTriggerReplan
}) => {
  if (!stop) return null;

  const isLateRisk = (stop.predicted_late_prob_max || 0) > 0.4 || stop.status === 'at_risk';

  return (
    <div className="glass-panel-strong rounded-3xl p-5 border border-white/80 shadow-2xl space-y-4 max-w-md w-full select-none text-xs">
      {/* Header */}
      <div className="flex items-start justify-between pb-3 border-b border-black/5">
        <div>
          <div className="flex items-center space-x-2">
            <span className="px-2 py-0.5 bg-[#17231F] text-white rounded-md font-bold text-[11px] tabular-nums">
              Stop #{stop.sequence_idx}
            </span>
            <span
              className={`px-2 py-0.5 rounded-full font-medium text-[11px] ${
                stop.is_completed
                  ? 'bg-[#E2F4EE] text-[#205A46]'
                  : isLateRisk
                  ? 'bg-[#FBECEB] text-[#C6534C]'
                  : 'bg-[#EFF4FF] text-[#2F5BEA]'
              }`}
            >
              {stop.status.toUpperCase()}
            </span>
          </div>
          <h3 className="font-bold text-sm text-[#15211D] mt-1.5 leading-snug">
            {stop.customer_name}
          </h3>
          <p className="text-[#68756F] flex items-center gap-1 mt-0.5">
            <MapPin className="w-3.5 h-3.5 text-[#205A46] shrink-0" />
            <span className="truncate">{stop.address}</span>
          </p>
        </div>

        <button
          onClick={onClose}
          className="p-1.5 rounded-xl hover:bg-black/5 text-[#68756F] hover:text-[#15211D] transition-colors"
          title="Close details"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* SLA & Time Windows */}
      <div className="grid grid-cols-2 gap-2.5">
        <div className="bg-white/80 border border-black/5 rounded-2xl p-3">
          <div className="text-[10px] uppercase font-bold text-[#929E99] flex items-center gap-1">
            <Clock className="w-3 h-3 text-[#205A46]" />
            Commitment Window
          </div>
          <div className="font-bold text-xs text-[#15211D] mt-1 tabular-nums">
            {stop.window_start} – {stop.window_end}
          </div>
          <div className="text-[10px] text-[#68756F] mt-0.5">Contractual SLA</div>
        </div>

        <div className="bg-white/80 border border-black/5 rounded-2xl p-3">
          <div className="text-[10px] uppercase font-bold text-[#929E99] flex items-center gap-1">
            <Clock className="w-3 h-3 text-[#2F5BEA]" />
            Predicted ETA
          </div>
          <div className="font-bold text-xs text-[#15211D] mt-1 tabular-nums">
            {stop.eta_min} – {stop.eta_max}
          </div>
          <div className="text-[10px] text-[#2F5BEA] font-medium mt-0.5">
            {stop.predicted_late_prob_max === null ? 'Risk score unavailable' : 'simulated interval'}
          </div>
        </div>
      </div>

      {/* Risk & Financial Exposure */}
      <div className="bg-white/80 border border-black/5 rounded-2xl p-3.5 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[#68756F] font-medium">Late Probability:</span>
          <span className="tabular-nums font-bold text-[#15211D] flex items-center gap-1">
            {stop.predicted_late_prob_min === null || stop.predicted_late_prob_max === null
              ? 'Unavailable'
              : `[${Math.round(stop.predicted_late_prob_min * 100)}%, ${Math.round(stop.predicted_late_prob_max * 100)}%]`}
            {stop.predicted_late_prob_min !== null && <span className="text-[10px] text-[#68756F] font-normal">(simulated)</span>}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-[#68756F] font-medium">SLA Penalty Exposure:</span>
          <span className="tabular-nums font-extrabold text-[#C6534C] text-sm">
            {stop.penalty_exposure_label ?? `₹${stop.penalty_exposure_inr.toLocaleString('en-IN')}`}
            <span className="text-[10px] text-[#68756F] font-normal ml-1">
              {stop.penalty_exposure_label ? '(scenario objective units)' : '(assumed rate ₹83/$)'}
            </span>
          </span>
        </div>

        <div className="flex items-center justify-between pt-1 border-t border-black/5">
          <span className="text-[#68756F] font-medium flex items-center gap-1">
            <Package className="w-3.5 h-3.5 text-[#205A46]" />
            Payload Demand:
          </span>
          <span className="tabular-nums font-bold text-[#15211D]">
            {stop.demand_label ?? `${stop.demand_kg} kg`}
          </span>
        </div>
      </div>

      {/* Assigned Vehicle & Locked Status */}
      <div className="flex items-center justify-between px-3 py-2 bg-black/5 rounded-2xl text-[11px]">
        <div className="flex items-center space-x-2">
          {stop.is_locked ? (
            <span className="flex items-center text-[#68756F] gap-1 font-medium">
              <Lock className="w-3.5 h-3.5 text-[#205A46]" />
              Locked (served)
            </span>
          ) : (
            <span className="flex items-center text-[#205A46] gap-1 font-medium">
              <Unlock className="w-3.5 h-3.5" />
              Dynamic (reschedulable)
            </span>
          )}
        </div>
        <div className="font-semibold text-[#15211D]">
          {vehicle?.callsign || 'Vehicle Unassigned'}
        </div>
      </div>

      {/* Operational Note */}
      {stop.notes && (
        <div className="p-2.5 bg-[#FDF4EB] border border-[#FBD6B0] rounded-2xl text-[11px] text-[#9A5B00] flex items-start gap-1.5">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{stop.notes}</span>
        </div>
      )}

      {/* Actions */}
      {!stop.is_completed && onTriggerReplan && (
        <button
          onClick={() => onTriggerReplan(stop.id, stop.vehicle_id)}
          className="w-full py-2.5 px-4 bg-[#17231F] hover:bg-[#205A46] text-white rounded-2xl font-bold transition-all flex items-center justify-center space-x-2 shadow-sm cursor-pointer"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Trigger Mid-Shift Re-Plan from Stop #{stop.sequence_idx}</span>
        </button>
      )}
    </div>
  );
};
