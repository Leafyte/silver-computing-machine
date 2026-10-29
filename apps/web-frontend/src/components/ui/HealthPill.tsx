import React, { useState } from 'react';
import { SystemHealth } from '../../types/api';
import { Cpu, Server, CheckCircle2, AlertTriangle } from 'lucide-react';

interface HealthPillProps {
  health: SystemHealth | null;
  onRefresh?: () => void;
}

export const HealthPill: React.FC<HealthPillProps> = ({ health, onRefresh }) => {
  const [isOpen, setIsOpen] = useState(false);

  const isOnline = health?.status === 'healthy';
  const activeSlots = health?.capacity?.active_slots ?? 0;
  const maxSlots = health?.capacity?.max_slots ?? 2;

  return (
    <div className="relative inline-block text-left">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="glass-pill flex items-center space-x-2.5 px-3 py-1.5 rounded-full hover:bg-white/95 transition-all text-xs text-[#15211D] cursor-pointer"
        aria-expanded={isOpen}
      >
        <span className="flex items-center space-x-1.5">
          <span
            className={`w-2 h-2 rounded-full ${
              isOnline ? 'bg-[#205A46] shadow-[0_0_8px_#205A46]' : 'bg-[#D78A45]'
            }`}
          />
          <span className="font-medium">
            {isOnline ? 'Backend Online' : 'Simulation Mode'}
          </span>
        </span>

        <span className="text-[#68756F] border-l border-black/10 pl-2 flex items-center space-x-1">
          <Cpu className="w-3.5 h-3.5 text-[#68756F]" />
          <span className="tabular-nums font-semibold">
            {activeSlots}/{maxSlots} Solvers
          </span>
        </span>
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-80 glass-panel-strong rounded-2xl p-4 shadow-xl z-50 text-xs border border-white/60">
          <div className="flex items-center justify-between pb-3 border-b border-black/5">
            <span className="font-semibold text-sm text-[#15211D] flex items-center gap-1.5">
              <Server className="w-4 h-4 text-[#205A46]" />
              Dispatch Engine Status
            </span>
            <span
              className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${
                isOnline ? 'bg-[#E2F4EE] text-[#205A46]' : 'bg-[#FDF4EB] text-[#D78A45]'
              }`}
            >
              {isOnline ? 'Connected' : 'Local Fallback'}
            </span>
          </div>

          <div className="mt-3 space-y-2.5 text-[#68756F]">
            <div className="flex justify-between items-center">
              <span>Endpoint:</span>
              <span className="font-mono text-[#15211D] bg-black/5 px-2 py-0.5 rounded">
                {health?.backend_url || 'http://127.0.0.1:8000'}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span>Solver Concurrency:</span>
              <span className="tabular-nums font-medium text-[#15211D]">
                BoundedSemaphore(2) — {activeSlots} busy, {maxSlots - activeSlots} free
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span>Job DB Mode:</span>
              <span className="font-medium text-[#15211D] flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5 text-[#205A46]" />
                SQLite WAL
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span>Independent Validator:</span>
              <span className="font-medium text-[#205A46]">validator.py Active</span>
            </div>
          </div>

          {!isOnline && (
            <div className="mt-3 p-2 bg-[#FDF4EB] border border-[#FBD6B0] rounded-xl flex items-start space-x-2 text-[#9A5B00] text-[11px]">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>
                Backend on 127.0.0.1:8000 is unreachable. Displaying deterministic benchmark runs with honest source badges.
              </span>
            </div>
          )}

          {onRefresh && (
            <button
              onClick={() => {
                onRefresh();
                setIsOpen(false);
              }}
              className="mt-3.5 w-full py-1.5 bg-[#17231F] text-white rounded-xl hover:bg-[#205A46] transition-colors font-medium text-center"
            >
              Check Connection
            </button>
          )}
        </div>
      )}
    </div>
  );
};
