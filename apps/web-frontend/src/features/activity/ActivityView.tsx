import React, { useState } from 'react';
import { OperationalEvent } from '../../types/api';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Info,
  RefreshCw,
  Search,
  User
} from 'lucide-react';

interface ActivityViewProps {
  events: OperationalEvent[];
  syncError?: string | null;
  lastSyncedAt?: string | null;
  onRefresh?: () => void;
}

export const ActivityView: React.FC<ActivityViewProps> = ({ events, syncError, lastSyncedAt, onRefresh }) => {
  const [filterCategory, setFilterCategory] = useState<string>('all');
  const [search, setSearch] = useState<string>('');
  const filteredEvents = events.filter((evt) => {
    const matchCategory = filterCategory === 'all' || evt.category === filterCategory;
    const matchSearch =
      evt.title.toLowerCase().includes(search.toLowerCase()) ||
      evt.description.toLowerCase().includes(search.toLowerCase()) ||
      evt.actor.toLowerCase().includes(search.toLowerCase());
    return matchCategory && matchSearch;
  });

  const getSeverityStyle = (sev: OperationalEvent['severity']) => {
    switch (sev) {
      case 'critical':
        return {
          bg: 'bg-[#FBECEB]',
          border: 'border-[#F2B8B5]',
          text: 'text-[#C6534C]',
          icon: <AlertTriangle className="w-4 h-4 text-[#C6534C]" />
        };
      case 'warning':
        return {
          bg: 'bg-[#FDF4EB]',
          border: 'border-[#FBD6B0]',
          text: 'text-[#D78A45]',
          icon: <AlertTriangle className="w-4 h-4 text-[#D78A45]" />
        };
      case 'success':
        return {
          bg: 'bg-[#E2F4EE]',
          border: 'border-[#A8DBC9]',
          text: 'text-[#205A46]',
          icon: <CheckCircle2 className="w-4 h-4 text-[#205A46]" />
        };
      case 'info':
      default:
        return {
          bg: 'bg-[#EFF4FF]',
          border: 'border-[#C7D7FE]',
          text: 'text-[#2F5BEA]',
          icon: <Info className="w-4 h-4 text-[#2F5BEA]" />
        };
    }
  };

  return (
    <div className="flex-1 flex flex-col overflow-y-auto p-6 gap-6 select-none text-xs">
      {/* Top Header Card */}
      <div className="bg-white rounded-3xl p-6 border border-[#EAE6DF] shadow-xs">
        <div className="flex flex-wrap items-center justify-between pb-4 border-b border-[#EAE6DF] gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <Activity className="w-5 h-5 text-[#1E5A44]" />
              <h2 className="text-xl font-extrabold text-[#141F1A] tracking-tight">
                Operational Event Stream & Audit Trail
              </h2>
            </div>
            <p className="text-xs text-[#7D8D85] mt-1">
              Backend solver lifecycle and driver-reported stop or delay events
            </p>
          </div>

          {/* Search & Refresh */}
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-[#7D8D85] absolute left-3.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search audit trail…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-9 pr-3.5 py-1.5 text-xs bg-[#F5F3ED]/80 border border-[#E5E0D8] rounded-full text-[#141F1A] placeholder-[#7D8D85] focus:outline-none focus:bg-white focus:ring-2 focus:ring-[#1E5A44]/20 focus:border-[#1E5A44]"
              />
            </div>

            {onRefresh && (
              <button
                onClick={onRefresh}
                className="p-2 rounded-full bg-[#F5F3ED] hover:bg-black/5 border border-[#E5E0D8] text-[#141F1A] transition-all cursor-pointer"
                title="Refresh Events"
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[10px]">
          <span className={syncError ? 'text-[#C6534C]' : 'text-[#7D8D85]'}>
            {syncError ? `Driver event sync failed: ${syncError}` : `Live backend feed · ${lastSyncedAt ? `synced ${new Date(lastSyncedAt).toLocaleTimeString()}` : 'connecting…'}`}
          </span>
          <span className="text-[#7D8D85]">Phone reports refresh automatically every 3 seconds.</span>
        </div>

        {/* Category Pills (Atlas Style) */}
        <div className="flex items-center space-x-2 mt-4 overflow-x-auto pb-1">
          {[
            { id: 'all', label: 'All Events' },
            { id: 'delay_alert', label: 'Delay Alerts' },
            { id: 'recovery', label: 'Recovery Re-plans' },
            { id: 'status_change', label: 'Stop Completions' },
            { id: 'assignment', label: 'Driver messages' },
            { id: 'job_lifecycle', label: 'Solver Lifecycle' }
          ].map((cat) => (
            <button
              key={cat.id}
              onClick={() => setFilterCategory(cat.id)}
              className={`px-3.5 py-1 rounded-full font-semibold transition-all cursor-pointer whitespace-nowrap text-[11px] ${
                filterCategory === cat.id
                  ? 'bg-[#141F1A] text-white shadow-xs'
                  : 'bg-[#F5F3ED] hover:bg-black/5 text-[#4D5C54] border border-[#E5E0D8]'
              }`}
            >
              {cat.label}
            </button>
          ))}
        </div>
      </div>

      {/* Events Stream Feed */}
      <div className="bg-white rounded-3xl p-6 border border-[#EAE6DF] shadow-xs space-y-3">
        {filteredEvents.length === 0 ? (
          <div className="text-center py-12 text-[#7D8D85] space-y-2">
            <Activity className="w-8 h-8 text-[#A8DBC9] mx-auto" />
            <div className="font-bold text-sm text-[#141F1A]">No operational events match this filter</div>
            <p className="text-[11px]">Phone arrival, completion, delay, and driver-message reports appear here after the backend accepts them.</p>
          </div>
        ) : (
          filteredEvents.map((evt) => {
            const sev = getSeverityStyle(evt.severity);
            const timeFormatted = evt.timestamp.includes('T')
              ? evt.timestamp.split('T')[1].slice(0, 8) + ' IST'
              : evt.timestamp;

            return (
              <div
                key={evt.id}
                className="p-3.5 bg-white/85 border border-black/5 rounded-2xl flex items-start space-x-3.5 hover:bg-white transition-all shadow-xs"
              >
                {/* Severity Icon Pill */}
                <div className={`p-2 rounded-xl shrink-0 ${sev.bg} ${sev.border} border`}>
                  {sev.icon}
                </div>

                {/* Event Details */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-baseline justify-between">
                    <h4 className="font-extrabold text-xs text-[#15211D]">
                      {evt.title}
                    </h4>
                    <span className="text-[11px] font-mono text-[#929E99] tabular-nums">
                      {timeFormatted}
                    </span>
                  </div>

                  <p className="text-[#68756F] text-[11px] mt-0.5 leading-relaxed">
                    {evt.description}
                  </p>

                  <div className="mt-2 flex items-center space-x-2 text-[10px] text-[#929E99]">
                    <span className="flex items-center gap-1 font-semibold text-[#15211D] bg-black/5 px-2 py-0.5 rounded-md">
                      <User className="w-3 h-3 text-[#205A46]" />
                      {evt.actor}
                    </span>
                    <span>·</span>
                    <span className="uppercase tracking-wider font-medium">
                      Category: {evt.category.replace('_', ' ')}
                    </span>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
