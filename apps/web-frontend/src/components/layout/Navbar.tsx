import React, { useState, useEffect } from 'react';
import { HealthPill } from '../ui/HealthPill';
import { SystemHealth, PlanRun } from '../../types/api';
import { Layers, Bell, Search, Sparkles } from 'lucide-react';

interface NavbarProps {
  currentPlan: PlanRun;
  allPlans: PlanRun[];
  onSelectPlan: (planId: string) => void;
  health: SystemHealth | null;
  onRefreshHealth: () => void;
  searchQuery: string;
  onSearchChange: (q: string) => void;
  activeJobsCount: number;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentPlan,
  allPlans,
  onSelectPlan,
  health,
  onRefreshHealth,
  searchQuery,
  onSearchChange,
  activeJobsCount
}) => {
  const [timeStr, setTimeStr] = useState<string>('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeStr(
        now.toLocaleTimeString('en-IN', {
          timeZone: 'Asia/Kolkata',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
          hour12: false
        }) + ' IST'
      );
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="h-16 px-6 bg-white/90 backdrop-blur-md border-b border-[#EAE6DF] flex items-center justify-between z-30 select-none">
      {/* Left: Brand + Date Widget + Scenario Switcher */}
      <div className="flex items-center space-x-5">
        <div className="flex items-baseline space-x-2">
          <span className="text-2xl font-black tracking-tight text-[#141F1A]">
            <span className="text-[#1E5A44]">ar</span>Q
          </span>
          <span className="text-[10px] uppercase font-mono font-bold tracking-widest text-[#1E5A44] bg-[#E8F5EE] px-2.5 py-0.5 rounded-full border border-[#A8DBC9]">
            DISPATCH OS · QML-01
          </span>
        </div>

        {/* Atlas Date Pill */}
        <div className="hidden lg:flex items-center space-x-2 pl-3 border-l border-black/5 text-xs">
          <div className="w-7 h-7 rounded-full bg-[#141F1A] text-white flex items-center justify-center font-bold text-[11px] tabular-nums">
            29
          </div>
          <div className="leading-tight">
            <div className="font-bold text-[11px] text-[#141F1A]">Wed, October</div>
            <div className="text-[10px] text-[#7D8D85] tabular-nums">{timeStr || '14:40:00 IST'}</div>
          </div>
        </div>

        {/* Scenario Switcher Dropdown */}
        <div className="relative flex items-center space-x-2 bg-[#F5F3ED] border border-[#E5E0D8] rounded-full px-3 py-1.5 text-xs shadow-xs">
          <Layers className="w-3.5 h-3.5 text-[#1E5A44]" />
          <select
            value={currentPlan.id}
            onChange={(e) => onSelectPlan(e.target.value)}
            className="bg-transparent font-medium text-[#141F1A] focus:outline-none cursor-pointer pr-1"
          >
            {allPlans.map((p) => (
              <option key={p.id} value={p.id}>
                {p.scenario_name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Center Search Input */}
      <div className="flex-1 max-w-md mx-6 hidden md:block">
        <div className="relative">
          <Search className="w-3.5 h-3.5 text-[#7D8D85] absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search vehicles, stops, waypoints, SLA deadlines…"
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            className="w-full pl-9 pr-10 py-1.5 text-xs rounded-full bg-[#F5F3ED]/80 border border-[#E5E0D8] text-[#141F1A] placeholder-[#7D8D85] focus:outline-none focus:bg-white focus:ring-2 focus:ring-[#1E5A44]/20 focus:border-[#1E5A44] transition-all"
          />
          <kbd className="absolute right-3.5 top-1/2 -translate-y-1/2 text-[9px] font-mono text-[#7D8D85] bg-black/5 px-1.5 py-0.5 rounded border border-black/5">
            /
          </kbd>
        </div>
      </div>

      {/* Right Controls: Active Jobs, Health, Notifications & Profile */}
      <div className="flex items-center space-x-3">
        {/* Solver Jobs Activity Trigger */}
        {activeJobsCount > 0 && (
          <div className="flex items-center space-x-1 px-3 py-1 bg-[#EFF4FF] border border-[#C7D7FE] rounded-full text-xs text-[#2563EB] font-semibold animate-pulse">
            <Sparkles className="w-3.5 h-3.5" />
            <span>{activeJobsCount} Active Job</span>
          </div>
        )}

        {/* Backend & Solver Health Pill */}
        <HealthPill health={health} onRefresh={onRefreshHealth} />

        {/* Notification Bell */}
        <button
          className="relative w-8 h-8 rounded-full bg-[#F5F3ED] hover:bg-black/5 border border-[#E5E0D8] flex items-center justify-center text-[#141F1A] transition-all cursor-pointer"
          title="Audit Notifications"
        >
          <Bell className="w-3.5 h-3.5 text-[#141F1A]" />
          <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full bg-[#DC2626]" />
        </button>

        {/* Dispatcher Profile (Atlas Logist Style) */}
        <div className="hidden sm:flex items-center space-x-2 pl-2 border-l border-black/5">
          <div className="w-8 h-8 rounded-full bg-[#1E5A44] text-[#A8DBC9] font-bold text-xs flex items-center justify-center border border-[#141F1A]/10 shadow-xs">
            AR
          </div>
          <div className="leading-tight hidden xl:block">
            <div className="font-bold text-xs text-[#141F1A]">Aditya R.</div>
            <div className="text-[10px] text-[#7D8D85] font-medium">Lead Dispatcher</div>
          </div>
        </div>
      </div>
    </header>
  );
};
