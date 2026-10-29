import React from 'react';
import {
  Compass,
  Truck,
  Route as RouteIcon,
  ShieldAlert,
  BrainCircuit,
  Activity,
  Cpu,
  Sun,
  Moon
} from 'lucide-react';

export type NavView =
  | 'command'
  | 'fleet'
  | 'dispatch'
  | 'risk'
  | 'intelligence'
  | 'activity'
  | 'jobs';

interface SidebarRailProps {
  currentView: NavView;
  onSelectView: (view: NavView) => void;
  activeJobsCount?: number;
}

export const SidebarRail: React.FC<SidebarRailProps> = ({
  currentView,
  onSelectView,
  activeJobsCount = 0
}) => {
  const [isDarkMode, setIsDarkMode] = React.useState(false);

  const navItems: { id: NavView; label: string; icon: React.ReactNode; badge?: number }[] = [
    {
      id: 'command',
      label: 'Command Center',
      icon: <Compass className="w-4 h-4" />
    },
    {
      id: 'fleet',
      label: 'Fleet Operations',
      icon: <Truck className="w-4 h-4" />
    },
    {
      id: 'dispatch',
      label: 'Route Planning',
      icon: <RouteIcon className="w-4 h-4" />
    },
    {
      id: 'risk',
      label: 'SLA & Exposure',
      icon: <ShieldAlert className="w-4 h-4" />
    },
    {
      id: 'intelligence',
      label: 'QML Intelligence',
      icon: <BrainCircuit className="w-4 h-4" />
    },
    {
      id: 'activity',
      label: 'Audit Events',
      icon: <Activity className="w-4 h-4" />
    },
    {
      id: 'jobs',
      label: 'Solver Jobs',
      icon: <Cpu className="w-4 h-4" />,
      badge: activeJobsCount > 0 ? activeJobsCount : undefined
    }
  ];

  return (
    <aside className="w-18 md:w-56 h-[calc(100vh-4rem)] bg-white/70 backdrop-blur-md border-r border-[#EAE6DF] flex flex-col justify-between p-3.5 select-none z-20 shrink-0">
      <div className="space-y-1">
        <div className="hidden md:block px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-[#7D8D85]">
          Overview
        </div>

        {navItems.map((item) => {
          const isActive = currentView === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectView(item.id)}
              className={`w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-2xl text-xs font-semibold transition-all cursor-pointer ${
                isActive
                  ? 'bg-[#141F1A] text-white shadow-xs'
                  : 'text-[#4D5C54] hover:bg-black/5 hover:text-[#141F1A]'
              }`}
              title={item.label}
            >
              <span className={isActive ? 'text-[#A8DBC9]' : 'text-[#7D8D85]'}>
                {item.icon}
              </span>
              <span className="hidden md:inline-block truncate flex-1 text-left font-medium">
                {item.label}
              </span>
              {item.badge !== undefined && (
                <span className="hidden md:inline-flex items-center justify-center px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-[#2563EB] text-white">
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Bottom Area: Engine status & Atlas Light/Dark toggle pill */}
      <div className="space-y-3">
        {/* Engine status pill */}
        <div className="hidden md:block p-3 rounded-2xl bg-[#F5F3ED] border border-[#E5E0D8] text-xs text-[#4D5C54]">
          <div className="text-[10px] font-bold uppercase tracking-wider text-[#7D8D85] mb-1">
            Engine Standard
          </div>
          <div className="text-[#141F1A] font-semibold flex items-center justify-between text-[11px]">
            <span>QpiAI Opt · QUBO</span>
            <span className="w-1.5 h-1.5 rounded-full bg-[#1E5A44]" />
          </div>
          <div className="text-[10px] text-[#7D8D85] mt-0.5">
            2-Qubit VQC · Statevector
          </div>
        </div>

        {/* Atlas Bottom Theme Switch Pill (ref_01_atlas_dashboard.png) */}
        <div className="flex items-center justify-center md:justify-start">
          <div className="flex items-center p-1 bg-[#F5F3ED] border border-[#E5E0D8] rounded-full">
            <button
              onClick={() => setIsDarkMode(false)}
              className={`p-1.5 rounded-full transition-all cursor-pointer ${
                !isDarkMode ? 'bg-white shadow-xs text-[#141F1A]' : 'text-[#7D8D85] hover:text-[#141F1A]'
              }`}
              title="Light Mode"
            >
              <Sun className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setIsDarkMode(true)}
              className={`p-1.5 rounded-full transition-all cursor-pointer ${
                isDarkMode ? 'bg-[#141F1A] text-white shadow-xs' : 'text-[#7D8D85] hover:text-[#141F1A]'
              }`}
              title="Dark Mode (Simulated)"
            >
              <Moon className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </aside>
  );
};
