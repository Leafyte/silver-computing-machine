import { useState, useEffect } from 'react';
import { PlanRun, SolverJob, SystemHealth, OperationalEvent } from './types/api';
import { apiClient } from './api/client';
import { Navbar } from './components/layout/Navbar';
import { SidebarRail, NavView } from './components/layout/SidebarRail';
import { CommandCenterView } from './features/dashboard/CommandCenterView';
import { FleetOperationsView } from './features/fleet/FleetOperationsView';
import { DispatchPlanningView } from './features/dispatch/DispatchPlanningView';
import { SlaRiskView } from './features/risk/SlaRiskView';
import { ActivityView } from './features/activity/ActivityView';
import { JobsView } from './features/jobs/JobsView';
import { LoginView } from './features/auth/LoginView';
import { CheckCircle2, AlertTriangle, X } from 'lucide-react';

import {
  BENGALURU_SCENARIO,
  SOLOMON_SCENARIO,
  DEFAULT_HEALTH
} from './data/mockScenarios';

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(() => apiClient.hasSession());
  const [plans, setPlans] = useState<PlanRun[]>([BENGALURU_SCENARIO, SOLOMON_SCENARIO]);
  const [activePlanId, setActivePlanId] = useState<string>('');
  const [currentView, setCurrentView] = useState<NavView>('command');
  const [health, setHealth] = useState<SystemHealth>(DEFAULT_HEALTH);
  const [jobs, setJobs] = useState<SolverJob[]>([]);
  const [events, setEvents] = useState<OperationalEvent[]>([]);
  const [auditError, setAuditError] = useState<string | null>(null);
  const [auditLastSyncedAt, setAuditLastSyncedAt] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedVehicleId, setSelectedVehicleId] = useState<string | null>(null);
  const [selectedStopId, setSelectedStopId] = useState<string | null>(null);
  const [notification, setNotification] = useState<{ message: string; type: 'success' | 'warning' } | null>(null);

  const refreshAuditEvents = async () => {
    try {
      setEvents(await apiClient.getOperationalEvents());
      setAuditError(null);
      setAuditLastSyncedAt(new Date().toISOString());
    } catch (error) {
      setAuditError(error instanceof Error ? error.message : 'Driver audit feed could not be loaded.');
    }
  };

  // Initial load
  useEffect(() => {
    if (!isAuthenticated) return;
    async function loadData() {
      const initialPlans = await apiClient.getRuns();
      setPlans(initialPlans);
      if (initialPlans.length > 0) {
        const newestBackendRun = initialPlans.find((plan) => plan.data_source === 'backend');
        const selectedPlan = newestBackendRun ?? initialPlans[0];
        setActivePlanId(selectedPlan.id);
        setSelectedVehicleId(selectedPlan.vehicles[0]?.id || null);
      }

      const initialHealth = await apiClient.getHealth();
      setHealth(initialHealth);

      const initialJobs = await apiClient.getJobs();
      setJobs(initialJobs);

      await refreshAuditEvents();
    }
    loadData();
  }, [isAuthenticated]);

  // Periodic polling for jobs, health, and events
  useEffect(() => {
    if (!isAuthenticated) return;
    const interval = setInterval(async () => {
      const updatedJobs = await apiClient.getJobs();
      setJobs(updatedJobs);

      const updatedHealth = await apiClient.getHealth();
      setHealth(updatedHealth);

      await refreshAuditEvents();
      const refreshedPlans = await apiClient.getRuns();
      setPlans(refreshedPlans);
      const newestBackendRun = refreshedPlans.find((plan) => plan.data_source === 'backend');
      if (newestBackendRun) {
        setActivePlanId((prev) => refreshedPlans.some((plan) => plan.id === prev && plan.data_source === 'backend') ? prev : newestBackendRun.id);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [isAuthenticated]);

  const activePlan = plans.find((p) => p.id === activePlanId) || plans[0];

  const handlePlanSelect = (id: string) => {
    setActivePlanId(id);
    const chosenPlan = plans.find((p) => p.id === id);
    if (chosenPlan && chosenPlan.vehicles.length > 0) {
      setSelectedVehicleId(chosenPlan.vehicles[0].id);
    } else {
      setSelectedVehicleId(null);
    }
    setSelectedStopId(null);
  };

  const handleTriggerReplan = async (stopId: string, vehicleId: string) => {
    if (!activePlan) return;
    const res = await apiClient.executeRecoveryReplan({
      plan: activePlan,
      vehicle_id: vehicleId,
      delayed_stop_id: stopId,
      delay_units: 1,
      locked_stop_ids: activePlan.routes
        .find((r) => r.vehicle_id === vehicleId)
        ?.stops.filter((s) => s.is_completed || s.is_locked)
        .map((s) => s.id) || []
    });

    if (res.success) {
      if (res.recovered_plan) {
        setPlans((prev) => [res.recovered_plan!, ...prev.filter((plan) => plan.id !== res.recovered_plan!.id)]);
        setActivePlanId(res.recovered_plan.id);
      }
      setNotification({
        message: `Recovery validated. Objective change: ${res.objective_delta_units.toFixed(2)} units · ${res.audit_log}`,
        type: 'success'
      });
      await refreshAuditEvents();
    } else {
      setNotification({ message: res.audit_log, type: 'warning' });
    }
  };

  const runningJobsCount = jobs.filter((j) => j.status === 'running' || j.status === 'queued').length;

  if (!isAuthenticated) return <LoginView onAuthenticated={() => setIsAuthenticated(true)} />;

  return (
    <div className="flex flex-col h-screen w-screen bg-[#F5F3ED] text-[#141F1A] font-sans antialiased overflow-hidden">
      {/* Global Toast Notification */}
      {notification && (
        <div className="fixed top-18 right-6 z-50 glass-panel-strong px-4 py-3 rounded-2xl shadow-xl border border-black/5 flex items-center space-x-3 text-xs animate-in slide-in-from-top-2 duration-200">
          {notification.type === 'success' ? (
            <CheckCircle2 className="w-4 h-4 text-[#1E5A44] shrink-0" />
          ) : (
            <AlertTriangle className="w-4 h-4 text-[#D97706] shrink-0" />
          )}
          <span className="font-semibold text-[#141F1A] max-w-sm">
            {notification.message}
          </span>
          <button
            onClick={() => setNotification(null)}
            className="p-1 rounded-lg hover:bg-black/5 text-[#7D8D85]"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Top Navigation Bar */}
      {activePlan && (
        <Navbar
          currentPlan={activePlan}
          allPlans={plans}
          onSelectPlan={handlePlanSelect}
          health={health}
          onRefreshHealth={async () => {
            const h = await apiClient.getHealth();
            setHealth(h);
          }}
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          activeJobsCount={runningJobsCount}
          onLogout={() => { apiClient.logout(); setIsAuthenticated(false); }}
        />
      )}

      {/* Body: Sidebar Rail + View Area */}
      <div className="flex-1 flex overflow-hidden">
        <SidebarRail
          currentView={currentView}
          onSelectView={setCurrentView}
          activeJobsCount={runningJobsCount}
        />

        {/* View Switcher Container */}
        <main className="flex-1 flex flex-col overflow-hidden relative">
          {activePlan ? (
            <>
              {currentView === 'command' && (
                <CommandCenterView
                  plan={activePlan}
                  selectedVehicleId={selectedVehicleId}
                  selectedStopId={selectedStopId}
                  onSelectVehicle={setSelectedVehicleId}
                  onSelectStop={setSelectedStopId}
                  searchQuery={searchQuery}
                  onTriggerReplan={handleTriggerReplan}
                />
              )}

              {currentView === 'fleet' && (
                <FleetOperationsView
                  plan={activePlan}
                  selectedVehicleId={selectedVehicleId}
                  searchQuery={searchQuery}
                  onSelectVehicleForMap={(vId) => {
                    setSelectedVehicleId(vId);
                    setSelectedStopId(null);
                    setCurrentView('command');
                  }}
                />
              )}

              {currentView === 'dispatch' && (
                <DispatchPlanningView
                  plan={activePlan}
                  onSelectRoute={(vehicleId) => {
                    setSelectedVehicleId(vehicleId);
                    setSelectedStopId(null);
                    setCurrentView('command');
                  }}
                  onJobTriggered={async () => {
                    const updatedJobs = await apiClient.getJobs();
                    setJobs(updatedJobs);
                  }}
                />
              )}

              {currentView === 'risk' && (
                <SlaRiskView
                  plan={activePlan}
                  onInspectStop={(stopId, vehicleId) => {
                    setSelectedVehicleId(vehicleId);
                    setSelectedStopId(stopId);
                    setCurrentView('command');
                  }}
                />
              )}

              {currentView === 'activity' && (
                <ActivityView
                  events={events}
                  syncError={auditError}
                  lastSyncedAt={auditLastSyncedAt}
                  onRefresh={refreshAuditEvents}
                />
              )}

              {currentView === 'jobs' && (
                <JobsView
                  jobs={jobs}
                  health={health}
                  onRefreshJobs={async () => {
                    const j = await apiClient.getJobs();
                    setJobs(j);
                  }}
                />
              )}
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center text-xs text-[#68756F]">
              Loading arQ logistics scenario…
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
