import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DataBadge } from '../components/ui/DataBadge';
import { HealthPill } from '../components/ui/HealthPill';
import { StopDrawer } from '../components/stops/StopDrawer';
import { RouteMap } from '../components/map/RouteMap';
import { CommandCenterView } from '../features/dashboard/CommandCenterView';
import { FleetOperationsView } from '../features/fleet/FleetOperationsView';
import { SlaRiskView } from '../features/risk/SlaRiskView';
import { DispatchPlanningView } from '../features/dispatch/DispatchPlanningView';
import { BENGALURU_SCENARIO, SOLOMON_SCENARIO, DEFAULT_HEALTH } from '../data/mockScenarios';

describe('DataBadge UI Component', () => {
  it('renders Validated sentence-case badge', () => {
    render(<DataBadge badge="Validated" />);
    expect(screen.getByText('Validated')).toBeInTheDocument();
  });

  it('renders QpiAI Opt local solver badge', () => {
    render(<DataBadge badge="QpiAI Opt · local solver" />);
    expect(screen.getByText('QpiAI Opt · local solver')).toBeInTheDocument();
  });

  it('renders VQC local statevector badge', () => {
    render(<DataBadge badge="VQC · local statevector" />);
    expect(screen.getByText('VQC · local statevector')).toBeInTheDocument();
  });

  it('renders Simulated held-out run badge', () => {
    render(<DataBadge badge="Simulated · held-out run" />);
    expect(screen.getByText('Simulated · held-out run')).toBeInTheDocument();
  });
});

describe('HealthPill UI Component', () => {
  it('renders active solvers slot capacity', () => {
    render(<HealthPill health={DEFAULT_HEALTH} />);
    expect(screen.getByText(/2 Solvers/)).toBeInTheDocument();
  });
});

describe('StopDrawer Component', () => {
  it('renders stop commitment window and simulated ETA interval', () => {
    const stop = BENGALURU_SCENARIO.routes[0].stops[0];
    const vehicle = BENGALURU_SCENARIO.vehicles[0];

    render(
      <StopDrawer
        stop={stop}
        vehicle={vehicle}
        onClose={() => {}}
        onTriggerReplan={() => {}}
      />
    );

    expect(screen.getByText(stop.customer_name)).toBeInTheDocument();
    expect(screen.getByText(`${stop.window_start} – ${stop.window_end}`)).toBeInTheDocument();
    expect(screen.getByText(`${stop.eta_min} – ${stop.eta_max}`)).toBeInTheDocument();
    expect(screen.getByText(/simulated interval/i)).toBeInTheDocument();
    expect(screen.getByText(/assumed rate ₹83\/\$/i)).toBeInTheDocument();
  });

  it('renders replan action and triggers onTriggerReplan', () => {
    const atRiskStop = BENGALURU_SCENARIO.routes[0].stops.find((s) => !s.is_completed) || BENGALURU_SCENARIO.routes[0].stops[3];
    const vehicle = BENGALURU_SCENARIO.vehicles[0];
    const handleReplan = vi.fn();

    render(
      <StopDrawer
        stop={atRiskStop}
        vehicle={vehicle}
        onClose={() => {}}
        onTriggerReplan={handleReplan}
      />
    );

    const replanBtn = screen.getByRole('button', { name: /Trigger Mid-Shift Re-Plan/i });
    expect(replanBtn).toBeInTheDocument();
    fireEvent.click(replanBtn);
    expect(handleReplan).toHaveBeenCalledWith(atRiskStop.id, atRiskStop.vehicle_id);
  });
});

describe('RouteMap Component (SVG Cartesian Benchmark)', () => {
  it('renders flat metric space with depot (40,50) and stop sequence pins', () => {
    const handleSelectVehicle = vi.fn();
    const handleSelectStop = vi.fn();

    render(
      <RouteMap
        plan={SOLOMON_SCENARIO}
        selectedVehicleId={null}
        selectedStopId={null}
        onSelectVehicle={handleSelectVehicle}
        onSelectStop={handleSelectStop}
      />
    );

    expect(screen.getByText(/Solomon Euclidean Metric Space/i)).toBeInTheDocument();
    expect(screen.getByText(/Depot \(40, 50\)/i)).toBeInTheDocument();

    // Click depot or vehicle marker
    const svgDepot = screen.getByText('D');
    expect(svgDepot).toBeInTheDocument();
  });
});

describe('CommandCenterView Component', () => {
  it('renders operational metrics and ordered stop itinerary', () => {
    const handleReplan = vi.fn();
    const handleSelectStop = vi.fn();

    render(
      <CommandCenterView
        plan={BENGALURU_SCENARIO}
        selectedVehicleId={BENGALURU_SCENARIO.vehicles[0].id}
        selectedStopId={null}
        onSelectVehicle={() => {}}
        onSelectStop={handleSelectStop}
        onTriggerReplan={handleReplan}
      />
    );

    expect(screen.getByText(BENGALURU_SCENARIO.scenario_name)).toBeInTheDocument();
    expect(screen.getByText(/Fleet In-Field/i)).toBeInTheDocument();
    expect(screen.getByText(/SLA Exposure/i)).toBeInTheDocument();
    expect(screen.getByText(BENGALURU_SCENARIO.vehicles[0].callsign)).toBeInTheDocument();
  });

  it('filters ordered stops when searchQuery is supplied', () => {
    render(
      <CommandCenterView
        plan={BENGALURU_SCENARIO}
        selectedVehicleId={BENGALURU_SCENARIO.vehicles[0].id}
        selectedStopId={null}
        searchQuery="HealthKart"
        onTriggerReplan={() => {}}
      />
    );

    expect(screen.getByText(/Filtered by: "HealthKart"/i)).toBeInTheDocument();
    expect(screen.getByText(/HealthKart Fulfilment/i)).toBeInTheDocument();
  });
});

describe('FleetOperationsView Component', () => {
  it('filters vehicles by status and allows locating vehicle on command map', () => {
    const handleLocate = vi.fn();

    render(
      <FleetOperationsView
        plan={BENGALURU_SCENARIO}
        onSelectVehicleForMap={handleLocate}
      />
    );

    expect(screen.getByText(/Fleet Operations & Telemetry/i)).toBeInTheDocument();

    // Click vehicle row
    const firstVeh = screen.getByText(BENGALURU_SCENARIO.vehicles[0].callsign);
    fireEvent.click(firstVeh);

    // Click Locate on Map button in drawer
    const locateBtn = screen.getByRole('button', { name: /Locate on Command Center Map/i });
    expect(locateBtn).toBeInTheDocument();
    fireEvent.click(locateBtn);

    expect(handleLocate).toHaveBeenCalledWith(BENGALURU_SCENARIO.vehicles[0].id);
  });
});

describe('SlaRiskView Component', () => {
  it('renders route exposure cards and triggers stop inspection on map', () => {
    const handleInspect = vi.fn();

    render(
      <SlaRiskView
        plan={BENGALURU_SCENARIO}
        onInspectStop={handleInspect}
      />
    );

    expect(screen.getByText(/SLA Violation Risk & Contractual Exposure/i)).toBeInTheDocument();
    expect(screen.getByText(/Total Financial Exposure/i)).toBeInTheDocument();

    // Inspect buttons should exist for at-risk commitments
    const inspectButtons = screen.getAllByRole('button', { name: /Inspect/i });
    expect(inspectButtons.length).toBeGreaterThan(0);

    fireEvent.click(inspectButtons[0]);
    expect(handleInspect).toHaveBeenCalled();
  });
});

describe('DispatchPlanningView Component', () => {
  it('allows solver architecture selection and publishing candidate manifest', () => {
    const handleUpdate = vi.fn();

    render(
      <DispatchPlanningView
        plan={{ ...BENGALURU_SCENARIO, status: 'draft' }}
        onPlanUpdate={handleUpdate}
      />
    );

    expect(screen.getByText(/Optimization Solver Engine/i)).toBeInTheDocument();
    expect(screen.getByText(/DRAFT CANDIDATE/i)).toBeInTheDocument();

    // Publish Plan to Fleet
    const publishBtn = screen.getByRole('button', { name: /Publish Plan to Fleet/i });
    fireEvent.click(publishBtn);

    expect(handleUpdate).toHaveBeenCalledWith(
      expect.objectContaining({ status: 'active' })
    );
  });
});
