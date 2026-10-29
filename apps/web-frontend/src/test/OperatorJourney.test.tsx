import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from '../App';

describe('Operator Journey Full Workflow Integration', () => {
  it('executes the full dispatcher journey with selection state preserved across views', async () => {
    render(<App />);

    // 1. App loads with scenario name
    await waitFor(() => {
      const titles = screen.getAllByText(/Bengaluru Urban Express/i);
      expect(titles.length).toBeGreaterThan(0);
    });

    // 2. Default vehicle KA-03 is active; inspect its at-risk stop
    const zeptoStop = screen.getByText(/Zepto Dark Store #8/i);
    expect(zeptoStop).toBeInTheDocument();
    fireEvent.click(zeptoStop);

    // 3. Verify stop drawer opens with SLA context and simulated interval
    await waitFor(() => {
      expect(screen.getByText(/Contractual SLA/i)).toBeInTheDocument();
      expect(screen.getByText(/simulated interval/i)).toBeInTheDocument();
      expect(screen.getByText(/assumed rate ₹83\/\$/i)).toBeInTheDocument();
    });

    // 4. Select vehicle KA-04 from the map legend
    const vanRouteButton = screen.getByRole('button', { name: /KA-04/i });
    expect(vanRouteButton).toBeInTheDocument();
    fireEvent.click(vanRouteButton);

    // 5. Navigate to Route Planning (Dispatch View)
    const routePlanNav = screen.getByRole('button', { name: /Route Planning/i });
    fireEvent.click(routePlanNav);

    await waitFor(() => {
      expect(screen.getByText(/Optimization Solver Engine/i)).toBeInTheDocument();
      expect(screen.getByText(/PUBLISHED MANIFEST/i)).toBeInTheDocument();
    });

    // 6. Navigate to QML Intelligence / Evidence view
    const qmlNav = screen.getByRole('button', { name: /QML Intelligence/i });
    fireEvent.click(qmlNav);

    await waitFor(() => {
      expect(screen.getByText(/Quantum Machine Learning & Optimization Evidence/i)).toBeInTheDocument();
      expect(screen.getByText(/Platt Reliability Diagram/i)).toBeInTheDocument();
      expect(screen.getAllByText(/Brier Score: 0.142/i).length).toBeGreaterThan(0);
      expect(screen.getByText(/Parameterized Quantum Circuit \(VQC\)/i)).toBeInTheDocument();
    });

    // 7. Navigate to SLA & Exposure view
    const slaNav = screen.getByRole('button', { name: /SLA & Exposure/i });
    fireEvent.click(slaNav);

    await waitFor(() => {
      expect(screen.getByText(/SLA Violation Risk & Contractual Exposure/i)).toBeInTheDocument();
    });

    // 8. Click Inspect on an at-risk stop to return to Command Center
    const inspectButtons = screen.getAllByRole('button', { name: /Inspect/i });
    expect(inspectButtons.length).toBeGreaterThan(0);
    fireEvent.click(inspectButtons[0]);

    // 9. Verify we returned to Command Center and Stop Drawer is active with selection preserved
    await waitFor(() => {
      expect(screen.getByText(/Contractual SLA/i)).toBeInTheDocument();
      expect(screen.getByText(/assumed rate ₹83\/\$/i)).toBeInTheDocument();
    });

    // 10. Switch to Fleet Operations view
    const fleetNav = screen.getByRole('button', { name: /Fleet Operations/i });
    fireEvent.click(fleetNav);

    await waitFor(() => {
      expect(screen.getByText(/Fleet Operations & Telemetry/i)).toBeInTheDocument();
    });

    // Select second vehicle and locate on map
    const secondVehRow = screen.getByText(/KA-04-MB-4819/i);
    fireEvent.click(secondVehRow);

    const locateBtn = screen.getByRole('button', { name: /Locate on Command Center Map/i });
    fireEvent.click(locateBtn);

    // Verify back in Command Center with KA-04-MB-4819 focused
    await waitFor(() => {
      expect(screen.getByText(/KA-04-MB-4819/i)).toBeInTheDocument();
    });
  });
});
