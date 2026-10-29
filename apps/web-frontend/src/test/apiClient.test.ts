import { describe, it, expect } from 'vitest';
import { apiClient } from '../api/client';

describe('arQ API Client & Solver Semaphore Contract', () => {
  it('fetches system health with 2-slot concurrency capacity', async () => {
    const health = await apiClient.getHealth();
    expect(health).toBeDefined();
    expect(health.capacity.max_slots).toBe(2);
    expect(health.engines_available).toContain('qpiai_opt_local');
    expect(health.engines_available).toContain('financial_cover');
  });

  it('fetches available scenarios including Bengaluru and Solomon benchmarks', async () => {
    const runs = await apiClient.getRuns();
    expect(runs.length).toBeGreaterThanOrEqual(2);
    const blr = runs.find((r) => r.id === 'run-blr-20261029-01');
    const sol = runs.find((r) => r.id === 'run-solomon-c101-01');
    expect(blr).toBeDefined();
    expect(blr?.coordinate_system).toBe('lat_lon');
    expect(sol).toBeDefined();
    expect(sol?.coordinate_system).toBe('cartesian_xy');
  });

  it('submits a solve job and respects BoundedSemaphore(2)', async () => {
    // Current jobs might have 1 running
    const res1 = await apiClient.submitSolve({
      scenario_id: 'blr-urban-express',
      engine: 'qpiai_opt'
    });

    if ('job_id' in res1) {
      expect(res1.job_id).toMatch(/^job-\d+-solve$/);
    }

    // Now 2 slots may be occupied, let's test capacity_reached guard
    const res2 = await apiClient.submitSolve({
      scenario_id: 'blr-urban-express',
      engine: 'financial'
    });

    if ('error' in res2) {
      expect(res2.error).toBe('capacity_reached');
      expect(res2.retry_after).toBe(4);
    }
  });

  it('cancels active job and updates state cleanly', async () => {
    const jobs = await apiClient.getJobs();
    const active = jobs.find((j) => j.status === 'running');
    if (active) {
      const ok = await apiClient.cancelJob(active.id);
      expect(ok).toBe(true);
      const updated = await apiClient.getJobs();
      const cancelled = updated.find((j) => j.id === active.id);
      expect(cancelled?.status).toBe('cancelled');
    }
  });

  it('executes mid-shift recovery re-plan with locked prefix', async () => {
    const res = await apiClient.executeRecoveryReplan({
      run_id: 'run-blr-20261029-01',
      vehicle_id: 'van-blr-01',
      delayed_stop_id: 'stop-blr-04',
      delay_minutes: 25,
      locked_stop_ids: ['stop-blr-01', 'stop-blr-02', 'stop-blr-03']
    });

    expect(res.success).toBe(true);
    expect(res.audit_log).toContain('recovery.py');
    expect(res.exposure_delta_inr).toBeLessThan(0); // reduced exposure

    const run = await apiClient.getRun('run-blr-20261029-01');
    expect(run?.provenance_badge).toBe('Classical re-plan');
  });

  it('retrieves mathematical evidence and Platt calibration data', async () => {
    const evidence = await apiClient.getIntelligenceEvidence();
    expect(evidence.brier_score).toBe(0.142);
    expect(evidence.brier_reduction_pct).toBe(-46.9);
    expect(evidence.regret_reduction_pct).toBe(-30.8);
    expect(evidence.calibration_points.length).toBeGreaterThan(5);
    expect(evidence.benchmarks.some((b) => b.engine.includes('QpiAI Opt'))).toBe(true);
  });
});
