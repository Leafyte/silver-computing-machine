import React, { useEffect, useState } from 'react';
import { PlanRun } from '../../types/api';
import { apiClient } from '../../api/client';
import { DispatchScenario, ScenarioValidation, VqcPlanResult } from '../../api/client';
import { DataBadge } from '../../components/ui/DataBadge';
import { Play, Sliders, ShieldCheck, Eye } from 'lucide-react';

interface DispatchPlanningViewProps {
  plan: PlanRun;
  onSelectRoute?: (vehicleId: string) => void;
  onJobTriggered?: () => void;
}

export const DispatchPlanningView: React.FC<DispatchPlanningViewProps> = ({
  plan,
  onSelectRoute,
  onJobTriggered
}) => {
  const [selectedEngine, setSelectedEngine] = useState<string>('exact');
  const [shots, setShots] = useState<number>(128);
  const [angleGrid, setAngleGrid] = useState<number>(4);
  const [isSolving, setIsSolving] = useState<boolean>(false);
  const [solveMessage, setSolveMessage] = useState<string | null>(null);
  const [scenarioText, setScenarioText] = useState('Loading bundled scenario…');
  const [scenario, setScenario] = useState<DispatchScenario | null>(null);
  const [scenarioValidation, setScenarioValidation] = useState<ScenarioValidation | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [vqcResult, setVqcResult] = useState<VqcPlanResult | null>(null);
  const [dispatcherToken, setDispatcherToken] = useState('');
  const [publishedPlanId, setPublishedPlanId] = useState<string | null>(null);
  const [isPublishing, setIsPublishing] = useState(false);
  const isPublished = plan.data_source === 'backend' && (plan.status === 'active' || publishedPlanId !== null);

  useEffect(() => {
    let mounted = true;
    void apiClient.getExampleScenario().then((example) => {
      if (!mounted) return;
      setScenario(example);
      setScenarioText(JSON.stringify(example, null, 2));
    }).catch(() => {
      if (mounted) setScenarioText('Could not load the bundled scenario. Paste a dispatch scenario JSON object here.');
    });
    return () => { mounted = false; };
  }, []);

  const parseScenario = (): DispatchScenario | null => {
    try {
      const parsed = JSON.parse(scenarioText) as DispatchScenario;
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('Scenario must be a JSON object.');
      return parsed;
    } catch {
      setScenarioValidation({ valid: false, errors: ['Scenario JSON is invalid.'], execution_type: 'not_run' });
      return null;
    }
  };

  const handleValidateScenario = async () => {
    const parsed = parseScenario();
    setScenario(parsed);
    setVqcResult(null);
    if (!parsed) return;
    setIsValidating(true);
    try {
      const result = await apiClient.validateScenario(parsed);
      setScenarioValidation(result);
      setSolveMessage(result.valid ? 'Scenario passed backend validation.' : `Scenario rejected: ${result.errors.join(', ')}`);
    } catch (error) {
      setScenarioValidation(null);
      setSolveMessage(error instanceof Error ? error.message : 'Scenario validation request failed.');
    } finally {
      setIsValidating(false);
    }
  };

  const handleActivateManifest = async () => {
    if (!dispatcherToken.trim()) {
      setSolveMessage('Enter a dispatcher/admin token. The backend must be running in token auth mode.');
      return;
    }
    setIsPublishing(true);
    try {
      const response = await apiClient.registerSyntheticManifest(plan, dispatcherToken.trim());
      setPublishedPlanId(typeof response.plan_id === 'string' ? response.plan_id : null);
      setDispatcherToken('');
      setSolveMessage(`Synthetic manifest ${response.plan_id ?? ''} activated at version ${response.version ?? 1}. It is separate from live vehicle dispatch.`);
    } catch (error) {
      setSolveMessage(error instanceof Error ? error.message : 'Manifest activation failed.');
    } finally {
      setIsPublishing(false);
    }
  };

  const handleRunSolve = async () => {
    setIsSolving(true);
    setVqcResult(null);
    setSolveMessage('Submitting optimization request to dispatch pipeline…');

    const activeScenario = scenario ?? parseScenario();
    if (!activeScenario) {
      setIsSolving(false);
      return;
    }
    const validation = scenarioValidation?.valid ? scenarioValidation : await apiClient.validateScenario(activeScenario).catch(() => null);
    if (!validation?.valid) {
      setScenarioValidation(validation);
      setSolveMessage('Validate a supported scenario before running a solver.');
      setIsSolving(false);
      return;
    }

    if (selectedEngine === 'vqc') {
      const result = await apiClient.runVqc(activeScenario);
      if ('error' in result) {
        setSolveMessage(`QpiAI local VQC request failed (${result.error}). Check SDK availability and backend logs.`);
      } else {
        setVqcResult(result);
        setSolveMessage(`Synchronous backend result · ${result.execution_type} · independently validated: ${result.validation?.valid ? 'yes' : 'no'}.`);
      }
      setIsSolving(false);
      return;
    }

    const res = await apiClient.submitSolve({
      scenario_id: plan.scenario_id,
      scenario: activeScenario,
      engine: selectedEngine,
      shots,
      angle_grid: angleGrid,
    });

    if ('error' in res) {
      const detail = res.error === 'capacity_reached'
        ? `Solver is busy. Retry in ${res.retry_after ?? 2} seconds.`
        : res.error === 'backend_unreachable'
          ? `Can't reach backend at ${apiClient.getBaseUrl()}.`
          : `The backend could not submit this run (${res.error}).`;
      setSolveMessage(detail);
      setIsSolving(false);
      return;
    }

    if ('job_id' in res) {
      setSolveMessage(`Backend job ${res.job_id} submitted. Its status is available in Jobs.`);
      if (onJobTriggered) onJobTriggered();
    }

    setTimeout(() => {
      setIsSolving(false);
    }, 2000);
  };

  return (
    <div className="flex-1 flex overflow-hidden p-6 gap-6 select-none text-xs">
      {/* Left: Solver Engine & Parameter Configuration */}
      <div className="w-96 bg-white rounded-3xl p-6 border border-[#EAE6DF] shadow-xs flex flex-col justify-between shrink-0">
        <div>
          <div className="pb-3 border-b border-[#EAE6DF]">
            <div className="flex items-center space-x-2">
              <Sliders className="w-4 h-4 text-[#1E5A44]" />
              <h2 className="text-base font-extrabold text-[#141F1A]">
                Optimization Solver Engine
              </h2>
            </div>
          </div>

          <div className="mt-4 space-y-4">
            {/* Engine Selection */}
            <div>
              <label className="text-[10px] uppercase font-bold text-[#7D8D85] block mb-2">
                Target Solver Architecture
              </label>
              <div className="space-y-2">
                {[
                  { id: 'exact', label: 'Exact Enumeration', desc: 'Bounded local CPU job with independent validation' },
                  { id: 'qaoa', label: 'Warm-Start QAOA', desc: 'Local Python statevector job with exact validation' },
                  { id: 'vqc', label: 'QpiAI VQC · local simulator', desc: 'Synchronous QpiAI Quantum inference; no cloud job or API key' }
                ].map((eng) => (
                  <button
                    key={eng.id}
                    onClick={() => setSelectedEngine(eng.id)}
                    className={`w-full p-3 rounded-2xl border text-left transition-all cursor-pointer ${
                      selectedEngine === eng.id
                        ? 'bg-[#141F1A] text-white border-transparent shadow-xs'
                        : 'bg-[#F5F3ED]/70 border-[#E5E0D8] text-[#141F1A] hover:bg-[#F5F3ED]'
                    }`}
                  >
                    <div className="font-bold text-xs">{eng.label}</div>
                    <div className={`text-[10px] mt-0.5 ${selectedEngine === eng.id ? 'text-white/70' : 'text-[#7D8D85]'}`}>
                      {eng.desc}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Bounded QAOA controls */}
            <div className="space-y-3 pt-2">
              <label className="block">
                <span className="flex justify-between text-[11px] font-semibold text-[#141F1A] mb-1">
                  <span>QAOA shots</span><span className="font-mono text-[#1E5A44]">{shots}</span>
                </span>
                <input type="range" min="32" max="256" step="32" value={shots}
                  onChange={(e) => setShots(Number(e.target.value))}
                  disabled={selectedEngine !== 'qaoa'} className="w-full accent-[#1E5A44] disabled:opacity-40" />
              </label>
              <label className="block">
                <span className="flex justify-between text-[11px] font-semibold text-[#141F1A] mb-1">
                  <span>Angle grid</span><span className="font-mono text-[#1E5A44]">{angleGrid}</span>
                </span>
                <input type="range" min="2" max="8" step="1" value={angleGrid}
                  onChange={(e) => setAngleGrid(Number(e.target.value))}
                  disabled={selectedEngine !== 'qaoa'} className="w-full accent-[#1E5A44] disabled:opacity-40" />
              </label>
              <p className="text-[10px] leading-relaxed text-[#7D8D85]">The API does not report percentage progress. Jobs show the server’s actual state.</p>
            </div>

            <div className="pt-3 border-t border-[#EAE6DF] space-y-2">
              <div className="flex items-center justify-between">
                <label htmlFor="dispatch-scenario" className="text-[10px] uppercase font-bold text-[#7D8D85]">Scenario JSON · validated by backend</label>
                <button type="button" onClick={() => { void apiClient.getExampleScenario().then((example) => { setScenario(example); setScenarioText(JSON.stringify(example, null, 2)); setScenarioValidation(null); }); }} className="text-[10px] font-bold text-[#1E5A44] hover:underline">Load example</button>
              </div>
              <textarea
                id="dispatch-scenario"
                value={scenarioText}
                onChange={(event) => { setScenarioText(event.target.value); setScenario(null); setScenarioValidation(null); setVqcResult(null); }}
                spellCheck={false}
                className="w-full h-32 resize-y rounded-xl border border-[#E5E0D8] bg-[#FBFAF7] p-2.5 font-mono text-[10px] leading-relaxed text-[#29352F] outline-none focus:border-[#1E5A44]"
                aria-label="Dispatch scenario JSON"
              />
              <div className="flex items-center justify-between gap-2">
                <button type="button" onClick={handleValidateScenario} disabled={isValidating} className="px-3 py-1.5 bg-white border border-[#D9E6DC] rounded-full font-bold text-[10px] text-[#1E5A44] disabled:opacity-50">{isValidating ? 'Validating…' : 'Validate scenario'}</button>
                {scenarioValidation && <span className={`text-[10px] font-bold ${scenarioValidation.valid ? 'text-[#1E5A44]' : 'text-[#C6534C]'}`}>{scenarioValidation.valid ? 'Valid' : scenarioValidation.errors.join(', ')}</span>}
              </div>
            </div>

            {/* Status message */}
            {solveMessage && (
              <div className="p-3 bg-[#F5F3ED] border border-[#E5E0D8] rounded-2xl text-[11px] text-[#141F1A] leading-relaxed">
                {solveMessage}
              </div>
            )}
          </div>
        </div>

        {/* Submit Solver Run Action */}
        <div className="pt-4 border-t border-[#EAE6DF]">
          <button
            onClick={handleRunSolve}
            disabled={isSolving}
            className="w-full py-3 bg-[#141F1A] hover:bg-[#1E5A44] text-white rounded-full font-bold transition-all flex items-center justify-center space-x-2 shadow-xs cursor-pointer disabled:opacity-50"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>{isSolving ? 'Dispatching Solver…' : 'Dispatch Optimization Run'}</span>
          </button>
        </div>
      </div>

      {/* Right: Candidate Plan Review & Manifest Publishing */}
      <div className="flex-1 bg-white rounded-3xl p-6 border border-[#EAE6DF] shadow-xs flex flex-col justify-between overflow-hidden">
        <div>
          {/* Top Plan Status & Publication Bar */}
          <div className="flex items-center justify-between pb-4 border-b border-[#EAE6DF]">
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-lg font-extrabold text-[#141F1A]">
                  {plan.scenario_name}
                </span>
                <span
                  className={`px-3 py-0.5 rounded-full text-[10px] font-bold ${
                    isPublished ? 'bg-[#E8F5EE] text-[#1E5A44]' : 'bg-[#FEF3C7] text-[#D97706]'
                  }`}
                >
                  {isPublished ? 'PUBLISHED MANIFEST' : plan.data_source === 'backend' ? 'VALIDATED DRAFT' : 'PREVIEW ONLY'}
                </span>
              </div>
              <div className="text-[#7D8D85] text-[11px] mt-1 flex items-center space-x-2">
                <DataBadge badge={plan.provenance_badge} size="sm" />
                <span>·</span>
                <span className="tabular-nums">Total stops: {plan.total_stops}</span>
                <span>·</span>
                <span className="tabular-nums">{plan.data_source === 'backend' ? plan.assumed_rate_label : 'Preview data'}</span>
              </div>
            </div>

            {/* Publish Manifest CTA */}
            {!isPublished ? (
              plan.data_source === 'backend' && plan.backend_payload && plan.validator_result.passed ? (
                <div className="flex items-center gap-2">
                  <input type="password" autoComplete="off" value={dispatcherToken} onChange={(event) => setDispatcherToken(event.target.value)} placeholder="Dispatcher token" aria-label="Dispatcher token for manifest activation" className="w-36 px-3 py-2 border border-[#E5E0D8] rounded-full text-[10px] outline-none focus:border-[#1E5A44]" />
                  <button onClick={handleActivateManifest} disabled={isPublishing} title="Activates a synthetic ETA manifest after backend validation." className="px-4 py-2.5 bg-[#141F1A] hover:bg-[#1E5A44] text-white rounded-full font-bold flex items-center gap-2 disabled:opacity-50">
                    <ShieldCheck className="w-3.5 h-3.5" /><span>{isPublishing ? 'Activating…' : 'Activate synthetic manifest'}</span>
                  </button>
                </div>
              ) : <button disabled title="Only a validated backend run can be registered as a manifest." className="px-5 py-2.5 bg-[#E5E0D8] text-[#7D8D85] rounded-full font-bold flex items-center space-x-2 cursor-not-allowed"><ShieldCheck className="w-3.5 h-3.5" /><span>Preview cannot activate</span></button>
            ) : (
              <div className="flex items-center space-x-1.5 text-[#1E5A44] font-bold">
                <ShieldCheck className="w-4 h-4" />
                <span>Manifest Active</span>
              </div>
            )}
          </div>

          {vqcResult && (
            <div className="mt-4 p-4 rounded-2xl border border-[#D9E6DC] bg-[#F4F8F4] text-[11px] space-y-1.5">
              <div className="font-extrabold text-[#141F1A]">QpiAI VQC response</div>
              <div>Execution: <strong>{vqcResult.execution_type}</strong> · Model: <strong>{vqcResult.model?.execution_backend ?? 'not reported'}</strong></div>
              <div>Routes returned: <strong>{vqcResult.plan?.routes?.length ?? 0}</strong> · Validation: <strong>{vqcResult.validation?.valid ? 'passed' : 'not passed'}</strong></div>
              <div className="text-[#7D8D85]">Caller-supplied model parameters; this result is synchronous and is not stored in job history.</div>
            </div>
          )}

          {/* Connected backend capabilities */}
          <div className="mt-4 p-4 bg-[#F5F3ED]/60 border border-[#E5E0D8] rounded-2xl">
            <div className="flex items-center justify-between pb-2 mb-2 border-b border-[#EAE6DF]">
              <span className="text-[10px] uppercase font-mono font-bold tracking-wider text-[#7D8D85]">
                Connected execution paths
              </span>
              <span className="text-[10px] text-[#7D8D85] font-mono">Live response for this request appears in Jobs or below</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px]">
              <div className="p-2.5 bg-white rounded-xl border border-[#EAE6DF]">
                <span className="font-bold text-[#141F1A] block">Exact enumeration</span>
                <span className="text-[#7D8D85]">Durable local CPU job · independently validated</span>
              </div>
              <div className="p-2.5 bg-white rounded-xl border border-[#EAE6DF]">
                <span className="font-bold text-[#141F1A] block">Warm-start QAOA</span>
                <span className="text-[#7D8D85]">Durable local Python statevector job · exact validation</span>
              </div>
              <div className="p-2.5 bg-white rounded-xl border border-[#EAE6DF]">
                <span className="font-bold text-[#141F1A] block">QpiAI VQC</span>
                <span className="text-[#7D8D85]">Synchronous local SDK inference · no cloud job</span>
              </div>
              <div className="p-2.5 bg-white rounded-xl border border-[#EAE6DF]">
                <span className="font-bold text-[#141F1A] block">QpiAI Cloud / Opt Cloud</span>
                <span className="text-[#7D8D85]">Not connected to this dashboard job flow</span>
              </div>
            </div>
          </div>

          {/* Assigned Candidate Routes Breakdown */}
          <div className="mt-4 overflow-y-auto max-h-[calc(100vh-380px)] space-y-3 pr-2">
            {plan.routes.map((route, i) => {
              const v = plan.vehicles.find((veh) => veh.id === route.vehicle_id);

              return (
                <div
                  key={route.id}
                  className="p-4 bg-[#F5F3ED]/40 border border-[#E5E0D8] rounded-2xl shadow-xs space-y-2.5"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2.5">
                      <span
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: route.color_token }}
                      />
                      <span className="font-bold text-sm text-[#141F1A]">
                        Route #{i + 1} — {v?.callsign}
                      </span>
                      <span className="text-[11px] text-[#7D8D85]">
                        ({v?.driver_name})
                      </span>
                    </div>

                    <div className="flex items-center space-x-3">
                      <div className="text-right tabular-nums">
                        <span className="font-bold text-[#141F1A]">
                          {route.total_distance_km} km
                        </span>
                        <span className="text-[#7D8D85] ml-1.5">
                          · {route.total_duration_minutes} min
                        </span>
                      </div>

                      {onSelectRoute && (
                        <button
                          onClick={() => onSelectRoute(route.vehicle_id)}
                          className="px-3 py-1.5 bg-white hover:bg-[#141F1A] hover:text-white border border-[#E5E0D8] rounded-full font-bold text-[10px] flex items-center gap-1 transition-all cursor-pointer shadow-xs"
                        >
                          <Eye className="w-3 h-3" />
                          <span>View on Map</span>
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Horizontal Sequence Ribbon */}
                  <div className="flex items-center space-x-1.5 overflow-x-auto py-1">
                    {route.stops.map((stop, sIdx) => (
                      <React.Fragment key={stop.id}>
                        <div
                          className={`px-2.5 py-1 rounded-xl font-medium shrink-0 flex items-center space-x-1 text-[11px] tabular-nums ${
                            stop.is_completed
                              ? 'bg-[#E8F5EE] text-[#1E5A44]'
                              : stop.status === 'at_risk'
                              ? 'bg-[#FEF3C7] text-[#DC2626] border border-[#FDE68A]'
                              : 'bg-white border border-[#E5E0D8] text-[#141F1A]'
                          }`}
                        >
                          <span className="font-bold">#{stop.sequence_idx}</span>
                          <span className="truncate max-w-[100px]">{stop.customer_name}</span>
                        </div>
                        {sIdx < route.stops.length - 1 && (
                          <span className="text-[#7D8D85] font-bold">→</span>
                        )}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
