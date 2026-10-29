import React from 'react';
import { IntelligenceEvidence } from '../../types/api';
import { DataBadge } from '../../components/ui/DataBadge';
import {
  BrainCircuit,
  BarChart3,
  CheckCircle2,
  Binary
} from 'lucide-react';

interface IntelligenceViewProps {
  evidence: IntelligenceEvidence;
}

export const IntelligenceView: React.FC<IntelligenceViewProps> = ({ evidence }) => {
  return (
    <div className="flex-1 flex flex-col overflow-y-auto p-6 gap-6 select-none text-xs">
      {/* Top Header Card */}
      <div className="bg-white rounded-3xl p-6 border border-[#EAE6DF] shadow-xs">
        <div className="flex flex-wrap items-center justify-between pb-4 border-b border-[#EAE6DF] gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <BrainCircuit className="w-5 h-5 text-[#1E5A44]" />
              <h2 className="text-xl font-extrabold text-[#141F1A] tracking-tight">
                Quantum Machine Learning & Optimization Evidence
              </h2>
            </div>
            <p className="text-xs text-[#7D8D85] mt-1">
              Fixture evidence from local research artifacts; it is not tied to the selected backend job.
            </p>
          </div>

          <div className="flex items-center space-x-2">
            <DataBadge badge="Preview" size="md" />
          </div>
        </div>

        {/* Claim Verification Callout Box */}
        <div className="mt-4 p-4 rounded-2xl bg-[#F5F3ED]/70 border border-[#E5E0D8] flex flex-wrap items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="text-[10px] uppercase font-bold text-[#7D8D85]">
              FIXTURE METRICS · NOT CURRENT JOB RESULTS
            </div>
            <div className="font-bold text-xs text-[#141F1A]">
              Brier Score: {evidence.brier_score} ({evidence.brier_reduction_pct}% relative error) · Regret: {evidence.regret_reduction_pct}% · 0.0% Fallback
            </div>
            <div className="text-[11px] text-[#4D5C54]">
              {evidence.user_claim_note}
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <span className="px-3.5 py-1 bg-[#E8F5EE] border border-[#A8DBC9] text-[#1E5A44] rounded-full font-bold text-[11px] flex items-center gap-1.5 shadow-xs">
              <CheckCircle2 className="w-3.5 h-3.5" />
              Preview evidence
            </span>
          </div>
        </div>
      </div>

      {/* Grid: Platt Calibration Curve (Left) + Solver Benchmarks (Right) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left: Calibration Curve Plot */}
        <div className="bg-white rounded-3xl p-6 border border-[#EAE6DF] shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-black/5">
              <div className="font-extrabold text-sm text-[#15211D] flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-[#205A46]" />
                Platt Reliability Diagram (Calibration)
              </div>
              <DataBadge badge="Preview" size="sm" />
            </div>

            <p className="text-[11px] text-[#68756F] mt-2">
              Predicted SLA breach probability vs empirical observed breach rate on held-out validation partitions.
            </p>

            {/* SVG Reliability Plot */}
            <div className="mt-4 w-full h-56 bg-white/70 rounded-2xl border border-black/5 p-3 flex items-center justify-center">
              <svg viewBox="0 0 100 100" className="w-full h-full select-none">
                {/* 45-degree Perfect Calibration Line */}
                <line
                  x1="10"
                  y1="90"
                  x2="90"
                  y2="10"
                  stroke="#D9E6DC"
                  strokeWidth="1.2"
                  strokeDasharray="2, 2"
                />

                {/* Empirical Calibration Curve */}
                <polyline
                  points={evidence.calibration_points
                    .map((pt) => `${10 + pt.predicted_prob * 80},${90 - pt.observed_freq * 80}`)
                    .join(' ')}
                  fill="none"
                  stroke="#205A46"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />

                {/* Data Points */}
                {evidence.calibration_points.map((pt, i) => (
                  <circle
                    key={i}
                    cx={10 + pt.predicted_prob * 80}
                    cy={90 - pt.observed_freq * 80}
                    r="2"
                    fill="#17231F"
                    stroke="#FFFFFF"
                    strokeWidth="0.8"
                  />
                ))}

                {/* Axis markers */}
                <text x="10" y="96" fontSize="3" fill="#929E99" textAnchor="middle">0.0</text>
                <text x="50" y="96" fontSize="3" fill="#929E99" textAnchor="middle">0.5</text>
                <text x="90" y="96" fontSize="3" fill="#929E99" textAnchor="middle">1.0</text>
                <text x="4" y="91" fontSize="3" fill="#929E99">0.0</text>
                <text x="4" y="52" fontSize="3" fill="#929E99">0.5</text>
                <text x="4" y="12" fontSize="3" fill="#929E99">1.0</text>
              </svg>
            </div>
          </div>

          <div className="mt-3 text-[11px] text-[#68756F] flex items-center justify-between">
            <span>Dashed line: Ideal calibration</span>
            <span className="font-semibold text-[#205A46]">Brier Score: {evidence.brier_score}</span>
          </div>
        </div>

        {/* Right: Quantum VQC Architecture & Circuit Details */}
        <div className="bg-white rounded-3xl p-6 border border-[#EAE6DF] shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-[#EAE6DF]">
              <div className="font-extrabold text-sm text-[#141F1A] flex items-center gap-2">
                <Binary className="w-4 h-4 text-[#8E3BD1]" />
                Parameterized Quantum Circuit (VQC)
              </div>
              <DataBadge badge={evidence.quantum_circuit_summary.execution_badge} size="sm" />
            </div>

            <div className="mt-4 space-y-3">
              <div className="p-3.5 bg-[#F5F3ED]/70 border border-[#E5E0D8] rounded-2xl flex justify-between items-center">
                <span className="text-[#7D8D85]">Circuit Topology:</span>
                <span className="font-bold text-[#141F1A]">
                  {evidence.quantum_circuit_summary.type}
                </span>
              </div>

              <div className="p-3.5 bg-[#F5F3ED]/70 border border-[#E5E0D8] rounded-2xl flex justify-between items-center">
                <span className="text-[#7D8D85]">Statevector Qubits:</span>
                <span className="font-bold text-[#141F1A] tabular-nums">
                  {evidence.quantum_circuit_summary.qubits} Qubits (Entangled Ry-Rz Ansätz)
                </span>
              </div>

              <div className="p-3.5 bg-[#F5F3ED]/70 border border-[#E5E0D8] rounded-2xl flex justify-between items-center">
                <span className="text-[#7D8D85]">Execution Runtime:</span>
                <span className="font-bold text-[#8E3BD1]">
                  {evidence.quantum_circuit_summary.backend_mode}
                </span>
              </div>

              <div className="p-3.5 bg-[#FEF3C7]/60 border border-[#FDE68A] rounded-2xl text-[11px] text-[#9A5B00] leading-relaxed">
                <strong>Execution Transparency:</strong> VQC evaluated on 2-qubit statevector simulator.
                Never implies QCloud physical QPU execution unless explicitly returned in job logs.
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-[#F5F3ED]/80 rounded-2xl text-[11px] text-[#7D8D85]">
            {evidence.disclaimer}
          </div>
        </div>
      </div>

      {/* Solver Benchmarks Table */}
      <div className="bg-white rounded-3xl p-6 border border-[#EAE6DF] shadow-xs">
        <div className="flex items-center justify-between pb-3 border-b border-[#EAE6DF]">
          <div>
            <h3 className="font-extrabold text-base text-[#141F1A]">
              Optimization Engine Benchmark Matrix
            </h3>
            <p className="text-[11px] text-[#7D8D85] mt-0.5">
              Fixture comparison across local solver artifacts. Values below are not measured for the selected backend job.
            </p>
          </div>
        </div>

        <div className="overflow-x-auto mt-3">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-[#EAE6DF] text-[#7D8D85] text-[10px] uppercase font-bold tracking-wider">
                <th className="pb-3 font-bold">Solver Engine</th>
                <th className="pb-3 font-bold">Provenance Badge</th>
                <th className="pb-3 font-bold">Gap to Optimum</th>
                <th className="pb-3 font-bold">Feasibility Rate</th>
                <th className="pb-3 font-bold">Execution Runtime</th>
                <th className="pb-3 font-bold">Objective Cost</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#EAE6DF]">
              {evidence.benchmarks.map((bm, idx) => (
                <tr key={idx} className="hover:bg-[#F5F3ED]/40 transition-colors">
                  <td className="py-4 pr-3 font-extrabold text-[#141F1A]">
                    {bm.engine}
                  </td>
                  <td className="py-4 pr-3">
                    <DataBadge badge={bm.provenance} size="sm" />
                  </td>
                  <td className="py-4 pr-3 tabular-nums font-bold text-[#141F1A]">
                    {bm.gap_to_optimum_pct === 0.0 ? '0.0% (optimal)' : `+${bm.gap_to_optimum_pct}%`}
                  </td>
                  <td className="py-4 pr-3 tabular-nums font-semibold text-[#1E5A44]">
                    {bm.feasibility_pct}%
                  </td>
                  <td className="py-4 pr-3 tabular-nums font-mono text-[#7D8D85]">
                    {bm.runtime_seconds}s
                  </td>
                  <td className="py-4 pr-3 tabular-nums font-bold text-[#141F1A]">
                    {bm.objective_cost}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
