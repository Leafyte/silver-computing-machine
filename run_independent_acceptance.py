"""Executable Script to Run Independent Classical vs Quantum Acceptance Benchmark.

Produces:
- benchmark_results.json
- runtime_breakdown.json
- Prints structured comparison tables and acceptance status.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from benchmark_harness.evaluator import MasterBenchmarkEvaluator


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    evaluator = MasterBenchmarkEvaluator(repo_root)

    print("================================================================================")
    print("DEBUGLEAF — INDEPENDENT CLASSICAL VS QUANTUM ACCEPTANCE BENCHMARK")
    print("================================================================================")

    benchmark_results, runtime_breakdown = evaluator.run_full_acceptance_audit()

    # Write deliverables to repo root
    out_results = repo_root / "benchmark_results.json"
    out_runtime = repo_root / "runtime_breakdown.json"

    out_results.write_text(json.dumps(benchmark_results, indent=2) + "\n", encoding="utf-8")
    out_runtime.write_text(json.dumps(runtime_breakdown, indent=2) + "\n", encoding="utf-8")

    # Also write copies to parent workspace root for accessibility
    parent_root = repo_root.parent
    if parent_root.is_dir() and parent_root != repo_root:
        try:
            (parent_root / "benchmark_results.json").write_text(
                json.dumps(benchmark_results, indent=2) + "\n", encoding="utf-8"
            )
            (parent_root / "runtime_breakdown.json").write_text(
                json.dumps(runtime_breakdown, indent=2) + "\n", encoding="utf-8"
            )
        except Exception:
            pass

    print(f"\n[OUTPUT] Written benchmark_results.json -> {out_results}")
    print(f"[OUTPUT] Written runtime_breakdown.json -> {out_runtime}")

    # Summary table
    v1_1 = benchmark_results["v1_1_benchmark"]
    pipes = v1_1["pipelines_evaluated"]

    print("\n" + "=" * 115)
    print("TABLE 1: 7-WAY HELD-OUT COMPARATOR (N = 30 Test Scenarios, 5 Replicates)")
    print("=" * 115)
    headers = ["Pipe", "Total ($)", "Trans ($)", "Labor ($)", "SLA ($)", "Regret ± 95%CI ($)", "Valid", "RawSucc", "Fallback", "QAOA Feas", "Gap ($)", "CPU (s)"]
    print(f"{headers[0]:<26} | {headers[1]:<9} | {headers[2]:<9} | {headers[3]:<9} | {headers[4]:<7} | {headers[5]:<20} | {headers[6]:<5} | {headers[7]:<7} | {headers[8]:<8} | {headers[9]:<9} | {headers[10]:<7} | {headers[11]:<7}")
    print("-" * 145)

    for pid, row in pipes.items():
        regret_str = f"{row['mean_decision_regret_usd']:.4f} +/- {row['regret_95ci_halfwidth_usd']:.4f}"
        valid_pct = f"{row['valid_plan_rate']*100:.1f}%"
        raw_pct = f"{row['raw_solver_success_rate']*100:.1f}%"
        fall_pct = f"{row['fallback_rate']*100:.1f}%"
        qaoa_feas = f"{row['mean_qaoa_shot_feasibility']*100:.2f}%" if row['mean_shots'] > 0 else "N/A"
        gap_str = f"{row['mean_subproblem_gap_usd']:.4f}"
        cpu_str = f"{row['mean_decision_cpu_seconds']:.5f}"

        print(f"{pid:<26} | {row['total_financial_cost_usd']:<9.4f} | {row['transport_cost_usd']:<9.4f} | {row['labor_cost_usd']:<9.4f} | {row['contractual_penalties_usd']:<7.4f} | {regret_str:<20} | {valid_pct:<5} | {raw_pct:<7} | {fall_pct:<8} | {qaoa_feas:<9} | {gap_str:<7} | {cpu_str:<7}")

    print("=" * 145)

    # Calibration table
    calib = v1_1["calibration_audit"]
    print("\n" + "=" * 80)
    print("TABLE 2: HELD-OUT PROBABILITY CALIBRATION (420 Route Scenarios)")
    print("=" * 80)
    print(f"{'Predictor':<32} | {'Brier (lower)':<14} | {'ECE (lower)':<12} | {'Status'}")
    print("-" * 80)
    for name, c in calib.items():
        print(f"{name:<32} | {c['brier_score']:<14.5f} | {c['ece']:<12.5f} | {c['status']}")
    print("=" * 80)

    # Provenance Summary
    prov = benchmark_results["dataset_provenance"]
    print("\n" + "=" * 80)
    print("TABLE 3: SOLOMON & DABIA DATASET PROVENANCE AUDIT")
    print("=" * 80)
    print(f"Solomon instances evaluated: {prov['solomon_instance_count']}")
    print(f"Dabia instances evaluated:   {prov['dabia_instance_count']}")
    print(f"Overlapping base topologies: {prov['overlapping_topology_count']} / {prov['solomon_instance_count']} (100% overlap)")
    print(f"Finding: {prov['conclusion']}")
    print("=" * 80)

    # RUN 1 Gate Status
    run1 = benchmark_results["run1_audit"]
    print("\n" + "=" * 80)
    print("TABLE 4: RUN 1 IMPLEMENTATION & GATE STATUS")
    print("=" * 80)
    print(f"MAMUT BKS Checker & Native Traces: {run1['native_benchmarks_verification']}")
    print(f"Solomon-Dabia Topology Audit:     {run1['solomon_dabia_topology_audit']}")
    print(f"Pilot VQC Gradient Verification:   {run1['pilot_gradient_verification']}")
    print(f"Pilot QpiAI SDK Statevector Parity:{run1['pilot_qpiai_sdk_parity']}")
    print(f"Full Held-Out Test Evaluation:     {run1['full_heldout_training_evaluation']}")
    print(f"Blocker / Failure Detail: {run1['blocker_reason']}")
    print("=" * 80)

    if run1.get("full_run_record"):
        fr = run1["full_run_record"]
        tm = fr.get("test_metrics_by_model", {})
        regret = fr.get("same_replay_candidate_oracle_regret", {})
        qaoa = fr.get("qaoa_summary", {})
        print("\n" + "=" * 80)
        print("TABLE 5: RUN 1 HELD-OUT GENERALIZATION & DECISION REGRET (Held-out C1, N=1532)")
        print("=" * 80)
        print(f"{'Model':<26} | {'Brier (lower)':<14} | {'ECE (lower)':<12} | {'LogLoss':<10} | {'Regret (SCU)':<12}")
        print("-" * 80)
        for m in sorted(tm.keys()):
            brier_v = f"{tm[m]['brier']:.4f}" if tm[m].get("brier") is not None else "N/A"
            ece_v = f"{tm[m]['ece']:.4f}" if tm[m].get("ece") is not None else "N/A"
            ll_v = f"{tm[m]['log_loss']:.4f}" if tm[m].get("log_loss") is not None else "N/A"
            reg_v = f"{regret.get(m, 0.0):.4f}" if m in regret else "N/A"
            print(f"{m:<26} | {brier_v:<14} | {ece_v:<12} | {ll_v:<10} | {reg_v:<12}")
        print("=" * 80)
        print(f"QAOA Dispatch: Plan Valid={qaoa.get('selected_plan_valid')}, Feasible={qaoa.get('selected_plan_feasible')}, Fallback Used={qaoa.get('fallback_used')}")
        print(f"QAOA Quality:  Gap={qaoa.get('optimality_gap')*100:.3f}%, Approx Ratio={qaoa.get('approximation_ratio')*100:.3f}%, Feasible Shot Prob={qaoa.get('feasible_sample_probability')*100:.2f}%")
        print(f"Decision Cost: Realized={qaoa.get('realized_decision_cost'):.3f} SCU, Realized Regret={qaoa.get('realized_decision_regret'):.3f} SCU")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("FINAL ACCEPTANCE DECISION:")
    print(f"  Criterion 8 (Classical vs Quantum Evaluation): {benchmark_results['acceptance_criteria']['overall_gate_decision']}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
