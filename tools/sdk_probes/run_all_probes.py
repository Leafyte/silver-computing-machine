#!/usr/bin/env python3
"""
tools/sdk_probes/run_all_probes.py

Unified SDK Capability Probe Runner for DebugLeaf.
Orchestrates:
  - QpiAI Quantum probe (probe_qpiai_quantum.py)
  - QpiAI Opt probe (probe_qpiai_opt.py)
  - Google OR-Tools probe (probe_ortools.py)

Separates:
  1. Import Availability
  2. Circuit / Model Construction
  3. Local Execution (Simulator / Offline Solver)
  4. Authenticated Cloud Status

SAFETY ASSURANCE:
- Never executes cloud jobs or heavy workloads.
- Never dumps or prints credentials.
- Handles missing packages gracefully with clear diagnostic notes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Support relative import regardless of invocation cwd
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from probe_ortools import probe_ortools
from probe_qpiai_opt import probe_opt
from probe_qpiai_quantum import probe_quantum


def run_all() -> Dict[str, Any]:
    # Probe ortools before highspy/qpiai_opt to prevent dylib symbol collision on macOS
    return {
        "ortools": probe_ortools(),
        "qpiai_quantum": probe_quantum(),
        "qpiai_opt": probe_opt(),
    }


def format_table(results: Dict[str, Any]) -> str:
    lines = [
        "# DebugLeaf SDK Capability Scorecard",
        "",
        "| SDK Component | Stage 1: Import | Stage 2: Construction | Stage 3: Local Execution | Stage 4: Cloud Status |",
        "|---|---|---|---|---|",
    ]

    for name, r in results.items():
        s1 = "PASS" if r["stage_1_import"]["success"] else "FAIL"
        if r["stage_1_import"].get("version"):
            s1 += f" ({r['stage_1_import']['version']})"

        s2 = "PASS" if r["stage_2_construction"]["success"] else ("FAIL" if r["stage_1_import"]["success"] else "SKIPPED")
        
        # Local execution status
        if r["stage_3_local_execution"]["success"]:
            s3 = "PASS (Local)"
        elif not r["stage_1_import"]["success"]:
            s3 = "SKIPPED"
        elif r["stage_3_local_execution"].get("status") == "REQUIRES_OPT_KEY":
            s3 = "REQUIRES_KEY"
        else:
            s3 = "FAIL"

        # Cloud status
        if name == "ortools":
            s4 = "LOCAL_ONLY"
        elif r.get("stage_4_cloud_status", {}).get("credentials_configured"):
            s4 = "CONFIGURED (Deferred)"
        else:
            s4 = "NO_KEY (Expected)"

        lines.append(f"| `{r['sdk']}` | {s1} | {s2} | {s3} | {s4} |")

    lines.extend([
        "",
        "## Detailed Diagnostics",
        "",
        "### 1. QpiAI Quantum",
        f"- Import: {r_summary(results['qpiai_quantum']['stage_1_import'])}",
        f"- Construction: {r_summary(results['qpiai_quantum']['stage_2_construction'])}",
        f"- Local Execution: {r_summary(results['qpiai_quantum']['stage_3_local_execution'])}",
        f"- Cloud: {results['qpiai_quantum'].get('stage_4_cloud_status', {}).get('action', 'N/A')}",
        "",
        "### 2. QpiAI Opt",
        f"- Import: {r_summary(results['qpiai_opt']['stage_1_import'])}",
        f"- Construction: {r_summary(results['qpiai_opt']['stage_2_construction'])}",
        f"- Local Execution: {r_summary(results['qpiai_opt']['stage_3_local_execution'])}",
        f"- Cloud / Platform: {results['qpiai_opt'].get('stage_4_cloud_status', {}).get('action', 'N/A')}",
        "",
        "### 3. Google OR-Tools",
        f"- Import: {r_summary(results['ortools']['stage_1_import'])}",
        f"- Construction: {r_summary(results['ortools']['stage_2_construction'])}",
        f"- Local Execution: {r_summary(results['ortools']['stage_3_local_execution'])}",
        f"- Cloud / Platform: {results['ortools'].get('stage_4_cloud_status', {}).get('action', 'N/A')}",
        "",
    ])

    return "\n".join(lines)


def r_summary(stage: Dict[str, Any]) -> str:
    if stage.get("note"):
        return stage["note"]
    if stage.get("error"):
        return f"Error: {stage['error']}"
    if stage.get("success"):
        return "Success"
    return "Not executed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all SDK capability probes.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output.")
    args = parser.parse_args()

    results = run_all()
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_table(results))

    return 0


if __name__ == "__main__":
    sys.exit(main())
