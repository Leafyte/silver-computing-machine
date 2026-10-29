#!/usr/bin/env python3
"""DebugLeaf - Execution Evidence Generator for Quant-A-Maze 3.0.

Provides the CLI interface for generating official judge/live execution evidence,
preserving the complete verified QpiAI quantum, GBDT, VQC, and QAOA execution pipeline.
Outputs to evidence/DEBUGLEAF_LIVE_EXECUTION.txt and evidence/DEBUGLEAF_QPIAI_EXECUTION.txt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tools.generate_judge_evidence import EvidenceBuilder


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Quant-A-Maze 3.0 execution evidence for DebugLeaf."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live verification against local QpiAI / FastAPI pipeline.",
    )
    parser.add_argument(
        "--allow-cloud",
        action="store_true",
        help="Enable optional remote QCloud execution (default: local simulator).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_root / "evidence" / "DEBUGLEAF_LIVE_EXECUTION.txt"),
        help="Path for output evidence report.",
    )
    args = parser.parse_args()

    builder = EvidenceBuilder(allow_cloud=args.allow_cloud)
    report_text = builder.build_report()

    # Write primary target
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")

    # Also sync DEBUGLEAF_QPIAI_EXECUTION.txt
    qpiai_out = _root / "evidence" / "DEBUGLEAF_QPIAI_EXECUTION.txt"
    qpiai_out.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\n[OK] Evidence successfully written to: {out_path} and {qpiai_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
