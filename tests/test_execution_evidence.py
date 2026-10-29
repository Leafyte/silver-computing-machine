"""Test execution evidence report generation and format integrity."""

from pathlib import Path
import pytest
from tools.generate_judge_evidence import EvidenceBuilder

_root = Path(__file__).resolve().parent.parent


def test_generate_execution_evidence_produces_valid_reports():
    builder = EvidenceBuilder(allow_cloud=False)
    report = builder.build_report()

    assert len(report.splitlines()) >= 200
    assert "DEBUGLEAF - QPIAI QUANTUM & OPT EXECUTION EVIDENCE REPORT" in report
    assert "SECTION 1: ENVIRONMENT AND PROVENANCE" in report
    assert "SECTION 2: REAL CLASSICAL GBDT INFERENCE" in report
    assert "SECTION 3: FOUR-FEATURE QUANTUM FEATURE MAP" in report
    assert "SECTION 4: CUSTOM QUANTUM CIRCUIT EXECUTION" in report
    assert "SECTION 5: HYBRID ETA CALCULATION" in report
    assert "SECTION 6: DISPATCH-RISK 2-QUBIT DECISION VQC" in report
    assert "SECTION 7: QPIAI OPT COMBINATORIAL QUBO" in report
    assert "SECTION 8: QAOA EXECUTION VIA QPIAI OPT" in report
    assert "SECTION 9: LIVE FASTAPI DISPATCH -> QPIAI OPT EXECUTION TRACE" in report
    assert "SECTION 10: SDK INTEGRATION AND QCLOUD PROVENANCE" in report
    assert "SECTION 11: FINAL VERIFICATION SUMMARY" in report
    assert "Overall Evaluation Verdict:  [PASS]" in report

    # Verify both evidence files exist
    live_file = _root / "evidence" / "DEBUGLEAF_LIVE_EXECUTION.txt"
    qpiai_file = _root / "evidence" / "DEBUGLEAF_QPIAI_EXECUTION.txt"
    assert live_file.is_file(), f"Missing {live_file}"
    assert qpiai_file.is_file(), f"Missing {qpiai_file}"
