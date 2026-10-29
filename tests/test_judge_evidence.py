"""Test judge evidence report generator."""

from pathlib import Path
import pytest
from tools.generate_judge_evidence import EvidenceBuilder

_root = Path(__file__).resolve().parent.parent


def test_generate_judge_evidence_builds_valid_report():
    builder = EvidenceBuilder(allow_cloud=False)
    report = builder.build_report()

    assert len(report.splitlines()) >= 200
    assert len(report.splitlines()) <= 500
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

    # Verify output file generation
    out_file = _root / "evidence" / "DEBUGLEAF_QPIAI_EXECUTION.txt"
    assert out_file.is_file()

