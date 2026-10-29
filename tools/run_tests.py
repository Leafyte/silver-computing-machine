#!/usr/bin/env python3
"""DebugLeaf - Focused Test Runner for Submission Verification.

Runs the isolated QML and evidence verification tests without unnecessary full-suite overhead.
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

_root = Path(__file__).resolve().parent.parent

FOCUSED_TESTS = [
    "tests/test_isolated_qml.py",
    "tests/test_execution_evidence.py",
    "tests/test_judge_evidence.py",
    "tests/test_domain.py",
]


def main() -> int:
    print(f"Running DebugLeaf focused test suite on {len(FOCUSED_TESTS)} test modules...")
    # Prefer python interpreter from virtualenv if available
    python_bin = sys.executable
    venv_python = _root / ".venv" / "bin" / "python"
    if venv_python.is_file():
        python_bin = str(venv_python)

    cmd = [python_bin, "-m", "pytest", "-q"] + FOCUSED_TESTS
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(_root))
    if result.returncode == 0:
        print("\n[PASS] All focused tests passed successfully.")
    else:
        print(f"\n[FAIL] Test run exited with code {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
