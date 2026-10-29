#!/usr/bin/env python3
"""Profile two-qubit CPU kernels on an explicitly synthetic numeric fixture.

This command is throughput-only. It does not train, compare ETA quality, select a
model, or stand in for the India dataset workload. Pass --synthetic-throughput-only
to make that scope explicit before the script will run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments_india_qml.quantum import (
    CircuitConfig,
    QuantumResidualRegressor,
    profile_cpu,
    profile_gradient_cpu,
    qpiai_expectations,
    validate_batched_forward,
)


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(512, dtype=np.int64)
    features = np.column_stack((
        ((index * 37) % 509) / 508.0,
        ((index * 73 + 19) % 509) / 508.0,
    ))
    targets = 8.0 * np.sin(2.3 * features[:, 0] - 1.1 * features[:, 1])
    targets += 1.75 * np.cos(4.1 * features[:, 1])
    return features, targets


def run(output: Path, repeats: int) -> dict[str, object]:
    features, residuals = _fixture()
    label = (
        "deterministic synthetic throughput fixture only; not India ETA data, "
        "not a trained model, no efficacy metric"
    )
    result: dict[str, object] = {
        "captured_note": (
            "CPU kernel profile only; no optimizer updates or model selection. "
            f"Each timed kernel is warmed once; {repeats} repeats per batch. "
            "This is not India ETA training throughput."
        ),
        "fixture": {
            "label": label,
            "rows": len(features),
            "features": ["synthetic_x0_scaled", "synthetic_x1_scaled"],
            "target_usage": "synthetic minutes-shaped values exercise only the MSE gradient kernel",
            "seed_formula": (
                "x0=((i*37)%509)/508; x1=((i*73+19)%509)/508; "
                "y=8*sin(2.3*x0-1.1*x1)+1.75*cos(4.1*x1)"
            ),
        },
        "host": {
            "python": os.sys.version,
            "logical_cpu_count": os.cpu_count(),
            "wall_clock": "perf_counter",
            "cpu_clock": "process_time; aggregate process CPU time",
        },
        "architectures": {},
    }
    architectures = result["architectures"]
    assert isinstance(architectures, dict)
    for architecture in ("one_block", "reupload"):
        model = QuantumResidualRegressor(
            CircuitConfig(
                architecture=architecture,
                feature_names=("synthetic_x0_scaled", "synthetic_x1_scaled"),
                residual_scale_minutes=8.0,
                seed=13,
            ),
            seed=13,
        )
        model.theta = np.linspace(-0.37, 0.41, model.theta.size)
        model.readout = np.asarray([0.29, -0.17, 0.08])
        batch32 = validate_batched_forward(model, features)
        gradient32 = model.gradient_check(features[:32], residuals[:32], h=1e-5)
        sdk = qpiai_expectations(features[:8], model.theta, architecture)
        local = model.forward_expectations(features[:8])

        # Warm both the batched forward and complete parameter-shift kernels.
        for batch_size in (128, 256, 512):
            model.forward_expectations(features[:batch_size])
            model._normalized_mse_gradient(
                features[:batch_size], residuals[:batch_size] / 8.0,
            )
        architectures[architecture] = {
            "parameter_count": model.parameter_count,
            "numerical_correctness_32": batch32,
            "gradient_check_32": gradient32,
            "qpiai_local_statevector_parity_8": {
                "max_abs_expectation_error": float(np.max(np.abs(sdk - local))),
                "execution": "QpiAI Quantum local StatevectorSimulator",
            },
            "cpu_forward_profile": profile_cpu(
                model, features, batch_sizes=(128, 256, 512), repeats=repeats,
            ),
            "cpu_forward_and_gradient_profile": profile_gradient_cpu(
                model, features, residuals,
                batch_sizes=(128, 256, 512), repeats=repeats,
                workload_label=label,
            ),
        }

    evidence_root = Path(__file__).resolve().parents[1] / "artifacts" / "run2_india"
    environment_path = evidence_root / "environment.json"
    if environment_path.is_file():
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        cuda_runtime = environment.get("runtime_packages", {}).get("cuda_runtime", {})
        numba_version = environment.get("runtime_packages", {}).get("distributions", {}).get("numba")
    else:
        cuda_runtime, numba_version = {}, None
    result["cuda_capability"] = {
        "nvidia_gpu_visible": True,
        "pytorch_available": False,
        "numba_version": numba_version,
        "numba_cuda_available": cuda_runtime.get("numba_cuda_available", False),
        "cuda_kernel_executed": False,
        "reason": "profiled Python runtime exposes no usable CUDA execution backend",
    }
    result["validation_units_regression_note"] = (
        "Validation RMSE compares predicted residual minutes directly to observed "
        "residual minutes; observed residuals are not rescaled a second time."
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-throughput-only", action="store_true",
        help="explicitly acknowledge this is not India ETA training evidence",
    )
    parser.add_argument("--repeats", type=int, default=256)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "run2_india" / "quantum_profile_synthetic.json",
    )
    args = parser.parse_args()
    if not args.synthetic_throughput_only:
        parser.error("pass --synthetic-throughput-only to run this numeric microbenchmark")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    result = run(args.output, args.repeats)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "evidence_label": result["fixture"]["label"],
        "architectures": sorted(result["architectures"]),
        "cuda_kernel_executed": result["cuda_capability"]["cuda_kernel_executed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
