#!/usr/bin/env python3
"""Capture bounded local environment evidence for RUN2 India QML.

This profiler does not install packages, load serialized models, access credentials,
run a quantum workload, or benchmark substitute data. Its default report is written
to artifacts/run2_india/environment.json in the current trainingv3 checkout.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".joblib", ".pkl", ".pickle", ".onnx",
    ".model", ".ubj", ".npz", ".npy", ".safetensors", ".json",
}
ARTIFACT_DIR_NAMES = {"artifacts", "models", "checkpoints", "run2_india"}
EXCLUDED_PARTS = {".git", ".venv", "venv", "site-packages", ".agents"}


def _run(command: list[str], timeout: int = 15) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return {"available": False, "error": type(exc).__name__}
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _venv_target(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        result["is_symlink"] = path.is_symlink()
        result["is_junction"] = path.is_junction() if hasattr(path, "is_junction") else None
        try:
            result["resolved_path"] = str(path.resolve(strict=True))
        except OSError as exc:
            result["resolve_error"] = type(exc).__name__
    return result


def _versions() -> dict[str, Any]:
    names = (
        "torch", "pennylane", "qpiai-quantum", "qpiai-opt", "numpy", "pandas",
        "scikit-learn", "pytest", "xgboost", "lightgbm", "numba", "cupy",
        "jax", "jaxlib", "tensorflow", "cuda-python",
    )
    packages: dict[str, str | None] = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    modules: dict[str, Any] = {}
    for name in ("torch", "pennylane", "qpiai_quantum", "qpiai_opt", "numba", "cupy", "jax", "tensorflow"):
        try:
            spec = importlib.util.find_spec(name)
            modules[name] = spec.origin if spec else None
        except (ImportError, ValueError):
            modules[name] = None
    imports: dict[str, str] = {}
    for name in ("qpiai_quantum", "qpiai_opt"):
        try:
            __import__(name)
            imports[name] = "imported"
        except Exception as exc:  # preserve safe type only, never error contents
            imports[name] = f"unavailable: {type(exc).__name__}"
    torch: dict[str, Any] | None = None
    try:
        import torch as torch_module

        torch = {
            "version": torch_module.__version__,
            "cuda_built": torch_module.backends.cuda.is_built(),
            "cuda_available": torch_module.cuda.is_available(),
            "cuda_runtime": torch_module.version.cuda,
            "device_count": torch_module.cuda.device_count(),
            "devices": [
                torch_module.cuda.get_device_name(i)
                for i in range(torch_module.cuda.device_count())
            ],
        }
    except Exception as exc:
        torch = {"available": False, "error_type": type(exc).__name__}
    cuda: dict[str, Any] = {"available": False, "frameworks": ["PyTorch", "CuPy", "Numba CUDA", "JAX", "TensorFlow"]}
    try:
        from numba import cuda as numba_cuda

        cuda["numba_cuda_available"] = bool(numba_cuda.is_available())
        cuda["devices"] = [str(device) for device in numba_cuda.gpus] if cuda["numba_cuda_available"] else []
        cuda["available"] = cuda["numba_cuda_available"]
    except Exception as exc:
        # Keep only the bounded error text needed to diagnose policy/runtime loading.
        cuda["numba_cuda_probe_error_type"] = type(exc).__name__
        cuda["numba_cuda_probe_error"] = str(exc)[:300]
    return {
        "distributions": packages,
        "modules": modules,
        "imports": imports,
        "torch_runtime": torch,
        "cuda_runtime": cuda,
    }


def _artifact_inventory(roots: list[Path]) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        try:
            candidates = root.rglob("*")
            for path in candidates:
                if not path.is_file() or path.suffix.lower() not in MODEL_SUFFIXES:
                    continue
                if any(part.lower() in EXCLUDED_PARTS for part in path.parts):
                    continue
                stem = path.stem.lower()
                if path.suffix.lower() == ".json" and not any(
                    term in stem for term in (
                        "model", "checkpoint", "predictor", "gbdt", "eta", "india",
                        "booster", "lgbm", "catboost", "xgb",
                    )
                ):
                    continue
                if not any(part.lower() in ARTIFACT_DIR_NAMES for part in path.parts):
                    # Only inspect files with India/ETA/model provenance in their path.
                    if not any(term in str(path).lower() for term in ("india", "eta", "delivery", "checkpoint")):
                        continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                found[str(path.resolve())] = {"path": str(path.resolve()), "size_bytes": stat.st_size}
        except OSError:
            continue
    return [found[key] for key in sorted(found)]


def collect(original_checkout: Path) -> dict[str, Any]:
    checkout = original_checkout.resolve()
    root = Path(__file__).resolve().parents[1]
    original_venv = checkout / ".venv"
    venv_python = original_venv / "Scripts" / "python.exe"
    env = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_checkout": str(root),
        "original_checkout": str(checkout),
        "host": {
            "os": platform.platform(),
            "architecture": platform.machine(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "processor": platform.processor(),
        },
        "original_runtime_paths": {
            "venv": _venv_target(original_venv),
            "venv_python": _venv_target(venv_python),
            "runtime": _venv_target(checkout / ".runtime"),
        },
        "runtime_packages": _versions(),
        "gpu": {
            "nvidia_smi": _run(["nvidia-smi"]),
            "compute_processes": _run([
                "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]),
        },
        "candidate_model_artifacts": _artifact_inventory([
            root / "artifacts", root / "data", root / "models", root / "checkpoints",
            checkout / "artifacts", checkout / "data", checkout / "models",
            checkout / "checkpoints", checkout / "experiments", checkout / ".runtime" / "run2_india",
        ]),
        "performance_profile": {
            "status": "not_run",
            "reason": "The frozen Indian ETA checkpoint is not present among the searched model artifact paths; benchmarking an invented replacement workload is not valid.",
            "requested_batch_sizes": [128, 256, 512],
            "requested_correctness_batch": 32,
            "requested_throughput_batch": 512,
            "cpu_gpu_comparison": "unavailable until frozen model and its actual two-qubit workload are supplied",
        },
        "execution_boundaries": {
            "packages_installed": False,
            "serialized_models_loaded": False,
            "quantum_workload_run": False,
            "qpu_or_cloud_jobs_submitted": False,
            "credentials_read_or_recorded": False,
        },
    }
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-checkout", type=Path,
        default=Path(__file__).resolve().parents[2] / "silver-computing-machine",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "run2_india" / "environment.json",
    )
    args = parser.parse_args()
    result = collect(args.original_checkout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "gpu_available": result["gpu"]["nvidia_smi"]["available"],
        "artifact_candidates": len(result["candidate_model_artifacts"]),
        "performance_profile": result["performance_profile"]["status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
