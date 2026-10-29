"""Portable, NumPy-only inference and integrity checks for RUN2 releases.

Training may use PyTorch/autograd. The shipped residual circuit needs only
NumPy for CPU inference, so a Dell installation need not carry the training
stack. Classical estimator loading is intentionally separate and must happen
only after its artifact digest has been verified.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .independent_acceptance import sha256_file, verify_manifest_hashes


CHECKPOINT_FORMAT = "india_eta_torch_residual_v2"
FEATURES_2 = ("vqc_x0_distance", "vqc_x1_congestion")
FEATURES_4 = (*FEATURES_2, "vqc_x2_multidelivery", "vqc_x3_preptime")


def _checked_features(values: Any, n_features: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[1] != n_features or len(x) == 0:
        raise ValueError(f"expected a non-empty (n,{n_features}) scaled feature matrix")
    if not np.isfinite(x).all() or np.any(x < -1e-12) or np.any(x > 1 + 1e-12):
        raise ValueError("scaled VQC features must be finite and lie in [0,1]")
    return np.clip(x, 0.0, 1.0)


def _ry(states: np.ndarray, wire: int, angle: np.ndarray | float) -> np.ndarray:
    angle = np.asarray(angle, dtype=np.float64)
    if angle.ndim == 0:
        angle = np.full(len(states), float(angle), dtype=np.float64)
    c, s = np.cos(angle / 2)[:, None], np.sin(angle / 2)[:, None]
    if wire == 0:
        return np.stack((c[:, 0] * states[:, 0] - s[:, 0] * states[:, 1],
                         s[:, 0] * states[:, 0] + c[:, 0] * states[:, 1],
                         c[:, 0] * states[:, 2] - s[:, 0] * states[:, 3],
                         s[:, 0] * states[:, 2] + c[:, 0] * states[:, 3]), axis=1)
    if wire == 1:
        return np.stack((c[:, 0] * states[:, 0] - s[:, 0] * states[:, 2],
                         c[:, 0] * states[:, 1] - s[:, 0] * states[:, 3],
                         s[:, 0] * states[:, 0] + c[:, 0] * states[:, 2],
                         s[:, 0] * states[:, 1] + c[:, 0] * states[:, 3]), axis=1)
    raise ValueError("wire must be 0 or 1")


def _cx(states: np.ndarray, control: int, target: int) -> np.ndarray:
    if (control, target) == (0, 1):
        return states[:, [0, 3, 2, 1]]
    if (control, target) == (1, 0):
        return states[:, [0, 1, 3, 2]]
    raise ValueError("distinct two-qubit CNOT wires required")


def _expectations(x: np.ndarray, theta: np.ndarray, architecture: str) -> np.ndarray:
    states = np.zeros((len(x), 4), dtype=np.float64)
    states[:, 0] = 1.0
    states = _ry(states, 0, np.pi * x[:, 0])
    states = _ry(states, 1, np.pi * x[:, 1])
    if architecture == "one_block_2feature":
        states = _cx(states, 0, 1)
        states = _ry(states, 0, theta[0])
        states = _ry(states, 1, theta[1])
        states = _cx(states, 1, 0)
    elif architecture == "two_stage_reupload_4feature":
        states = _ry(states, 0, theta[0])
        states = _ry(states, 1, theta[1])
        states = _cx(states, 0, 1)
        states = _ry(states, 0, np.pi * x[:, 2])
        states = _ry(states, 1, np.pi * x[:, 3])
        states = _ry(states, 0, theta[2])
        states = _ry(states, 1, theta[3])
        states = _cx(states, 1, 0)
    else:
        raise ValueError(f"unsupported residual circuit architecture: {architecture}")
    probability = states * states
    z0 = probability[:, 0] - probability[:, 1] + probability[:, 2] - probability[:, 3]
    z1 = probability[:, 0] + probability[:, 1] - probability[:, 2] - probability[:, 3]
    return np.column_stack((z0, z1))


def load_residual_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read JSON only, checking shapes, finite values, feature order and scale."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported residual checkpoint format")
    architecture = payload.get("architecture")
    expected_features = FEATURES_2 if architecture == "one_block_2feature" else FEATURES_4 if architecture == "two_stage_reupload_4feature" else None
    if expected_features is None or tuple(payload.get("feature_names", ())) != expected_features:
        raise ValueError("residual architecture or ordered feature schema is unsupported")
    config = payload.get("config", {})
    if tuple(config.get("feature_names", ())) != expected_features or config.get("architecture") != architecture:
        raise ValueError("checkpoint config and feature schema disagree")
    theta = np.asarray(payload.get("theta"), dtype=np.float64)
    readout = np.asarray(payload.get("readout"), dtype=np.float64)
    angle_count = 2 if architecture == "one_block_2feature" else 4
    scale = float(payload.get("residual_scale_minutes"))
    if theta.shape != (angle_count,) or readout.shape != (3,):
        raise ValueError("checkpoint parameter shape mismatch")
    if not np.isfinite(theta).all() or not np.isfinite(readout).all() or not math.isfinite(scale) or scale <= 0:
        raise ValueError("checkpoint parameters and minute scale must be finite; scale must be positive")
    return {"path": str(source.resolve()), "sha256": sha256_file(source), "payload": payload,
            "architecture": architecture, "feature_names": expected_features,
            "theta": theta, "readout": readout, "residual_scale_minutes": scale}


def predict_residual_minutes(checkpoint: str | Path | Mapping[str, Any], features: Any) -> np.ndarray:
    """Predict minute-valued residuals from the frozen two-qubit checkpoint."""
    model = load_residual_checkpoint(checkpoint) if isinstance(checkpoint, (str, Path)) else checkpoint
    x = _checked_features(features, len(model["feature_names"]))
    expectations = _expectations(x, model["theta"], model["architecture"])
    readout = model["readout"]
    prediction = model["residual_scale_minutes"] * (expectations @ readout[:2] + readout[2])
    if not np.isfinite(prediction).all():
        raise ValueError("residual prediction is not finite")
    return prediction.astype(np.float64, copy=False)


def load_release_manifest(release_dir: str | Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Verify release inventory before returning any deserialization paths."""
    root = Path(release_dir).resolve()
    manifest_path = root / "release_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "run2.india.portable_release.v1":
        raise ValueError("unsupported portable release manifest schema")
    verified = verify_manifest_hashes(root, manifest)
    return manifest, verified


def validate_release_inventory(release_dir: str | Path, manifest: Mapping[str, Any]) -> None:
    """Reject raw training/evaluation data and large intermediates in a release."""
    root = Path(release_dir).resolve()
    inventory = manifest.get("artifacts") or manifest.get("files")
    if not isinstance(inventory, Mapping):
        raise ValueError("release manifest lacks an artifact inventory")
    forbidden_parts = {"train.csv", "indian_delivery_clean.parquet", "indian_delivery_gbdt.parquet",
                       "indian_delivery_vqc.parquet", "oof_predictions.parquet", "validation_predictions.parquet",
                       "test_predictions.parquet"}
    for relative in inventory:
        name = Path(str(relative)).name.lower()
        if name in forbidden_parts:
            raise ValueError(f"raw data or large intermediate is included in release: {relative}")
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "release_manifest.json":
            continue
        if path.suffix.lower() in {".csv", ".parquet", ".feather"}:
            raise ValueError(f"tabular training/evaluation data is present in release: {path.relative_to(root)}")
        if path.stat().st_size > 25 * 1024 * 1024:
            raise ValueError(f"release artifact exceeds 25 MiB: {path.relative_to(root)}")


def predict_residual_qpiai(checkpoint: str | Path | Mapping[str, Any], features: Any) -> np.ndarray:
    """Independent actual local QpiAI StatevectorSimulator minute predictions."""
    model = load_residual_checkpoint(checkpoint) if isinstance(checkpoint, (str, Path)) else checkpoint
    x = _checked_features(features, len(model["feature_names"]))
    try:
        from qpiai_quantum import Circuit
        from qpiai_quantum.simulator import StatevectorSimulator
    except ImportError as exc:
        raise RuntimeError("QpiAI Quantum local simulator is unavailable") from exc
    simulator = StatevectorSimulator()
    theta, readout = model["theta"], model["readout"]
    output: list[float] = []
    for row in x:
        circuit = Circuit(2, name="RUN2IndiaPortableReleaseCheck")
        circuit.ry(0, float(np.pi * row[0]))
        circuit.ry(1, float(np.pi * row[1]))
        if model["architecture"] == "one_block_2feature":
            circuit.cx(0, 1)
            circuit.ry(0, float(theta[0]))
            circuit.ry(1, float(theta[1]))
            circuit.cx(1, 0)
        else:
            circuit.ry(0, float(theta[0]))
            circuit.ry(1, float(theta[1]))
            circuit.cx(0, 1)
            circuit.ry(0, float(np.pi * row[2]))
            circuit.ry(1, float(np.pi * row[3]))
            circuit.ry(0, float(theta[2]))
            circuit.ry(1, float(theta[3]))
            circuit.cx(1, 0)
        result = simulator.run(circuit, shots=1, seed=0)
        state = np.asarray(result.statevector, dtype=np.complex128)
        if state.shape != (4,) or not np.isfinite(state.real).all() or not np.isfinite(state.imag).all():
            raise RuntimeError("QpiAI local simulator returned an invalid statevector")
        probability = np.abs(state) ** 2
        z0 = probability[0] - probability[1] + probability[2] - probability[3]
        z1 = probability[0] + probability[1] - probability[2] - probability[3]
        output.append(float(model["residual_scale_minutes"] * (readout[0] * z0 + readout[1] * z1 + readout[2])))
    return np.asarray(output, dtype=np.float64)

