"""Dependency-light CPU inference for frozen RUN2 India residual checkpoints.

Only NumPy is required. The code follows the exact little-endian, real-valued
two-qubit gate order used by ``torch_residual.py``; no training framework,
SDK, CUDA, QAOA, or cloud client is imported.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


BASE_FEATURES = ("vqc_x0_distance", "vqc_x1_congestion")
REUPLOAD_FEATURES = (
    "vqc_x0_distance", "vqc_x1_congestion",
    "vqc_x2_multidelivery", "vqc_x3_preptime",
)


def _ry(states: np.ndarray, wire: int, angle: float | np.ndarray) -> np.ndarray:
    angle_array = np.asarray(angle, dtype=np.float64)
    if angle_array.ndim == 0:
        angle_array = np.full(len(states), float(angle_array), dtype=np.float64)
    c, s = np.cos(angle_array / 2.0), np.sin(angle_array / 2.0)
    c, s = c[:, None], s[:, None]
    out = states.copy()
    if wire == 0:
        out[:, 0] = c[:, 0] * states[:, 0] - s[:, 0] * states[:, 1]
        out[:, 1] = s[:, 0] * states[:, 0] + c[:, 0] * states[:, 1]
        out[:, 2] = c[:, 0] * states[:, 2] - s[:, 0] * states[:, 3]
        out[:, 3] = s[:, 0] * states[:, 2] + c[:, 0] * states[:, 3]
    elif wire == 1:
        out[:, 0] = c[:, 0] * states[:, 0] - s[:, 0] * states[:, 2]
        out[:, 2] = s[:, 0] * states[:, 0] + c[:, 0] * states[:, 2]
        out[:, 1] = c[:, 0] * states[:, 1] - s[:, 0] * states[:, 3]
        out[:, 3] = s[:, 0] * states[:, 1] + c[:, 0] * states[:, 3]
    else:
        raise ValueError("wire must be 0 or 1")
    return out


def _cx(states: np.ndarray, control: int, target: int) -> np.ndarray:
    if (control, target) == (0, 1):
        return states[:, [0, 3, 2, 1]]
    if (control, target) == (1, 0):
        return states[:, [0, 1, 3, 2]]
    raise ValueError("distinct two-qubit CNOT wires required")


def predict_residual_minutes(checkpoint: str | Path | dict[str, Any], features: Any) -> np.ndarray:
    """Load frozen JSON weights and return a NumPy vector of residual minutes."""
    payload = (json.loads(Path(checkpoint).read_text(encoding="utf-8"))
               if isinstance(checkpoint, (str, Path)) else checkpoint)
    if payload.get("format") != "india_eta_torch_residual_v2":
        raise ValueError("unsupported RUN2 residual checkpoint format")
    architecture = payload.get("architecture")
    n_features = 2 if architecture == "one_block_2feature" else 4 if architecture == "two_stage_reupload_4feature" else 0
    if not n_features:
        raise ValueError("unsupported RUN2 circuit architecture")
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[1] != n_features or x.shape[0] == 0:
        raise ValueError(f"features must have shape (n,{n_features})")
    if not np.isfinite(x).all() or np.any(x < -1e-12) or np.any(x > 1.0 + 1e-12):
        raise ValueError("features must be finite and scaled to [0,1]")
    x = np.clip(x, 0.0, 1.0)
    theta = np.asarray(payload["theta"], dtype=np.float64)
    readout = np.asarray(payload["readout"], dtype=np.float64)
    expected_angles = 2 if n_features == 2 else 4
    if theta.shape != (expected_angles,) or readout.shape != (3,) or not np.isfinite(theta).all() or not np.isfinite(readout).all():
        raise ValueError("checkpoint angles/readout are invalid")
    scale = float(payload["residual_scale_minutes"])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("checkpoint residual scale must be finite positive minutes")
    states = np.zeros((len(x), 4), dtype=np.float64)
    states[:, 0] = 1.0
    states = _ry(states, 0, math.pi * x[:, 0])
    states = _ry(states, 1, math.pi * x[:, 1])
    if n_features == 2:
        states = _cx(states, 0, 1)
        states = _ry(states, 0, float(theta[0]))
        states = _ry(states, 1, float(theta[1]))
        states = _cx(states, 1, 0)
    else:
        states = _ry(states, 0, float(theta[0]))
        states = _ry(states, 1, float(theta[1]))
        states = _cx(states, 0, 1)
        states = _ry(states, 0, math.pi * x[:, 2])
        states = _ry(states, 1, math.pi * x[:, 3])
        states = _ry(states, 0, float(theta[2]))
        states = _ry(states, 1, float(theta[3]))
        states = _cx(states, 1, 0)
    probs = states ** 2
    z0 = probs[:, 0] - probs[:, 1] + probs[:, 2] - probs[:, 3]
    z1 = probs[:, 0] + probs[:, 1] - probs[:, 2] - probs[:, 3]
    normalized = readout[0] * z0 + readout[1] * z1 + readout[2]
    return scale * normalized

