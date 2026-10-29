"""Portable NumPy CPU circuit correctness independent of a torch runtime."""
from __future__ import annotations

import math

import numpy as np
import pytest

from experiments_india_qml.portable_inference import predict_residual_minutes
from experiments_india_qml.quantum import _state_expectations


def _payload(architecture, feature_names, theta):
    return {
        "format": "india_eta_torch_residual_v2",
        "architecture": architecture,
        "config": {"architecture": architecture, "feature_names": list(feature_names), "seed": 5, "dtype": "float64"},
        "theta": theta,
        "readout": [0.18, -0.11, 0.04],
        "residual_scale_minutes": 7.5,
    }


def test_portable_baseline_matches_existing_numpy_two_feature_reference():
    x = np.asarray([[0.03, 0.91], [0.42, 0.58], [0.99, 0.12]], dtype=np.float64)
    theta = [0.17, -0.43]
    payload = _payload("one_block_2feature", ["vqc_x0_distance", "vqc_x1_congestion"], theta)
    z = _state_expectations(x, np.asarray(theta), "one_block")
    expected = 7.5 * (0.18 * z[:, 0] - 0.11 * z[:, 1] + 0.04)
    got = predict_residual_minutes(payload, x)
    assert np.max(np.abs(got - expected)) <= 1e-12


def test_portable_four_feature_reupload_matches_actual_qpiai_local_simulator():
    pytest.importorskip("qpiai_quantum")
    from qpiai_quantum import Circuit
    from qpiai_quantum.simulator import StatevectorSimulator

    x = np.asarray([[0.03, 0.91, 0.16, 0.73], [0.42, 0.58, 0.84, 0.25]], dtype=np.float64)
    theta = [0.17, -0.43, 0.29, 0.51]
    payload = _payload("two_stage_reupload_4feature", [
        "vqc_x0_distance", "vqc_x1_congestion", "vqc_x2_multidelivery", "vqc_x3_preptime",
    ], theta)
    local = predict_residual_minutes(payload, x)
    sim = StatevectorSimulator()
    sdk = []
    for row in x:
        circuit = Circuit(2, name="RUN2Portable4FeatureParity")
        circuit.ry(0, math.pi * row[0]); circuit.ry(1, math.pi * row[1])
        circuit.ry(0, theta[0]); circuit.ry(1, theta[1]); circuit.cx(0, 1)
        circuit.ry(0, math.pi * row[2]); circuit.ry(1, math.pi * row[3])
        circuit.ry(0, theta[2]); circuit.ry(1, theta[3]); circuit.cx(1, 0)
        state = np.asarray(sim.run(circuit, shots=1, seed=0).statevector, dtype=np.complex128)
        probs = np.abs(state) ** 2
        z0 = probs[0] - probs[1] + probs[2] - probs[3]
        z1 = probs[0] + probs[1] - probs[2] - probs[3]
        sdk.append(7.5 * (0.18 * z0 - 0.11 * z1 + 0.04))
    assert np.max(np.abs(local - np.asarray(sdk))) <= 1e-12
