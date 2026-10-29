"""Independent numerical acceptance for the RUN2 4-qubit circuit.

This test uses an explicit 16x16 matrix construction rather than the module's
wire-index statevector updates. Feature rows are read from the train-only OOF
file with a feature-only projection; no labels or validation/test rows enter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments_india_vqc4.circuit import (
    FEATURE_NAMES,
    READOUT_BASIS,
    jacobian_rank_report,
    numpy_expectations,
    numpy_statevector,
)


def _single_qubit_matrix(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    if axis == "rx":
        return np.asarray([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    if axis == "ry":
        return np.asarray([[c, -s], [s, c]], dtype=np.complex128)
    if axis == "rz":
        return np.diag([np.exp(-1j * angle / 2), np.exp(1j * angle / 2)])
    raise AssertionError(axis)


def _wire_matrix(local: np.ndarray, wire: int) -> np.ndarray:
    """Expand a 2x2 gate into a little-endian 16x16 matrix."""
    full = np.zeros((16, 16), dtype=np.complex128)
    for source in range(16):
        low = source & ~(1 << wire)
        high = low | (1 << wire)
        bit = (source >> wire) & 1
        full[low, source] = local[0, bit]
        full[high, source] = local[1, bit]
    return full


def _controlled_x_matrix(control: int, target: int) -> np.ndarray:
    full = np.zeros((16, 16), dtype=np.complex128)
    for source in range(16):
        destination = source ^ ((1 << target) if source & (1 << control) else 0)
        full[destination, source] = 1.0
    return full


def _matrix_reference(row: np.ndarray, angles: np.ndarray, variant: str) -> np.ndarray:
    """Direct unitary-product reference, intentionally separate from circuit.py."""
    state = np.zeros(16, dtype=np.complex128)
    state[0] = 1.0
    forward = ((0, 1), (1, 2), (2, 3), (3, 0))
    reverse = ((0, 3), (3, 2), (2, 1), (1, 0))
    for block in range(5):
        for q in range(4):
            state = _wire_matrix(_single_qubit_matrix("ry", np.pi * row[q]), q) @ state
        for q in range(4):
            offset = 3 * (4 * block + q)
            state = _wire_matrix(_single_qubit_matrix("rx", angles[offset]), q) @ state
        for control, target in forward:
            state = _controlled_x_matrix(control, target) @ state
        for q in range(4):
            offset = 3 * (4 * block + q)
            state = _wire_matrix(_single_qubit_matrix("ry", angles[offset + 1]), q) @ state
        for control, target in reverse:
            state = _controlled_x_matrix(control, target) @ state
        for q in range(4):
            offset = 3 * (4 * block + q)
            state = _wire_matrix(_single_qubit_matrix("rz", angles[offset + 2]), q) @ state
        if variant == "extended76" and block < 4:
            for q in range(4):
                state = _wire_matrix(_single_qubit_matrix("rx", angles[60 + 4 * block + q]), q) @ state
    for axis, angle in READOUT_BASIS:
        for q in range(4):
            state = _wire_matrix(_single_qubit_matrix(axis, angle), q) @ state
    return state


def test_statevector_and_readouts_match_independent_dense_unitary_product() -> None:
    rng = np.random.default_rng(420)
    rows = rng.uniform(0.01, 0.99, size=(6, 4))
    angles60 = rng.normal(0, 0.27, size=60)
    angles76 = np.concatenate((angles60, rng.normal(0, 0.2, size=16)))
    signs = np.asarray([[1 - 2 * ((basis >> q) & 1) for q in range(4)] for basis in range(16)])

    for variant, angles in (("baseline60", angles60), ("extended76", angles76)):
        expected_states = np.stack([_matrix_reference(row, angles, variant) for row in rows])
        actual_states = numpy_statevector(rows, angles, variant)
        np.testing.assert_allclose(actual_states, expected_states, rtol=0, atol=2e-12)
        np.testing.assert_allclose(np.sum(np.abs(actual_states) ** 2, axis=1), 1.0, rtol=0, atol=2e-12)
        expected_z = (np.abs(expected_states) ** 2) @ signs
        np.testing.assert_allclose(numpy_expectations(rows, angles, variant), expected_z, rtol=0, atol=2e-12)

    zero_extended = np.concatenate((angles60, np.zeros(16)))
    np.testing.assert_array_equal(
        numpy_expectations(rows, angles60, "baseline60"),
        numpy_expectations(rows, zero_extended, "extended76"),
    )


def test_60_and_76_jacobians_are_full_rank_on_train_oof_features() -> None:
    # Read only the four train-OOF feature columns. Target/residual columns and
    # validation/test partitions are never requested from the Parquet reader.
    frame = pd.read_parquet(
        "artifacts/india_eta_recovery/oof_predictions.parquet",
        columns=list(FEATURE_NAMES),
        engine="fastparquet",
    )
    positions = np.linspace(0, len(frame) - 1, 64, dtype=int)
    rows = frame.iloc[positions].to_numpy(dtype=np.float64)
    assert rows.shape == (64, 4)
    assert np.isfinite(rows).all() and np.all((rows >= 0) & (rows <= 1))

    angles = np.random.default_rng(641).normal(0, 0.23, size=60)
    report = jacobian_rank_report(rows, angles, step=1e-5, relative_tolerance=1e-8)
    assert report["baseline_local_rank"] == 60
    assert report["extended_local_rank"] == 76
    assert report["rank_increment"] == 16
    assert report["all_extension_angles_sensitive"]
    assert report["all_extension_angles_add_local_rank"]
    assert min(report["extension_incremental_l2_by_angle"]) > 0.5
