"""Numerical and interface checks for the separately versioned 4-qubit RUN2 circuit."""
from __future__ import annotations

import math

import numpy as np
import pytest

from experiments_india_vqc4.circuit import (
    ANGLE_COUNT,
    BASELINE_ANGLE_COUNT,
    EXTENSION_ANGLE_COUNT,
    CircuitConfig,
    NumpyFourQubitResidualVQC,
    TorchFourQubitVQC,
    _cnot_numpy,
    expand_baseline_payload,
    jacobian_rank_report,
    load_payload,
    numpy_expectations,
    numpy_statevector,
    parameter_names,
    qpiai_expectations,
    save_payload,
)


def _fixtures(n: int = 64):
    rng = np.random.default_rng(20261029)
    x = rng.uniform(0.02, 0.98, size=(n, 4))
    base = np.random.default_rng(531).normal(0, 0.23, size=60)
    return x, base


def test_cnot_uses_little_endian_wire_order() -> None:
    state = np.zeros((1, 16), dtype=np.complex128)
    state[0, 1] = 1.0  # |q0=1,q1=q2=q3=0>
    out = _cnot_numpy(state, control=0, target=1)
    assert out[0, 3] == 1.0  # target q1 toggles, index bit1
    assert np.count_nonzero(out) == 1


def test_statevector_stays_normalized_and_outputs_are_bounded_at_extreme_inputs() -> None:
    x = np.asarray([[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1]], dtype=np.float64)
    for variant in ("baseline60", "extended76"):
        p = np.zeros(ANGLE_COUNT[variant], dtype=np.float64)
        state = numpy_statevector(x, p, variant)
        assert state.shape == (3, 16)
        np.testing.assert_allclose(np.sum(np.abs(state) ** 2, axis=1), 1.0, atol=2e-14, rtol=0)
        z = numpy_expectations(x, p, variant)
        assert z.shape == (3, 4)
        assert np.isfinite(z).all()
        assert np.max(np.abs(z)) <= 1.0 + 1e-12


def test_zero_extended_angles_exactly_reproduce_all_baseline_expectations() -> None:
    x, p60 = _fixtures()
    p76 = np.concatenate((p60, np.zeros(EXTENSION_ANGLE_COUNT)))
    base = numpy_expectations(x, p60, "baseline60")
    extended = numpy_expectations(x, p76, "extended76")
    np.testing.assert_array_equal(extended, base)


def test_candidate_rx_extension_adds_16_independent_jacobian_directions() -> None:
    x, p60 = _fixtures()
    report = jacobian_rank_report(x, p60, step=1e-5, relative_tolerance=1e-8)
    assert report["baseline_local_rank"] == 60
    assert report["extended_local_rank"] == 76
    assert report["rank_increment"] == 16
    assert report["all_extension_angles_sensitive"] is True
    assert report["all_extension_angles_add_local_rank"] is True
    assert min(report["extension_gradient_l2_by_angle"]) > 1.0
    assert min(report["extension_incremental_l2_by_angle"]) > 0.2


def test_batched_numpy_output_matches_qpiai_for_both_variants() -> None:
    pytest.importorskip("qpiai_quantum")
    x, p60 = _fixtures(3)
    for variant, n in (("baseline60", 60), ("extended76", 76)):
        angles = p60 if n == 60 else np.concatenate((p60, np.random.default_rng(9).normal(0, 0.2, 16)))
        local = numpy_expectations(x, angles, variant)
        sdk = qpiai_expectations(x, angles, variant)
        np.testing.assert_allclose(local, sdk, atol=2e-12, rtol=0)


def test_checkpoint_roundtrip_and_exact_zero_expansion(tmp_path) -> None:
    x, p60 = _fixtures(5)
    source = NumpyFourQubitResidualVQC(
        CircuitConfig("baseline60"), quantum_angles=p60,
        readout=[0.11, -0.04, 0.03, 0.08, -0.2], residual_scale_minutes=6.3,
    )
    path = save_payload(source.export_payload(), tmp_path / "baseline60.json")
    loaded_payload = load_payload(path)
    restored = NumpyFourQubitResidualVQC.from_payload(loaded_payload)
    np.testing.assert_array_equal(restored.predict_residual_minutes(x), source.predict_residual_minutes(x))
    expanded_payload = expand_baseline_payload(loaded_payload)
    expanded = NumpyFourQubitResidualVQC.from_payload(expanded_payload)
    np.testing.assert_array_equal(expanded.predict_residual_minutes(x), source.predict_residual_minutes(x))
    assert len(parameter_names("baseline60")) == BASELINE_ANGLE_COUNT
    assert len(parameter_names("extended76")) == BASELINE_ANGLE_COUNT + EXTENSION_ANGLE_COUNT
    assert expanded_payload["expansion"]["exact_zero_extension"] is True


def test_torch_batched_autograd_matches_numpy_and_finite_differences() -> None:
    torch = pytest.importorskip("torch")
    from experiments_india_vqc4.circuit import numpy_expectations

    x_np, p60 = _fixtures(3)
    x = torch.as_tensor(x_np, dtype=torch.float64)
    y = torch.as_tensor([0.3, -0.7, 0.1], dtype=torch.float64)
    for variant in ("baseline60", "extended76"):
        model = TorchFourQubitVQC(variant, residual_scale_minutes=4.5, seed=117, dtype="float64")
        if variant == "extended76":
            with torch.no_grad():
                model.quantum_angles[:60].copy_(torch.as_tensor(p60, dtype=torch.float64))
                model.quantum_angles[60:].copy_(torch.as_tensor(np.linspace(-0.08, 0.1, 16), dtype=torch.float64))
        got = model.forward_expectations(x).detach().numpy()
        expected = numpy_expectations(x_np, model.quantum_angles.detach().numpy(), variant)
        np.testing.assert_allclose(got, expected, atol=2e-12, rtol=0)
        model.zero_grad(set_to_none=True)
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        analytic_angles = model.quantum_angles.grad.detach().numpy().copy()
        analytic_readout = model.readout.grad.detach().numpy().copy()
        all_analytic = np.concatenate((analytic_angles, analytic_readout))
        original = model.export_payload()
        param = np.asarray(original["quantum_angles"], dtype=np.float64)
        head = np.asarray(original["readout"], dtype=np.float64)
        step = 1e-5
        numeric = []
        for index in range(len(param) + len(head)):
            plus_angles, minus_angles = param.copy(), param.copy()
            plus_head, minus_head = head.copy(), head.copy()
            if index < len(param):
                plus_angles[index] += step
                minus_angles[index] -= step
            else:
                plus_head[index - len(param)] += step
                minus_head[index - len(param)] -= step
            def objective(a, w):
                z = numpy_expectations(x_np, a, variant)
                prediction = 4.5 * (z @ w[:4] + w[4])
                return float(np.mean((prediction - y.numpy()) ** 2))
            numeric.append((objective(plus_angles, plus_head) - objective(minus_angles, minus_head)) / (2 * step))
        error = np.abs(all_analytic - np.asarray(numeric))
        assert np.max(error) <= 2e-5


def test_torch_60_to_76_expansion_copies_weights_and_preserves_prediction() -> None:
    torch = pytest.importorskip("torch")
    x_np, _ = _fixtures(5)
    x = torch.as_tensor(x_np, dtype=torch.float64)
    base = TorchFourQubitVQC("baseline60", residual_scale_minutes=3.7, seed=91, dtype="float64")
    expanded = TorchFourQubitVQC.expand_baseline(base)
    assert expanded.quantum_parameter_count == 76
    assert expanded.parameter_count == 81
    torch.testing.assert_close(expanded.quantum_angles[:60], base.quantum_angles, rtol=0, atol=0)
    assert torch.count_nonzero(expanded.quantum_angles[60:]).item() == 0
    torch.testing.assert_close(expanded.readout, base.readout, rtol=0, atol=0)
    torch.testing.assert_close(expanded(x), base(x), rtol=0, atol=1e-12)
