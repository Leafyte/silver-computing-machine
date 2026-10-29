"""Numerical and checkpoint tests for the separate RUN2 torch circuit path."""
from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch", reason="RUN2 isolated torch runtime is required")

from experiments_india_qml.quantum import _state_expectations as numpy_expectations
from experiments_india_qml.torch_residual import (
    BASE_FEATURES,
    REUPLOAD_FEATURES,
    TorchCircuitConfig,
    TorchResidualVQC,
    qpiai_residual_parity,
)


@pytest.mark.parametrize("architecture,feature_names", [
    ("one_block_2feature", BASE_FEATURES),
    ("two_stage_reupload_4feature", REUPLOAD_FEATURES),
])
def test_torch_autograd_matches_centered_finite_difference(architecture, feature_names):
    rng = np.random.default_rng(22)
    x = rng.uniform(0.05, 0.95, size=(9, len(feature_names)))
    y = 7.0 * np.sin(x[:, 0] + x[:, 1]) - 3.0 * np.cos(x[:, -1])
    model = TorchResidualVQC(TorchCircuitConfig(architecture, tuple(feature_names), seed=9))
    result = model.gradient_check(x, y, h=1e-5)
    assert result["passed_abs_1e-5"]
    assert result["max_absolute_error"] < 1e-5
    assert result["parameters_checked"] == model.parameter_count


def test_one_block_torch_expectations_match_existing_numpy_reference():
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 1.0, size=(32, 2))
    model = TorchResidualVQC(TorchCircuitConfig("one_block_2feature", BASE_FEATURES, seed=5))
    theta = np.asarray([0.21, -0.37])
    with torch.no_grad():
        model.net.theta.copy_(torch.as_tensor(theta, dtype=model.dtype))
    got = model.forward_expectations(x).detach().cpu().numpy()
    expected = numpy_expectations(x, theta, "one_block")
    assert np.max(np.abs(got - expected)) <= 1e-12


def test_training_checkpoint_reloads_exact_minute_predictions(tmp_path):
    rng = np.random.default_rng(77)
    x = rng.uniform(0.02, 0.98, size=(48, 4))
    y = 4.0 * np.sin(2.0 * x[:, 0] - x[:, 3]) + np.linspace(-0.3, 0.3, len(x))
    checkpoint = tmp_path / "reupload.json"
    progress = tmp_path / "epochs.jsonl"
    model = TorchResidualVQC(TorchCircuitConfig("two_stage_reupload_4feature", REUPLOAD_FEATURES, seed=11))
    info = model.fit(x[:40], y[:40], x[40:], y[40:], max_epochs=4, batch_size=16,
                     patience=4, seed=20261029, checkpoint_path=checkpoint, progress_path=progress)
    before = model.predict_residual_minutes(x[40:])
    restored = TorchResidualVQC.load(checkpoint)
    after = restored.predict_residual_minutes(x[40:])
    assert info["epochs_run"] == 4
    assert checkpoint.is_file() and progress.read_text(encoding="utf-8").count("\n") == 4
    assert np.array_equal(before, after)


def test_qpiai_local_parity_on_selected_circuit_if_sdk_is_installed():
    pytest.importorskip("qpiai_quantum")
    rng = np.random.default_rng(1)
    x = rng.uniform(0.1, 0.9, size=(5, 4))
    model = TorchResidualVQC(TorchCircuitConfig("two_stage_reupload_4feature", REUPLOAD_FEATURES, seed=2))
    model.residual_scale_minutes = 9.0
    parity = qpiai_residual_parity(model, x)
    assert parity["passed"]
    assert parity["remote_calls"] == 0 and parity["qpu_jobs"] == 0
