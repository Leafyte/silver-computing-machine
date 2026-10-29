"""Batched two-qubit statevector regressors for India ETA residuals.

Inputs are exactly two pre-delivery features scaled by the caller into [0, 1].
The model predicts a residual correction in minutes, not a duration or risk score.
All local numerical simulation uses NumPy; no SDK calls occur in training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np


Architecture = Literal["one_block", "reupload"]
N_QUBITS = 2
N_FEATURES = 2
PARAMETER_SHIFT = math.pi / 2.0


@dataclass(frozen=True)
class CircuitConfig:
    architecture: Architecture
    feature_names: tuple[str, str] = ("feature_0", "feature_1")
    residual_scale_minutes: float | None = None
    seed: int = 17

    def __post_init__(self) -> None:
        if self.architecture not in ("one_block", "reupload"):
            raise ValueError("architecture must be 'one_block' or 'reupload'")
        if len(self.feature_names) != N_FEATURES or len(set(self.feature_names)) != N_FEATURES:
            raise ValueError("exactly two distinct feature names are required")
        scale = self.residual_scale_minutes
        if scale is not None and (not math.isfinite(float(scale)) or float(scale) <= 0):
            raise ValueError("residual_scale_minutes must be finite and positive")


def _validate_features(features: Any) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[1] != N_FEATURES:
        raise ValueError("features must have shape (n, 2), in configured order")
    if x.shape[0] == 0 or not np.isfinite(x).all():
        raise ValueError("features must be non-empty and finite")
    if np.any(x < -1e-12) or np.any(x > 1.0 + 1e-12):
        raise ValueError("both input features must be scaled into [0, 1]")
    return np.clip(x, 0.0, 1.0)


def _validate_theta(theta: Any, architecture: Architecture) -> np.ndarray:
    t = np.asarray(theta, dtype=np.float64)
    expected = 2 if architecture == "one_block" else 4
    if t.shape != (expected,) or not np.isfinite(t).all():
        raise ValueError(f"{architecture} requires {expected} finite trainable angles")
    return t


def _ry(states: np.ndarray, wire: int, angle: float | np.ndarray) -> None:
    """In-place RY on wire 0/1; state index is |q1 q0>, q0 is the low bit."""
    mask = 1 << wire
    c = np.cos(np.asarray(angle, dtype=np.float64) / 2.0)
    s = np.sin(np.asarray(angle, dtype=np.float64) / 2.0)
    if np.ndim(c) == 0:
        c = np.full((states.shape[0], 1), c)
        s = np.full((states.shape[0], 1), s)
    elif np.ndim(c) == 1:
        c, s = c[:, None], s[:, None]
    for i in range(4):
        if i & mask == 0:
            j = i | mask
            a = states[:, i].copy()
            b = states[:, j].copy()
            states[:, i] = c[:, 0] * a - s[:, 0] * b
            states[:, j] = s[:, 0] * a + c[:, 0] * b


def _cx(states: np.ndarray, control: int, target: int) -> None:
    """In-place CNOT on little-endian wires q0=LSB, q1=MSB."""
    control_mask, target_mask = 1 << control, 1 << target
    for i in range(4):
        if i & control_mask and not i & target_mask:
            j = i | target_mask
            states[:, [i, j]] = states[:, [j, i]]


def _expectations_from_state(states: np.ndarray) -> np.ndarray:
    probabilities = np.abs(states) ** 2
    z0 = probabilities[:, 0] - probabilities[:, 1] + probabilities[:, 2] - probabilities[:, 3]
    z1 = probabilities[:, 0] + probabilities[:, 1] - probabilities[:, 2] - probabilities[:, 3]
    return np.column_stack((z0, z1))


def _state_expectations(features: np.ndarray, theta: np.ndarray, architecture: Architecture) -> np.ndarray:
    x = _validate_features(features)
    t = _validate_theta(theta, architecture)
    states = np.zeros((x.shape[0], 4), dtype=np.complex128)
    states[:, 0] = 1.0
    # Stage 1: two feature angles are encoded directly onto their corresponding wires.
    _ry(states, 0, math.pi * x[:, 0])
    _ry(states, 1, math.pi * x[:, 1])
    if architecture == "one_block":
        # Existing one-block ordering: entangle, trainable local rotations, reverse entangle.
        _cx(states, 0, 1)
        _ry(states, 0, t[0])
        _ry(states, 1, t[1])
        _cx(states, 1, 0)
    else:
        # Genuine data reupload: a second encoding occurs between trainable stages.
        _ry(states, 0, t[0])
        _ry(states, 1, t[1])
        _cx(states, 0, 1)
        _ry(states, 0, math.pi * x[:, 0])
        _ry(states, 1, math.pi * x[:, 1])
        _ry(states, 0, t[2])
        _ry(states, 1, t[3])
        _cx(states, 1, 0)
    return _expectations_from_state(states)


def qpiai_expectations(
    features: Any,
    theta: Any,
    architecture: Architecture = "one_block",
) -> np.ndarray:
    """Execute this circuit on QpiAI's local statevector simulator, row by row.

    This helper is for small parity checks only; fitting uses the batched NumPy path.
    It never creates a cloud client or submits a QPU job.
    """
    x = _validate_features(features)
    t = _validate_theta(theta, architecture)
    try:
        from qpiai_quantum import Circuit
        from qpiai_quantum.simulator import StatevectorSimulator
    except ImportError as exc:
        raise RuntimeError("QpiAI Quantum local simulator is unavailable") from exc

    simulator = StatevectorSimulator()
    rows: list[tuple[float, float]] = []
    for row in x:
        circuit = Circuit(N_QUBITS, name="IndiaETAResidualVQC")
        circuit.ry(0, math.pi * float(row[0]))
        circuit.ry(1, math.pi * float(row[1]))
        if architecture == "one_block":
            circuit.cx(0, 1)
            circuit.ry(0, float(t[0]))
            circuit.ry(1, float(t[1]))
            circuit.cx(1, 0)
        else:
            circuit.ry(0, float(t[0]))
            circuit.ry(1, float(t[1]))
            circuit.cx(0, 1)
            circuit.ry(0, math.pi * float(row[0]))
            circuit.ry(1, math.pi * float(row[1]))
            circuit.ry(0, float(t[2]))
            circuit.ry(1, float(t[3]))
            circuit.cx(1, 0)
        result = simulator.run(circuit, shots=1, seed=0)
        state = getattr(result, "statevector", None)
        if state is None or len(state) != 4:
            raise RuntimeError("QpiAI StatevectorSimulator returned no 4-amplitude statevector")
        states = np.asarray([state], dtype=np.complex128)
        norm = float(np.sum(np.abs(states) ** 2))
        if abs(norm - 1.0) > 1e-9:
            raise RuntimeError("QpiAI statevector failed normalization")
        rows.append(tuple(float(v) for v in _expectations_from_state(states)[0]))
    return np.asarray(rows, dtype=np.float64)


def hybrid_predict_minutes(
    classical_eta_minutes: Any,
    quantum_residual_minutes: Any,
    alpha: float,
    gate_mask: Any | None = None,
) -> np.ndarray:
    """Fuse ETA and a selected residual correction, clipping durations at zero.

    The caller selects ``alpha`` on validation data only; the inference function
    enforces the release range and never fits or calibrates it.
    """
    eta = np.asarray(classical_eta_minutes, dtype=np.float64)
    residual = np.asarray(quantum_residual_minutes, dtype=np.float64)
    if eta.shape != residual.shape or eta.size == 0:
        raise ValueError("classical ETA and quantum residual arrays must have the same non-empty shape")
    if not np.isfinite(eta).all() or not np.isfinite(residual).all():
        raise ValueError("ETA and residual predictions must be finite")
    a = float(alpha)
    if not math.isfinite(a) or not 0.0 <= a <= 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1]")
    if gate_mask is not None:
        mask = np.asarray(gate_mask, dtype=bool)
        if mask.shape != residual.shape:
            raise ValueError("gate_mask must have the same shape as residual predictions")
        residual = np.where(mask, residual, 0.0)
    return np.maximum(0.0, eta + a * residual)


class QuantumResidualRegressor:
    """Two-qubit variational residual regressor with analytic parameter-shift gradients."""

    def __init__(self, config: CircuitConfig, seed: int | None = None):
        self.config = config
        self.seed = config.seed if seed is None else int(seed)
        self._rng = np.random.default_rng(self.seed)
        ntheta = 2 if config.architecture == "one_block" else 4
        self.theta = self._rng.uniform(-0.15, 0.15, size=ntheta)
        # Coefficients of measured Pauli Z on q0 and q1, plus a trainable bias.
        self.readout = np.asarray([0.03, -0.03, 0.0], dtype=np.float64)
        self.residual_scale_minutes = (
            float(config.residual_scale_minutes)
            if config.residual_scale_minutes is not None else None
        )
        self.fit_info: dict[str, Any] = {}
        self.circuit_batch_evaluations = 0
        self.circuit_example_evaluations = 0

    @property
    def architecture(self) -> Architecture:
        return self.config.architecture

    @property
    def parameter_count(self) -> int:
        return int(self.theta.size + self.readout.size)

    @property
    def circuit_description(self) -> dict[str, Any]:
        if self.architecture == "one_block":
            gates = [
                "RY(pi*x0) q0", "RY(pi*x1) q1", "CX(q0,q1)",
                "RY(theta0) q0", "RY(theta1) q1", "CX(q1,q0)",
            ]
        else:
            gates = [
                "RY(pi*x0) q0", "RY(pi*x1) q1",
                "RY(theta0) q0", "RY(theta1) q1", "CX(q0,q1)",
                "RY(pi*x0) q0", "RY(pi*x1) q1",
                "RY(theta2) q0", "RY(theta3) q1", "CX(q1,q0)",
            ]
        return {
            "qubits": N_QUBITS,
            "statevector_indexing": "little-endian; q0 is least-significant bit; index=2*q1+q0",
            "feature_order": list(self.config.feature_names),
            "gates_in_execution_order": gates,
            "observable_readout": "bias + w0*<Z(q0)> + w1*<Z(q1)>",
            "readout_parameters": ["w0", "w1", "bias"],
            "parameter_count": self.parameter_count,
            "parameter_shift": "(f(theta+pi/2)-f(theta-pi/2))/2 per RY angle",
        }

    def _evaluate(self, features: np.ndarray, theta: np.ndarray) -> np.ndarray:
        self.circuit_batch_evaluations += 1
        self.circuit_example_evaluations += int(features.shape[0])
        return _state_expectations(features, theta, self.architecture)

    def forward_expectations(self, features: Any) -> np.ndarray:
        x = _validate_features(features)
        return self._evaluate(x, self.theta)

    def _predict_normalized(self, expectations: np.ndarray) -> np.ndarray:
        return self.readout[2] + expectations @ self.readout[:2]

    def predict_minutes(self, features: Any) -> np.ndarray:
        if self.residual_scale_minutes is None:
            raise RuntimeError("fit must set residual_scale_minutes before minute predictions")
        expectation = self.forward_expectations(features)
        return self.residual_scale_minutes * self._predict_normalized(expectation)

    def predict_hybrid_minutes(
        self,
        classical_eta_minutes: Any,
        features: Any,
        alpha: float,
        *,
        gate_feature: str = "vqc_x1_congestion",
        gate_operator: Literal[">=", "<="] = ">=",
        gate_threshold: float = 0.0,
    ) -> dict[str, Any]:
        """Run QML only when a pre-circuit input-feature gate permits invocation."""
        x = _validate_features(features)
        eta = np.asarray(classical_eta_minutes, dtype=np.float64)
        if eta.ndim == 0 and len(x) == 1:
            eta = eta.reshape(1)
        if eta.shape != (len(x),) or not np.isfinite(eta).all():
            raise ValueError("classical_eta_minutes must be a finite vector aligned with features")
        a = float(alpha)
        if not math.isfinite(a) or not 0.0 <= a <= 1.0:
            raise ValueError("alpha must be finite and lie in [0, 1]")
        if gate_feature not in self.config.feature_names:
            raise ValueError(f"gate feature {gate_feature!r} is not part of the configured feature schema")
        if gate_operator not in (">=", "<="):
            raise ValueError("gate_operator must be '>=' or '<='")
        threshold = float(gate_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("gate_threshold must be finite and lie in [0, 1]")
        index = self.config.feature_names.index(gate_feature)
        mask = x[:, index] >= threshold if gate_operator == ">=" else x[:, index] <= threshold
        residual = np.zeros(len(x), dtype=np.float64)
        invoked = mask if a > 0.0 else np.zeros_like(mask)
        # alpha=0 preserves the safe fallback and performs zero circuit calls.
        if np.any(invoked):
            residual[invoked] = self.predict_minutes(x[invoked])
        prediction = hybrid_predict_minutes(eta, residual, a, gate_mask=invoked)
        return {
            "eta_minutes": prediction,
            "quantum_residual_minutes": residual,
            "quantum_invoked": invoked,
            "invocation_count": int(np.sum(invoked)),
        }

    def _normalized_mse_gradient(self, features: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
        expectation = self._evaluate(features, self.theta)
        prediction = self._predict_normalized(expectation)
        error = prediction - targets
        grad = np.zeros(self.parameter_count, dtype=np.float64)
        grad[: self.theta.size] = 0.0
        weighted_error = 2.0 * error / len(targets)
        for k in range(self.theta.size):
            plus = self.theta.copy()
            minus = self.theta.copy()
            plus[k] += PARAMETER_SHIFT
            minus[k] -= PARAMETER_SHIFT
            z_plus = self._evaluate(features, plus)
            z_minus = self._evaluate(features, minus)
            dz = (z_plus - z_minus) * 0.5
            dpred = dz @ self.readout[:2]
            grad[k] = float(weighted_error @ dpred)
        grad[self.theta.size : self.theta.size + 2] = weighted_error @ expectation
        grad[-1] = float(np.sum(weighted_error))
        return float(np.mean(error**2)), grad

    def gradient_check(self, features: Any, residual_minutes: Any, h: float = 1e-5) -> dict[str, Any]:
        """Compare parameter-shift MSE gradients with centered finite differences."""
        x = _validate_features(features)
        y = np.asarray(residual_minutes, dtype=np.float64).reshape(-1)
        if len(y) != len(x) or not np.isfinite(y).all() or len(y) < 1:
            raise ValueError("residual_minutes must be finite and align with features")
        if not math.isfinite(h) or h <= 0:
            raise ValueError("h must be finite and positive")
        scale = self.residual_scale_minutes or max(float(np.std(y)), 1.0)
        target = y / scale
        base_theta, base_readout = self.theta.copy(), self.readout.copy()
        try:
            _, analytic = self._normalized_mse_gradient(x, target)
            numeric = np.empty_like(analytic)
            parameters = np.concatenate((self.theta, self.readout))
            for k in range(len(parameters)):
                plus, minus = parameters.copy(), parameters.copy()
                plus[k] += h
                minus[k] -= h
                self.theta = plus[: self.theta.size]
                self.readout = plus[self.theta.size :]
                loss_plus = self._loss_only(x, target)
                self.theta = minus[: self.theta.size]
                self.readout = minus[self.theta.size :]
                loss_minus = self._loss_only(x, target)
                numeric[k] = (loss_plus - loss_minus) / (2.0 * h)
            errors = np.abs(analytic - numeric)
            return {
                "examples": len(x),
                "step": h,
                "analytic_gradient": analytic.tolist(),
                "finite_difference_gradient": numeric.tolist(),
                "max_absolute_error": float(np.max(errors)),
                "max_relative_error": float(np.max(errors / np.maximum(np.abs(numeric), 1e-8))),
                "passed_abs_1e-5": bool(np.max(errors) < 1e-5),
            }
        finally:
            self.theta, self.readout = base_theta, base_readout

    def _loss_only(self, features: np.ndarray, target: np.ndarray) -> float:
        expectation = self._evaluate(features, self.theta)
        error = self._predict_normalized(expectation) - target
        return float(np.mean(error**2))

    def fit(
        self,
        features_train: Any,
        residual_train_minutes: Any,
        features_validation: Any,
        residual_validation_minutes: Any,
        *,
        max_updates: int = 120,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        patience: int = 15,
        min_delta: float = 1e-6,
        checkpoint_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Fit only on supplied train targets and select the step on supplied validation targets."""
        x_train = _validate_features(features_train)
        x_val = _validate_features(features_validation)
        y_train = np.asarray(residual_train_minutes, dtype=np.float64).reshape(-1)
        y_val = np.asarray(residual_validation_minutes, dtype=np.float64).reshape(-1)
        if len(y_train) != len(x_train) or len(y_val) != len(x_val):
            raise ValueError("feature and residual row counts must match")
        if len(y_train) < 2 or len(y_val) < 1 or not np.isfinite(y_train).all() or not np.isfinite(y_val).all():
            raise ValueError("train and validation residuals must be non-empty and finite")
        if max_updates <= 0 or batch_size <= 0 or patience <= 0 or learning_rate <= 0:
            raise ValueError("updates, batch size, patience, and learning rate must be positive")
        if min_delta < 0 or not math.isfinite(min_delta):
            raise ValueError("min_delta must be finite and non-negative")
        if self.config.residual_scale_minutes is None:
            self.residual_scale_minutes = max(float(np.std(y_train)), 1.0)
        else:
            self.residual_scale_minutes = float(self.config.residual_scale_minutes)
        yt, yv = y_train / self.residual_scale_minutes, y_val / self.residual_scale_minutes

        params = np.concatenate((self.theta, self.readout))
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        best_params = params.copy()
        best_val = math.inf
        best_update = 0
        stale = 0
        history: list[dict[str, float | int]] = []
        order = self._rng.permutation(len(x_train))
        cursor = 0
        started_wall, started_cpu = time.perf_counter(), time.process_time()
        starting_batch_evaluations = self.circuit_batch_evaluations
        starting_example_evaluations = self.circuit_example_evaluations
        for update in range(1, max_updates + 1):
            if cursor + batch_size > len(order):
                order = self._rng.permutation(len(x_train))
                cursor = 0
            indices = order[cursor : cursor + min(batch_size, len(x_train))]
            cursor += len(indices)
            self.theta = params[: self.theta.size].copy()
            self.readout = params[self.theta.size :].copy()
            train_loss, gradient = self._normalized_mse_gradient(x_train[indices], yt[indices])
            # Adam, with a fixed seed and no dependency on an autodiff framework.
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            m = beta1 * m + (1.0 - beta1) * gradient
            v = beta2 * v + (1.0 - beta2) * gradient**2
            mhat, vhat = m / (1.0 - beta1**update), v / (1.0 - beta2**update)
            params = params - learning_rate * mhat / (np.sqrt(vhat) + eps)
            self.theta = params[: self.theta.size].copy()
            self.readout = params[self.theta.size :].copy()
            pred_val = self.residual_scale_minutes * self._predict_normalized(self._evaluate(x_val, self.theta))
            val_rmse = float(np.sqrt(np.mean((pred_val - y_val) ** 2)))
            train_rmse = float(np.sqrt(train_loss) * self.residual_scale_minutes)
            history.append({"update": update, "train_batch_rmse_minutes": train_rmse, "validation_rmse_minutes": val_rmse})
            if val_rmse < best_val - min_delta:
                best_val, best_update, best_params, stale = val_rmse, update, params.copy(), 0
                if checkpoint_path is not None:
                    self.theta = best_params[: self.theta.size].copy()
                    self.readout = best_params[self.theta.size :].copy()
                    self.fit_info = {
                        "status": "fitting",
                        "updates_run": update,
                        "best_update": best_update,
                        "best_validation_rmse_minutes": best_val,
                        "train_rows": len(x_train),
                        "validation_rows": len(x_val),
                        "residual_scale_minutes": self.residual_scale_minutes,
                    }
                    self.save(checkpoint_path)
            else:
                stale += 1
                if stale >= patience:
                    break

        self.theta = best_params[: self.theta.size].copy()
        self.readout = best_params[self.theta.size :].copy()
        pred_val = self.residual_scale_minutes * self._predict_normalized(self._evaluate(x_val, self.theta))
        final_val_rmse = float(np.sqrt(np.mean((pred_val - y_val) ** 2)))
        self.fit_info = {
            "updates_run": len(history),
            "best_update": best_update,
            "best_validation_rmse_minutes": final_val_rmse,
            "validation_rows": len(x_val),
            "train_rows": len(x_train),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "residual_scale_minutes": self.residual_scale_minutes,
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "history": history,
            "circuit_batch_evaluations": self.circuit_batch_evaluations - starting_batch_evaluations,
            "circuit_example_evaluations": self.circuit_example_evaluations - starting_example_evaluations,
        }
        if checkpoint_path is not None:
            self.save(checkpoint_path)
        return dict(self.fit_info)

    def save(self, path: str | Path) -> Path:
        if self.residual_scale_minutes is None:
            raise RuntimeError("cannot save an unfitted model without a residual scale")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": "india_eta_quantum_residual_v1",
            "config": asdict(self.config),
            "seed": self.seed,
            "architecture": self.architecture,
            "residual_scale_minutes": self.residual_scale_minutes,
            "circuit": self.circuit_description,
            "fit_info": self.fit_info,
        }
        np.savez_compressed(
            destination,
            theta=self.theta,
            readout=self.readout,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        # np.savez appends .npz when omitted; return the actual path.
        return destination if destination.exists() else destination.with_suffix(destination.suffix + ".npz")

    @classmethod
    def load(cls, path: str | Path) -> "QuantumResidualRegressor":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("format") != "india_eta_quantum_residual_v1":
                raise ValueError("unsupported quantum residual checkpoint format")
            config_values = metadata["config"]
            config_values["feature_names"] = tuple(config_values["feature_names"])
            model = cls(CircuitConfig(**config_values), seed=int(metadata["seed"]))
            model.theta = np.asarray(archive["theta"], dtype=np.float64).copy()
            model.readout = np.asarray(archive["readout"], dtype=np.float64).copy()
            model.residual_scale_minutes = float(metadata["residual_scale_minutes"])
            model.fit_info = dict(metadata.get("fit_info", {}))
        _validate_theta(model.theta, model.architecture)
        if model.readout.shape != (3,) or not np.isfinite(model.readout).all():
            raise ValueError("invalid checkpoint readout parameters")
        return model


def validate_batched_forward(model: QuantumResidualRegressor, features: Any) -> dict[str, Any]:
    """Compare a real 32-row batch against row-wise CPU execution."""
    x = _validate_features(features)
    if len(x) < 32:
        raise ValueError("at least 32 actual feature rows are required")
    batch = x[:32]
    got = model.forward_expectations(batch)
    expected = np.vstack([model.forward_expectations(row)[0] for row in batch])
    return {
        "examples": len(batch),
        "max_abs_expectation_error": float(np.max(np.abs(got - expected))),
        "passed_1e-12": bool(np.max(np.abs(got - expected)) <= 1e-12),
    }


def profile_cpu(
    model: QuantumResidualRegressor,
    features: Any,
    *,
    batch_sizes: tuple[int, ...] = (128, 256, 512),
    repeats: int = 3,
) -> dict[str, Any]:
    """Measure batched forward throughput using supplied real feature records."""
    x = _validate_features(features)
    if len(x) < max(batch_sizes):
        raise ValueError(f"profile requires at least {max(batch_sizes)} actual feature rows")
    if repeats < 1 or any(size < 1 for size in batch_sizes):
        raise ValueError("batch sizes and repeats must be positive")
    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        batch = x[:batch_size]
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        before_batches = model.circuit_batch_evaluations
        before_examples = model.circuit_example_evaluations
        for _ in range(repeats):
            model.forward_expectations(batch)
        wall = time.perf_counter() - start_wall
        cpu = time.process_time() - start_cpu
        examples = batch_size * repeats
        rows.append({
            "batch_size": batch_size,
            "repeats": repeats,
            "examples_evaluated": examples,
            "circuit_batches": model.circuit_batch_evaluations - before_batches,
            "circuit_example_evaluations": model.circuit_example_evaluations - before_examples,
            "wall_seconds": wall,
            "cpu_seconds": cpu,
            "examples_per_second": examples / wall if wall else None,
            "backend": "NumPy batched statevector / CPU",
            "cuda": "unavailable; PyTorch is not installed in the profiled runtime",
            "vram_bytes": None,
        })
    return {
        "profile_type": "forward_throughput",
        "architecture": model.architecture,
        "real_feature_rows": len(x),
        "circuit": model.circuit_description,
        "results": rows,
    }


def profile_gradient_cpu(
    model: QuantumResidualRegressor,
    features: Any,
    residual_minutes: Any,
    *,
    batch_sizes: tuple[int, ...] = (128, 256, 512),
    repeats: int = 5,
    workload_label: str = "caller-supplied data; semantics must be stated by caller",
) -> dict[str, Any]:
    """Time the complete batched MSE plus parameter-shift gradient kernel.

    This is a profiling helper only; it performs no optimizer updates or model
    selection. Throughput claims must retain the caller's workload label.
    """
    x = _validate_features(features)
    y = np.asarray(residual_minutes, dtype=np.float64).reshape(-1)
    if len(y) != len(x) or not np.isfinite(y).all():
        raise ValueError("residual minutes must be finite and align with profile features")
    if len(x) < max(batch_sizes):
        raise ValueError(f"gradient profile requires {max(batch_sizes)} supplied rows")
    if model.residual_scale_minutes is None:
        raise ValueError("set a train-only residual_scale_minutes before profiling")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        xb = x[:batch_size]
        yb = y[:batch_size] / model.residual_scale_minutes
        before_batches = model.circuit_batch_evaluations
        before_examples = model.circuit_example_evaluations
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        for _ in range(repeats):
            model._normalized_mse_gradient(xb, yb)
        wall = time.perf_counter() - start_wall
        cpu = time.process_time() - start_cpu
        examples = batch_size * repeats
        rows.append({
            "batch_size": batch_size,
            "gradient_steps": repeats,
            "examples_per_gradient_step": batch_size,
            "training_examples_processed": examples,
            "circuit_batches": model.circuit_batch_evaluations - before_batches,
            "circuit_example_evaluations": model.circuit_example_evaluations - before_examples,
            "wall_seconds": wall,
            "cpu_seconds": cpu,
            "examples_per_second": examples / wall if wall else None,
            "backend": "NumPy batched statevector / CPU",
            "cuda_supported_by_profiled_runtime": False,
            "vram_bytes": None,
        })
    return {
        "profile_type": "MSE_plus_parameter_shift_gradient_kernel",
        "evidence_label": workload_label,
        "architecture": model.architecture,
        "parameter_count": model.parameter_count,
        "real_feature_rows_supplied": len(x),
        "optimizer_updates": 0,
        "fit_performed": False,
        "circuit": model.circuit_description,
        "results": rows,
    }
