"""RUN 1 model wrapper around the reviewed V1.1 two-qubit VQC.

The model keeps the established two-feature/two-circuit-angle map and adds
the V1.1 affine readout parameters. No gates or feature dimensions are changed
here. Raw features are transformed only by a scaler fitted on training rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Sequence

from backend.quantum_learning import _cnot, _ry, expectation_z0


FEATURE_COUNT = 2
V1_1_PARAMETER_COUNT = 4
LOGIT_LIMIT = 30.0
PARAMETER_SHIFT_EVALUATIONS = 5
BASE_VARIANT = "v1_1_base"
DEPTH2_VARIANT = "run1_two_ansatz_blocks"
NON_ENTANGLING_VARIANT = "run1_non_entangling_ablation"


def _finite_pair(values: Sequence[float], name: str) -> tuple[float, float]:
    """Validate at least two finite float values (VQC contract), returning the first pair."""
    if len(values) < 2:
        raise ValueError(f"{name} must contain at least two values")
    pair = (float(values[0]), float(values[1]))
    if not all(math.isfinite(value) for value in pair):
        raise ValueError(f"{name} values must be finite")
    return pair


def _finite_vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    """Validate N finite float values for expanded feature sets."""
    vec = tuple(float(v) for v in values)
    if not vec:
        raise ValueError(f"{name} must be non-empty")
    if not all(math.isfinite(v) for v in vec):
        raise ValueError(f"{name} values must be finite")
    return vec


@dataclass(frozen=True)
class FeatureScaler:
    """N-feature min/max scaler with bounds learned from train rows only.

    Generalised from the original 2-feature version to support expanded
    feature sets.  The VQC still consumes features[0:2] via its own
    _finite_pair guard; GBDT and logistic use all N features.
    """

    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    fit_group_ids: tuple[str, ...] = ()

    @classmethod
    def fit(
        cls,
        rows: Iterable[Sequence[float]],
        *,
        group_ids: Iterable[str] = (),
    ) -> "FeatureScaler":
        values = [_finite_vector(row, "feature row") for row in rows]
        if not values:
            raise ValueError("cannot fit feature scaler without training rows")
        dim = len(values[0])
        if any(len(row) != dim for row in values):
            raise ValueError("all feature rows must have the same dimensionality")
        return cls(
            minimum=tuple(min(row[i] for row in values) for i in range(dim)),
            maximum=tuple(max(row[i] for row in values) for i in range(dim)),
            fit_group_ids=tuple(sorted(set(str(value) for value in group_ids))),
        )

    @property
    def feature_count(self) -> int:
        return len(self.minimum)

    def transform(self, features: Sequence[float]) -> tuple[float, ...]:
        raw = _finite_vector(features, "features")
        if len(raw) != self.feature_count:
            raise ValueError(
                f"expected {self.feature_count} features, got {len(raw)}"
            )
        scaled = []
        for index, value in enumerate(raw):
            span = self.maximum[index] - self.minimum[index]
            normalized = 0.5 if span == 0.0 else (value - self.minimum[index]) / span
            # Values outside the observed train range are clipped, never refit.
            scaled.append(min(1.0, max(0.0, normalized)))
        return tuple(scaled)

    def transform_pair(self, features: Sequence[float]) -> tuple[float, float]:
        """Scale all N features, then return only the first two for the VQC."""
        full = self.transform(features)
        return (full[0], full[1])

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "train_only_min_max",
            "feature_count": self.feature_count,
            "minimum": list(self.minimum),
            "maximum": list(self.maximum),
            "fit_group_ids": list(self.fit_group_ids),
            "out_of_range": "clip_to_[0,1]",
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class VQCModel:
    """V1.1 two-qubit VQC plus explicitly versioned RUN 1 ablations.

    Parameter order for V1.1 is ``(theta0, theta1, w_obs, b_obs)``. The
    baseline is the reviewed one-block ansatz:
    RY(pi*x0) q0, RY(pi*x1) q1, CX(0,1), RY(theta0) q0,
    RY(theta1) q1, CX(1,0), measured as <Z0>. The readout is
    sigmoid(w_obs * <Z0> + b_obs). The fixed-readout baseline uses the
    original two angles and sigmoid(-2 * <Z0>). The two-block and
    non-entangling variants are separate ablations and never replace V1.1.
    """

    parameters: tuple[float, ...]
    scaler: FeatureScaler | None = None
    model_version: str = "debugleaf-vqc-v1.1-run1"
    circuit_variant: str = BASE_VARIANT

    def __post_init__(self) -> None:
        allowed = {
            BASE_VARIANT: (2, 4),
            DEPTH2_VARIANT: (6,),
            NON_ENTANGLING_VARIANT: (4,),
        }
        if self.circuit_variant not in allowed or len(self.parameters) not in allowed[self.circuit_variant]:
            raise ValueError("parameter count does not match the selected circuit variant")
        self.parameters = tuple(float(value) for value in self.parameters)
        if not all(math.isfinite(value) for value in self.parameters):
            raise ValueError("model parameters must be finite")

    @property
    def trainable_readout(self) -> bool:
        return len(self.parameters) > self._circuit_angle_count

    @property
    def _circuit_angle_count(self) -> int:
        return 4 if self.circuit_variant == DEPTH2_VARIANT else 2

    @property
    def circuit_spec(self) -> dict[str, object]:
        profile = {
            BASE_VARIANT: {"ansatz_blocks": 1, "ry_gates": 4, "cnot_gates": 2, "depth": 4},
            DEPTH2_VARIANT: {"ansatz_blocks": 2, "ry_gates": 6, "cnot_gates": 4, "depth": 7},
            NON_ENTANGLING_VARIANT: {"ansatz_blocks": 1, "ry_gates": 4, "cnot_gates": 0, "depth": 2},
        }[self.circuit_variant]
        return {
            "architecture": self.circuit_variant,
            "feature_count": FEATURE_COUNT,
            "feature_encoding": ["RY(pi*x0) q0", "RY(pi*x1) q1"],
            "gates": _gate_sequence(self.circuit_variant),
            "ansatz_blocks": profile["ansatz_blocks"],
            "gate_counts": {"RY": profile["ry_gates"], "CNOT": profile["cnot_gates"]},
            "depth": profile["depth"],
            "trainable_parameters": list(self.parameters),
            "observable": "Pauli Z on q0",
            "readout": (
                "sigmoid(w_obs * <Z0> + b_obs)"
                if self.trainable_readout else "sigmoid(-2 * <Z0>)"
            ),
            "execution": "exact local statevector; zero shots",
        }

    def transform(self, raw_features: Sequence[float]) -> tuple[float, float]:
        if self.scaler is not None:
            return self.scaler.transform_pair(raw_features)
        features = _finite_pair(raw_features, "features")
        if any(value < 0.0 or value > 1.0 for value in features):
            raise ValueError("unscaled features must lie in [0, 1]")
        return features

    def predict_proba(self, raw_features: Sequence[float]) -> float:
        return self.predict_proba_scaled(self.transform(raw_features))

    def predict_proba_scaled(self, features: Sequence[float]) -> float:
        x = _finite_pair(features, "scaled features")
        if any(value < 0.0 or value > 1.0 for value in x):
            raise ValueError("scaled features must lie in [0, 1]")
        z0 = _variant_expectation(x, self.parameters[:self._circuit_angle_count], self.circuit_variant)
        if not self.trainable_readout:
            return _sigmoid(-2.0 * z0)
        weight, bias = self.parameters[-2:]
        return _sigmoid(_clip_logit(weight * z0 + bias))

    def predict_proba_qpiai_local(self, raw_features: Sequence[float]) -> float:
        """Execute this model's circuit in QpiAI's local statevector simulator.

        The affine readout is evaluated classically after SDK statevector
        execution. Experimental circuit variants have no parity claim here.
        """
        try:
            from qpiai_quantum import Circuit
            from qpiai_quantum.simulator import StatevectorSimulator
        except ImportError as exc:
            raise RuntimeError("QpiAI Quantum SDK is unavailable in this environment") from exc
        features = self.transform(raw_features)
        circuit = Circuit(2, name="RUN1_VQC_Parity")
        circuit.ry(0, math.pi * features[0])
        circuit.ry(1, math.pi * features[1])
        angle_count = self._circuit_angle_count
        blocks = 2 if self.circuit_variant == DEPTH2_VARIANT else 1
        entangled = self.circuit_variant != NON_ENTANGLING_VARIANT
        if entangled:
            circuit.cx(0, 1)
        for block in range(blocks):
            circuit.ry(0, self.parameters[2 * block])
            circuit.ry(1, self.parameters[2 * block + 1])
            if entangled:
                circuit.cx(1, 0)
                if block + 1 < blocks:
                    circuit.cx(0, 1)
        result = StatevectorSimulator().run(circuit, shots=1, seed=0)
        state = tuple(complex(value) for value in result.statevector)
        z0 = sum((1.0 if index & 1 == 0 else -1.0) * abs(amplitude) ** 2
                 for index, amplitude in enumerate(state))
        if not self.trainable_readout:
            return _sigmoid(-2.0 * z0)
        weight, bias = self.parameters[-2:]
        return _sigmoid(_clip_logit(weight * z0 + bias))

    def probability_and_gradient(
        self, scaled_features: Sequence[float]
    ) -> tuple[float, tuple[float, ...], int]:
        """Return probability, exact parameter gradient, and circuit eval count.

        The affine readout's logit clipping is part of the function. Outside
        the open clipping interval its derivative is therefore zero; this
        avoids the saturated-logit gradient mismatch in the historical helper.
        """
        x = _finite_pair(scaled_features, "scaled features")
        if any(value < 0.0 or value > 1.0 for value in x):
            raise ValueError("scaled features must lie in [0, 1]")
        angle_count = self._circuit_angle_count
        angles = self.parameters[:angle_count]
        z0 = _variant_expectation(x, angles, self.circuit_variant)
        if not self.trainable_readout:
            raw_logit = -2.0 * z0
        else:
            weight, bias = self.parameters[-2:]
            raw_logit = weight * z0 + bias
        probability = _sigmoid(_clip_logit(raw_logit))
        if raw_logit <= -LOGIT_LIMIT or raw_logit >= LOGIT_LIMIT:
            return probability, tuple(0.0 for _ in self.parameters), 1 + 2 * angle_count
        dz = []
        for index in range(angle_count):
            plus, minus = list(angles), list(angles)
            plus[index] += math.pi / 2.0
            minus[index] -= math.pi / 2.0
            dz.append(0.5 * (
                _variant_expectation(x, plus, self.circuit_variant)
                - _variant_expectation(x, minus, self.circuit_variant)
            ))
        dp_dlogit = probability * (1.0 - probability)
        if self.trainable_readout:
            gradient = tuple(dp_dlogit * weight * value for value in dz) + (dp_dlogit * z0, dp_dlogit)
        else:
            gradient = tuple(-2.0 * dp_dlogit * value for value in dz)
        return probability, gradient, 1 + 2 * angle_count

    def with_parameters(self, values: Sequence[float]) -> "VQCModel":
        return VQCModel(tuple(float(value) for value in values), self.scaler, self.model_version, self.circuit_variant)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_version": self.model_version,
            "parameters": list(self.parameters),
            "trainable_readout": self.trainable_readout,
            "circuit_variant": self.circuit_variant,
            "scaler": self.scaler.to_dict() if self.scaler else None,
            "circuit": self.circuit_spec,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class LogisticModel:
    """Classical probabilistic linear baseline using the same train scaler."""

    weights: tuple[float, ...]
    bias: float
    scaler: FeatureScaler
    model_version: str = "run1-logistic-v1"

    def predict_proba(self, raw_features: Sequence[float]) -> float:
        x = self.scaler.transform(raw_features)
        logit = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return _sigmoid(logit)


@dataclass
class ScaledEstimatorModel:
    """Adapter for the existing V1.1 NewtonGBDT implementation."""

    estimator: object
    scaler: FeatureScaler
    model_version: str = "run1-v1.1-newton-gbdt"

    def predict_proba(self, raw_features: Sequence[float]) -> float:
        x = self.scaler.transform(raw_features)
        probability = float(self.estimator.predict_proba(x))  # type: ignore[attr-defined]
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("classical estimator returned an invalid probability")
        return probability


def _clip_logit(value: float) -> float:
    return min(LOGIT_LIMIT, max(-LOGIT_LIMIT, value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _variant_expectation(
    features: Sequence[float], angles: Sequence[float], variant: str
) -> float:
    if variant == BASE_VARIANT:
        return expectation_z0(features, angles)
    expected_angles = 4 if variant == DEPTH2_VARIANT else 2
    if len(features) != FEATURE_COUNT or len(angles) != expected_angles:
        raise ValueError("feature or angle count does not match circuit variant")
    if any(not math.isfinite(float(value)) for value in (*features, *angles)):
        raise ValueError("circuit inputs must be finite")
    if any(float(value) < 0.0 or float(value) > 1.0 for value in features):
        raise ValueError("features must lie in [0, 1]")
    state = [1 + 0j, 0j, 0j, 0j]
    _ry(state, 0, math.pi * float(features[0]))
    _ry(state, 1, math.pi * float(features[1]))
    blocks = 2 if variant == DEPTH2_VARIANT else 1
    entangled = variant != NON_ENTANGLING_VARIANT
    for block in range(blocks):
        if entangled:
            _cnot(state, 0, 1)
        _ry(state, 0, float(angles[2 * block]))
        _ry(state, 1, float(angles[2 * block + 1]))
        if entangled:
            _cnot(state, 1, 0)
    return sum((1.0 if index & 1 == 0 else -1.0) * abs(amplitude) ** 2
               for index, amplitude in enumerate(state))


def _gate_sequence(variant: str) -> list[str]:
    gates = ["RY(pi*x0) q0", "RY(pi*x1) q1"]
    blocks = 2 if variant == DEPTH2_VARIANT else 1
    entangled = variant != NON_ENTANGLING_VARIANT
    for block in range(blocks):
        suffix = f"[{block + 1}]" if blocks > 1 else ""
        if entangled:
            gates.append("CX(0,1)" + suffix)
        gates.extend((f"RY(theta{2 * block}) q0", f"RY(theta{2 * block + 1}) q1"))
        if entangled:
            gates.append("CX(1,0)" + suffix)
    return gates
