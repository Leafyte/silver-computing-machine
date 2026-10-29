"""Four-qubit, five-block India ETA residual VQC, versioned independently.

The module exposes an auditable NumPy statevector, a batched PyTorch/autograd
model, exact 60->76 transfer, checkpoint payload helpers and a QpiAI local
simulator builder. No cloud client or test partition is involved.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

try:  # Keep the NumPy/reference and SDK paths usable without PyTorch installed.
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - exercised in the bare SDK environment
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]


Variant = Literal["baseline60", "extended76"]
FEATURE_NAMES = (
    "vqc_x0_distance", "vqc_x1_congestion",
    "vqc_x2_multidelivery", "vqc_x3_preptime",
)
N_QUBITS = 4
N_BLOCKS = 5
BASELINE_ANGLE_COUNT = N_BLOCKS * N_QUBITS * 3
EXTENSION_ANGLE_COUNT = 4 * N_QUBITS
ANGLE_COUNT = {"baseline60": BASELINE_ANGLE_COUNT, "extended76": BASELINE_ANGLE_COUNT + EXTENSION_ANGLE_COUNT}
READOUT_COUNT = N_QUBITS + 1
FORMAT = "india_eta_vqc4_run2_v1"
AXES = ("rx", "ry", "rz")
ENTANGLER_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))
REVERSE_ENTANGLER_EDGES = tuple((target, control) for control, target in reversed(ENTANGLER_EDGES))
READOUT_BASIS = (("ry", math.pi / 7.0), ("rx", math.pi / 9.0))


@dataclass(frozen=True)
class CircuitConfig:
    variant: Variant
    feature_names: tuple[str, ...] = FEATURE_NAMES
    seed: int = 20261029
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.variant not in ANGLE_COUNT:
            raise ValueError("variant must be 'baseline60' or 'extended76'")
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise ValueError(f"feature_names must exactly match {FEATURE_NAMES}")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")


def _checked_features(features: Any) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[1] != 4 or x.shape[0] == 0:
        raise ValueError("features must have non-empty shape [batch,4]")
    if not np.isfinite(x).all() or np.any(x < -1e-12) or np.any(x > 1.0 + 1e-12):
        raise ValueError("four VQC features must be finite and already scaled to [0,1]")
    return np.clip(x, 0.0, 1.0)


def _checked_angles(angles: Any, variant: Variant) -> np.ndarray:
    p = np.asarray(angles, dtype=np.float64).reshape(-1)
    if p.shape != (ANGLE_COUNT[variant],) or not np.isfinite(p).all():
        raise ValueError(f"{variant} requires {ANGLE_COUNT[variant]} finite quantum angles")
    return p


def _checked_readout(readout: Any) -> np.ndarray:
    w = np.asarray(readout, dtype=np.float64).reshape(-1)
    if w.shape != (READOUT_COUNT,) or not np.isfinite(w).all():
        raise ValueError("readout must contain four finite Z weights and one bias")
    return w


def _rotation_numpy(states: np.ndarray, wire: int, axis: str, angle: float | np.ndarray) -> np.ndarray:
    """Apply one vectorized single-qubit rotation; basis index is little-endian."""
    stride = 1 << wire
    a = np.asarray(angle, dtype=np.float64)
    if a.ndim == 0:
        c, s = math.cos(float(a) / 2.0), math.sin(float(a) / 2.0)
        c = np.full(states.shape[0], c, dtype=np.float64)
        s = np.full(states.shape[0], s, dtype=np.float64)
    else:
        if a.shape != (states.shape[0],):
            raise ValueError("batched gate angle must have one value per state")
        c, s = np.cos(a / 2.0), np.sin(a / 2.0)
    out = states.copy()
    for base in range(16):
        if base & stride:
            continue
        partner = base | stride
        left, right = states[:, base], states[:, partner]
        if axis == "rx":
            out[:, base] = c * left - 1j * s * right
            out[:, partner] = -1j * s * left + c * right
        elif axis == "ry":
            out[:, base] = c * left - s * right
            out[:, partner] = s * left + c * right
        elif axis == "rz":
            out[:, base] = (c - 1j * s) * left
            out[:, partner] = (c + 1j * s) * right
        else:
            raise ValueError(f"unknown rotation axis {axis!r}")
    return out


def _cnot_numpy(states: np.ndarray, control: int, target: int) -> np.ndarray:
    control_mask, target_mask = 1 << control, 1 << target
    permutation = [i ^ target_mask if (i & control_mask) else i for i in range(16)]
    return states[:, permutation]


def _forward_numpy_states(features: Any, quantum_angles: Any, variant: Variant) -> np.ndarray:
    x = _checked_features(features)
    angles = _checked_angles(quantum_angles, variant)
    batch = len(x)
    states = np.zeros((batch, 16), dtype=np.complex128)
    states[:, 0] = 1.0
    for block in range(N_BLOCKS):
        # Genuine four-feature data re-upload on every block.
        for wire in range(N_QUBITS):
            states = _rotation_numpy(states, wire, "ry", math.pi * x[:, wire])
        for wire in range(N_QUBITS):
            offset = (block * N_QUBITS + wire) * 3
            states = _rotation_numpy(states, wire, "rx", float(angles[offset]))
        for control, target in ENTANGLER_EDGES:
            states = _cnot_numpy(states, control, target)
        for wire in range(N_QUBITS):
            offset = (block * N_QUBITS + wire) * 3
            states = _rotation_numpy(states, wire, "ry", float(angles[offset + 1]))
        for control, target in REVERSE_ENTANGLER_EDGES:
            states = _cnot_numpy(states, control, target)
        for wire in range(N_QUBITS):
            offset = (block * N_QUBITS + wire) * 3
            states = _rotation_numpy(states, wire, "rz", float(angles[offset + 2]))
        if variant == "extended76" and block < 4:
            extra_offset = BASELINE_ANGLE_COUNT + block * N_QUBITS
            for wire in range(N_QUBITS):
                states = _rotation_numpy(states, wire, "rx", float(angles[extra_offset + wire]))
    # Rotate into a fixed noncommuting basis so final Z readout is not a bare-Z gauge.
    for axis, angle in READOUT_BASIS:
        for wire in range(N_QUBITS):
            states = _rotation_numpy(states, wire, axis, angle)
    return states


def numpy_expectations(features: Any, quantum_angles: Any, variant: Variant = "baseline60") -> np.ndarray:
    """Return four batched Pauli-Z expectations from the custom circuit."""
    states = _forward_numpy_states(features, quantum_angles, variant)
    probabilities = np.abs(states) ** 2
    basis = np.arange(16, dtype=np.uint8)
    signs = np.stack([1.0 - 2.0 * ((basis >> wire) & 1) for wire in range(N_QUBITS)], axis=1)
    return probabilities @ signs


def numpy_statevector(features: Any, quantum_angles: Any, variant: Variant = "baseline60") -> np.ndarray:
    """Return the normalized complex statevector for independent inspection."""
    return _forward_numpy_states(features, quantum_angles, variant)


def parameter_names(variant: Variant) -> tuple[str, ...]:
    names = [f"block{b}.q{q}.{axis}" for b in range(N_BLOCKS) for q in range(N_QUBITS) for axis in AXES]
    if variant == "extended76":
        names.extend(f"extension.block{b}.q{q}.rx" for b in range(4) for q in range(N_QUBITS))
    elif variant != "baseline60":
        raise ValueError("unsupported circuit variant")
    return tuple(names)


def numerical_observable_jacobian(
    features: Any, quantum_angles: Any, variant: Variant, *, step: float = 1e-5,
) -> np.ndarray:
    """Central-difference Jacobian `[sample,observable,quantum_angle]` for audit only."""
    x = _checked_features(features)
    base = _checked_angles(quantum_angles, variant)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("finite-difference step must be positive and finite")
    jac = np.empty((len(x), N_QUBITS, len(base)), dtype=np.float64)
    for index in range(len(base)):
        plus, minus = base.copy(), base.copy()
        plus[index] += step
        minus[index] -= step
        jac[:, :, index] = (numpy_expectations(x, plus, variant) - numpy_expectations(x, minus, variant)) / (2.0 * step)
    return jac


class NumpyFourQubitResidualVQC:
    """Simple batch reference and portable predictor for the two circuit variants."""

    def __init__(self, config: CircuitConfig, *, quantum_angles: Any | None = None,
                 readout: Any | None = None, residual_scale_minutes: float = 1.0) -> None:
        self.config = config
        self.variant = config.variant
        self.residual_scale_minutes = float(residual_scale_minutes)
        if not math.isfinite(self.residual_scale_minutes) or self.residual_scale_minutes <= 0:
            raise ValueError("residual scale must be positive finite minutes")
        rng = np.random.default_rng(config.seed)
        self.quantum_angles = (_checked_angles(quantum_angles, self.variant) if quantum_angles is not None
                               else rng.uniform(-0.12, 0.12, size=ANGLE_COUNT[self.variant]))
        if self.variant == "extended76" and quantum_angles is None:
            self.quantum_angles[BASELINE_ANGLE_COUNT:] = 0.0
        self.readout = (_checked_readout(readout) if readout is not None
                        else np.asarray([0.02, -0.02, 0.015, -0.015, 0.0], dtype=np.float64))

    @property
    def quantum_parameter_count(self) -> int:
        return int(len(self.quantum_angles))

    @property
    def parameter_count(self) -> int:
        return self.quantum_parameter_count + READOUT_COUNT

    def forward_expectations(self, features: Any, quantum_angles: Any | None = None) -> np.ndarray:
        angles = self.quantum_angles if quantum_angles is None else quantum_angles
        return numpy_expectations(features, angles, self.variant)

    def predict_residual_minutes(self, features: Any) -> np.ndarray:
        z = self.forward_expectations(features)
        return self.residual_scale_minutes * (z @ self.readout[:N_QUBITS] + self.readout[-1])

    def export_payload(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "config": asdict(self.config),
            "variant": self.variant,
            "feature_names": list(FEATURE_NAMES),
            "angle_order": list(parameter_names(self.variant)),
            "quantum_angles": self.quantum_angles.tolist(),
            "readout": self.readout.tolist(),
            "residual_scale_minutes": self.residual_scale_minutes,
            "units": {"feature_angles": "radians (pi*x)", "trainable_angles": "radians",
                      "target_and_prediction": "minutes"},
            "circuit": {"qubits": N_QUBITS, "blocks": N_BLOCKS,
                        "entangler_edges": [list(e) for e in ENTANGLER_EDGES],
                        "readout": "four Pauli-Z expectations with affine 4-weight + bias head"},
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NumpyFourQubitResidualVQC":
        verified = validate_payload(payload)
        config = verified["config"]
        return cls(CircuitConfig(verified["variant"], tuple(config["feature_names"]),
                                int(config["seed"]), str(config["dtype"])),
                   quantum_angles=verified["quantum_angles"], readout=verified["readout"],
                   residual_scale_minutes=float(verified["residual_scale_minutes"]))


def jacobian_rank_report(
    features: Any, base_angles: Any, *, extended_angles: Any | None = None,
    step: float = 1e-5, relative_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Compare numerical local rank and extension sensitivity on fixed inputs."""
    x = _checked_features(features)
    base = _checked_angles(base_angles, "baseline60")
    if extended_angles is None:
        extended = np.concatenate((base, np.zeros(EXTENSION_ANGLE_COUNT, dtype=np.float64)))
    else:
        extended = _checked_angles(extended_angles, "extended76")
    base_j = numerical_observable_jacobian(x, base, "baseline60", step=step)
    extended_j = numerical_observable_jacobian(x, extended, "extended76", step=step)
    base_matrix = base_j.reshape(-1, BASELINE_ANGLE_COUNT)
    full_matrix = extended_j.reshape(-1, ANGLE_COUNT["extended76"])
    base_s = np.linalg.svd(base_matrix, compute_uv=False)
    full_s = np.linalg.svd(full_matrix, compute_uv=False)
    def rank(s: np.ndarray) -> int:
        return int(np.sum(s > (s[0] * relative_tolerance if len(s) else 0.0)))
    extension_columns = full_matrix[:, BASELINE_ANGLE_COUNT:]
    extension_norms = np.linalg.norm(extension_columns, axis=0)
    # Per-column residual after least-squares projection onto the baseline tangent.
    projection = base_matrix @ np.linalg.pinv(base_matrix, rcond=relative_tolerance) @ extension_columns
    incremental_norms = np.linalg.norm(extension_columns - projection, axis=0)
    return {
        "variant_baseline": "baseline60", "variant_extension": "extended76",
        "input_rows": int(len(x)), "observable_count": N_QUBITS,
        "finite_difference_step_radians": float(step), "relative_rank_tolerance": relative_tolerance,
        "baseline_local_rank": rank(base_s), "extended_local_rank": rank(full_s),
        "rank_increment": rank(full_s) - rank(base_s),
        "extension_angle_count": EXTENSION_ANGLE_COUNT,
        "extension_gradient_l2_by_angle": extension_norms.tolist(),
        "extension_incremental_l2_by_angle": incremental_norms.tolist(),
        "all_extension_angles_sensitive": bool(np.all(extension_norms > 1e-8)),
        "all_extension_angles_add_local_rank": bool(np.all(incremental_norms > 1e-8)),
        "baseline_singular_values": base_s.tolist(), "extended_singular_values": full_s.tolist(),
        "method": "centered finite differences of four batched Z expectations over diverse rows; no labels used",
    }


def build_qpiai_circuit(features_row: Any, quantum_angles: Any, variant: Variant = "baseline60") -> Any:
    """Build one QpiAI SDK circuit for a small independent parity check."""
    x = _checked_features(features_row)
    if len(x) != 1:
        raise ValueError("SDK builder accepts exactly one feature row; use numpy/torch for batches")
    angles = _checked_angles(quantum_angles, variant)
    try:
        from qpiai_quantum import Circuit
    except ImportError as exc:
        raise RuntimeError("QpiAI Quantum SDK is not installed in this interpreter") from exc
    circuit = Circuit(N_QUBITS, name=f"IndiaETA4Q-{variant}")
    for block in range(N_BLOCKS):
        for wire in range(N_QUBITS):
            circuit.ry(wire, math.pi * float(x[0, wire]))
        for wire in range(N_QUBITS):
            offset = (block * N_QUBITS + wire) * 3
            circuit.rx(wire, float(angles[offset]))
        for control, target in ENTANGLER_EDGES:
            circuit.cx(control, target)
        for wire in range(N_QUBITS):
            offset = (block * N_QUBITS + wire) * 3
            circuit.ry(wire, float(angles[offset + 1]))
        for control, target in REVERSE_ENTANGLER_EDGES:
            circuit.cx(control, target)
        for wire in range(N_QUBITS):
            offset = (block * N_QUBITS + wire) * 3
            circuit.rz(wire, float(angles[offset + 2]))
        if variant == "extended76" and block < 4:
            extra_offset = BASELINE_ANGLE_COUNT + block * N_QUBITS
            for wire in range(N_QUBITS):
                circuit.rx(wire, float(angles[extra_offset + wire]))
    for axis, angle in READOUT_BASIS:
        for wire in range(N_QUBITS):
            getattr(circuit, axis)(wire, angle)
    return circuit


def qpiai_expectations(features: Any, quantum_angles: Any, variant: Variant = "baseline60") -> np.ndarray:
    """Independent local QpiAI statevector expectations; intended for tiny checks."""
    from qpiai_quantum.simulator import StatevectorSimulator
    x = _checked_features(features)
    angles = _checked_angles(quantum_angles, variant)
    simulator = StatevectorSimulator()
    results: list[list[float]] = []
    for row in x:
        result = simulator.run(build_qpiai_circuit(row, angles, variant), shots=1, seed=0)
        state = np.asarray(result.statevector, dtype=np.complex128)
        if state.shape != (16,) or not np.isfinite(state.real).all() or not np.isfinite(state.imag).all():
            raise RuntimeError("QpiAI simulator returned an invalid four-qubit statevector")
        if abs(float(np.vdot(state, state).real) - 1.0) > 1e-10:
            raise RuntimeError("QpiAI statevector is not normalized")
        probability = np.abs(state) ** 2
        results.append([float(sum((1.0 if not ((basis >> q) & 1) else -1.0) * probability[basis]
                                  for basis in range(16))) for q in range(N_QUBITS)])
    return np.asarray(results, dtype=np.float64)


def _torch_required() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is unavailable; install it in the isolated RUN2 runtime before training")
    return torch


def _torch_rotation(states: Any, wire: int, axis: str, angle: Any) -> Any:
    """Differentiable batched rotation on [batch,16], with static wire loops."""
    t = _torch_required()
    stride = 1 << wire
    if not isinstance(angle, t.Tensor):
        angle = t.as_tensor(angle, dtype=states.real.dtype, device=states.device)
    if angle.ndim == 0:
        angle = angle.expand(states.shape[0])
    c, s = t.cos(angle / 2.0)[:, None, None], t.sin(angle / 2.0)[:, None, None]
    blocks = states.reshape(states.shape[0], 16 // (2 * stride), 2, stride)
    left, right = blocks[:, :, 0, :], blocks[:, :, 1, :]
    if axis == "rx":
        new_left = c * left - 1j * s * right
        new_right = -1j * s * left + c * right
    elif axis == "ry":
        new_left = c * left - s * right
        new_right = s * left + c * right
    elif axis == "rz":
        new_left = (c - 1j * s) * left
        new_right = (c + 1j * s) * right
    else:
        raise ValueError(f"unknown rotation axis {axis!r}")
    return t.stack((new_left, new_right), dim=2).reshape(states.shape[0], 16)


def _torch_cnot(states: Any, control: int, target: int) -> Any:
    t = _torch_required()
    cm, tm = 1 << control, 1 << target
    permutation = [i ^ tm if i & cm else i for i in range(16)]
    return states.index_select(1, t.as_tensor(permutation, dtype=t.long, device=states.device))


def torch_expectations(features: Any, quantum_angles: Any, variant: Variant) -> Any:
    """Batched statevector map returning `[batch,4]` differentiable Z values."""
    t = _torch_required()
    if features.ndim != 2 or features.shape[1] != 4:
        raise ValueError("Torch features must have shape [batch,4]")
    if quantum_angles.ndim != 1 or quantum_angles.numel() != ANGLE_COUNT[variant]:
        raise ValueError(f"{variant} requires {ANGLE_COUNT[variant]} trainable angles")
    batch = features.shape[0]
    states = t.zeros((batch, 16), dtype=t.complex128 if features.dtype == t.float64 else t.complex64,
                     device=features.device)
    initial = t.cat((t.ones((batch, 1), dtype=states.dtype, device=states.device),
                     t.zeros((batch, 15), dtype=states.dtype, device=states.device)), dim=1)
    states = states + initial
    for block in range(N_BLOCKS):
        for wire in range(N_QUBITS):
            states = _torch_rotation(states, wire, "ry", math.pi * features[:, wire])
        for wire in range(N_QUBITS):
            offset = (block * N_QUBITS + wire) * 3
            states = _torch_rotation(states, wire, "rx", quantum_angles[offset])
        for control, target in ENTANGLER_EDGES:
            states = _torch_cnot(states, control, target)
        for wire in range(N_QUBITS):
            offset = (block * N_QUBITS + wire) * 3
            states = _torch_rotation(states, wire, "ry", quantum_angles[offset + 1])
        for control, target in REVERSE_ENTANGLER_EDGES:
            states = _torch_cnot(states, control, target)
        for wire in range(N_QUBITS):
            offset = (block * N_QUBITS + wire) * 3
            states = _torch_rotation(states, wire, "rz", quantum_angles[offset + 2])
        if variant == "extended76" and block < 4:
            offset = BASELINE_ANGLE_COUNT + block * N_QUBITS
            for wire in range(N_QUBITS):
                states = _torch_rotation(states, wire, "rx", quantum_angles[offset + wire])
    for axis, angle in READOUT_BASIS:
        for wire in range(N_QUBITS):
            states = _torch_rotation(states, wire, axis, angle)
    probabilities = states.abs().square()
    basis = t.arange(16, device=features.device, dtype=t.long)
    signs = t.stack([1.0 - 2.0 * ((basis >> wire) & 1) for wire in range(N_QUBITS)], dim=1)
    return probabilities @ signs.to(dtype=features.dtype)


if torch is not None:
    class TorchFourQubitVQC(torch.nn.Module):
        """Autograd model with batched complex statevector and minute residual output."""

        def __init__(self, variant: Variant = "baseline60", *,
                     residual_scale_minutes: float = 1.0, seed: int = 20261029,
                     dtype: str = "float64", device: str = "cpu") -> None:
            super().__init__()
            self.config = CircuitConfig(variant, seed=seed, dtype=dtype)
            self.variant = variant
            self.residual_scale_minutes = float(residual_scale_minutes)
            if not math.isfinite(self.residual_scale_minutes) or self.residual_scale_minutes <= 0:
                raise ValueError("residual_scale_minutes must be finite and positive")
            real_dtype = torch.float64 if dtype == "float64" else torch.float32
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            angles = torch.empty(ANGLE_COUNT[variant], dtype=real_dtype)
            angles[:BASELINE_ANGLE_COUNT].uniform_(-0.12, 0.12, generator=generator)
            if variant == "extended76":
                angles[BASELINE_ANGLE_COUNT:] = 0.0
            self.quantum_angles = torch.nn.Parameter(angles.to(device))
            head = torch.zeros(READOUT_COUNT, dtype=real_dtype)
            head[:N_QUBITS] = torch.tensor([0.02, -0.02, 0.015, -0.015], dtype=real_dtype)
            self.readout = torch.nn.Parameter(head.to(device))

        @property
        def quantum_parameter_count(self) -> int:
            return int(self.quantum_angles.numel())

        @property
        def parameter_count(self) -> int:
            return self.quantum_parameter_count + READOUT_COUNT

        def forward_expectations(self, features: Any) -> Any:
            return torch_expectations(features, self.quantum_angles, self.variant)

        def forward_normalized(self, features: Any) -> Any:
            return self.forward_expectations(features) @ self.readout[:N_QUBITS] + self.readout[-1]

        def forward(self, features: Any) -> Any:
            return self.forward_normalized(features) * self.residual_scale_minutes

        def export_payload(self) -> dict[str, Any]:
            return {
                "format": FORMAT,
                "config": asdict(self.config),
                "variant": self.variant,
                "feature_names": list(FEATURE_NAMES),
                "angle_order": list(parameter_names(self.variant)),
                "quantum_angles": self.quantum_angles.detach().cpu().to(torch.float64).numpy().tolist(),
                "readout": self.readout.detach().cpu().to(torch.float64).numpy().tolist(),
                "residual_scale_minutes": self.residual_scale_minutes,
                "units": {"feature_angles": "radians (pi*x)", "trainable_angles": "radians",
                          "target_and_prediction": "minutes"},
                "circuit": {"qubits": N_QUBITS, "blocks": N_BLOCKS, "entangler_edges": [list(e) for e in ENTANGLER_EDGES],
                            "readout": "four Pauli-Z expectations with affine 4-weight + bias head"},
            }

        def load_payload(self, payload: Mapping[str, Any]) -> None:
            payload = validate_payload(payload)
            if payload.get("variant") != self.variant:
                raise ValueError("checkpoint format/variant does not match this VQC")
            angles = _checked_angles(payload.get("quantum_angles"), self.variant)
            readout = _checked_readout(payload.get("readout"))
            scale = float(payload.get("residual_scale_minutes"))
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("checkpoint residual scale must be positive finite minutes")
            with torch.no_grad():
                self.quantum_angles.copy_(torch.as_tensor(angles, dtype=self.quantum_angles.dtype,
                                                           device=self.quantum_angles.device))
                self.readout.copy_(torch.as_tensor(readout, dtype=self.readout.dtype, device=self.readout.device))
                self.residual_scale_minutes = scale

        @classmethod
        def from_payload(cls, payload: Mapping[str, Any], *, device: str = "cpu") -> "TorchFourQubitVQC":
            config = payload.get("config", {})
            model = cls(payload["variant"], residual_scale_minutes=float(payload["residual_scale_minutes"]),
                        seed=int(config.get("seed", 20261029)), dtype=config.get("dtype", "float64"), device=device)
            model.load_payload(payload)
            return model

        @staticmethod
        def expand_baseline(baseline: "TorchFourQubitVQC", *, device: str | None = None) -> "TorchFourQubitVQC":
            if baseline.variant != "baseline60":
                raise ValueError("only a trained baseline60 model can initialize extended76")
            destination = device or str(baseline.quantum_angles.device)
            extended = TorchFourQubitVQC("extended76", residual_scale_minutes=baseline.residual_scale_minutes,
                                         seed=baseline.config.seed, dtype=baseline.config.dtype, device=destination)
            with torch.no_grad():
                extended.quantum_angles[:BASELINE_ANGLE_COUNT].copy_(baseline.quantum_angles.to(destination))
                extended.quantum_angles[BASELINE_ANGLE_COUNT:].zero_()
                extended.readout.copy_(baseline.readout.to(destination))
            return extended
else:  # pragma: no cover - only used without the optional dependency
    class TorchFourQubitVQC:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _torch_required()


def save_payload(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically write a JSON checkpoint with SHA-256 returned to caller."""
    payload = validate_payload(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_payload(path: str | Path) -> dict[str, Any]:
    return validate_payload(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a JSON-compatible checkpoint without importing or loading Torch."""
    if payload.get("format") != FORMAT or payload.get("variant") not in ANGLE_COUNT:
        raise ValueError("unsupported VQC checkpoint format")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("checkpoint feature order is incompatible")
    variant = payload["variant"]
    angles = _checked_angles(payload.get("quantum_angles"), variant)
    readout = _checked_readout(payload.get("readout"))
    scale = float(payload.get("residual_scale_minutes"))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("checkpoint residual scale must be positive finite minutes")
    expected_names = list(parameter_names(variant))
    if payload.get("angle_order") != expected_names:
        raise ValueError("checkpoint parameter ordering is incompatible")
    config = payload.get("config")
    if not isinstance(config, Mapping) or tuple(config.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("checkpoint circuit config is missing the committed feature order")
    # Return a plain JSON-compatible copy with normalized vector/list values.
    result = dict(payload)
    result["quantum_angles"] = angles.tolist()
    result["readout"] = readout.tolist()
    return result


def expand_baseline_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create the exact 76-angle zero-extension of a frozen 60-angle payload."""
    base = validate_payload(payload)
    if base["variant"] != "baseline60":
        raise ValueError("only baseline60 can be expanded to extended76")
    result = dict(base)
    result["variant"] = "extended76"
    result["config"] = dict(base["config"])
    result["config"]["variant"] = "extended76"
    result["angle_order"] = list(parameter_names("extended76"))
    result["quantum_angles"] = list(base["quantum_angles"]) + [0.0] * EXTENSION_ANGLE_COUNT
    result["expansion"] = {
        "source_variant": "baseline60", "appended_angles": "all 16 RX extension angles initialized to exact zero",
        "exact_zero_extension": True,
    }
    return result


def payload_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(canonical).hexdigest()
