"""Differentiable batched two-qubit ETA residual circuits for RUN2 India.

This file is separate from ``quantum.py``: it provides an autograd/PyTorch
implementation while leaving the earlier NumPy parameter-shift reference
unchanged. It has no QAOA or cloud path. Model outputs are residual minutes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Literal, Sequence

import numpy as np

try:  # Import stays possible in the base runtime where torch is not installed.
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - exercised only in minimal runtime
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]


Architecture = Literal["one_block_2feature", "two_stage_reupload_4feature"]
BASE_FEATURES = ("vqc_x0_distance", "vqc_x1_congestion")
REUPLOAD_FEATURES = (
    "vqc_x0_distance", "vqc_x1_congestion",
    "vqc_x2_multidelivery", "vqc_x3_preptime",
)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is unavailable; use the isolated RUN2 training runtime")
    return torch


@dataclass(frozen=True)
class TorchCircuitConfig:
    architecture: Architecture
    feature_names: tuple[str, ...]
    seed: int = 17
    dtype: str = "float64"

    def __post_init__(self) -> None:
        expected = BASE_FEATURES if self.architecture == "one_block_2feature" else REUPLOAD_FEATURES
        if self.architecture not in ("one_block_2feature", "two_stage_reupload_4feature"):
            raise ValueError("unsupported RUN2 torch architecture")
        if tuple(self.feature_names) != expected:
            raise ValueError(f"{self.architecture} requires ordered features {expected}")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")


def _features_numpy(features: Any, n_features: int) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[1] != n_features or x.shape[0] == 0:
        raise ValueError(f"features must have shape (n,{n_features}) and be non-empty")
    if not np.isfinite(x).all() or np.any(x < -1e-12) or np.any(x > 1.0 + 1e-12):
        raise ValueError("scaled VQC features must be finite and lie in [0,1]")
    return np.clip(x, 0.0, 1.0)


def _torch_ry(states: Any, wire: int, angle: Any) -> Any:
    """Batched, differentiable real-valued RY on little-endian wire 0 or 1."""
    t = _require_torch()
    if angle.ndim == 0:
        angle = angle.expand(states.shape[0])
    c, s = t.cos(angle / 2.0), t.sin(angle / 2.0)
    c, s = c[:, None], s[:, None]
    a = states
    if wire == 0:
        return t.stack((c[:, 0] * a[:, 0] - s[:, 0] * a[:, 1],
                        s[:, 0] * a[:, 0] + c[:, 0] * a[:, 1],
                        c[:, 0] * a[:, 2] - s[:, 0] * a[:, 3],
                        s[:, 0] * a[:, 2] + c[:, 0] * a[:, 3]), dim=1)
    if wire == 1:
        return t.stack((c[:, 0] * a[:, 0] - s[:, 0] * a[:, 2],
                        c[:, 0] * a[:, 1] - s[:, 0] * a[:, 3],
                        s[:, 0] * a[:, 0] + c[:, 0] * a[:, 2],
                        s[:, 0] * a[:, 1] + c[:, 0] * a[:, 3]), dim=1)
    raise ValueError("wire must be 0 or 1")


def _torch_cx(states: Any, control: int, target: int) -> Any:
    if (control, target) == (0, 1):
        return states[:, [0, 3, 2, 1]]
    if (control, target) == (1, 0):
        return states[:, [0, 1, 3, 2]]
    raise ValueError("distinct two-qubit CNOT wires required")


def _torch_expectations(x: Any, theta: Any, architecture: Architecture) -> Any:
    t = _require_torch()
    n_features = 2 if architecture == "one_block_2feature" else 4
    if x.ndim != 2 or x.shape[1] != n_features:
        raise ValueError(f"{architecture} feature tensor has wrong shape")
    n_angles = 2 if architecture == "one_block_2feature" else 4
    if theta.ndim != 1 or theta.shape[0] != n_angles:
        raise ValueError(f"{architecture} requires {n_angles} angles")
    states = t.zeros((x.shape[0], 4), dtype=x.dtype, device=x.device)
    states = states + t.as_tensor([1.0, 0.0, 0.0, 0.0], dtype=x.dtype, device=x.device)
    states = _torch_ry(states, 0, math.pi * x[:, 0])
    states = _torch_ry(states, 1, math.pi * x[:, 1])
    if architecture == "one_block_2feature":
        states = _torch_cx(states, 0, 1)
        states = _torch_ry(states, 0, theta[0])
        states = _torch_ry(states, 1, theta[1])
        states = _torch_cx(states, 1, 0)
    else:
        # Stage 1 uses distance/congestion; stage 2 reuploads workload/prep.
        states = _torch_ry(states, 0, theta[0])
        states = _torch_ry(states, 1, theta[1])
        states = _torch_cx(states, 0, 1)
        states = _torch_ry(states, 0, math.pi * x[:, 2])
        states = _torch_ry(states, 1, math.pi * x[:, 3])
        states = _torch_ry(states, 0, theta[2])
        states = _torch_ry(states, 1, theta[3])
        states = _torch_cx(states, 1, 0)
    probs = states.square()
    z0 = probs[:, 0] - probs[:, 1] + probs[:, 2] - probs[:, 3]
    z1 = probs[:, 0] + probs[:, 1] - probs[:, 2] - probs[:, 3]
    return t.stack((z0, z1), dim=1)


def _make_net(config: TorchCircuitConfig, dtype: Any, device: Any) -> Any:
    t = _require_torch()

    class _Net(t.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            angle_count = 2 if config.architecture == "one_block_2feature" else 4
            generator = t.Generator(device="cpu").manual_seed(config.seed)
            angles = t.empty(angle_count, dtype=dtype).uniform_(-0.15, 0.15, generator=generator)
            self.theta = t.nn.Parameter(angles.to(device))
            self.readout = t.nn.Parameter(t.tensor([0.03, -0.03, 0.0], dtype=dtype, device=device))

        def forward(self, x: Any) -> Any:
            exp = _torch_expectations(x, self.theta, config.architecture)
            return exp @ self.readout[:2] + self.readout[2]

    return _Net().to(device=device, dtype=dtype)


class TorchResidualVQC:
    """Torch-autograd residual VQC; prediction units are minutes."""

    def __init__(self, config: TorchCircuitConfig, *, device: str = "cpu") -> None:
        t = _require_torch()
        self.config = config
        self.device = t.device(device)
        if self.device.type == "cuda" and not t.cuda.is_available():
            raise RuntimeError("CUDA device requested but unavailable in this PyTorch runtime")
        self.dtype = t.float64 if config.dtype == "float64" else t.float32
        self.net = _make_net(config, self.dtype, self.device)
        self.residual_scale_minutes: float | None = None
        self.fit_info: dict[str, Any] = {}

    @property
    def architecture(self) -> Architecture:
        return self.config.architecture

    @property
    def parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.net.parameters())

    def forward_normalized(self, features: Any) -> Any:
        x_np = _features_numpy(features, len(self.config.feature_names))
        x = _require_torch().as_tensor(x_np, dtype=self.dtype, device=self.device)
        return self.net(x)

    def forward_expectations(self, features: Any) -> Any:
        x_np = _features_numpy(features, len(self.config.feature_names))
        x = _require_torch().as_tensor(x_np, dtype=self.dtype, device=self.device)
        return _torch_expectations(x, self.net.theta, self.architecture)

    def predict_residual_minutes(self, features: Any, *, batch_size: int = 4096) -> np.ndarray:
        t = _require_torch()
        if self.residual_scale_minutes is None:
            raise RuntimeError("residual scale has not been fitted")
        x = _features_numpy(features, len(self.config.feature_names))
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.net.eval()
        rows: list[np.ndarray] = []
        with t.no_grad():
            for start in range(0, len(x), batch_size):
                xt = t.as_tensor(x[start:start + batch_size], dtype=self.dtype, device=self.device)
                rows.append((self.net(xt) * self.residual_scale_minutes).cpu().numpy())
        return np.concatenate(rows).astype(np.float64, copy=False)

    def gradient_check(self, features: Any, residual_minutes: Any, *, h: float = 1e-5) -> dict[str, Any]:
        t = _require_torch()
        x_np = _features_numpy(features, len(self.config.feature_names))
        y_np = np.asarray(residual_minutes, dtype=np.float64).reshape(-1)
        if len(x_np) != len(y_np) or not np.isfinite(y_np).all() or h <= 0:
            raise ValueError("gradient-check targets must be finite and align with features")
        scale = self.residual_scale_minutes or max(float(np.std(y_np)), 1.0)
        x = t.as_tensor(x_np, dtype=self.dtype, device=self.device)
        y = t.as_tensor(y_np / scale, dtype=self.dtype, device=self.device)
        self.net.zero_grad(set_to_none=True)
        loss = t.mean((self.net(x) - y) ** 2)
        loss.backward()
        params = [p for p in self.net.parameters()]
        analytic = np.concatenate([p.grad.detach().cpu().numpy().reshape(-1) for p in params])
        numeric: list[float] = []
        with t.no_grad():
            for parameter in params:
                flat = parameter.view(-1)
                for idx in range(flat.numel()):
                    original = float(flat[idx].item())
                    flat[idx] = original + h
                    plus = float(t.mean((self.net(x) - y) ** 2).item())
                    flat[idx] = original - h
                    minus = float(t.mean((self.net(x) - y) ** 2).item())
                    flat[idx] = original
                    numeric.append((plus - minus) / (2.0 * h))
        errors = np.abs(analytic - np.asarray(numeric))
        return {
            "architecture": self.architecture,
            "parameters_checked": int(len(analytic)),
            "examples": int(len(x_np)),
            "step": float(h),
            "max_absolute_error": float(errors.max(initial=0.0)),
            "max_relative_error": float((errors / np.maximum(np.abs(numeric), 1e-8)).max(initial=0.0)),
            "passed_abs_1e-5": bool(errors.max(initial=0.0) <= 1e-5),
            "device": str(self.device),
            "method": "PyTorch autograd versus centered finite differences of normalized MSE",
        }

    def fit(
        self, features_train: Any, residual_train_minutes: Any,
        features_validation: Any, residual_validation_minutes: Any, *,
        max_epochs: int = 100, batch_size: int = 256, learning_rate: float = 0.01,
        patience: int = 12, seed: int = 20261029,
        checkpoint_path: str | Path | None = None,
        progress_path: str | Path | None = None,
    ) -> dict[str, Any]:
        t = _require_torch()
        x_train_np = _features_numpy(features_train, len(self.config.feature_names))
        x_val_np = _features_numpy(features_validation, len(self.config.feature_names))
        y_train_np = np.asarray(residual_train_minutes, dtype=np.float64).reshape(-1)
        y_val_np = np.asarray(residual_validation_minutes, dtype=np.float64).reshape(-1)
        if len(y_train_np) != len(x_train_np) or len(y_val_np) != len(x_val_np):
            raise ValueError("feature and residual row counts must match")
        if len(y_train_np) < 2 or len(y_val_np) < 1 or not np.isfinite(y_train_np).all() or not np.isfinite(y_val_np).all():
            raise ValueError("residual labels must be finite and contain at least two train rows")
        if max_epochs <= 0 or batch_size <= 0 or patience <= 0 or learning_rate <= 0:
            raise ValueError("epochs, batch size, patience, and learning rate must be positive")
        # Train-only scale keeps the minibatch objective well-conditioned while
        # `predict_residual_minutes` and checkpoints remain in minute units.
        self.residual_scale_minutes = max(float(np.std(y_train_np)), 1.0)
        x_train = t.as_tensor(x_train_np, dtype=self.dtype, device=self.device)
        x_val = t.as_tensor(x_val_np, dtype=self.dtype, device=self.device)
        y_train = t.as_tensor(y_train_np / self.residual_scale_minutes, dtype=self.dtype, device=self.device)
        y_val = t.as_tensor(y_val_np / self.residual_scale_minutes, dtype=self.dtype, device=self.device)
        generator = t.Generator(device="cpu").manual_seed(int(seed))
        optimizer = t.optim.Adam(self.net.parameters(), lr=learning_rate)
        best_state: dict[str, Any] | None = None
        best_rmse = math.inf
        best_epoch, stale = 0, 0
        history: list[dict[str, float | int]] = []
        checkpoint = Path(checkpoint_path) if checkpoint_path else None
        progress = Path(progress_path) if progress_path else None
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if progress:
            progress.parent.mkdir(parents=True, exist_ok=True)
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        train_circuit_examples = 0
        validation_circuit_examples = 0
        for epoch in range(1, max_epochs + 1):
            self.net.train()
            permutation = t.randperm(len(x_train), generator=generator, device="cpu").to(self.device)
            batch_loss_sum, batch_n = 0.0, 0
            for start in range(0, len(x_train), batch_size):
                idx = permutation[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                pred = self.net(x_train[idx])
                loss = t.mean((pred - y_train[idx]) ** 2)
                loss.backward()
                optimizer.step()
                n = int(len(idx))
                batch_loss_sum += float(loss.detach().item()) * n
                batch_n += n
                # Forward plus one parameter-shift-equivalent autograd circuit
                # evaluation per row; report examples and batch calls separately.
                train_circuit_examples += n
            self.net.eval()
            with t.no_grad():
                val_pred = self.net(x_val)
                val_rmse = float(t.sqrt(t.mean((val_pred - y_val) ** 2)).item() * self.residual_scale_minutes)
            validation_circuit_examples += len(x_val)
            row = {"epoch": epoch,
                   "train_batch_rmse_minutes": math.sqrt(batch_loss_sum / max(batch_n, 1)) * self.residual_scale_minutes,
                   "validation_rmse_minutes": val_rmse}
            history.append(row)
            if progress:
                with progress.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
            if val_rmse < best_rmse - 1e-6:
                best_rmse, best_epoch, stale = val_rmse, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in self.net.state_dict().items()}
            else:
                stale += 1
            # Checkpoint each epoch, including current and best parameters,
            # so an interrupted run has a recoverable progress boundary.
            self.fit_info = {
                "status": "fitting", "epoch": epoch, "best_epoch": best_epoch,
                "best_validation_rmse_minutes": best_rmse,
                "train_rows": len(x_train_np), "validation_rows": len(x_val_np),
                "residual_scale_minutes": self.residual_scale_minutes,
                "architecture": self.architecture,
            }
            if checkpoint:
                self._save_json(checkpoint, best_state or self.net.state_dict(), self.fit_info)
            if stale >= patience:
                break
        if best_state is None:
            raise RuntimeError("VQC training produced no finite validation checkpoint")
        self.net.load_state_dict(best_state)
        self.net.eval()
        self.fit_info = {
            "status": "completed", "epochs_run": len(history), "best_epoch": best_epoch,
            "best_validation_rmse_minutes": best_rmse,
            "train_rows": len(x_train_np), "validation_rows": len(x_val_np),
            "batch_size": batch_size, "learning_rate": learning_rate, "patience": patience,
            "seed": int(seed), "residual_scale_minutes": self.residual_scale_minutes,
            "wall_seconds": time.perf_counter() - start_wall,
            "cpu_seconds": time.process_time() - start_cpu,
            "train_circuit_evaluations": train_circuit_examples,
            "validation_circuit_evaluations": validation_circuit_examples,
            "circuit_evaluations": train_circuit_examples + validation_circuit_examples,
            "circuit_execution_batches": (math.ceil(len(x_train_np) / batch_size) + 1) * len(history),
            "device": str(self.device), "dtype": self.config.dtype, "history": history,
        }
        if checkpoint:
            self.save(checkpoint)
        return dict(self.fit_info)

    def _payload(self, state: Any | None = None, fit_info: dict[str, Any] | None = None) -> dict[str, Any]:
        source = state if state is not None else self.net.state_dict()
        return {
            "format": "india_eta_torch_residual_v2",
            "model_version": "run2_india_torch_residual_v2",
            "config": asdict(self.config),
            "architecture": self.architecture,
            "feature_names": list(self.config.feature_names),
            "parameter_count": self.parameter_count,
            "theta": source["theta"].detach().cpu().numpy().tolist(),
            "readout": source["readout"].detach().cpu().numpy().tolist(),
            "residual_scale_minutes": self.residual_scale_minutes,
            "fit_info": fit_info if fit_info is not None else self.fit_info,
            "circuit": self.circuit_description(),
        }

    def circuit_description(self) -> dict[str, Any]:
        if self.architecture == "one_block_2feature":
            gates = ["RY(pi*x0) q0", "RY(pi*x1) q1", "CX(0,1)", "RY(theta0) q0", "RY(theta1) q1", "CX(1,0)"]
        else:
            gates = ["RY(pi*x0) q0", "RY(pi*x1) q1", "RY(theta0) q0", "RY(theta1) q1", "CX(0,1)",
                     "RY(pi*x2) q0", "RY(pi*x3) q1", "RY(theta2) q0", "RY(theta3) q1", "CX(1,0)"]
        return {"qubits": 2, "architecture": self.architecture,
                "gate_count": len(gates),
                "gate_counts": {"RY": sum(g.startswith("RY(") for g in gates),
                                "CNOT": sum(g.startswith("CX(") for g in gates)},
                "depth": 4 if self.architecture == "one_block_2feature" else 6,
                "depth_definition": "parallel disjoint single-qubit RY gates scheduled together; CNOTs are separate layers",
                "gates": gates, "feature_names": list(self.config.feature_names),
                "observable": ["Z(q0)", "Z(q1)"],
                "readout": "trainable affine combination of two Pauli-Z expectations; minute residual scale applied outside circuit",
                "training_autodiff": "PyTorch reverse-mode; local statevector only"}

    @staticmethod
    def _save_json(path: Path, state: Any, fit_info: dict[str, Any]) -> None:
        payload = {
            "format": "india_eta_torch_residual_v2_progress",
            "config": asdict(self.config),
            "residual_scale_minutes": self.residual_scale_minutes,
            "state": {key: value.detach().cpu().numpy().tolist() for key, value in state.items()},
            "fit_info": fit_info,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def save(self, path: str | Path) -> Path:
        if self.residual_scale_minutes is None:
            raise RuntimeError("cannot save a model without its train-only residual scale")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self._payload()
        tmp = destination.with_suffix(destination.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "TorchResidualVQC":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "india_eta_torch_residual_v2":
            raise ValueError("unsupported final Torch residual checkpoint format")
        config_payload = dict(payload["config"])
        config_payload["feature_names"] = tuple(config_payload["feature_names"])
        model = cls(TorchCircuitConfig(**config_payload), device=device)
        t = _require_torch()
        with t.no_grad():
            model.net.theta.copy_(t.as_tensor(payload["theta"], dtype=model.dtype, device=model.device))
            model.net.readout.copy_(t.as_tensor(payload["readout"], dtype=model.dtype, device=model.device))
        model.residual_scale_minutes = float(payload["residual_scale_minutes"])
        model.fit_info = dict(payload.get("fit_info", {}))
        model.net.eval()
        return model

    def parameter_sha256(self) -> str:
        t = _require_torch()
        payload = {
            "format": "india_eta_torch_residual_model_v2",
            "config": asdict(self.config),
            "theta": self.net.theta.detach().cpu().numpy().tolist(),
            "readout": self.net.readout.detach().cpu().numpy().tolist(),
            "residual_scale_minutes": self.residual_scale_minutes,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def qpiai_residual_parity(
    model: TorchResidualVQC, features: Any, *, tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Compare reloaded model predictions with QpiAI local statevector output."""
    t = _require_torch()
    if model.residual_scale_minutes is None:
        raise RuntimeError("residual scale is missing")
    x = _features_numpy(features, len(model.config.feature_names))
    try:
        from qpiai_quantum import Circuit
        from qpiai_quantum.simulator import StatevectorSimulator
        from qpiai_quantum import __version__ as qpi_version
    except (ImportError, AttributeError):
        from importlib.metadata import version
        try:
            from qpiai_quantum import Circuit
            from qpiai_quantum.simulator import StatevectorSimulator
        except ImportError as exc:
            raise RuntimeError("QpiAI Quantum local simulator is unavailable") from exc
        qpi_version = version("qpiai-quantum")
    theta = model.net.theta.detach().cpu().numpy()
    readout = model.net.readout.detach().cpu().numpy()
    simulator = StatevectorSimulator()
    sdk_predictions: list[float] = []
    model_predictions = model.predict_residual_minutes(x)
    for row in x:
        circuit = Circuit(2, name="RUN2IndiaResidualParity")
        circuit.ry(0, math.pi * float(row[0]))
        circuit.ry(1, math.pi * float(row[1]))
        if model.architecture == "one_block_2feature":
            circuit.cx(0, 1)
            circuit.ry(0, float(theta[0]))
            circuit.ry(1, float(theta[1]))
            circuit.cx(1, 0)
        else:
            circuit.ry(0, float(theta[0]))
            circuit.ry(1, float(theta[1]))
            circuit.cx(0, 1)
            circuit.ry(0, math.pi * float(row[2]))
            circuit.ry(1, math.pi * float(row[3]))
            circuit.ry(0, float(theta[2]))
            circuit.ry(1, float(theta[3]))
            circuit.cx(1, 0)
        result = simulator.run(circuit, shots=1, seed=0)
        state = np.asarray(result.statevector, dtype=np.complex128)
        if state.shape != (4,) or abs(float(np.sum(np.abs(state) ** 2)) - 1.0) > 1e-9:
            raise RuntimeError("QpiAI returned an invalid local statevector")
        probability = np.abs(state) ** 2
        z0 = probability[0] - probability[1] + probability[2] - probability[3]
        z1 = probability[0] + probability[1] - probability[2] - probability[3]
        sdk_predictions.append(float(model.residual_scale_minutes * (readout[0] * z0 + readout[1] * z1 + readout[2])))
    errors = np.abs(model_predictions.reshape(-1) - np.asarray(sdk_predictions))
    return {
        "execution_type": "QpiAI Quantum local StatevectorSimulator",
        "sdk_version": str(qpi_version), "architecture": model.architecture,
        "row_count": len(x), "max_absolute_error_minutes": float(errors.max(initial=0.0)),
        "tolerance_minutes": float(tolerance),
        "passed": bool(errors.max(initial=0.0) <= tolerance),
        "remote_calls": 0, "qpu_jobs": 0,
        "model_sha256": model.parameter_sha256(),
    }


def benchmark_torch_backends(
    config: TorchCircuitConfig, features: Any, residual_minutes: Any, *,
    repetitions: int = 30, warmup: int = 5,
) -> dict[str, Any]:
    """Compare identical full-batch forward+autograd workloads on CPU/CUDA.

    This is a hardware/backend comparison, not a CodSpeed claim. CUDA is
    reported unavailable when the isolated runtime cannot execute it.
    """
    t = _require_torch()
    x_np = _features_numpy(features, len(config.feature_names))
    y_np = np.asarray(residual_minutes, dtype=np.float64).reshape(-1)
    if len(x_np) != len(y_np) or repetitions < 1 or warmup < 0:
        raise ValueError("benchmark inputs or repeat counts are invalid")
    results = []
    cuda_attempt: dict[str, Any] = {
        "torch_build_cuda_version": getattr(t.version, "cuda", None),
        "cuda_is_available": bool(t.cuda.is_available()),
        "attempted": True,
    }
    try:
        probe = t.ones((1,), dtype=t.float32, device="cuda")
        t.cuda.synchronize()
        cuda_attempt.update({"status": "kernel_passed", "device_name": t.cuda.get_device_name(0),
                             "probe_value": float(probe.cpu().item())})
    except Exception as exc:  # record a real device allocation/kernel attempt
        cuda_attempt.update({"status": "unavailable", "error_type": type(exc).__name__,
                             "error": str(exc)[:600]})
    devices = ["cpu"] + (["cuda"] if t.cuda.is_available() else [])
    for device in devices:
        model = TorchResidualVQC(config, device=device)
        model.residual_scale_minutes = max(float(np.std(y_np)), 1.0)
        x = t.as_tensor(x_np, dtype=model.dtype, device=model.device)
        y = t.as_tensor(y_np / model.residual_scale_minutes, dtype=model.dtype, device=model.device)
        def step() -> None:
            model.net.zero_grad(set_to_none=True)
            loss = t.mean((model.net(x) - y) ** 2)
            loss.backward()
        for _ in range(warmup):
            step()
        if device == "cuda":
            t.cuda.synchronize()
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        for _ in range(repetitions):
            step()
        if device == "cuda":
            t.cuda.synchronize()
        wall, cpu = time.perf_counter() - start_wall, time.process_time() - start_cpu
        results.append({"device": device, "batch_rows": len(x_np), "repetitions": repetitions,
                        "warmup": warmup, "wall_seconds": wall, "cpu_seconds": cpu,
                        "examples_per_second": len(x_np) * repetitions / wall if wall else None,
                        "circuit_evaluations": len(x_np) * repetitions,
                        "torch_version": str(t.__version__),
                        "cuda_version": getattr(t.version, "cuda", None),
                        "gpu_name": t.cuda.get_device_name(0) if device == "cuda" else None})
    return {"workload": "same-batch two-qubit statevector MSE forward+autograd gradient; no optimizer update",
            "architecture": config.architecture, "rows": len(x_np), "repetitions": repetitions,
            "cuda_available": bool(t.cuda.is_available()), "cuda_attempt": cuda_attempt,
            "results": results}
