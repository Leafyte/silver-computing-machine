"""Unified QpiAI Quantum and Optimization Service for DebugLeaf.

Provides seamless integration with:
  1. QpiAI Quantum SDK & QpiAI Cloud (StatevectorSimulator, remote QCloud job dispatch)
  2. QpiAI Opt SDK (Algebraic modeling, QUBO generation, Warm-Start QAOA)

Includes automatic credential loading from `.env` and graceful fallbacks from
cloud endpoints to local high-precision simulators.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

# Ensure .env is loaded
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


class QpiAIService:
    """Manager for QpiAI Quantum and Optimization capabilities."""

    def __init__(
        self,
        opt_api_key: str | None = None,
        qcloud_api_key: str | None = None,
    ) -> None:
        self.opt_api_key = opt_api_key or os.environ.get("QPIAI_OPT_API_KEY", "")
        self.qcloud_api_key = qcloud_api_key or os.environ.get("QPIAI_QCLOUD_API_KEY", "")
        self._quantum_authenticated = False
        self._opt_authenticated = False

    # -------------------------------------------------------------------------
    # Authentication & Diagnostics
    # -------------------------------------------------------------------------

    def check_connection(self) -> dict[str, Any]:
        """Perform a full diagnostic check on both Quantum and Opt services."""
        status: dict[str, Any] = {
            "quantum": {"configured": bool(self.qcloud_api_key), "authenticated": False, "details": None},
            "opt": {"configured": bool(self.opt_api_key), "authenticated": False, "details": None},
        }

        # Check Quantum Cloud
        if self.qcloud_api_key:
            try:
                from qpiai_quantum import QpiAIQuantumAuth
                QpiAIQuantumAuth.login(api_key=self.qcloud_api_key)
                user = QpiAIQuantumAuth.me()
                self._quantum_authenticated = True
                status["quantum"]["authenticated"] = True
                status["quantum"]["details"] = {
                    "user_name": user.get("name", "Unknown"),
                    "email": user.get("email", "Unknown"),
                    "supported_backends": [
                        "QpiAI-QSV-Simulator",
                        "QpiAI-QDM-Simulator",
                        "QpiAI-Indus-1",
                    ],
                }
            except Exception as exc:
                status["quantum"]["error"] = str(exc)

        # Check Opt Platform
        if self.opt_api_key:
            try:
                from qpiai_opt.opt_solvers.client import Client
                client = Client(api_key=self.opt_api_key)
                user = client.me()
                self._opt_authenticated = True
                status["opt"]["authenticated"] = True
                sub = user.get("active_subscription", {})
                status["opt"]["details"] = {
                    "user_name": user.get("name", "Unknown"),
                    "email": user.get("email", "Unknown"),
                    "organization_id": user.get("organization_id"),
                    "subscription_status": sub.get("status", "inactive"),
                    "cloud_credits": sub.get("credits", 0),
                }
            except Exception as exc:
                status["opt"]["error"] = str(exc)

        return status

    # -------------------------------------------------------------------------
    # Quantum Circuit Execution (Local & Cloud)
    # -------------------------------------------------------------------------

    def predict_route_risk(
        self,
        features: Sequence[float] = (0.35, 0.25),
        theta: Sequence[float] = (0.31, -0.47),
        *,
        shots: int = 100,
        use_cloud: bool = False,
    ) -> dict[str, Any]:
        """Execute the 2-qubit decision VQC to predict route SLA breach risk.

        Args:
            features: [time_buffer_slack, stop_density] in [0, 1]
            theta: Trainable variational angles in radians
            shots: Measurement shot budget
            use_cloud: Whether to dispatch to QpiAI Quantum Cloud (falls back to local if unavailable)
        """
        if len(features) != 2 or len(theta) != 2:
            raise ValueError("Expected 2 features and 2 theta parameters")

        x0, x1 = float(features[0]), float(features[1])
        th0, th1 = float(theta[0]), float(theta[1])

        from qpiai_quantum import Circuit
        from qpiai_quantum.simulator import StatevectorSimulator

        # Build parameterized VQC
        circuit = Circuit(2, name="DebugLeafDecisionVQC")
        circuit.ry(0, math.pi * x0)
        circuit.ry(1, math.pi * x1)
        circuit.cx(0, 1)
        circuit.ry(0, th0)
        circuit.ry(1, th1)
        circuit.cx(1, 0)

        # 1. Local Statevector Simulation
        sim = StatevectorSimulator()
        sim_result = sim.run(circuit, shots=shots, seed=42)
        statevector = tuple(complex(a) for a in getattr(sim_result, "statevector", ()))

        # Expectation value <Z0> on wire 0 (bit 0)
        z0 = sum((1.0 if idx & 1 == 0 else -1.0) * abs(amp) ** 2 for idx, amp in enumerate(statevector))
        # Logistic sigmoid risk: P(breach) = 1 / (1 + exp(2 * <Z0>))
        risk_probability = 1.0 / (1.0 + math.exp(2.0 * z0))

        result: dict[str, Any] = {
            "status": "success",
            "execution_backend": "QpiAI-QSV-Local",
            "features": {"time_slack": x0, "density": x1},
            "theta": {"th0": th0, "th1": th1},
            "expectation_z0": float(z0),
            "breach_risk_probability": float(risk_probability),
            "dispatch_recommendation": (
                "SAFE_DISPATCH" if risk_probability < 0.35
                else "MODERATE_RISK" if risk_probability < 0.65
                else "HIGH_RISK_BREACH"
            ),
            "state_probabilities": {
                f"{i:02b}": float(abs(a) ** 2) for i, a in enumerate(statevector)
            },
        }

        # 2. Optional Cloud Dispatch
        if use_cloud and self.qcloud_api_key:
            try:
                from qpiai_quantum import JobManager, QpiAIQuantumAuth
                QpiAIQuantumAuth.login(api_key=self.qcloud_api_key)
                
                cloud_circuit = Circuit(2, name="DebugLeafDecisionVQC")
                cloud_circuit.ry(0, math.pi * x0)
                cloud_circuit.ry(1, math.pi * x1)
                cloud_circuit.cx(0, 1)
                cloud_circuit.ry(0, th0)
                cloud_circuit.ry(1, th1)
                cloud_circuit.cx(1, 0)
                cloud_circuit.measure_all()

                job_mgr = JobManager()
                sub = job_mgr.submit_circuit_job(
                    circuit=cloud_circuit,
                    experiment_name="Default Experiment",
                    shots=shots,
                    device_name="QpiAI-QSV-Simulator",
                )
                result["cloud_job"] = {
                    "status": "submitted",
                    "job_id": sub.get("job_id"),
                    "device": "QpiAI-QSV-Simulator",
                }
                result["execution_backend"] = "QpiAI-QSV-Simulator (Cloud)"
            except Exception as exc:
                result["cloud_job"] = {
                    "status": "fallback_to_local",
                    "note": f"Cloud submission deferred ({exc}); local exact simulation preserved.",
                }

        return result

    # -------------------------------------------------------------------------
    # Optimization & QUBO Modeling (QpiAI Opt)
    # -------------------------------------------------------------------------

    def build_routing_qubo(
        self,
        candidate_costs: Sequence[float],
        overlap_matrix: Mapping[tuple[int, int], float] | None = None,
        relaxed_solution: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Build a routing candidate selection QUBO using QpiAI Opt.

        Args:
            candidate_costs: Cost (distance/time) for each candidate route
            overlap_matrix: Pairwise overlap penalty between candidate routes
            relaxed_solution: Continuous LP relaxation vector for QAOA warm-start
        """
        from qpiai_opt.opt_modeller import Model
        from qpiai_opt.opt_modeller.variables import Binary
        from qpiai_opt.opt_solvers.qubo.models import QUBOProblem
        from qpiai_opt.quantum import QAOA, WarmStart, build_circuit

        n = len(candidate_costs)
        model = Model("DebugLeafRouteQUBO")
        x = [Binary(f"x{i}") for i in range(n)]

        # Objective: minimize candidate route costs
        obj = 0.0
        for i, cost in enumerate(candidate_costs):
            obj = obj + cost * x[i]

        # Add pairwise route overlap penalties
        if overlap_matrix:
            for (i, j), penalty in overlap_matrix.items():
                if 0 <= i < n and 0 <= j < n and i != j:
                    obj = obj + penalty * x[i] * x[j]

        model.minimize(obj)

        qubo_dict, offset = model.to_qubo_dict(matrix_format="upper")
        qubo_problem = QUBOProblem.from_model(model)

        # Build QAOA warm-start circuit if relaxation is supplied
        circuit_info = None
        if relaxed_solution and len(relaxed_solution) == n:
            try:
                warm = WarmStart(relaxed_solution=list(relaxed_solution))
                qaoa = QAOA(reps=1, warm_start=warm)
                circuit = build_circuit(qubo_problem, algorithm=qaoa, parameters=[0.15, 0.35])
                circuit_info = {
                    "num_qubits": n,
                    "circuit_type": type(circuit).__name__,
                }
            except Exception as exc:
                circuit_info = {"error": str(exc)}

        return {
            "num_candidates": n,
            "terms_count": len(qubo_dict),
            "qubo_offset": float(offset),
            "qubo_terms": {f"({k[0]},{k[1]})": float(v) for k, v in qubo_dict.items()},
            "warm_start_circuit": circuit_info,
        }
