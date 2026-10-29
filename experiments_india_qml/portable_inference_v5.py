"""Small portable inference API for the RUN2 India recovery bundle.

Inputs are the already transformed 59-column GBDT feature matrix and the two
train-scaled VQC features in the manifest's exact order. No labels are needed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from experiments_india_qml.quantum import QuantumResidualRegressor, hybrid_predict_minutes


def load_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir)
    manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    baseline = joblib.load(root / manifest["files"]["recovered_gbdt"]["path"])
    candidates = {
        arch: QuantumResidualRegressor.load(root / file_info["path"])
        for arch, file_info in manifest["files"]["quantum_candidates"].items()
    }
    return {"root": root, "manifest": manifest, "baseline": baseline, "quantum": candidates}


def predict_minutes(
    bundle: dict[str, Any],
    gbdt_features: Any,
    q_features: Any,
    *,
    mode: str = "operational_release",
) -> dict[str, np.ndarray]:
    """Predict ETA with the portable release or frozen validation candidates.

    Modes are `operational_release` (alpha=0), `candidate_hybrid` (ungated
    selected validation candidate), and `candidate_selective` (same candidate
    with its frozen input-only feature gate).
    """
    manifest = bundle["manifest"]
    xgb = np.asarray(gbdt_features, dtype=np.float32)
    xq = np.asarray(q_features, dtype=np.float64)
    names = manifest["gbdt_feature_order"]
    if xgb.ndim != 2 or xgb.shape[1] != len(names) or not np.isfinite(xgb).all():
        raise ValueError(f"gbdt_features must be a finite matrix with {len(names)} columns")
    if xq.ndim != 2 or xq.shape != (len(xgb), 2) or not np.isfinite(xq).all():
        raise ValueError("q_features must be a finite n x 2 matrix")
    if np.any(xq < 0.0) or np.any(xq > 1.0):
        raise ValueError("q_features must use the frozen train-only [0,1] scaler")
    baseline = np.maximum(0.0, bundle["baseline"]["model"].predict(xgb))
    if mode == "operational_release":
        return {"eta_minutes": baseline, "quantum_invoked": np.zeros(len(xgb), dtype=bool)}
    candidate = manifest["validation_selected_quantum_candidate"]
    arch = candidate["architecture"]
    model = bundle["quantum"][arch]
    alpha = float(candidate["alpha"])
    if mode == "candidate_hybrid":
        residual = model.predict_minutes(xq)
        return {"eta_minutes": hybrid_predict_minutes(baseline, residual, alpha), "quantum_residual_minutes": residual,
                "quantum_invoked": np.ones(len(xgb), dtype=bool)}
    if mode == "candidate_selective":
        if not candidate.get("gate_feature") or candidate.get("gate_threshold") is None:
            raise ValueError("frozen candidate has no selective gate")
        result = model.predict_hybrid_minutes(
            baseline, xq, alpha, gate_feature=candidate["gate_feature"],
            gate_operator=candidate["gate_operator"], gate_threshold=float(candidate["gate_threshold"]),
        )
        return {"eta_minutes": result["eta_minutes"], "quantum_residual_minutes": result["quantum_residual_minutes"],
                "quantum_invoked": result["quantum_invoked"]}
    raise ValueError(f"unsupported inference mode {mode!r}")
