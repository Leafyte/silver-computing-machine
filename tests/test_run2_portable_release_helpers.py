from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments_india_qml.release import (
    load_residual_checkpoint,
    predict_residual_minutes,
    predict_residual_qpiai,
)


@pytest.mark.parametrize(
    ("architecture", "feature_names", "theta"),
    [
        ("one_block_2feature", ["vqc_x0_distance", "vqc_x1_congestion"], [0.31, -0.17]),
        ("two_stage_reupload_4feature", ["vqc_x0_distance", "vqc_x1_congestion", "vqc_x2_multidelivery", "vqc_x3_preptime"], [0.31, -0.17, 0.23, 0.41]),
    ],
)
def test_portable_numpy_reload_matches_actual_qpiai_local_simulator(tmp_path: Path, architecture, feature_names, theta) -> None:
    pytest.importorskip("qpiai_quantum")
    path = tmp_path / "residual.json"
    payload = {
        "format": "india_eta_torch_residual_v2",
        "model_version": "run2_india_torch_residual_v2",
        "config": {"architecture": architecture, "feature_names": feature_names, "seed": 3, "dtype": "float64"},
        "architecture": architecture,
        "feature_names": feature_names,
        "parameter_count": len(theta) + 3,
        "theta": theta,
        "readout": [0.07, -0.11, 0.025],
        "residual_scale_minutes": 7.5,
        "fit_info": {"status": "completed"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    n = len(feature_names)
    features = np.random.default_rng(91).uniform(0.03, 0.97, size=(5, n))
    model = load_residual_checkpoint(path)
    portable = predict_residual_minutes(model, features)
    sdk = predict_residual_qpiai(model, features)
    assert portable.shape == sdk.shape == (5,)
    assert np.allclose(portable, sdk, atol=1e-12, rtol=1e-12)


def test_portable_loader_rejects_out_of_range_features_and_bad_scale(tmp_path: Path) -> None:
    path = tmp_path / "residual.json"
    payload = {
        "format": "india_eta_torch_residual_v2", "architecture": "one_block_2feature",
        "config": {"architecture": "one_block_2feature", "feature_names": ["vqc_x0_distance", "vqc_x1_congestion"]},
        "feature_names": ["vqc_x0_distance", "vqc_x1_congestion"],
        "theta": [0.1, 0.2], "readout": [0.1, 0.2, 0.3], "residual_scale_minutes": 4.0,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"lie in \[0,1\]"):
        predict_residual_minutes(path, [[0.5, 1.1]])
    payload["residual_scale_minutes"] = -1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        load_residual_checkpoint(path)

