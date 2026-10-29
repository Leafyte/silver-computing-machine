"""Integrity and leakage checks for the bounded India ETA reconstruction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from joblib import load

from experiments_india_eta.recovery import EXPECTED_COUNTS, VQC_FEATURES, load_verified_data


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "india_eta_recovery"


def test_verified_inputs_are_key_joined_and_test_is_not_exposed() -> None:
    bundle = load_verified_data(ROOT)
    assert len(bundle.train) == EXPECTED_COUNTS["train"]
    assert len(bundle.validation) == EXPECTED_COUNTS["val"]
    assert set(bundle.train.split_partition) == {"train"}
    assert set(bundle.validation.split_partition) == {"val"}
    assert "test" not in set(bundle.train.split_partition) | set(bundle.validation.split_partition)
    assert bundle.feature_names == json.loads((ROOT / "data" / "feature_schema.json").read_text())["gbdt_view"]["features"]
    assert len(bundle.feature_names) == 59
    assert not ({"ID", "Delivery_person_ID", "split_partition", "delivery_duration_min"} & set(bundle.feature_names))
    assert all(name in bundle.train.columns and name in bundle.validation.columns for name in VQC_FEATURES)
    assert bundle.train.ID.is_unique and bundle.validation.ID.is_unique
    assert set(bundle.train.ID).isdisjoint(bundle.validation.ID)
    assert set(bundle.train.Delivery_person_ID).isdisjoint(bundle.validation.Delivery_person_ID)
    assert bundle.provenance["source_csv"]["canonical_git_sha256_lf"] == "4baa699e1c09d286f06790010a8c0b284ed3376cbde7e346e19d0ded16a240e3"


def test_validation_role_split_is_exact_hash_ranked_and_group_disjoint() -> None:
    bundle = load_verified_data(ROOT)
    couriers = sorted(bundle.validation.Delivery_person_ID.astype(str).unique())
    ranked = sorted(
        couriers,
        key=lambda courier: (
            hashlib.sha256(b"india-eta-val-role-v1|20261029|" + courier.encode("utf-8")).digest(),
            courier,
        ),
    )
    expected_calibration = set(ranked[: len(ranked) // 3])
    expected = {
        courier: ("fusion_calibration" if courier in expected_calibration else "model_selection")
        for courier in couriers
    }
    assert bundle.validation_roles == expected
    assert sum(role == "fusion_calibration" for role in expected.values()) == len(couriers) // 3
    role_by_row = bundle.validation.Delivery_person_ID.astype(str).map(bundle.validation_roles)
    assert not role_by_row.isna().any()
    selection_groups = {c for c, r in expected.items() if r == "model_selection"}
    calibration_groups = {c for c, r in expected.items() if r == "fusion_calibration"}
    assert selection_groups.isdisjoint(calibration_groups)


def test_recovered_artifacts_are_reloadable_and_oof_groups_are_held_out() -> None:
    manifest_path = ARTIFACTS / "recovery_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("Run the authorized recovery first; no frozen checkpoint manifest yet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact_group in ("model_artifacts", "prediction_artifacts"):
        for artifact in manifest[artifact_group].values():
            artifact_path = ROOT / artifact["path"]
            assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact["sha256"]
    model_info = manifest["model_artifacts"]["model"]
    model_path = ROOT / model_info["path"]
    model = load(model_path)
    schema = json.loads((ROOT / manifest["model_artifacts"]["feature_schema"]["path"]).read_text(encoding="utf-8"))
    assert schema["estimator_identity"] == "sklearn.ensemble.HistGradientBoostingRegressor"
    assert len(schema["feature_names"]) == 59
    val_path = ROOT / manifest["prediction_artifacts"]["validation"]["path"]
    oof_path = ROOT / manifest["prediction_artifacts"]["oof"]["path"]
    val = pd.read_parquet(val_path)
    oof = pd.read_parquet(oof_path)
    assert len(val) == EXPECTED_COUNTS["val"]
    assert len(oof) == EXPECTED_COUNTS["train"]
    assert set(oof.split_partition) == {"train"}
    assert set(val.split_partition) == {"val"}
    assert val.delivery_record_id.is_unique and oof.delivery_record_id.is_unique
    assert {"model_selection", "fusion_calibration"} == set(val.validation_role)
    assert all(name in val.columns and name in oof.columns for name in VQC_FEATURES)
    fold_path = ROOT / manifest["prediction_artifacts"]["oof_fold_manifest"]["path"]
    folds = json.loads(fold_path.read_text(encoding="utf-8"))["folds"]
    assert set(oof.fold.astype(int).unique()) == {int(f) for f in folds}
    for fold, details in folds.items():
        held = set(details["heldout_couriers"])
        fit = set(details["train_couriers"])
        assert held.isdisjoint(fit)
        assert set(oof.loc[oof.fold.astype(int) == int(fold), "courier_id"].astype(str)) == held
    role_manifest_path = ROOT / manifest["prediction_artifacts"]["validation_role_assignment"]["path"]
    role_manifest = json.loads(role_manifest_path.read_text(encoding="utf-8"))
    assert role_manifest["courier_to_role"] == load_verified_data(ROOT).validation_roles
    assert role_manifest["record_id_to_role"] == dict(zip(val.delivery_record_id.astype(str), val.validation_role))
    assert role_manifest["assignment_sha256"] == manifest["validation_role_assignment_sha256"]
    assert role_manifest["record_assignment_sha256"]
    bundle = load_verified_data(ROOT)
    lookup = bundle.validation.set_index("ID")
    validation_ids = val.delivery_record_id.tolist()[:32]
    sample_x = lookup.loc[validation_ids, schema["feature_names"]].to_numpy(dtype=np.float64)
    reloaded = model.predict(sample_x)
    saved = val.gbdt_pred_min.to_numpy(dtype=np.float64)[:32]
    np.testing.assert_array_equal(reloaded, saved)
