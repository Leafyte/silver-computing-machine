"""Run 2 India ETA data/release acceptance gates; no test scores are computed here."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUN_ID = os.environ.get("RUN2_ACCEPTANCE_RUN_ID", "luna_hgbr_recovery_20261029T131809Z_v5")
RUN_ARTIFACTS = ROOT / "artifacts" / "run2_india" / RUN_ID
RUN_RUNTIME = ROOT / ".runtime" / "run2" / RUN_ID
PRETEST_RELEASE_SHA256 = "e8c8f679761018036f1ebe3eff3251ea44572fab7c766057625b7302b53ff189"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _canonical_id_sha(values) -> str:
    encoded = json.dumps(sorted(map(str, values)), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metric_triplet(y_true, prediction) -> dict[str, float]:
    import numpy as np
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "mae_minutes": float(np.mean(absolute)),
        "rmse_minutes": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error_minutes": float(np.median(absolute)),
    }


def test_eta_target_is_minutes_and_synthetic_label_is_separate() -> None:
    schema = _json(DATA / "feature_schema.json")
    target = schema["target"]
    assert target["name"] == "delivery_duration_min"
    assert target["unit"].lower() == "minutes"
    assert target["type"].startswith("float")
    scenario = schema["synthetic_scenario_target"]
    assert scenario["name"] == "is_sla_breach_synthetic"
    assert scenario["name"] != target["name"]


def test_gbdt_feature_schema_excludes_target_and_row_group_metadata() -> None:
    schema = _json(DATA / "feature_schema.json")
    names = set(schema["gbdt_view"]["features"])
    forbidden = {
        "delivery_duration_min",
        "target_normalized",
        "is_sla_breach_synthetic",
        "sla_threshold_min",
        "ID",
        "Delivery_person_ID",
        "split_partition",
    }
    assert names.isdisjoint(forbidden), f"Leakage/metadata columns in GBDT feature list: {sorted(names & forbidden)}"
    assert len(names) == schema["gbdt_view"]["total_features"]


def test_frozen_split_manifest_has_group_safe_declared_partitions() -> None:
    manifest = _json(DATA / "split_manifest.json")
    assert manifest["split_method"] == "GroupShuffleSplit on Delivery_person_ID"
    counts = manifest["split_counts"]
    assert counts["train"] > 0 and counts["val"] > 0 and counts["test"] > 0
    assert counts["train"] + counts["val"] + counts["test"] == manifest["eligible_model_rows"]
    overlaps = manifest["courier_counts"]
    assert overlaps["overlap_train_val"] == 0
    assert overlaps["overlap_train_test"] == 0
    assert overlaps["overlap_val_test"] == 0


def test_parquet_split_row_ids_and_courier_groups_match_manifest() -> None:
    """Independently recompute split membership without reading duration labels."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("fastparquet")

    manifest = _json(DATA / "split_manifest.json")
    frame = pd.read_parquet(
        DATA / "indian_delivery_clean.parquet",
        engine="fastparquet",
        columns=["ID", "Delivery_person_ID", "split_partition", "is_model_eligible"],
    )
    eligible = frame.loc[frame["is_model_eligible"].astype(bool)]
    assert eligible["ID"].is_unique, "eligible source row IDs must be unique"
    assert len(eligible) == manifest["eligible_model_rows"]

    counts = eligible["split_partition"].value_counts().to_dict()
    assert counts["train"] == manifest["split_counts"]["train"]
    assert counts["val"] == manifest["split_counts"]["val"]
    assert counts["test"] == manifest["split_counts"]["test"]

    groups = {
        partition: set(eligible.loc[eligible["split_partition"] == partition, "Delivery_person_ID"])
        for partition in ("train", "val", "test")
    }
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("indian_delivery_gbdt.parquet", "bd03b47c3d3e23050329cefc69d752aad607031cacdb9696151d237e3534a627"),
        ("indian_delivery_vqc.parquet", "a633677472e6e0e874c2797a66eb7b011e3a307285553f8fa00c6ee664df4b4e"),
        ("indian_delivery_clean.parquet", "9034d24389c7162f0ea0e275e5b4cc9ec16f04bba75c0f239dbcef8535a1f7c1"),
    ],
)
def test_input_parquet_hashes_match_preprocessing_manifest(filename: str, expected: str) -> None:
    manifest = _json(DATA / "split_manifest.json")
    manifest_key = {
        "indian_delivery_gbdt.parquet": "gbdt_parquet_sha256",
        "indian_delivery_vqc.parquet": "vqc_parquet_sha256",
        "indian_delivery_clean.parquet": "clean_parquet_sha256",
    }[filename]
    path = DATA / filename
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected
    assert manifest["file_digests"][manifest_key].lower() == actual


def test_recovered_gbdt_checkpoint_is_manifested_and_hash_verified() -> None:
    """Check the authorized one-time recovery artifact once the trainer publishes it."""
    manifest_path = RUN_ARTIFACTS / "recovered_gbdt_manifest.json"
    if not manifest_path.is_file():
        pytest.skip(
            "India ETA recovery artifacts have not been written by the training owner yet."
        )
    manifest = _json(manifest_path)
    checkpoint_path = ROOT / manifest["model"]["checkpoint_path"]
    assert checkpoint_path.is_file(), f"Frozen GBDT checkpoint missing: {checkpoint_path}"
    actual = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert manifest["model"]["checkpoint_sha256"].lower() == actual
    assert manifest["target_name"] == "delivery_duration_min"
    assert manifest["target_unit"].lower() == "minutes"
    assert manifest["recovery_fit_authorized_missing_frozen_checkpoint"] is True
    assert manifest["parent_split_manifest_sha256"].lower() == hashlib.sha256((DATA / "split_manifest.json").read_bytes()).hexdigest()
    assert manifest["model"]["fit_partition"].lower() in {"train", "original train only"}
    assert manifest["model"]["features"] == _json(DATA / "feature_schema.json")["gbdt_view"]["features"]
    assert manifest["crossfit"]["prediction_path"] == f".runtime/run2/{RUN_ID}/classical_development_predictions.parquet"
    assert manifest["test_partition_read_or_used"] is False

    # Hash and provenance checks precede trusted local joblib deserialization.
    pytest.importorskip("sklearn", reason="run this checkpoint reload under the pinned training runtime")
    joblib = pytest.importorskip("joblib")
    payload = joblib.load(checkpoint_path)
    model = payload.get("model", payload) if isinstance(payload, dict) else payload
    assert getattr(model, "n_features_in_", len(manifest["model"]["features"])) == len(manifest["model"]["features"])


def test_recovered_checkpoint_reload_matches_validation_predictions_in_minutes() -> None:
    manifest_path = RUN_ARTIFACTS / "recovered_gbdt_manifest.json"
    prediction_path = RUN_RUNTIME / "classical_development_predictions.parquet"
    if not manifest_path.is_file() or not prediction_path.is_file():
        pytest.skip("recovered checkpoint or development prediction artifact is not ready")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("fastparquet")
    import numpy as np

    manifest = _json(manifest_path)
    checkpoint_path = ROOT / manifest["model"]["checkpoint_path"]
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == manifest["model"]["checkpoint_sha256"].lower()
    pytest.importorskip("sklearn", reason="run this checkpoint reload under the pinned training runtime")
    joblib = pytest.importorskip("joblib")
    payload = joblib.load(checkpoint_path)
    model = payload.get("model", payload) if isinstance(payload, dict) else payload
    features = manifest["model"]["features"]
    from fastparquet import ParquetFile
    validation = ParquetFile(DATA / "indian_delivery_gbdt.parquet").to_pandas(
        columns=["ID", "split_partition", *features],
        filters=[[ ("split_partition", "==", "val") ]],
        row_filter=True,
    )
    assert set(validation["split_partition"]) == {"val"}
    predictions = np.asarray(model.predict(validation[features].to_numpy()), dtype=np.float64)
    assert predictions.shape == (len(validation),)
    assert np.isfinite(predictions).all() and np.all(predictions >= 0.0)

    saved = pd.read_parquet(
        prediction_path,
        filters=[("split_partition", "==", "val")],
        engine="fastparquet",
    )
    saved = saved.set_index(saved["ID"].astype(str))
    validation_ids = validation["ID"].astype(str).to_numpy()
    expected = saved.loc[validation_ids, "gbdt_pred_minutes"].to_numpy(dtype=np.float64)
    assert np.allclose(predictions, expected, rtol=1e-12, atol=1e-10)


def test_quantum_early_stop_and_fusion_selection_use_disjoint_validation_couriers() -> None:
    """Prevent QML stopping and alpha/gate selection from reusing validation drivers."""
    manifest_path = RUN_ARTIFACTS / "recovered_gbdt_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("recovered manifest/development role audit is not ready")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("fastparquet")
    manifest = _json(manifest_path)
    roles = manifest["crossfit"]["development_roles"]
    early_groups = set(map(str, roles["vqc_early_stop_group_ids"]))
    fusion_groups = set(map(str, roles["fusion_selection_group_ids"]))
    assert early_groups.isdisjoint(fusion_groups)

    from fastparquet import ParquetFile
    validation = ParquetFile(DATA / "indian_delivery_gbdt.parquet").to_pandas(
        columns=["ID", "Delivery_person_ID", "split_partition"],
        filters=[[ ("split_partition", "==", "val") ]],
        row_filter=True,
    )
    validation_groups = set(validation["Delivery_person_ID"].astype(str))
    assert early_groups | fusion_groups == validation_groups
    early_rows = int(validation["Delivery_person_ID"].astype(str).isin(early_groups).sum())
    fusion_rows = int(validation["Delivery_person_ID"].astype(str).isin(fusion_groups).sum())
    assert early_rows == int(roles["vqc_early_stop_row_count"])
    assert fusion_rows == int(roles["fusion_selection_row_count"])
    assert early_rows + fusion_rows == len(validation)
    early_ids = validation.loc[validation["Delivery_person_ID"].astype(str).isin(early_groups), "ID"].astype(str)
    fusion_ids = validation.loc[validation["Delivery_person_ID"].astype(str).isin(fusion_groups), "ID"].astype(str)
    all_ids = validation["ID"].astype(str)
    assert _canonical_id_sha(early_ids) == roles["vqc_early_stop_row_ids_sha256"]
    assert _canonical_id_sha(fusion_ids) == roles["fusion_selection_row_ids_sha256"]
    assert _canonical_id_sha(all_ids) == roles["all_validation_row_ids_sha256"]


def test_development_oof_predictions_are_train_only_group_safe_and_recomputable() -> None:
    """Verify row/group/fold provenance before any sealed-test evaluation."""
    prediction_path = RUN_RUNTIME / "classical_development_predictions.parquet"
    fold_manifest_path = RUN_ARTIFACTS / "crossfit_manifest.json"
    if not prediction_path.is_file() or not fold_manifest_path.is_file():
        pytest.skip("cross-fit prediction artifact and fold audit manifest are not ready")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("fastparquet")
    import numpy as np

    predictions = pd.read_parquet(prediction_path, engine="fastparquet")
    fold_manifest = _json(fold_manifest_path)
    recovered_manifest = _json(RUN_ARTIFACTS / "recovered_gbdt_manifest.json")
    assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() == recovered_manifest["crossfit"]["prediction_sha256"]
    assert hashlib.sha256(fold_manifest_path.read_bytes()).hexdigest() == recovered_manifest["crossfit"]["fold_manifest_sha256"]
    assert recovered_manifest["crossfit"]["contains_test_rows"] is False
    assert recovered_manifest["test_partition_read_or_used"] is False
    folds = fold_manifest["folds"]
    required = {
        "ID", "Delivery_person_ID", "split_partition", "fold",
        "y_true_minutes", "gbdt_pred_minutes", "residual_minutes",
    }
    assert required.issubset(predictions.columns)
    assert set(predictions["split_partition"]) == {"train", "val"}, "pre-freeze predictions must not include test rows"
    assert predictions["ID"].astype(str).is_unique
    assert np.isfinite(predictions[["y_true_minutes", "gbdt_pred_minutes", "residual_minutes"]].to_numpy()).all()
    assert np.allclose(
        predictions["residual_minutes"].to_numpy(),
        predictions["y_true_minutes"].to_numpy() - predictions["gbdt_pred_minutes"].to_numpy(),
        rtol=0.0,
        atol=1e-9,
    )

    split = _json(DATA / "split_manifest.json")["split_counts"]
    train_rows = predictions.loc[predictions["split_partition"] == "train"]
    val_rows = predictions.loc[predictions["split_partition"] == "val"]
    assert len(train_rows) == split["train"]
    assert len(val_rows) == split["val"]
    assert set(train_rows["Delivery_person_ID"].astype(str)).isdisjoint(
        set(val_rows["Delivery_person_ID"].astype(str))
    )
    assert set(predictions["development_role"]) == {"train_oof", "vqc_early_stop", "fusion_selection"}
    assert train_rows["fold"].notna().all() and val_rows["fold"].isna().all()

    feature_index = pd.read_parquet(
        DATA / "indian_delivery_gbdt.parquet",
        columns=["ID", "Delivery_person_ID", "split_partition"],
        engine="fastparquet",
    )
    feature_index["ID"] = feature_index["ID"].astype(str)
    predictions["ID"] = predictions["ID"].astype(str)
    source_dev = feature_index.loc[feature_index["split_partition"].isin(["train", "val"])]
    expected = set(zip(source_dev["ID"], source_dev["split_partition"]))
    observed = set(zip(predictions["ID"], predictions["split_partition"]))
    assert observed == expected

    all_train_groups = set(train_rows["Delivery_person_ID"].astype(str))
    seen_heldout_groups: set[str] = set()
    for fold_info in folds:
        fold_key = str(fold_info["fold"])
        training_groups = set(map(str, fold_info["fit_group_ids"]))
        heldout_groups = set(map(str, fold_info["heldout_group_ids"]))
        assert training_groups.isdisjoint(heldout_groups)
        assert training_groups == all_train_groups - heldout_groups
        fold_rows = train_rows.loc[train_rows["fold"].astype(str) == fold_key]
        observed_groups = set(fold_rows["Delivery_person_ID"].astype(str))
        assert observed_groups == heldout_groups
        fit_rows = train_rows.loc[~train_rows["Delivery_person_ID"].astype(str).isin(heldout_groups)]
        assert len(fold_rows) == int(fold_info["heldout_rows"])
        assert len(fit_rows) == int(fold_info["fit_rows"])
        assert _canonical_id_sha(fold_rows["ID"]) == fold_info["heldout_row_ids_sha256"]
        assert _canonical_id_sha(fit_rows["ID"]) == fold_info["fit_row_ids_sha256"]
        preprocessing = fold_info["preprocessing"]
        assert set(map(str, preprocessing["fit_groups"])) == training_groups
        assert fold_info["fit_group_membership_matches_training_rows"] is True
        assert fold_info["preprocessing_settings_sha256"]
        assert not (seen_heldout_groups & observed_groups), "a courier cannot be held out in multiple folds"
        seen_heldout_groups |= observed_groups
    assert seen_heldout_groups == all_train_groups

    # The saved QML scaling map is fit on original-train covariates only and
    # is shared across OOF and validation rows.
    scaler_path = ROOT / fold_manifest["quantum_feature_scaler_path"]
    assert hashlib.sha256(scaler_path.read_bytes()).hexdigest() == fold_manifest["quantum_feature_scaler_sha256"]
    scaler = _json(scaler_path)
    assert scaler["fit_partition"] == "original train covariates only"
    assert scaler["fit_row_count"] == split["train"]


def test_oof_fold_preprocessing_statistics_use_only_fold_fit_couriers() -> None:
    """Recompute recorded fold medians/modes from target-free training features."""
    if not (RUN_RUNTIME / "classical_development_predictions.parquet").is_file():
        pytest.skip("v4 development rows are not ready for fold-statistic recomputation")
    if not (RUN_ARTIFACTS / "crossfit_manifest.json").is_file():
        pytest.skip("v4 fold manifest is not ready for fold-statistic recomputation")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("fastparquet")
    import numpy as np

    predictions = pd.read_parquet(RUN_RUNTIME / "classical_development_predictions.parquet", engine="fastparquet")
    fold_manifest = _json(RUN_ARTIFACTS / "crossfit_manifest.json")
    raw_columns = [
        "ID", "Delivery_person_ID", "split_partition", "delivery_person_age_clean",
        "delivery_person_ratings_clean", "order_to_pickup_min", "multiple_deliveries_clean",
        "traffic_density_ordinal", "weather_clean", "city_type_clean", "traffic_density_clean",
        "type_of_order_clean", "type_of_vehicle_clean", "city_code",
    ]
    # This is intentionally a target-free source-column read. Keep only original
    # train rows before computing independent fit-fold statistics.
    raw = pd.read_parquet(DATA / "indian_delivery_clean.parquet", engine="fastparquet", columns=raw_columns)
    raw = raw.loc[raw["split_partition"].eq("train")].copy()
    raw["ID"] = raw["ID"].astype(str)
    for fold_info in fold_manifest["folds"]:
        fold_number = int(fold_info["fold"])
        heldout = set(map(str, fold_info["heldout_group_ids"]))
        fold_fit_groups = set(map(str, fold_info["fit_group_ids"]))
        assert not fold_fit_groups & heldout
        fit = raw.loc[raw["Delivery_person_ID"].astype(str).isin(fold_fit_groups)]
        oof = predictions.loc[
            predictions["split_partition"].eq("train") & predictions["fold"].eq(fold_number)
        ]
        assert set(oof["Delivery_person_ID"].astype(str)) == heldout
        assert set(fit["Delivery_person_ID"].astype(str)) == fold_fit_groups
        imputers = fold_info["preprocessing"]["imputers"]
        numeric_sources = {
            "delivery_person_age_imp": "delivery_person_age_clean",
            "delivery_person_ratings_imp": "delivery_person_ratings_clean",
            "order_to_pickup_min_imp": "order_to_pickup_min",
            "multiple_deliveries_imp": "multiple_deliveries_clean",
        }
        for output, source in numeric_sources.items():
            expected = float(pd.to_numeric(fit[source], errors="coerce").median())
            assert imputers[output]["source"] == source
            assert np.isclose(float(imputers[output]["value"]), expected, rtol=0, atol=1e-12)
        traffic_expected = float(pd.to_numeric(fit["traffic_density_ordinal"], errors="coerce").mode().iloc[0])
        assert np.isclose(float(imputers["traffic_density_ordinal_imp"]["value"]), traffic_expected, rtol=0, atol=1e-12)
        categories = fold_info["preprocessing"]["fit_categories"]
        source_by_prefix = {
            "traffic_density_clean": "traffic_density_clean",
            "type_of_order_clean": "type_of_order_clean",
            "type_of_vehicle_clean": "type_of_vehicle_clean",
            "city_code": "city_code",
            "weather_clean_imp": "weather_clean",
            "city_type_clean_imp": "city_type_clean",
        }
        for prefix, source in source_by_prefix.items():
            expected = sorted(fit[source].dropna().astype(str).unique().tolist())
            assert categories[prefix] == expected


def test_validation_residual_candidates_reload_match_saved_minutes_predictions() -> None:
    """Independently reload both candidates and reproduce validation-only outputs."""
    predictions_path = RUN_ARTIFACTS / "validation_predictions.npz"
    selection_path = RUN_ARTIFACTS / "residual_selection.json"
    if not predictions_path.is_file() or not selection_path.is_file():
        pytest.skip("validation residual selection/reload artifacts are not ready")
    import numpy as np
    from experiments_india_qml.quantum import QuantumResidualRegressor, hybrid_predict_minutes

    selection = _json(selection_path)
    role_protocol_path = Path(selection["validation_role_protocol_path"])
    if not role_protocol_path.is_absolute():
        role_protocol_path = ROOT / role_protocol_path
    assert hashlib.sha256(role_protocol_path.read_bytes()).hexdigest() == selection["validation_role_protocol_sha256"]
    role_protocol = _json(role_protocol_path)
    assert role_protocol["written_before_qml_fitting"] is True
    assert role_protocol["development_data_sha256"] == selection["development_table_sha256"]
    assert role_protocol["recovery_manifest_sha256"] == selection["manifest_sha256"]
    with np.load(predictions_path, allow_pickle=False) as archive:
        required = {
            "row_id_sha256", "vqc_features", "y_true_minutes", "gbdt_eta_minutes",
            "quantum_residual_minutes", "quantum_hybrid_eta_minutes",
            "ridge_residual_minutes", "ridge_alpha", "ridge_hybrid_eta_minutes",
            "quantum_gate_mask", "quantum_gate_threshold", "quantum_gate_feature",
            "quantum_gate_operator",
            "selected_architecture", "selected_alpha",
        }
        assert required.issubset(archive.files)
        row_hashes = archive["row_id_sha256"].astype(str)
        features = np.asarray(archive["vqc_features"], dtype=np.float64)
        y_true = np.asarray(archive["y_true_minutes"], dtype=np.float64).reshape(-1)
        gbdt_eta = np.asarray(archive["gbdt_eta_minutes"], dtype=np.float64).reshape(-1)
        selected_residual = np.asarray(archive["quantum_residual_minutes"], dtype=np.float64).reshape(-1)
        selected_hybrid = np.asarray(archive["quantum_hybrid_eta_minutes"], dtype=np.float64).reshape(-1)
        selected_gate_mask = np.asarray(archive["quantum_gate_mask"], dtype=bool).reshape(-1)
        selected_gate_threshold = float(archive["quantum_gate_threshold"].item())
        selected_gate_feature = str(archive["quantum_gate_feature"].item())
        selected_gate_operator = str(archive["quantum_gate_operator"].item())
        selected_architecture = str(archive["selected_architecture"].item())
        selected_alpha = float(archive["selected_alpha"].item())

    rows = _json(DATA / "split_manifest.json")["split_counts"]["val"]
    assert len(row_hashes) == rows and len(set(row_hashes)) == rows
    assert features.shape == (rows, 2)
    assert np.isfinite(features).all() and np.all((features >= 0.0) & (features <= 1.0))
    assert np.isfinite(np.column_stack((y_true, gbdt_eta, selected_residual, selected_hybrid))).all()
    assert len(selected_gate_mask) == rows
    if selected_gate_feature:
        assert selected_gate_feature == "vqc_x1_congestion"
        assert 0.0 <= selected_gate_threshold <= 1.0 and np.isfinite(selected_gate_threshold)
        gate_values = features[:, 1]
        expected_gate = gate_values >= selected_gate_threshold if selected_gate_operator == ">=" else gate_values <= selected_gate_threshold
        assert selected_gate_operator in (">=", "<=")
    else:
        assert np.isnan(selected_gate_threshold)
        assert selected_gate_operator in ("all", "disabled")
        expected_gate = selected_gate_operator == "all" and np.ones(rows, dtype=bool)
        if selected_gate_operator == "disabled":
            expected_gate = np.zeros(rows, dtype=bool)
    if selected_alpha == 0.0:
        expected_gate = np.zeros(rows, dtype=bool)
    assert np.array_equal(selected_gate_mask, expected_gate)
    pd = pytest.importorskip("pandas")
    pytest.importorskip("fastparquet")
    role_frame = pd.read_parquet(
        RUN_RUNTIME / "classical_development_predictions.parquet",
        columns=["split_partition", "development_role", "vqc_x0_distance", "vqc_x1_congestion", "residual_minutes"],
        engine="fastparquet",
    )
    early_stop = role_frame.loc[
        role_frame["split_partition"].eq("val") & role_frame["development_role"].eq("vqc_early_stop")
    ]
    stop_features = early_stop[["vqc_x0_distance", "vqc_x1_congestion"]].to_numpy(dtype=np.float64)
    stop_residuals = early_stop["residual_minutes"].to_numpy(dtype=np.float64)
    assert len(stop_features) > 0
    candidates = selection["quantum_candidates"]
    for architecture in ("one_block", "reupload"):
        candidate = candidates[architecture]
        checkpoint = ROOT / candidate["checkpoint_path"]
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == candidate["checkpoint_sha256"].lower()
        restored = QuantumResidualRegressor.load(checkpoint)
        assert restored.architecture == architecture
        assert restored.config.feature_names == ("vqc_x0_distance", "vqc_x1_congestion")
        reloaded_residual = restored.predict_minutes(features)
        saved_candidate = np.asarray(candidate["validation_residual_predictions_minutes"], dtype=np.float64)
        assert saved_candidate.shape == (rows,)
        candidate_operator = candidate["gate_operator"]
        candidate_threshold = candidate["gate_threshold"]
        if candidate_operator == "all":
            candidate_mask = np.ones(rows, dtype=bool)
        elif candidate_operator == "disabled":
            candidate_mask = np.zeros(rows, dtype=bool)
        else:
            values = features[:, 1]
            candidate_mask = values >= float(candidate_threshold) if candidate_operator == ">=" else values <= float(candidate_threshold)
        candidate_alpha = float(candidate["alpha_validation_only"])
        if candidate_alpha == 0.0:
            candidate_mask[:] = False
        call_rows: list[int] = []
        original_predict_minutes = restored.predict_minutes

        def counted_predict_minutes(input_features):
            call_rows.append(len(input_features))
            return original_predict_minutes(input_features)

        restored.predict_minutes = counted_predict_minutes
        candidate_operator = candidate["gate_operator"]
        if candidate_operator == "disabled":
            inference_operator = ">="
            inference_threshold = 0.0
        elif candidate_operator == "all":
            inference_operator = ">="
            inference_threshold = 0.0
        else:
            inference_operator = candidate_operator
            inference_threshold = float(candidate_threshold)
        inference = restored.predict_hybrid_minutes(
            np.maximum(gbdt_eta, 0.0), features, candidate_alpha,
            gate_feature="vqc_x1_congestion",
            gate_operator=inference_operator,
            gate_threshold=inference_threshold,
        )
        expected_calls = int(np.sum(candidate_mask))
        assert sum(call_rows) == expected_calls
        assert inference["invocation_count"] == expected_calls
        assert np.array_equal(inference["quantum_invoked"], candidate_mask)
        reloaded_gated = np.asarray(inference["quantum_residual_minutes"], dtype=np.float64)
        assert np.allclose(reloaded_gated, saved_candidate, rtol=1e-12, atol=1e-10)
        independently_recomputed_rmse = float(np.sqrt(np.mean((restored.predict_minutes(stop_features) - stop_residuals) ** 2)))
        assert np.isclose(
            independently_recomputed_rmse,
            candidate["earlystop_residual_validation_metrics"]["rmse_minutes"],
            rtol=1e-10,
            atol=1e-10,
        )
        assert np.isclose(
            independently_recomputed_rmse,
            candidate["fit_info"]["best_validation_rmse_minutes"],
            rtol=1e-10,
            atol=1e-10,
        )

    release_gate = selection["selected_release_gate"]
    assert selected_architecture == release_gate["architecture"]
    assert selected_alpha == float(release_gate["alpha"])
    assert selected_gate_feature == (release_gate["gate_feature"] or "")
    assert selected_gate_operator == release_gate["gate_operator"]
    if selected_gate_feature:
        assert selected_gate_threshold == float(release_gate["gate_threshold"])
    selected_candidate = candidates[selected_architecture]
    selected_prediction = np.asarray(selected_candidate["validation_residual_predictions_minutes"], dtype=np.float64)
    assert np.allclose(selected_residual, selected_prediction, rtol=1e-12, atol=1e-10)
    assert np.array_equal(selected_gate_mask, np.asarray(selected_candidate["validation_gate_mask"], dtype=bool))
    assert np.isclose(
        float(np.mean(selected_gate_mask)),
        float(release_gate["quantum_invocation_rate_validation"]),
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.allclose(
        selected_hybrid,
        hybrid_predict_minutes(gbdt_eta, selected_residual, selected_alpha),
        rtol=1e-12,
        atol=1e-10,
    )
    assert release_gate["architecture_selection_role"] == "qml_earlystop_architecture"
    assert release_gate["gate_selection_role"] == "fusion_alpha_gate_calibration"
    selected_by_earlystop = min(
        candidates,
        key=lambda architecture: candidates[architecture]["earlystop_residual_validation_metrics"]["rmse_minutes"],
    )
    assert selected_architecture == selected_by_earlystop

    # Independently recompute full-validation reporting metrics for the fixed
    # candidate, GBDT baseline, and Ridge control. These are not used to tune.
    metric_tol = {"rtol": 1e-10, "atol": 1e-10}
    baseline_metrics = _metric_triplet(y_true, np.maximum(0.0, gbdt_eta))
    for metric, value in baseline_metrics.items():
        assert np.isclose(value, selection["classical_baseline_validation_metrics_reporting_only"][metric], **metric_tol)
    selected_full_metrics = _metric_triplet(y_true, selected_hybrid)
    for metric, value in selected_full_metrics.items():
        assert np.isclose(value, selected_candidate["validation_hybrid_metrics"][metric], **metric_tol)
    with np.load(predictions_path, allow_pickle=False) as archive:
        ridge_hybrid = np.asarray(archive["ridge_hybrid_eta_minutes"], dtype=np.float64)
    ridge_metrics = _metric_triplet(y_true, ridge_hybrid)
    for metric, value in ridge_metrics.items():
        assert np.isclose(value, selection["ridge_residual_control"]["selected"]["validation_hybrid_metrics"][metric], **metric_tol)


def test_fusion_alpha_and_feature_gate_metrics_recompute_on_heldout_validation_role() -> None:
    """Verify each gate/alpha against the disjoint validation calibration couriers."""
    predictions_path = RUN_ARTIFACTS / "validation_predictions.npz"
    selection_path = RUN_ARTIFACTS / "residual_selection.json"
    if not predictions_path.is_file() or not selection_path.is_file():
        pytest.skip("trained validation residual artifacts are not ready")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("fastparquet")
    import numpy as np
    from experiments_india_qml.quantum import QuantumResidualRegressor, hybrid_predict_minutes

    selection = _json(selection_path)
    development = pd.read_parquet(
        RUN_RUNTIME / "classical_development_predictions.parquet",
        columns=["Delivery_person_ID", "split_partition", "development_role", "y_true_minutes", "gbdt_pred_minutes", "vqc_x0_distance", "vqc_x1_congestion"],
        engine="fastparquet",
    )
    calibration = development.loc[
        development["split_partition"].eq("val") & development["development_role"].eq("fusion_selection")
    ]
    assert len(calibration) == _json(RUN_ARTIFACTS / "recovered_gbdt_manifest.json")["crossfit"]["development_roles"]["fusion_selection_row_count"]
    features = calibration[["vqc_x0_distance", "vqc_x1_congestion"]].to_numpy(dtype=np.float64)
    target = calibration["y_true_minutes"].to_numpy(dtype=np.float64)
    baseline = np.maximum(0.0, calibration["gbdt_pred_minutes"].to_numpy(dtype=np.float64))

    stop_groups = set(map(str, selection["validation_role_protocol"]["selection_roles"]["qml_earlystop_architecture"]["group_sha256"]))
    calibration_groups = set(map(str, selection["validation_role_protocol"]["selection_roles"]["fusion_alpha_gate_calibration"]["group_sha256"]))
    assert stop_groups.isdisjoint(calibration_groups)
    assert selection["validation_role_protocol"]["written_before_qml_fitting"] is True

    for architecture, candidate in selection["quantum_candidates"].items():
        checkpoint = ROOT / candidate["checkpoint_path"]
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == candidate["checkpoint_sha256"].lower()
        model = QuantumResidualRegressor.load(checkpoint)
        alpha = float(candidate["alpha_validation_only"])
        if candidate["gate_operator"] == "disabled" or alpha == 0.0:
            gated = np.zeros(len(features), dtype=np.float64)
            invocation = np.zeros(len(features), dtype=bool)
        else:
            predicted = model.predict_minutes(features)
            congestion = features[:, model.config.feature_names.index("vqc_x1_congestion")]
            operator = candidate["gate_operator"]
            if operator == "all":
                invocation = np.ones(len(features), dtype=bool)
            else:
                threshold = float(candidate["gate_threshold"])
                assert operator in (">=", "<=")
                invocation = congestion >= threshold if operator == ">=" else congestion <= threshold
            gated = np.where(invocation, predicted, 0.0)
        hybrid = hybrid_predict_minutes(baseline, gated, alpha)
        recomputed = _metric_triplet(target, hybrid)
        assert np.isclose(float(np.mean(invocation)), candidate["quantum_invocation_rate_calibration"], rtol=1e-12, atol=1e-12)
        assert int(np.sum(invocation)) == candidate["quantum_invocation_rows_calibration"]
        for metric, value in recomputed.items():
            assert np.isclose(value, candidate["hybrid_calibration_metrics"][metric], rtol=1e-10, atol=1e-10)


def test_pretest_release_manifest_separates_candidate_comparison_from_alpha_zero_deployment() -> None:
    """Check pre-test frozen config without opening final test prediction files."""
    manifest_path = RUN_ARTIFACTS / "release_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("pre-test frozen release manifest is not ready")
    import numpy as np
    manifest = _json(manifest_path)
    post_evaluation = manifest.get("test_evaluation_timestamp_utc") is not None
    if post_evaluation and os.environ.get("RUN2_FINAL_TEST_ACCEPTANCE") != "1":
        pytest.skip("release manifest has been updated after the authorized test run; use the guarded saved-output acceptance")
    selection = _json(RUN_ARTIFACTS / "residual_selection.json")
    candidate = selection["selected_release_gate"]
    assert manifest["freeze_state"] == "frozen_before_test"
    assert manifest["selection_partition"] == "val"
    assert manifest["model_selection_source"] == "validation"
    assert manifest["selected_architecture"] == candidate["architecture"] == "reupload"
    assert float(manifest["selected_alpha"]) == 0.0
    operational = manifest["operational_release"]
    assert float(operational["alpha"]) == 0.0
    assert operational["gate_enabled"] is False
    preserved = manifest["validation_selected_quantum_candidate"]
    assert preserved["candidate_kept_for_one_time_test_comparison"] is True
    assert preserved["architecture"] == candidate["architecture"]
    assert np.isclose(float(preserved["alpha"]), float(candidate["alpha"]), rtol=0.0, atol=1e-12)
    assert preserved["gate_feature"] == candidate["gate_feature"]
    assert preserved["gate_operator"] == candidate["gate_operator"]
    assert np.isclose(float(preserved["gate_threshold"]), float(candidate["gate_threshold"]), rtol=0.0, atol=1e-12)
    assert np.isclose(
        float(preserved["deployment_invocation_rate_validation"]),
        float(candidate["quantum_invocation_rate_validation"]),
        rtol=0.0,
        atol=1e-12,
    )
    if not post_evaluation:
        assert manifest["test_metrics"] is None
        assert manifest["test_evaluation_timestamp_utc"] is None
        assert manifest["test_predictions_path"] is None


def test_qpiai_parity_on_reloaded_validation_candidates_when_numpy2_sdk_is_available() -> None:
    predictions_path = RUN_ARTIFACTS / "validation_predictions.npz"
    selection_path = RUN_ARTIFACTS / "residual_selection.json"
    if not predictions_path.is_file() or not selection_path.is_file():
        pytest.skip("trained validation checkpoints are not ready for QpiAI parity")
    np = pytest.importorskip("numpy")
    if int(np.__version__.split(".", maxsplit=1)[0]) < 2:
        pytest.skip("QpiAI 0.3.0 local simulator requires NumPy 2; run this gate in the original SDK runtime")
    try:
        import qpiai_quantum
        from experiments_india_qml.quantum import QuantumResidualRegressor, qpiai_expectations
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"QpiAI/NumPy SDK capability is unavailable in this runtime: {exc}")

    selection = _json(selection_path)
    with np.load(predictions_path, allow_pickle=False) as archive:
        features = np.asarray(archive["vqc_features"], dtype=np.float64)
    assert qpiai_quantum.__version__ == "0.3.0"
    for architecture in ("one_block", "reupload"):
        candidate = selection["quantum_candidates"][architecture]
        checkpoint = ROOT / candidate["checkpoint_path"]
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == candidate["checkpoint_sha256"].lower()
        model = QuantumResidualRegressor.load(checkpoint)
        qpi = qpiai_expectations(features[:8], model.theta, architecture)
        numpy_reference = model.forward_expectations(features[:8])
        assert np.allclose(qpi, numpy_reference, rtol=1e-12, atol=1e-12)
        qpiai_residual_minutes = model.residual_scale_minutes * (
            model.readout[2] + qpi @ model.readout[:2]
        )
        numpy_residual_minutes = model.predict_minutes(features[:8])
        assert np.allclose(qpiai_residual_minutes, numpy_residual_minutes, rtol=1e-12, atol=1e-12)


def test_qpiai_local_statevector_matches_independent_little_endian_reference() -> None:
    """Execute SDK 0.3.0's local simulator and check it against explicit matrices."""
    np = pytest.importorskip("numpy")
    if int(np.__version__.split(".", maxsplit=1)[0]) < 2:
        pytest.skip("QpiAI 0.3.0 local simulator requires NumPy 2; run this gate in the original SDK runtime")
    circuit_module = pytest.importorskip("qpiai_quantum.circuit")
    simulator_module = pytest.importorskip("qpiai_quantum.simulator.statevector")

    x0, x1 = 0.23, 0.67
    theta0, theta1 = 0.41, -0.28

    def ry(angle: float):
        return np.asarray(
            [[np.cos(angle / 2), -np.sin(angle / 2)], [np.sin(angle / 2), np.cos(angle / 2)]],
            dtype=np.complex128,
        )

    def one_qubit(wire: int, angle: float):
        # Statevector index is |q1 q0>; q0 is the least-significant bit.
        return np.kron(ry(angle), np.eye(2)) if wire == 1 else np.kron(np.eye(2), ry(angle))

    def cnot(control: int, target: int):
        matrix = np.zeros((4, 4), dtype=np.complex128)
        for source in range(4):
            destination = source ^ ((1 << target) if ((source >> control) & 1) else 0)
            matrix[destination, source] = 1.0
        return matrix

    expected = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    gates = (
        one_qubit(0, np.pi * x0),
        one_qubit(1, np.pi * x1),
        cnot(0, 1),
        one_qubit(0, theta0),
        one_qubit(1, theta1),
        cnot(1, 0),
    )
    for gate in gates:
        expected = gate @ expected

    circuit = circuit_module.Circuit(2, name="run2-acceptance-local-parity")
    circuit.ry(0, np.pi * x0)
    circuit.ry(1, np.pi * x1)
    circuit.cx(0, 1)
    circuit.ry(0, theta0)
    circuit.ry(1, theta1)
    circuit.cx(1, 0)
    result = simulator_module.StatevectorSimulator().run(circuit, shots=0, seed=17)
    actual = np.asarray(result.statevector, dtype=np.complex128)

    assert result.method == "statevector"
    assert result.job_status == "completed"
    assert result.credits_used == 0
    assert np.isclose(np.linalg.norm(actual), 1.0, atol=1e-12)
    assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    ("architecture", "expected_gates", "expected_parameters"),
    [
        (
            "one_block",
            [
                "RY(pi*x0) q0", "RY(pi*x1) q1", "CX(q0,q1)",
                "RY(theta0) q0", "RY(theta1) q1", "CX(q1,q0)",
            ],
            5,
        ),
        (
            "reupload",
            [
                "RY(pi*x0) q0", "RY(pi*x1) q1",
                "RY(theta0) q0", "RY(theta1) q1", "CX(q0,q1)",
                "RY(pi*x0) q0", "RY(pi*x1) q1",
                "RY(theta2) q0", "RY(theta3) q1", "CX(q1,q0)",
            ],
            7,
        ),
    ],
)
def test_vqc_gate_sequence_feature_order_and_readout(architecture, expected_gates, expected_parameters) -> None:
    np = pytest.importorskip("numpy")
    from experiments_india_qml.quantum import CircuitConfig, QuantumResidualRegressor

    model = QuantumResidualRegressor(
        CircuitConfig(architecture, ("distance_km", "congestion"), residual_scale_minutes=4.0, seed=19)
    )
    circuit = model.circuit_description
    assert circuit["qubits"] == 2
    assert circuit["statevector_indexing"] == "little-endian; q0 is least-significant bit; index=2*q1+q0"
    assert circuit["feature_order"] == ["distance_km", "congestion"]
    assert circuit["gates_in_execution_order"] == expected_gates
    assert circuit["observable_readout"] == "bias + w0*<Z(q0)> + w1*<Z(q1)>"
    assert circuit["parameter_count"] == expected_parameters

    # Each input coordinate must alter at least one measured observable.
    x0_pair = np.asarray([[0.2, 0.6], [0.8, 0.6]])
    x1_pair = np.asarray([[0.6, 0.2], [0.6, 0.8]])
    assert not np.allclose(model.forward_expectations(x0_pair)[0], model.forward_expectations(x0_pair)[1])
    assert not np.allclose(model.forward_expectations(x1_pair)[0], model.forward_expectations(x1_pair)[1])


@pytest.mark.parametrize("architecture", ["one_block", "reupload"])
def test_parameter_shift_gradient_matches_centered_finite_difference(architecture) -> None:
    np = pytest.importorskip("numpy")
    from experiments_india_qml.quantum import CircuitConfig, QuantumResidualRegressor

    rng = np.random.default_rng(20261029)
    features = rng.uniform(0.05, 0.95, size=(32, 2))
    residual_minutes = 5.0 * np.sin(2.0 * features[:, 0] - features[:, 1]) + np.linspace(-0.5, 0.5, 32)
    model = QuantumResidualRegressor(
        CircuitConfig(architecture, ("distance_km", "congestion"), residual_scale_minutes=3.0, seed=19)
    )
    check = model.gradient_check(features, residual_minutes, h=1e-5)
    circuit_gradient = np.asarray(check["analytic_gradient"][: len(model.theta)])
    assert check["step"] == 1e-5
    assert check["max_absolute_error"] < 1e-5
    assert check["passed_abs_1e-5"] is True
    assert np.any(np.abs(circuit_gradient) > 1e-6), "the quantum-angle gradient path must be nonzero"

    # Independently finite-difference the public forward path and learned readout.
    scale = model.residual_scale_minutes
    target = residual_minutes / scale
    original_theta, original_readout = model.theta.copy(), model.readout.copy()
    parameters = np.concatenate((original_theta, original_readout))

    def normalized_loss(candidate):
        model.theta = candidate[: len(original_theta)].copy()
        model.readout = candidate[len(original_theta) :].copy()
        expectations = model.forward_expectations(features)
        predicted = model.readout[2] + expectations @ model.readout[:2]
        return float(np.mean((predicted - target) ** 2))

    numeric = np.empty_like(parameters)
    try:
        for index in range(len(parameters)):
            plus, minus = parameters.copy(), parameters.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            numeric[index] = (normalized_loss(plus) - normalized_loss(minus)) / (2e-5)
    finally:
        model.theta, model.readout = original_theta, original_readout
    analytic = np.asarray(check["analytic_gradient"])
    assert np.max(np.abs(analytic - numeric)) < 1e-5


@pytest.mark.parametrize("architecture", ["one_block", "reupload"])
def test_qpiai_sdk_local_expectations_match_batched_vqc(architecture) -> None:
    np = pytest.importorskip("numpy")
    if int(np.__version__.split(".", maxsplit=1)[0]) < 2:
        pytest.skip("QpiAI 0.3.0 local simulator requires NumPy 2; run this gate in the original SDK runtime")
    qpiai = pytest.importorskip("qpiai_quantum")
    from experiments_india_qml.quantum import CircuitConfig, QuantumResidualRegressor, qpiai_expectations

    assert qpiai.__version__ == "0.3.0"
    features = np.random.default_rng(7).uniform(0.05, 0.95, size=(8, 2))
    model = QuantumResidualRegressor(
        CircuitConfig(architecture, ("distance_km", "congestion"), residual_scale_minutes=4.0, seed=23)
    )
    local = qpiai_expectations(features, model.theta, architecture)
    reference = model.forward_expectations(features)
    assert local.shape == (8, 2)
    assert np.allclose(local, reference, atol=1e-12, rtol=1e-12)


def test_validation_metric_is_recomputed_in_raw_minutes_and_checkpoint_reload_is_exact(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    from experiments_india_qml.quantum import CircuitConfig, QuantumResidualRegressor

    rng = np.random.default_rng(41)
    x_train = rng.uniform(0.05, 0.95, size=(24, 2))
    y_train = 6.0 * (x_train[:, 0] - 0.5) + 2.0 * np.sin(np.pi * x_train[:, 1])
    x_val = rng.uniform(0.05, 0.95, size=(12, 2))
    y_val = 6.0 * (x_val[:, 0] - 0.5) + 2.0 * np.sin(np.pi * x_val[:, 1])
    model = QuantumResidualRegressor(
        CircuitConfig("reupload", ("distance_km", "congestion"), seed=29)
    )
    fit_info = model.fit(
        x_train,
        y_train,
        x_val,
        y_val,
        max_updates=3,
        batch_size=12,
        learning_rate=0.01,
        patience=3,
    )
    independent_rmse_minutes = float(np.sqrt(np.mean((model.predict_minutes(x_val) - y_val) ** 2)))
    assert np.isclose(fit_info["best_validation_rmse_minutes"], independent_rmse_minutes, rtol=1e-12, atol=1e-12)

    before = model.predict_minutes(x_val)
    checkpoint = model.save(tmp_path / "qml_fixture.npz")
    restored = QuantumResidualRegressor.load(checkpoint)
    after = restored.predict_minutes(x_val)
    assert np.array_equal(before, after)


def test_alpha_zero_fallback_returns_classical_eta_and_clamps_only_below_zero() -> None:
    np = pytest.importorskip("numpy")
    from experiments_india_qml.quantum import hybrid_predict_minutes

    classical = np.asarray([20.0, 5.0, 0.0])
    residual = np.asarray([4.0, -9.0, -3.0])
    fallback = hybrid_predict_minutes(classical, residual, alpha=0.0)
    assert np.array_equal(fallback, classical)
    assert np.array_equal(hybrid_predict_minutes(classical, residual, alpha=1.0), np.asarray([24.0, 0.0, 0.0]))
    gated = hybrid_predict_minutes(classical, np.asarray([4.0, -2.0, -3.0]), alpha=0.5, gate_mask=np.asarray([True, False, True]))
    assert np.array_equal(gated, np.asarray([22.0, 5.0, 0.0]))
    for invalid_alpha in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            hybrid_predict_minutes(classical, residual, alpha=invalid_alpha)
    with pytest.raises(ValueError):
        hybrid_predict_minutes([20.0, float("nan")], [1.0, 2.0], alpha=0.0)


def test_final_test_metrics_are_independently_recomputed_after_freeze() -> None:
    """Read only the saved final predictions after model/fusion selection is frozen."""
    if os.environ.get("RUN2_FINAL_TEST_ACCEPTANCE") != "1":
        pytest.skip("sealed-test metric recomputation is held until root explicitly authorizes it")
    manifest_path = RUN_ARTIFACTS / "release_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("release manifest/final test evaluation not ready")
    manifest = _json(manifest_path)
    test_ref = manifest.get("test_predictions_path")
    if not isinstance(test_ref, dict) or "path" not in test_ref:
        pytest.skip("authorized final prediction artifact has not been written")
    prediction_path = ROOT / test_ref["path"]
    if manifest.get("freeze_state") != "frozen_before_test" or not prediction_path.is_file():
        pytest.skip("frozen release and one-time final test predictions are not both present")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("fastparquet")
    import numpy as np
    from datetime import datetime

    frozen_at = datetime.fromisoformat(manifest["freeze_timestamp_utc"].replace("Z", "+00:00"))
    evaluated_at = datetime.fromisoformat(manifest["test_evaluation_timestamp_utc"].replace("Z", "+00:00"))
    assert frozen_at <= evaluated_at
    assert manifest["selection_partition"] == "val", "test observations cannot select circuit, alpha, or calibration"
    start_path = RUN_RUNTIME / "sealed_test_evaluation_started.json"
    assert start_path.is_file(), "single-evaluation start sentinel is required"
    start = _json(start_path)
    assert start["release_manifest_sha256"].lower() == PRETEST_RELEASE_SHA256
    prediction_hash = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    assert prediction_hash == test_ref["sha256"].lower()
    record_path = ROOT / manifest["test_evaluation_record_path"]
    assert hashlib.sha256(record_path.read_bytes()).hexdigest() == manifest["test_evaluation_record_sha256"].lower()
    record = _json(record_path)
    assert record["evaluation_attempt"] == 2
    assert record["prior_attempt_aborted_without_metrics"] is True
    assert record["selection_partition"] == "val"
    assert record["model_selection_source"] == "validation"
    assert record["evaluated_at_utc"] == manifest["test_evaluation_timestamp_utc"]
    assert record["prediction_artifact"]["sha256"].lower() == prediction_hash
    prior_attempt_path = ROOT / record["prior_attempt_evidence_path"]
    prior_attempt = _json(prior_attempt_path)
    assert prior_attempt["attempt"] == 1
    assert prior_attempt["status"] == "aborted_before_metrics_persisted"
    assert prior_attempt["test_data_was_read"] is True
    assert prior_attempt["metrics_or_predictions_persisted"] is False
    assert prior_attempt["test_outcomes_inspected_or_used_for_selection"] is False
    assert prior_attempt["recovery"] == "code-only mapping correction; no fitting, parameter selection, or frozen artifact change"
    retry_path = ROOT / record["evaluation_start_sentinel_path"]
    retry = _json(retry_path)
    assert retry["attempt"] == 2
    assert retry["resumes_failed_attempt"] is True
    assert retry["release_manifest_sha256"].lower() == PRETEST_RELEASE_SHA256
    assert record["quantum_invocations"]["one_block"] == 0
    assert record["quantum_invocations"]["reupload"] == int(round(record["quantum_invocation_rates"]["reupload"] * record["test_row_count"]))

    predictions = pd.read_parquet(prediction_path, engine="fastparquet")
    split = _json(DATA / "split_manifest.json")["split_counts"]
    assert len(predictions) == split["test"]
    assert predictions["ID"].astype(str).is_unique
    courier_count = _json(DATA / "split_manifest.json")["courier_counts"]["test"]
    assert predictions["Delivery_person_ID"].astype(str).nunique() == courier_count
    from fastparquet import ParquetFile
    source_test = ParquetFile(DATA / "indian_delivery_gbdt.parquet").to_pandas(
        columns=["ID", "Delivery_person_ID", "split_partition"],
        filters=[[ ("split_partition", "==", "test") ]],
        row_filter=True,
    )
    expected_test_rows = set(zip(
        source_test["ID"].astype(str),
        source_test["Delivery_person_ID"].astype(str),
    ))
    observed_test_rows = set(zip(
        predictions["ID"].astype(str),
        predictions["Delivery_person_ID"].astype(str),
    ))
    assert observed_test_rows == expected_test_rows
    columns = {
        "classical_gbdt": "prediction_classical_gbdt_minutes",
        "classical_plus_ridge": "prediction_classical_plus_ridge_minutes",
        "quantum_hybrid": "prediction_quantum_hybrid_minutes",
        "selective_quantum": "prediction_selective_quantum_minutes",
        "operational_release": "prediction_operational_release_minutes",
    }
    candidate_alpha = float(manifest["validation_selected_quantum_candidate"]["alpha"])
    assert manifest["freeze_state"] == "frozen_before_test"
    assert manifest["selection_partition"] == "val"
    assert manifest["selected_architecture"] == record["selected_architecture"] == "reupload"
    assert float(manifest["selected_alpha"]) == float(record["selected_alpha"]) == 0.0
    assert np.isclose(candidate_alpha, record["validation_candidate_alpha"], rtol=0.0, atol=1e-12)
    raw_residual_col = "predicted_residual_selected_quantum_minutes"
    gated_residual_col = "predicted_residual_selected_quantum_selective_minutes"
    assert set(predictions.columns).issuperset({
        "ID", "Delivery_person_ID", "y_true_minutes", *columns.values(),
        raw_residual_col, gated_residual_col,
    })
    values = predictions[["y_true_minutes", *columns.values(), raw_residual_col, gated_residual_col]].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all()
    assert float(manifest["selected_alpha"]) == 0.0
    classical = predictions[columns["classical_gbdt"]].to_numpy(dtype=np.float64)
    raw_residual = predictions[raw_residual_col].to_numpy(dtype=np.float64)
    gated_residual = predictions[gated_residual_col].to_numpy(dtype=np.float64)
    expected_candidate = np.maximum(0.0, classical + candidate_alpha * raw_residual)
    expected_selective = np.maximum(0.0, classical + candidate_alpha * gated_residual)
    assert np.allclose(predictions[columns["quantum_hybrid"]].to_numpy(dtype=np.float64), expected_candidate, rtol=1e-12, atol=1e-10)
    assert np.allclose(predictions[columns["selective_quantum"]].to_numpy(dtype=np.float64), expected_selective, rtol=1e-12, atol=1e-10)
    assert np.array_equal(predictions[columns["operational_release"]].to_numpy(dtype=np.float64), classical)
    assert np.array_equal(predictions["prediction_alpha_zero_fallback_minutes"].to_numpy(dtype=np.float64), classical)
    assert record["test_row_count"] == len(predictions)
    assert record["unique_test_ids"] == predictions["ID"].nunique()
    assert record["test_target_unit"] == "minutes"

    actual = predictions["y_true_minutes"].to_numpy(dtype=np.float64)
    recomputed = {}
    for metric_name, column in columns.items():
        error = predictions[column].to_numpy(dtype=np.float64) - actual
        recomputed[metric_name] = {
            "mae_minutes": float(np.mean(np.abs(error))),
            "rmse_minutes": float(np.sqrt(np.mean(error**2))),
            "median_absolute_error_minutes": float(np.median(np.abs(error))),
        }
        for metric, value in recomputed[metric_name].items():
            assert np.isclose(value, manifest["test_metrics"][metric_name][metric], rtol=1e-10, atol=1e-10)
            assert np.isclose(value, record["metrics"][metric_name][metric], rtol=1e-10, atol=1e-10)
