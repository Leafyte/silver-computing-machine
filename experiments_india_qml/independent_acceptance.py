"""Independent, outcome-blind RUN2 India artifact checks.

These helpers intentionally do not train/select models or open test labels.
The final test evaluator is a separate, explicitly post-freeze operation.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SOURCE_CSV_SHA256 = "4baa699e1c09d286f06790010a8c0b284ed3376cbde7e346e19d0ded16a240e3"
GBDT_PARQUET_SHA256 = "bd03b47c3d3e23050329cefc69d752aad607031cacdb9696151d237e3534a627"
VQC_PARQUET_SHA256 = "a633677472e6e0e874c2797a66eb7b011e3a307285553f8fa00c6ee664df4b4e"
EXPECTED_PARTITION_COUNTS = {"train": 29_592, "val": 5_989, "test": 5_941}
FORBIDDEN_FEATURES = {
    "ID", "delivery_record_id", "Delivery_person_ID", "courier_id",
    "split_partition", "target", "delivery_duration_min", "y_true_min",
    "delivery_duration", "is_sla_breach_synthetic", "target_normalized",
}
VQC_FEATURES_2 = ("vqc_x0_distance", "vqc_x1_congestion")
VQC_FEATURES_4 = (
    "vqc_x0_distance", "vqc_x1_congestion",
    "vqc_x2_multidelivery", "vqc_x3_preptime",
)
VALIDATION_ROLE_PREFIX = b"india-eta-val-role-v1|20261029|"


def validation_roles_for_couriers(courier_ids: Iterable[str]) -> dict[str, str]:
    """Freeze exact 1/3 calibration, 2/3 selection by deterministic courier hash."""
    unique = {str(courier) for courier in courier_ids}
    if len(unique) < 3:
        raise ValueError("at least three validation couriers are required")
    ranked = sorted(
        unique,
        key=lambda courier: (hashlib.sha256(VALIDATION_ROLE_PREFIX + courier.encode("utf-8")).digest(), courier),
    )
    calibration_count = len(ranked) // 3
    calibration = set(ranked[:calibration_count])
    return {
        courier: "fusion_calibration" if courier in calibration else "model_selection"
        for courier in unique
    }


def validate_validation_roles(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Ensure validation role assignment exactly follows the frozen courier hash."""
    expected_roles = validation_roles_for_couriers(str(row["courier_id"]) for row in records)
    roles: dict[str, str] = {}
    role_couriers: dict[str, set[str]] = {"model_selection": set(), "fusion_calibration": set()}
    ids: set[str] = set()
    for row in records:
        if row.get("split_partition") != "val":
            raise ValueError("validation role map may contain validation rows only")
        identity = str(row["delivery_record_id"])
        courier = str(row["courier_id"])
        if identity in ids:
            raise ValueError(f"duplicate validation role record ID {identity}")
        ids.add(identity)
        expected = expected_roles[courier]
        actual = str(row["validation_role"])
        if actual != expected:
            raise ValueError(f"validation role mismatch for courier {courier}: {actual} != {expected}")
        prior = roles.setdefault(courier, expected)
        if prior != actual:
            raise ValueError(f"courier {courier} spans validation roles")
        role_couriers[actual].add(courier)
    if role_couriers["model_selection"] & role_couriers["fusion_calibration"]:
        raise ValueError("courier groups overlap selection/calibration roles")
    return {
        "records": len(ids),
        "couriers": {role: len(groups) for role, groups in role_couriers.items()},
        "algorithm": "deduplicate validation courier_id; sort by (SHA256(b'india-eta-val-role-v1|20261029|' + courier_id UTF-8) digest bytes, courier_id); first floor(N/3) groups fusion_calibration, remainder model_selection",
        "passed": True,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout.decode("utf-8", errors="strict").strip()


def verify_committed_csv_source(repo: str | Path) -> dict[str, Any]:
    """Verify source CSV's committed LF blob and explain a CRLF worktree copy.

    Git's `core.autocrlf=true` can make the on-disk SHA differ from the
    committed blob. This reports both byte-level hashes and accepts conversion
    only when normalizing CRLF to LF reproduces the exact committed bytes.
    """
    root = Path(repo).resolve()
    source = root / "data" / "train.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    blob = subprocess.run(
        ["git", "-C", str(root), "show", "HEAD:data/train.csv"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    worktree = source.read_bytes()
    canonical = worktree.replace(b"\r\n", b"\n")
    blob_sha = hashlib.sha256(blob).hexdigest()
    disk_sha = hashlib.sha256(worktree).hexdigest()
    canonical_sha = hashlib.sha256(canonical).hexdigest()
    attrs = _git(root, "ls-files", "--eol", "--", "data/train.csv")
    autocrlf = _git(root, "config", "--get", "core.autocrlf")
    clean = not _git(root, "status", "--porcelain", "--", "data/train.csv")
    if blob_sha != SOURCE_CSV_SHA256:
        raise ValueError(f"committed data/train.csv blob SHA mismatch: {blob_sha}")
    if canonical != blob:
        raise ValueError("worktree CSV differs from committed blob beyond CRLF normalization")
    if not clean:
        raise ValueError("data/train.csv has uncommitted changes")
    return {
        "path": str(source), "working_copy_sha256": disk_sha,
        "canonical_lf_sha256": canonical_sha, "git_blob_sha256": blob_sha,
        "expected_sha256": SOURCE_CSV_SHA256, "git_eol": attrs,
        "core_autocrlf": autocrlf, "worktree_clean": clean,
        "conversion": "CRLF worktree bytes normalize exactly to committed LF blob"
        if worktree != blob else "none",
        "passed": True,
    }


def verify_data_hashes(repo: str | Path) -> dict[str, Any]:
    """Verify committed parquet bytes and CSV Git-blob provenance."""
    root = Path(repo).resolve()
    csv = verify_committed_csv_source(root)
    expected = {
        "data/indian_delivery_gbdt.parquet": GBDT_PARQUET_SHA256,
        "data/indian_delivery_vqc.parquet": VQC_PARQUET_SHA256,
    }
    parquet: dict[str, dict[str, Any]] = {}
    for relative, want in expected.items():
        path = root / relative
        actual = sha256_file(path)
        if actual != want:
            raise ValueError(f"{relative} SHA mismatch: expected {want}, got {actual}")
        parquet[relative] = {"sha256": actual, "expected_sha256": want, "passed": True}
    return {"source_csv": csv, "parquet": parquet}


def validate_feature_names(names: Sequence[str], *, allowed: Sequence[str] | None = None) -> None:
    """Reject target/identity/split fields and enforce exact frozen order."""
    values = tuple(str(name) for name in names)
    if len(values) != len(set(values)):
        raise ValueError("feature names must be unique")
    found = set(values) & FORBIDDEN_FEATURES
    if found:
        raise ValueError(f"identity, split, or target fields in predictors: {sorted(found)}")
    if allowed is not None and values != tuple(allowed):
        raise ValueError("feature names/order differ from the frozen schema")


def validate_partition_groups(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Recompute original split counts and courier isolation from row records."""
    counts = {key: 0 for key in EXPECTED_PARTITION_COUNTS}
    groups: dict[str, set[str]] = {key: set() for key in EXPECTED_PARTITION_COUNTS}
    ids: set[str] = set()
    for row in records:
        partition = str(row["split_partition"])
        if partition not in counts:
            continue
        identity = str(row["delivery_record_id"])
        courier = str(row["courier_id"])
        if identity in ids:
            raise ValueError(f"duplicate delivery_record_id: {identity}")
        ids.add(identity)
        counts[partition] += 1
        groups[partition].add(courier)
    if counts != EXPECTED_PARTITION_COUNTS:
        raise ValueError(f"partition row counts differ from frozen manifest: {counts}")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups[left] & groups[right]
        if overlap:
            raise ValueError(f"courier leakage across {left}/{right}: {len(overlap)} groups")
    return counts


def validate_oof_rows(
    records: Sequence[Mapping[str, Any]], *, expected_train_count: int = 29_592,
    fold_train_couriers: Mapping[int, set[str]] | None = None,
) -> dict[str, Any]:
    """Check OOF identity, residual arithmetic, finite minutes, and fold isolation."""
    if len(records) != expected_train_count:
        raise ValueError(f"OOF count {len(records)} != frozen train count {expected_train_count}")
    ids: set[str] = set()
    for row in records:
        identity = str(row["delivery_record_id"])
        if identity in ids:
            raise ValueError(f"duplicate OOF ID {identity}")
        ids.add(identity)
        if row.get("split_partition") != "train":
            raise ValueError("OOF rows must be from the original train partition only")
        courier = str(row["courier_id"])
        fold = int(row["fold"])
        if fold_train_couriers is not None:
            if fold not in fold_train_couriers:
                raise ValueError(f"unknown OOF fold {fold}")
            if courier in fold_train_couriers[fold]:
                raise ValueError(f"OOF prediction for {identity} trained on its courier")
        y = float(row["y_true_min"])
        pred = float(row["gbdt_pred_min"])
        residual_value = row.get("residual_true_min")
        if not math.isfinite(y) or not math.isfinite(pred):
            raise ValueError("OOF target and prediction must be finite minutes")
        residual = y - pred
        if residual_value is not None:
            residual = float(residual_value)
            if not math.isfinite(residual) or not math.isclose(residual, y - pred, rel_tol=1e-10, abs_tol=1e-9):
                raise ValueError(f"OOF residual arithmetic mismatch for {identity}")
        if "test" in str(row.get("split_partition", "")).lower():
            raise ValueError("sealed test row appeared in OOF artifact")
    return {"rows": len(records), "unique_ids": len(ids), "passed": True}


def keyed_join(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]],
    *, key: Sequence[str] = ("delivery_record_id", "split_partition"),
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Join records by stable identity keys and reject duplicates/missing rows."""
    def index(rows: Sequence[Mapping[str, Any]], label: str) -> dict[tuple[str, ...], Mapping[str, Any]]:
        result: dict[tuple[str, ...], Mapping[str, Any]] = {}
        for row in rows:
            k = tuple(str(row[name]) for name in key)
            if k in result:
                raise ValueError(f"duplicate {label} join key {k}")
            result[k] = row
        return result
    a, b = index(left, "left"), index(right, "right")
    if a.keys() != b.keys():
        raise ValueError(f"keyed join mismatch: left-only={len(a.keys() - b.keys())}, right-only={len(b.keys() - a.keys())}")
    return [(a[k], b[k]) for k in sorted(a)]


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float | int]:
    """Recompute requested regression metrics from aligned minute values."""
    import numpy as np

    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("finite non-empty aligned target/prediction arrays are required")
    error = p - y
    return {
        "n": int(len(y)), "mae_min": float(np.mean(np.abs(error))),
        "rmse_min": float(np.sqrt(np.mean(error ** 2))),
        "median_absolute_error_min": float(np.median(np.abs(error))),
        "mean_error_min": float(np.mean(error)),
    }


def verify_manifest_hashes(root: str | Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    """Verify a manifest inventory and refuse path traversal/symlink escape."""
    base = Path(root).resolve()
    inventory = manifest.get("artifacts") or manifest.get("files")
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("manifest must contain a non-empty artifacts/files mapping")
    verified: dict[str, str] = {}
    for relative, item in inventory.items():
        expected = item if isinstance(item, str) else item.get("sha256")
        candidate = (base / str(relative)).resolve()
        if candidate != base and base not in candidate.parents:
            raise ValueError(f"manifest path escapes release root: {relative}")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        actual = sha256_file(candidate)
        if not isinstance(expected, str) or actual != expected.lower():
            raise ValueError(f"artifact SHA mismatch for {relative}: {actual}")
        verified[str(relative)] = actual
    return verified


def _assert_close_metrics(actual: Mapping[str, float], expected: Mapping[str, float], label: str, tol: float = 1e-9) -> None:
    for key in ("mae_min", "rmse_min", "median_absolute_error_min"):
        if key not in expected or not math.isclose(float(actual[key]), float(expected[key]), rel_tol=tol, abs_tol=tol):
            raise ValueError(f"{label} metric {key} differs: recomputed={actual.get(key)}, manifest={expected.get(key)}")


def audit_classical_recovery(repo: str | Path) -> dict[str, Any]:
    """Independently verify the frozen GBDT/OOF/validation artifacts.

    This reads no sealed test target. It reads split/group metadata without
    labels to establish ID coverage, and reads only validation feature columns
    for checkpoint reload parity. Metrics are recomputed from the saved
    training OOF and validation prediction records.
    """
    root = Path(repo).resolve()
    manifest_path = root / "artifacts" / "india_eta_recovery" / "recovery_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "trained" or manifest.get("target_name", "delivery_duration_min") != "delivery_duration_min" or manifest.get("target_unit", "minutes") != "minutes":
        raise ValueError("recovery manifest status/target contract is invalid")
    prov = manifest["data_provenance"]
    source_audit = verify_committed_csv_source(root)
    source_commit = str(prov["source_commit"])
    committed_source = subprocess.run(
        ["git", "-C", str(root), "show", f"{source_commit}:data/train.csv"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    committed_source_sha = hashlib.sha256(committed_source).hexdigest()
    if committed_source_sha != SOURCE_CSV_SHA256:
        raise ValueError("recovery source_commit does not contain the verified canonical CSV blob")
    if source_audit["git_blob_sha256"] != prov["source_csv"]["canonical_git_sha256_lf"]:
        raise ValueError("recovery manifest source CSV provenance does not match committed Git blob")
    parquet_expected = {
        "indian_delivery_gbdt.parquet": GBDT_PARQUET_SHA256,
        "indian_delivery_vqc.parquet": VQC_PARQUET_SHA256,
        "indian_delivery_clean.parquet": prov["files_sha256"]["clean_parquet"],
    }
    data_hashes: dict[str, str] = {}
    for name, expected in parquet_expected.items():
        path = root / "data" / name
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"input parquet SHA mismatch for {name}: {actual}")
        data_hashes[name] = actual

    features = list(manifest["model_artifacts"].get("feature_names", []))
    schema_path = root / "artifacts" / "india_eta_recovery" / "gbdt" / "feature_schema.json"
    schema_doc = json.loads(schema_path.read_text(encoding="utf-8"))
    features = list(schema_doc["feature_names"])
    validate_feature_names(features)
    source_schema = json.loads((root / "data" / "feature_schema.json").read_text(encoding="utf-8-sig"))
    validate_feature_names(features, allowed=source_schema["gbdt_view"]["features"])
    if features != manifest["model_selection"]["chosen_config"].get("feature_names", features):
        # The chosen config intentionally contains hyperparameters only; tolerate
        # absence but never accept a contradictory embedded feature list.
        raise ValueError("feature schema and selection metadata disagree")

    file_entries = []
    for section in ("prediction_artifacts", "model_artifacts"):
        for key, entry in manifest[section].items():
            path = root / str(entry["path"])
            digest = sha256_file(path)
            if digest != entry["sha256"]:
                raise ValueError(f"{section}.{key} artifact SHA mismatch: {digest}")
            file_entries.append({"path": entry["path"], "sha256": digest})

    role_entry = manifest["prediction_artifacts"]["validation_role_assignment"]
    role_path = root / role_entry["path"]
    role_doc = json.loads(role_path.read_text(encoding="utf-8"))
    role_map = {str(k): str(v) for k, v in role_doc["courier_to_role"].items()}
    role_map_hash = hashlib.sha256(json.dumps(role_map, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if role_map_hash != manifest["data_provenance"]["validation_role_assignment_sha256"] or role_doc["assignment_sha256"] != role_map_hash:
        raise ValueError("validation role assignment hash does not match the courier map")
    if role_doc["algorithm"] != manifest["data_provenance"]["validation_role_algorithm"]:
        raise ValueError("validation role assignment algorithm differs from frozen manifest")

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas + fastparquet are required for row-level independent acceptance") from exc
    oof_path = root / manifest["prediction_artifacts"]["oof"]["path"]
    val_path = root / manifest["prediction_artifacts"]["validation"]["path"]
    oof_frame = pd.read_parquet(oof_path, engine="fastparquet")
    val_frame = pd.read_parquet(val_path, engine="fastparquet")
    oof_frame = oof_frame.rename(columns={"ID": "delivery_record_id", "Delivery_person_ID": "courier_id"})
    val_frame = val_frame.rename(columns={"ID": "delivery_record_id", "Delivery_person_ID": "courier_id"})
    oof = oof_frame.to_dict(orient="records")
    val = val_frame.to_dict(orient="records")
    fold_doc = json.loads((root / manifest["prediction_artifacts"]["oof_fold_manifest"]["path"]).read_text(encoding="utf-8"))
    folds = fold_doc["folds"]
    fold_train_groups = {int(k): set(v["train_couriers"]) for k, v in folds.items()}
    oof_audit = validate_oof_rows(oof, fold_train_couriers=fold_train_groups)
    if len(val) != EXPECTED_PARTITION_COUNTS["val"] or any(row["split_partition"] != "val" for row in val):
        raise ValueError("validation prediction artifact has incorrect row count/partition")
    role_audit = validate_validation_roles(val)
    if role_audit["couriers"] != role_doc["role_courier_counts"]:
        raise ValueError("validation role courier counts mismatch")
    if len(role_map) != EXPECTED_PARTITION_COUNTS["val"] and len(role_map) != role_audit["couriers"]["model_selection"] + role_audit["couriers"]["fusion_calibration"]:
        raise ValueError("validation role map unexpectedly contains record rows instead of courier rows")
    if set(role_map) != {str(row["courier_id"]) for row in val}:
        raise ValueError("validation role map courier set does not match validation predictions")

    # Independently establish the eligible source row ID/group/split inventory
    # without loading any duration or target-derived column.
    clean_path = root / "data" / "indian_delivery_clean.parquet"
    inventory = pd.read_parquet(clean_path, engine="fastparquet", columns=["ID", "Delivery_person_ID", "split_partition", "is_model_eligible"])
    inventory = inventory.loc[inventory["is_model_eligible"].astype(bool) & inventory["split_partition"].isin(["train", "val", "test"])]
    if inventory["ID"].duplicated().any() or len(inventory) != sum(EXPECTED_PARTITION_COUNTS.values()):
        raise ValueError("source eligible record inventory does not match frozen partition contract")
    src_train = set(inventory.loc[inventory.split_partition == "train", "ID"].astype(str))
    src_val = set(inventory.loc[inventory.split_partition == "val", "ID"].astype(str))
    src_test = set(inventory.loc[inventory.split_partition == "test", "ID"].astype(str))
    oof_ids = {str(row["delivery_record_id"]) for row in oof}
    val_ids = {str(row["delivery_record_id"]) for row in val}
    if oof_ids != src_train or val_ids != src_val or (oof_ids | val_ids) & src_test:
        raise ValueError("OOF/validation row IDs mismatch original source partitions or overlap test IDs")
    for part in ("train", "val", "test"):
        part_rows = inventory.loc[inventory.split_partition == part]
        courier_groups = set(part_rows.Delivery_person_ID.astype(str))
        if part == "train": train_groups = courier_groups
        elif part == "val": val_groups = courier_groups
        else: test_groups = courier_groups
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise ValueError("original split courier groups overlap")

    # Verify fold coverage and group partition directly from the fold manifest.
    oof_fold_by_courier: dict[str, int] = {}
    for row in oof:
        courier, fold = str(row["courier_id"]), int(row["fold"])
        prior = oof_fold_by_courier.setdefault(courier, fold)
        if prior != fold:
            raise ValueError(f"courier {courier} is split across OOF folds")
    for fold_key, record in folds.items():
        fold = int(fold_key)
        held = set(record["heldout_couriers"])
        fitted = set(record["train_couriers"])
        if held & fitted or held != {courier for courier, row_fold in oof_fold_by_courier.items() if row_fold == fold}:
            raise ValueError(f"OOF fold {fold} does not match heldout courier groups")
        held_rows = [r for r in oof if int(r["fold"]) == fold]
        if len(held_rows) != int(record["heldout_row_count"]):
            raise ValueError(f"OOF fold {fold} row count mismatch")

    # Recompute OOF/validation metrics using saved record-level outcomes.
    oof_metrics = regression_metrics([r["y_true_min"] for r in oof], [r["gbdt_pred_min"] for r in oof])
    val_y = [r["y_true_min"] for r in val]
    val_p = [r["gbdt_pred_min"] for r in val]
    val_roles = [str(r["validation_role"]) for r in val]
    val_metrics = regression_metrics(val_y, val_p)
    selection_index = [i for i, role in enumerate(val_roles) if role == "model_selection"]
    calibration_index = [i for i, role in enumerate(val_roles) if role == "fusion_calibration"]
    select_metrics = regression_metrics([val_y[i] for i in selection_index], [val_p[i] for i in selection_index])
    calibration_metrics = regression_metrics([val_y[i] for i in calibration_index], [val_p[i] for i in calibration_index])
    metric_doc = manifest["frozen_model_metrics"]
    _assert_close_metrics(val_metrics, metric_doc["all_validation_descriptive"], "all-validation")
    _assert_close_metrics(select_metrics, metric_doc["model_selection"], "model-selection")
    _assert_close_metrics(calibration_metrics, metric_doc["fusion_calibration"], "fusion-calibration")

    # Only after SHA verification, reload the trusted local model and independently
    # recompute validation predictions from feature columns (target is not read).
    model_entry = manifest["model_artifacts"]["model"]
    model_path = root / model_entry["path"]
    import joblib
    model = joblib.load(model_path)
    if type(model).__module__ != "sklearn.ensemble._hist_gradient_boosting.gradient_boosting" or type(model).__name__ != "HistGradientBoostingRegressor":
        raise ValueError(f"unexpected estimator type after trusted checksum: {type(model).__module__}.{type(model).__name__}")
    gbdt_data_path = root / "data" / "indian_delivery_gbdt.parquet"
    val_x = pd.read_parquet(gbdt_data_path, engine="fastparquet", columns=["ID", "Delivery_person_ID", "split_partition", *features], filters=[("split_partition", "==", "val")])
    val_x = val_x.loc[val_x.split_partition == "val"]
    if val_x["ID"].duplicated().any() or set(val_x["ID"].astype(str)) != val_ids:
        raise ValueError("reloaded model validation feature IDs do not match saved predictions")
    val_pred_independent = model.predict(val_x[features].to_numpy(dtype="float64"))
    aligned_saved = keyed_join(
        [{"delivery_record_id": str(r["ID"]), "split_partition": str(r["split_partition"]), "pred": float(pred)} for r, pred in zip(val_x.to_dict(orient="records"), val_pred_independent)],
        [{"delivery_record_id": str(r["delivery_record_id"]), "split_partition": str(r["split_partition"]), "pred": float(r["gbdt_pred_min"])} for r in val],
    )
    max_reload_error = max(abs(float(a["pred"]) - float(b["pred"])) for a, b in aligned_saved)
    if max_reload_error > 1e-10:
        raise ValueError(f"frozen GBDT reload differs from validation predictions by {max_reload_error}")

    return {
        "status": "PASS",
        "test_targets_read": False,
        "source_provenance": source_audit,
        "source_commit": source_commit,
        "source_commit_csv_blob_sha256": committed_source_sha,
        "input_parquet_sha256": data_hashes,
        "verified_model_and_prediction_files": file_entries,
        "source_partition_counts": {k: EXPECTED_PARTITION_COUNTS[k] for k in EXPECTED_PARTITION_COUNTS},
        "source_courier_groups": {"train": len(train_groups), "val": len(val_groups), "test": len(test_groups)},
        "oof": oof_audit,
        "oof_metrics_minutes_recomputed": oof_metrics,
        "validation_role_assignment": role_audit | {"sha256": role_map_hash},
        "validation_metrics_minutes_recomputed": {
            "all_validation_descriptive": val_metrics,
            "model_selection": select_metrics,
            "fusion_calibration": calibration_metrics,
        },
        "reloaded_model_validation_prediction_max_abs_error_minutes": max_reload_error,
        "limitations": ["release and quantum-model acceptance remain pending", "test evaluation remains embargoed until all model weights and fusion policy are frozen"],
    }


def write_classical_acceptance(repo: str | Path, output: str | Path | None = None) -> Path:
    """Run the independent pre-test audit and persist its evidence JSON."""
    root = Path(repo).resolve()
    result = audit_classical_recovery(root)
    result["scope"] = "classical checkpoint, source provenance, OOF, validation roles, and pretest prediction reload only"
    destination = Path(output).resolve() if output else root / "artifacts" / "run2_india" / "acceptance_classical.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return destination

