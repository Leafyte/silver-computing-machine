"""Read-only integrity preflight for India ETA recovery artifacts.

This validates a known local recovery manifest and its referenced files. It
never deserializes estimator objects, fits models, reads test rows, or
authorizes quantum training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping


DATA_FILES = (
    "data/train.csv",
    "data/indian_delivery_clean.parquet",
    "data/indian_delivery_gbdt.parquet",
    "data/indian_delivery_vqc.parquet",
    "data/feature_schema.json",
    "data/preprocessing_artifacts.json",
    "data/split_manifest.json",
)
RECOVERY_MANIFEST_RELATIVE = Path("artifacts/india_eta_recovery/recovery_manifest.json")
TARGET_NAME = "delivery_duration_min"
ETA_NAME = re.compile(r"india|indian|delivery|eta", re.IGNORECASE)
ARTIFACT_NAME = re.compile(r"gbdt|model|checkpoint|prediction|predictions|scaler|preprocess", re.IGNORECASE)
SERIALIZED_SUFFIXES = {".json", ".jsonl", ".npz", ".npy", ".joblib", ".pkl", ".pickle", ".pt", ".pth", ".onnx"}
SKIP_DIRS = {".git", ".venv", "venv", ".runtime", "node_modules", "site-packages", "__pycache__"}
EXPECTED_GROUPS = {
    "model_artifacts": {"model", "feature_schema", "preprocessing_artifacts"},
    "prediction_artifacts": {"oof", "validation", "oof_fold_manifest", "validation_role_assignment"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_lf(data: bytes) -> bytes:
    """Normalize only CRLF to LF; leave lone CR bytes visible as corruption."""
    return data.replace(b"\r\n", b"\n")


def source_identity(local_bytes: bytes, committed_bytes: bytes | None, expected_sha256: str | None) -> dict[str, Any]:
    canonical = canonical_lf(local_bytes)
    canonical_hash = hashlib.sha256(canonical).hexdigest()
    raw_hash = hashlib.sha256(local_bytes).hexdigest()
    git_blob_sha1 = hashlib.sha1(b"blob " + str(len(canonical)).encode("ascii") + b"\0" + canonical).hexdigest()
    committed_hash = hashlib.sha256(committed_bytes).hexdigest() if committed_bytes is not None else None
    return {
        "raw_sha256": raw_hash,
        "canonical_lf_sha256": canonical_hash,
        "git_blob_sha1": git_blob_sha1,
        "committed_blob_sha256": committed_hash,
        "expected_canonical_sha256": expected_sha256,
        "crlf_line_endings": local_bytes.count(b"\r\n"),
        "canonical_matches_expected": expected_sha256 is not None and canonical_hash == expected_sha256,
        "canonical_matches_git_blob": committed_bytes is not None and canonical == committed_bytes,
        "matches": (expected_sha256 is not None and canonical_hash == expected_sha256
                    and committed_bytes is not None and canonical == committed_bytes),
    }


def _git_blob(repo: Path, relative: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{relative}"], cwd=repo,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout


def discover_candidates(roots: list[Path]) -> list[dict[str, str]]:
    """Manual-only name search; candidates are listed and never auto-loaded."""
    candidates: dict[str, dict[str, str]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SERIALIZED_SUFFIXES:
                continue
            if any(part.lower() in SKIP_DIRS for part in path.parts):
                continue
            if ETA_NAME.search(str(path)) and ARTIFACT_NAME.search(path.name):
                candidates[str(path.resolve())] = {"path": str(path.resolve()), "sha256": sha256(path)}
    return [candidates[key] for key in sorted(candidates)]


def check_data(repo: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None,
                                    list[str], list[dict[str, Any]], dict[str, Any] | None]:
    files: list[dict[str, Any]] = []
    problems: list[str] = []
    for relative in DATA_FILES:
        path = repo / relative
        entry: dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
        if path.is_file():
            entry["sha256"] = sha256(path)
        files.append(entry)

    schema_path, split_path = repo / "data/feature_schema.json", repo / "data/split_manifest.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.is_file() else None
    split = json.loads(split_path.read_text(encoding="utf-8")) if split_path.is_file() else None
    if schema is None or schema.get("target", {}).get("name") != TARGET_NAME:
        problems.append(f"target schema is missing or does not name {TARGET_NAME!r}")
    if split is None:
        problems.append("split manifest is missing")
    missing = [item["path"] for item in files if not item["exists"]]
    if missing:
        problems.append("one or more frozen source-data files are missing")

    expected_hashes = {
        "data/indian_delivery_clean.parquet": split.get("file_digests", {}).get("clean_parquet_sha256") if split else None,
        "data/indian_delivery_gbdt.parquet": split.get("file_digests", {}).get("gbdt_parquet_sha256") if split else None,
        "data/indian_delivery_vqc.parquet": split.get("file_digests", {}).get("vqc_parquet_sha256") if split else None,
    }
    by_relative_path = {Path(item["path"]).relative_to(repo).as_posix(): item for item in files}
    integrity: list[dict[str, Any]] = []
    for relative, expected in expected_hashes.items():
        observed = by_relative_path[relative].get("sha256")
        check = {"path": str((repo / relative).resolve()), "expected_sha256": expected,
                 "observed_sha256": observed, "matches": expected is not None and observed == expected}
        integrity.append(check)
        if expected is not None and observed is not None and observed != expected:
            problems.append(f"recorded digest mismatch for {relative}: expected {expected}, observed {observed}")

    source_path = repo / "data/train.csv"
    source_expected = split.get("source_sha256") if split else None
    local_source = source_path.read_bytes() if source_path.is_file() else None
    committed_source = _git_blob(repo, "data/train.csv") if local_source is not None else None
    source_check = None
    if local_source is not None:
        source_check = source_identity(local_source, committed_source, source_expected)
        integrity.append({"path": str(source_path.resolve()), **source_check})
        if not source_check["matches"]:
            problems.append("source CSV canonical bytes do not match both the split-manifest SHA-256 and current HEAD Git blob")
    return files, schema, split, problems, integrity, source_check


def _safe_manifest_path(repo: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("artifact paths must be repository-relative")
    resolved = (repo / path).resolve()
    if not resolved.is_relative_to(repo.resolve()):
        raise ValueError("artifact path escapes repository root")
    return resolved


def verify_recovery_manifest(repo: Path, manifest_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Verify the known recovery manifest and every declared artifact hash."""
    problems: list[str] = []
    result: dict[str, Any] = {"path": str(manifest_path.resolve()), "exists": manifest_path.is_file(),
                              "artifacts": {}, "verified": False}
    if not manifest_path.is_file():
        return result, [f"trusted recovery manifest is missing: {manifest_path}"]
    result["sha256"] = sha256(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return result, [f"recovery manifest is unreadable: {exc}"]
    result["status"] = manifest.get("status")
    if manifest.get("status") != "trained":
        problems.append("recovery manifest status is not 'trained'")
    provenance = manifest.get("data_provenance", {})
    if manifest.get("estimator_identity") != "sklearn.ensemble.HistGradientBoostingRegressor":
        problems.append("recovery manifest does not identify the expected HistGradientBoostingRegressor")
    if provenance.get("target") != TARGET_NAME or provenance.get("target_unit") != "minutes":
        problems.append("recovery manifest target contract is not delivery_duration_min in minutes")
    if provenance.get("gbdt_feature_count") != 59:
        problems.append("recovery manifest must declare the verified 59-feature GBDT view")
    if manifest.get("sealed_test") is None or "not loaded" not in str(manifest.get("sealed_test")):
        problems.append("recovery manifest does not state that the test partition was sealed")

    resolved_artifacts: dict[str, dict[str, Any]] = {}
    for group, expected_names in EXPECTED_GROUPS.items():
        entries = manifest.get(group)
        if not isinstance(entries, Mapping) or set(entries) != expected_names:
            problems.append(f"recovery manifest {group} keys do not match the required verified artifact set")
            continue
        resolved_artifacts[group] = {}
        for name, detail in entries.items():
            if not isinstance(detail, Mapping) or not isinstance(detail.get("path"), str):
                problems.append(f"manifest artifact {group}.{name} has no relative path")
                continue
            expected_hash = detail.get("sha256")
            if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
                problems.append(f"manifest artifact {group}.{name} has an invalid SHA-256")
                continue
            try:
                path = _safe_manifest_path(repo, detail["path"])
            except ValueError as exc:
                problems.append(f"manifest artifact {group}.{name}: {exc}")
                continue
            observed = sha256(path) if path.is_file() else None
            okay = observed == expected_hash
            resolved_artifacts[group][name] = {
                "path": str(path), "manifest_path": detail["path"], "exists": path.is_file(),
                "expected_sha256": expected_hash, "observed_sha256": observed, "matches": okay,
            }
            if not okay:
                problems.append(f"manifest artifact {group}.{name} is missing or its SHA-256 does not match")
    result["artifacts"] = resolved_artifacts
    if not problems:
        result["verified"] = True
        result["target"] = TARGET_NAME
        result["target_unit"] = "minutes"
        result["feature_count"] = 59
        result["oof_rows"] = manifest.get("prediction_artifacts", {}).get("oof", {}).get("rows")
        result["validation_rows"] = manifest.get("prediction_artifacts", {}).get("validation", {}).get("rows")
    return result, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1], help="trainingv3 repository root")
    parser.add_argument("--checkpoint", type=Path, help="optional explicit checkpoint; must match the verified recovery manifest")
    parser.add_argument("--predictions", type=Path, action="append", default=[], help="optional explicit OOF/validation prediction; repeat to check both; must match manifest")
    parser.add_argument("--search-root", type=Path, action="append", default=[], help="manual-only candidate search root; not scanned unless explicitly supplied")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    data_files, schema, split, problems, integrity_checks, source_check = check_data(repo)
    recovery_manifest_path = repo / RECOVERY_MANIFEST_RELATIVE
    recovery, recovery_problems = verify_recovery_manifest(repo, recovery_manifest_path)
    problems.extend(recovery_problems)
    verified = recovery.get("verified", False)

    provenance = recovery_manifest = {}
    if recovery_manifest_path.is_file():
        try:
            recovery_manifest = json.loads(recovery_manifest_path.read_text(encoding="utf-8"))
            provenance = recovery_manifest.get("data_provenance", {})
        except (OSError, json.JSONDecodeError):
            pass
    provenance_hashes = {
        "data/indian_delivery_clean.parquet": provenance.get("files_sha256", {}).get("clean_parquet"),
        "data/indian_delivery_gbdt.parquet": provenance.get("files_sha256", {}).get("gbdt_parquet"),
        "data/indian_delivery_vqc.parquet": provenance.get("files_sha256", {}).get("vqc_parquet"),
        "data/feature_schema.json": provenance.get("schema_sha256"),
        "data/preprocessing_artifacts.json": provenance.get("preprocessing_artifacts_sha256"),
    }
    data_by_path = {Path(item["path"]).relative_to(repo).as_posix(): item for item in data_files}
    for relative, expected in provenance_hashes.items():
        if expected is None:
            problems.append(f"trusted recovery manifest lacks the source digest for {relative}")
            continue
        observed = data_by_path[relative].get("sha256")
        if observed != expected:
            problems.append(f"recovered-source digest mismatch for {relative}: expected {expected}, observed {observed}")
    source_meta = provenance.get("source_csv", {})
    if source_check is not None:
        if source_check.get("raw_sha256") != source_meta.get("worktree_sha256_crlf"):
            problems.append("raw source CSV SHA-256 differs from the recovery manifest's preserved worktree-byte hash")
        if source_check.get("canonical_lf_sha256") != source_meta.get("canonical_git_sha256_lf"):
            problems.append("canonical source CSV SHA-256 differs from the recovery manifest")

    gbdt_features = schema.get("gbdt_view", {}).get("features") if schema else None
    if not isinstance(gbdt_features, list) or len(gbdt_features) != 59 or len(set(gbdt_features)) != 59:
        problems.append("feature schema must contain exactly 59 unique ordered GBDT features")
    elif {"ID", "Delivery_person_ID", "split_partition", TARGET_NAME} & set(gbdt_features):
        problems.append("GBDT feature schema contains an identity, split, or target leakage column")

    canonical_model = recovery.get("artifacts", {}).get("model_artifacts", {}).get("model")
    canonical_predictions = recovery.get("artifacts", {}).get("prediction_artifacts", {})
    if args.checkpoint:
        supplied = args.checkpoint.resolve()
        if canonical_model is None or supplied != Path(canonical_model["path"]).resolve():
            problems.append("explicit --checkpoint does not exactly match the verified local recovery manifest")
        elif not canonical_model["matches"]:
            problems.append("explicit --checkpoint failed the recovery manifest hash check")
    for supplied_arg in args.predictions:
        supplied = supplied_arg.resolve()
        matches = [entry for entry in (canonical_predictions.get("oof"), canonical_predictions.get("validation"))
                   if entry is not None and supplied == Path(entry["path"]).resolve()]
        if not matches:
            problems.append(f"explicit --predictions path is not a manifest-verified OOF or validation artifact: {supplied}")
        elif not matches[0]["matches"]:
            problems.append(f"explicit --predictions artifact failed its manifest hash check: {supplied}")

    expected_rows = (29_592, 5_989)
    observed_rows = (recovery.get("oof_rows"), recovery.get("validation_rows"))
    if verified and observed_rows != expected_rows:
        problems.append(f"recovery manifest prediction row counts are unexpected: {observed_rows}")

    search_roots = [path.resolve() for path in args.search_root]
    candidates = discover_candidates(search_roots)
    report = {
        "status": "CLASSICAL_READY" if not problems else "BLOCKED",
        "repo": str(repo), "target": TARGET_NAME, "target_unit": "minutes",
        "data_files": data_files, "integrity_checks": integrity_checks,
        "source_identity": source_check, "split_manifest": split,
        "schema_target": schema.get("target") if schema else None,
        "gbdt_feature_count": len(gbdt_features) if isinstance(gbdt_features, list) else None,
        "recovery_manifest": recovery,
        "checkpoint": canonical_model,
        "oof_predictions": canonical_predictions.get("oof"),
        "validation_predictions": canonical_predictions.get("validation"),
        "candidate_artifacts_for_manual_review": candidates,
        "blockers": problems,
        "quantum_training_authorized": False,
        "quantum_status": "pending_external_Opus_architecture_review",
        "guardrail": "No estimator is deserialized; only manifest paths and bytes are hashed. Test rows are not opened. CLASSICAL_READY is not quantum-training authorization.",
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"India ETA preflight: {report['status']}")
        print(f"Repository: {repo}")
        print(f"Target: {TARGET_NAME} (minutes), features: {report['gbdt_feature_count']}")
        print(f"Recovery manifest verified: {verified}")
        print("Quantum training authorized: false (pending external Opus architecture review)")
        print(f"Manual candidate artifacts found: {len(candidates)}")
        for blocker in problems:
            print(f"BLOCKER: {blocker}")
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
