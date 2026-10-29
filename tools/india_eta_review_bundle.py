"""Package the uncommitted ETA implementation without staging or committing."""
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/india_eta"


def main():
    tracked = subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).splitlines()
    if set(tracked) - {"backend/api.py", "backend/auth.py"}:
        raise RuntimeError("Unexpected preexisting file changes; review ownership before packaging")
    protected = json.loads((OUT / "protected-file-check.json").read_text(encoding="utf-8-sig"))
    if protected["mismatches"]:
        raise RuntimeError("Protected-file verification did not pass")
    roots = ["backend/india_eta", "tests/india_eta", "fixtures/india_eta", "evidence/india_eta"]
    new = []
    excluded = {"implementation.patch", "review-manifest.json", "patch-check.txt"}
    for folder in roots:
        for file in (ROOT / folder).rglob("*"):
            if (file.is_file() and "__pycache__" not in file.parts and file.name not in excluded
                    and file.suffix in {".py", ".md", ".json", ".txt"}):
                new.append(file.relative_to(ROOT).as_posix())
    new += ["INDIA_ETA_INTEGRATION_REPORT.md", "tools/india_eta_http_smoke.py", "tools/india_eta_review_bundle.py"]
    new += ["docs/" + name for name in ("INDIA-ETA-CONTRACTS.md", "INDIA-ETA-PREDICTION.md", "INDIA-ETA-OPTIMIZATION.md", "INDIA-ETA-CONTROLLER.md")]
    new = sorted(set(new))
    patch = subprocess.check_output(["git", "diff", "--binary", "--", *tracked], cwd=ROOT, text=True)
    for rel in new:
        lines = (ROOT / rel).read_text(encoding="utf-8").splitlines(keepends=True)
        patch += f"diff --git a/{rel} b/{rel}\nnew file mode 100644\n"
        patch += "".join(difflib.unified_diff([], lines, fromfile="/dev/null", tofile="b/" + rel))
    patch_path = OUT / "implementation.patch"
    patch_path.write_text(patch, encoding="utf-8", newline="\n")
    records = []
    for rel in sorted(tracked + new):
        content = (ROOT / rel).read_bytes()
        records.append({"path": rel, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"schema_version": "india_eta.review_bundle.v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                "uncommitted": True, "tracked_changed": tracked, "new_files": new, "files": records,
                "patch_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
                "patch_bytes": patch_path.stat().st_size,
                "protected_file_count": protected["file_count"], "protected_hash_mismatches": []}
    (OUT / "review-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checked = subprocess.run(["git", "apply", "--reverse", "--check", str(patch_path)], cwd=ROOT, capture_output=True, text=True)
    (OUT / "patch-check.txt").write_text(f"git apply --reverse --check evidence/india_eta/implementation.patch\nexit_code={checked.returncode}\n" + checked.stdout + checked.stderr)
    if checked.returncode:
        raise RuntimeError("Review patch does not match the current worktree; inspect patch-check.txt")
    print(json.dumps({"files": len(records), "patch_bytes": manifest["patch_bytes"], "patch_reverse_check": "passed", "protected_files_unchanged": protected["file_count"]}))


if __name__ == "__main__":
    main()
