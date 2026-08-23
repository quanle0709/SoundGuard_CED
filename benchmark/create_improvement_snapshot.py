"""Create a recoverable pre-improvement snapshot without touching user work."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "improvement"

EXCLUDED_PREFIXES = (
    ".git/", ".venv/", "benchmark_data/external/",
    "benchmark_data/expanded_generated/", "benchmark_data/generated_audio/",
    "benchmark_results/", "__pycache__/",
)
EXCLUDED_FILES = {"benchmark/create_improvement_snapshot.py"}
PRODUCTION_FILES = (
    "app.py", "audio_pipeline.py", "display_transport.py", "emergency_system.py",
    "fusion_engine.py", "sound_classifier.py", "speech_enhancer.py",
    "speech_recognizer.py", "personalization/profile_generator.py",
    "personalization/profile_validator.py", "personalization/profile_manager.py",
    "personalization/sound_labels.py", "personalization/role_knowledge.py",
    "personalization/schemas.py", "personalization/web_server.py",
)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def relevant(relative: str) -> bool:
    value = relative.replace("\\", "/")
    return value not in EXCLUDED_FILES and not any(value.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    status = git("status", "--porcelain=v1", "-uall")
    branch = git("branch", "--show-current").strip()
    commit = git("rev-parse", "HEAD").strip()
    subprocess.run(["git", "diff", "--binary", f"--output={OUT / 'baseline_unstaged.patch'}"], cwd=ROOT, check=True)
    subprocess.run(["git", "diff", "--cached", "--binary", f"--output={OUT / 'baseline_staged.patch'}"], cwd=ROOT, check=True)

    tracked = set(git("ls-files").splitlines())
    untracked = set(git("ls-files", "--others", "--exclude-standard").splitlines())
    paths = sorted(path for path in tracked | untracked if relevant(path) and (ROOT / path).is_file())
    file_rows = [{"path": path, "sha256": digest(ROOT / path), "tracked": path in tracked,
                  "size_bytes": (ROOT / path).stat().st_size} for path in paths]
    production = {path: {"sha256": digest(ROOT / path), "size_bytes": (ROOT / path).stat().st_size}
                  for path in PRODUCTION_FILES if (ROOT / path).exists()}

    archive = OUT / "baseline_source_snapshot.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in paths:
            bundle.write(ROOT / path, f"workspace/{path}")
        bundle.writestr("snapshot_file_manifest.json", json.dumps(file_rows, indent=2, ensure_ascii=False))

    (OUT / "baseline_production_hashes.json").write_text(json.dumps({
        "git_commit": commit, "branch": branch, "production_files": production,
        "archive": archive.name, "archive_sha256": digest(archive),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "baseline_snapshot_file_manifest.json").write_text(
        json.dumps(file_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    untracked_relevant = [path for path in paths if path in untracked]
    staged_size = (OUT / "baseline_staged.patch").stat().st_size
    unstaged_size = (OUT / "baseline_unstaged.patch").stat().st_size
    report = f"""# SoundGuard pre-improvement worktree snapshot

- Baseline branch: `{branch}`
- Baseline commit: `{commit}`
- Worktree was dirty: `{bool(status.strip())}`
- Unstaged binary patch: `baseline_unstaged.patch` ({unstaged_size} bytes)
- Staged binary patch: `baseline_staged.patch` ({staged_size} bytes)
- Recoverable source snapshot: `baseline_source_snapshot.zip`
- Snapshot files: {len(file_rows)}
- Relevant untracked files captured: {len(untracked_relevant)}
- Production hashes: `baseline_production_hashes.json`
- Complete source manifest: `baseline_snapshot_file_manifest.json`

No commit or tag was created because the dirty worktree contains pre-existing user work that should not be bundled into an artificial baseline commit. The archive contains the exact relevant tracked and untracked source contents as they existed before production improvement work; large downloaded/generated benchmark data and result trees are intentionally excluded and remain preserved in place.

## Git status at freeze

```text
{status.rstrip()}
```

## Relevant untracked files captured

```text
{chr(10).join(untracked_relevant)}
```
"""
    (OUT / "baseline_worktree_snapshot.md").write_text(report, encoding="utf-8")
    print(f"snapshot_files={len(file_rows)} untracked={len(untracked_relevant)} archive_sha256={digest(archive)}")


if __name__ == "__main__":
    main()
