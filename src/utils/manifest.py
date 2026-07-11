from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.utils.hashing import sha256_file, sha256_json


TRACKED_PACKAGES = (
    "accelerate",
    "bitsandbytes",
    "joblib",
    "llama-cpp-python",
    "numpy",
    "pandas",
    "peft",
    "PyYAML",
    "safetensors",
    "scipy",
    "scikit-learn",
    "torch",
    "transformers",
)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_value(args: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def build_manifest(
    *,
    run_id: str,
    project_root: str | Path,
    config: dict[str, Any],
    input_files: Iterable[str | Path] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    inputs = {str(Path(p).resolve()): sha256_file(p) for p in input_files}
    status = _git_value(["status", "--short"], root)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "git_commit": _git_value(["rev-parse", "HEAD"], root),
        "dirty_worktree": bool(status) if status is not None else None,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "config": config,
        "config_sha256": sha256_json(config),
        "inputs": inputs,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(
    path: str | Path,
    manifest: dict[str, Any],
    *,
    create_only: bool = False,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        default=str,
        allow_nan=False,
    ) + "\n"
    if create_only:
        created = False
        try:
            handle = target.open("x", encoding="utf-8", newline="\n")
            created = True
            with handle:
                handle.write(rendered)
        except Exception:
            if created:
                target.unlink(missing_ok=True)
            raise
        return
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(target)
