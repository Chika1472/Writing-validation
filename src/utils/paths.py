from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def require_distinct_paths(**named_paths: Any) -> None:
    """Reject accidental input/output aliases, including case aliases on Windows."""

    seen: dict[str, tuple[str, Path]] = {}
    for name, value in named_paths.items():
        if value is None:
            continue
        path = Path(value).resolve()
        key = os.path.normcase(str(path))
        if key in seen:
            previous_name, previous_path = seen[key]
            raise ValueError(
                f"paths must be distinct: {previous_name} and {name} both resolve to "
                f"{previous_path}"
            )
        seen[key] = (name, path)


def require_new_paths(**named_paths: Any) -> None:
    """Require artifact targets to be absent so reruns cannot overwrite evidence."""

    existing = [
        (name, Path(value).resolve())
        for name, value in named_paths.items()
        if value is not None and Path(value).resolve().exists()
    ]
    if existing:
        rendered = ", ".join(f"{name}={path}" for name, path in existing)
        raise FileExistsError(f"artifact output path already exists: {rendered}")


def require_outside_roots(
    named_roots: Mapping[str, Any],
    **named_paths: Any,
) -> None:
    """Reject outputs located anywhere inside immutable artifact/input trees."""

    roots = {
        name: Path(value).resolve()
        for name, value in named_roots.items()
        if value is not None
    }
    for path_name, value in named_paths.items():
        if value is None:
            continue
        path = Path(value).resolve()
        for root_name, root in roots.items():
            if path == root or path.is_relative_to(root):
                raise ValueError(
                    f"{path_name} must not be inside {root_name}: {path} is under {root}"
                )
