from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML and attach resolved config/project paths.

    ``project_root`` is resolved relative to the config file, not the caller's
    current directory. This keeps every CLI stable when launched elsewhere.
    """

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")

    root_value = data.get("project_root", ".")
    project_root = (config_path.parent / str(root_value)).resolve()
    data["_config_path"] = config_path
    data["_project_root"] = project_root
    return data


def resolve_project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root = config.get("_project_root")
    if not isinstance(root, Path):
        raise ValueError("configuration was not loaded with load_yaml")
    return (root / path).resolve()

