from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.registry import (
    load_run_registry,
    validate_registry_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate immutable experiment inputs, every run manifest/checkpoint, and "
            "registry status consistency without loading a model."
        )
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail when any planned task artifact is absent or incomplete.",
    )
    parser.add_argument("--output", default=None, help="Optional new JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry_path = Path(args.registry).resolve()
    registry = load_run_registry(registry_path)
    report = {
        "registry": str(registry_path),
        **validate_registry_artifacts(
            registry,
            require_complete=args.require_complete,
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output).resolve()
        if output.exists():
            raise FileExistsError(f"validation report already exists: {output}")
        if output == registry_path:
            raise ValueError("validation report must not overwrite the registry")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    print(rendered, end="")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
