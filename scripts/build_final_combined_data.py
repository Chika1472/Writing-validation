from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.final_combined import (
    FinalCombinedDataError,
    build_final_combined_dataset,
    require_validation_label_training_acknowledgement,
)
from src.utils.config import load_yaml, resolve_project_path
from src.utils.hashing import sha256_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an immutable train+validation labeled dataset for final retraining. "
            "This is forbidden unless competition rules explicitly allow validation "
            "labels to be used for training."
        )
    )
    parser.add_argument("--config", default="configs/data_final_combined.yaml")
    parser.add_argument(
        "--acknowledge-rules-allow-validation-label-training",
        action="store_true",
        help=(
            "Confirm that the competition rules were checked and explicitly permit "
            "using validation labels for final model training."
        ),
    )
    return parser.parse_args(argv)


def _required_mapping(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FinalCombinedDataError(f"{where} must be a mapping")
    return value


def _required_path(mapping: dict[str, Any], key: str, *, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FinalCombinedDataError(f"{where}.{key} must be a non-empty path string")
    return value


def _require_artifact_destination(
    path: Path,
    *,
    artifact_root: Path,
    where: str,
) -> None:
    if path == artifact_root:
        raise FinalCombinedDataError(f"{where} must be a file below paths.artifacts")
    try:
        path.relative_to(artifact_root)
    except ValueError as error:
        raise FinalCombinedDataError(
            f"{where} must stay below paths.artifacts: {path} is outside {artifact_root}"
        ) from error


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # This check intentionally precedes even reading the configuration file.
    require_validation_label_training_acknowledgement(
        args.acknowledge_rules_allow_validation_label_training
    )

    config_path = Path(args.config).resolve()
    config_sha256 = sha256_file(config_path)
    config = load_yaml(config_path)
    if sha256_file(config_path) != config_sha256:
        raise FinalCombinedDataError(
            f"configuration changed while it was being read: {config_path}"
        )
    paths = _required_mapping(config.get("paths"), where="paths")
    final = _required_mapping(config.get("final_combined"), where="final_combined")
    train_source = resolve_project_path(
        config,
        _required_path(final, "train_source", where="final_combined"),
    )
    validation_source = resolve_project_path(
        config,
        _required_path(final, "validation_source", where="final_combined"),
    )
    output = resolve_project_path(
        config,
        _required_path(paths, "train", where="paths"),
    )
    artifact_root = resolve_project_path(
        config,
        _required_path(paths, "artifacts", where="paths"),
    )
    manifest = resolve_project_path(
        config,
        _required_path(final, "manifest", where="final_combined"),
    )
    _require_artifact_destination(
        output,
        artifact_root=artifact_root,
        where="paths.train",
    )
    _require_artifact_destination(
        manifest,
        artifact_root=artifact_root,
        where="final_combined.manifest",
    )
    payload = build_final_combined_dataset(
        train_source=train_source,
        validation_source=validation_source,
        output_path=output,
        manifest_path=manifest,
        project_root=config["_project_root"],
        config_path=config["_config_path"],
        validation_label_training_acknowledged=True,
        expected_config_sha256=config_sha256,
    )
    print(
        json.dumps(
            {
                "combined": payload["combined"]["path"],
                "manifest": str(manifest),
                "record_count": payload["combined"]["record_count"],
                "sha256": payload["combined"]["sha256"],
                "cohort_counts": payload["combined"]["cohort_counts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
