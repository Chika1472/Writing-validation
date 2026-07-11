from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.deployment import (
    PACKAGE_SCHEMA_VERSION,
    is_runtime_cache_file,
    load_deployment_config,
    locked_requirements,
    validate_artifact_contracts,
    validate_package_manifest,
)
from src.utils.hashing import sha256_file, sha256_json
from src.utils.paths import require_distinct_paths, require_new_paths, require_outside_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a new, hash-signed, offline submission directory. The command "
            "never edits source artifacts or an existing package."
        )
    )
    parser.add_argument("--config", default="configs/inference_l40s.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--requirements-lock",
        required=True,
        help="Exact, tested dependency lock copied as requirements-lock.txt.",
    )
    parser.add_argument(
        "--license-file",
        action="append",
        required=True,
        help="Repeat for model/dataset/code licenses and third-party notices.",
    )
    parser.add_argument(
        "--hf-home",
        required=True,
        help=(
            "Hugging Face cache root containing the pinned base revision. Cache "
            "symlinks are materialized into ordinary package files."
        ),
    )
    return parser.parse_args()


def _copy_file(source: Path, target: Path) -> None:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(f"package path collision: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _copy_group(files: Iterable[Path], source_root: Path, target_root: Path) -> None:
    source_root = source_root.resolve()
    for source in files:
        source = source.resolve()
        try:
            relative = source.relative_to(source_root)
        except ValueError as error:
            raise ValueError(f"artifact escapes expected root {source_root}: {source}") from error
        _copy_file(source, target_root / relative)


def _materialize_cache(source_root: Path, target_root: Path) -> int:
    source_root = source_root.resolve()
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError(f"--hf-home must be a real directory: {source_root}")
    copied = 0
    for entry in sorted(source_root.rglob("*")):
        if entry.is_dir() and not entry.is_symlink():
            continue
        resolved = entry.resolve()
        try:
            resolved.relative_to(source_root)
        except ValueError as error:
            raise ValueError(f"HF cache link escapes --hf-home: {entry}") from error
        if not resolved.is_file():
            raise ValueError(f"HF cache contains a non-file link/special entry: {entry}")
        relative = entry.relative_to(source_root)
        _copy_file(resolved, target_root / relative)
        copied += 1
    if copied == 0:
        raise ValueError("--hf-home contains no files")
    return copied


def _require_model_snapshot(hf_home: Path, model_id: str, revision: str) -> Path:
    parts = model_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("packaging currently requires a canonical Hugging Face org/model id")
    snapshot = (
        hf_home
        / "hub"
        / f"models--{parts[0]}--{parts[1]}"
        / "snapshots"
        / revision
    )
    if not snapshot.is_dir() or not any(path.is_file() for path in snapshot.rglob("*")):
        raise FileNotFoundError(
            f"pinned base snapshot is absent from --hf-home: {snapshot}"
        )
    return snapshot


def _source_files(project_root: Path) -> tuple[Path, ...]:
    required = (
        project_root / "pyproject.toml",
        project_root / "docs" / "SUBMISSION_RUNTIME.md",
        project_root / "scripts" / "run_submission.py",
        project_root / "scripts" / "smoke_test_submission.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"submission source closure is incomplete: {missing}")
    files = list(required)
    files.extend(path for path in (project_root / "src").rglob("*.py") if path.is_file())
    return tuple(dict.fromkeys(path.resolve() for path in files))


def _public_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("deployment config must contain a mapping")
    return value


def _all_payload_hashes(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    package_manifest = (root / "package.manifest.json").resolve()
    for path in sorted(root.rglob("*")):
        if path.resolve() == package_manifest:
            continue
        if path.is_symlink():
            raise ValueError(f"package payload must not contain symlinks: {path}")
        if path.is_file() and is_runtime_cache_file(path.relative_to(root)):
            continue
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha256_file(path)
    if not files:
        raise RuntimeError("submission package has no payload files")
    return files


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    lock_path = Path(args.requirements_lock).resolve()
    license_paths = tuple(Path(value).resolve() for value in args.license_file)
    hf_home = Path(args.hf_home).resolve()
    if len(set(license_paths)) != len(license_paths):
        raise ValueError("--license-file contains duplicates")
    if output_dir.parent == output_dir or not output_dir.name:
        raise ValueError("--output-dir must be a named child directory")
    staging = output_dir.with_name(output_dir.name + ".staging")
    require_distinct_paths(
        config=config_path,
        output=output_dir,
        staging=staging,
        requirements_lock=lock_path,
        hf_home=hf_home,
        **{f"license_{index}": path for index, path in enumerate(license_paths)},
    )
    require_new_paths(output=output_dir, staging=staging)
    config = load_deployment_config(config_path)
    validate_package_manifest(config)
    artifacts = validate_artifact_contracts(config)
    locked_requirements(lock_path)
    assert config.qwen.model_id is not None and config.qwen.model_revision is not None
    pinned_snapshot = _require_model_snapshot(
        hf_home, config.qwen.model_id, config.qwen.model_revision
    )
    source_files = _source_files(project_root)
    for path in (lock_path, *license_paths):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"package metadata must be a real file: {path}")
    require_outside_roots(
        {
            "config": config_path,
            "hf_home": hf_home,
            **{
                f"artifact_root_{index}": path
                for index, path in enumerate(artifacts.immutable_roots)
            },
        },
        output=output_dir,
        staging=staging,
    )

    staging.mkdir(parents=True, exist_ok=False)
    try:
        for source in source_files:
            _copy_file(source, staging / source.relative_to(project_root))
        _copy_file(lock_path, staging / "requirements-lock.txt")
        license_names: set[str] = set()
        for source in license_paths:
            if source.name in license_names:
                raise ValueError(f"duplicate license filename: {source.name}")
            license_names.add(source.name)
            _copy_file(source, staging / "licenses" / source.name)

        scorer_targets = []
        for index, (checkpoint, files) in enumerate(
            zip(artifacts.scorer_checkpoints, artifacts.scorer_files, strict=True)
        ):
            target = staging / "artifacts" / "scorer" / f"checkpoint_{index:02d}"
            _copy_group(files, checkpoint, target)
            scorer_targets.append(target.relative_to(staging).as_posix())

        calibrator_target = None
        if config.qwen.calibrator is not None:
            target_root = staging / "artifacts" / "scorer_calibrator"
            for source in artifacts.calibrator_files:
                _copy_file(source, target_root / source.name)
            calibrator_target = (
                target_root / config.qwen.calibrator.name
            ).relative_to(staging).as_posix()

        baseline_target = None
        if config.baseline_artifact is not None:
            target_root = staging / "artifacts" / "baseline"
            for source in artifacts.baseline_files:
                _copy_file(source, target_root / source.name)
            baseline_target = (
                target_root / config.baseline_artifact.name
            ).relative_to(staging).as_posix()

        anchor_target = None
        if config.anchor_artifact is not None:
            target_root = staging / "artifacts" / "anchor"
            for source in artifacts.anchor_files:
                _copy_file(source, target_root / source.name)
            anchor_target = (
                target_root / config.anchor_artifact.name
            ).relative_to(staging).as_posix()

        assessment_target = None
        if config.assessment_artifact is not None:
            target_root = staging / "artifacts" / "assessment"
            for source in artifacts.assessment_files:
                _copy_file(source, target_root / source.name)
            assessment_target = (
                target_root / config.assessment_artifact.name
            ).relative_to(staging).as_posix()

        stacker_target = None
        if config.stacker is not None:
            target_root = staging / "artifacts" / "stacker"
            for source in artifacts.stacker_files:
                _copy_file(source, target_root / source.name)
            stacker_target = (
                target_root / config.stacker.artifact.name
            ).relative_to(staging).as_posix()

        rationale_target = None
        if artifacts.rationale_checkpoint is not None:
            target_root = staging / "artifacts" / "rationale" / "checkpoint"
            _copy_group(
                artifacts.rationale_files,
                artifacts.rationale_checkpoint,
                target_root,
            )
            rationale_target = target_root.relative_to(staging).as_posix()

        model_repository = pinned_snapshot.parents[1]
        cache_file_count = _materialize_cache(
            model_repository,
            staging / "model_cache" / "hub" / model_repository.name,
        )
        hf_version = hf_home / "version.txt"
        if hf_version.is_file():
            _copy_file(hf_version, staging / "model_cache" / "version.txt")
            cache_file_count += 1
        packaged_config = _public_yaml(config_path)
        packaged_config["project_root"] = "."
        packaged_config["runtime"]["hf_home"] = "model_cache"
        packaged_config["runtime"]["package_manifest"] = "package.manifest.json"
        packaged_config["runtime"]["requirements_lock"] = "requirements-lock.txt"
        packaged_config["scoring"]["qwen"]["checkpoints"] = scorer_targets
        packaged_config["scoring"]["qwen"]["calibrator"] = calibrator_target
        packaged_config["scoring"]["baseline"]["artifact"] = baseline_target
        packaged_config["scoring"]["anchor"]["artifact"] = anchor_target
        packaged_config["scoring"]["assessment"]["artifact"] = assessment_target
        packaged_config["scoring"]["stacker"]["artifact"] = stacker_target
        packaged_config["rationale"]["checkpoint"] = rationale_target
        packaged_config_path = staging / "inference_l40s.yaml"
        packaged_config_path.write_text(
            yaml.safe_dump(packaged_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        file_hashes = _all_payload_hashes(staging)
        signature_payload = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "config_file": packaged_config_path.name,
            "config_sha256": sha256_file(packaged_config_path),
            "files": file_hashes,
        }
        manifest = {
            "artifact_type": "submission_package_v1",
            **signature_payload,
            "package_signature": sha256_json(signature_payload),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_config_sha256": sha256_file(config_path),
            "source_artifact_hashes": {
                str(path): sha256_file(path) for path in artifacts.all_files
            },
            "requirements_lock_source_sha256": sha256_file(lock_path),
            "license_source_hashes": {
                path.name: sha256_file(path) for path in license_paths
            },
            "hf_home_source": str(hf_home),
            "pinned_base_snapshot_source": str(pinned_snapshot),
            "packaged_model_repository_source": str(model_repository),
            "hf_cache_file_count": cache_file_count,
        }
        (staging / "package.manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except Exception:
        if staging.exists():
            resolved = staging.resolve()
            if resolved.parent != output_dir.parent or resolved.name != output_dir.name + ".staging":
                raise RuntimeError("refusing to clean an unexpected staging path")
            shutil.rmtree(resolved)
        raise

    print(
        json.dumps(
            {
                "package": str(output_dir),
                "manifest": str(output_dir / "package.manifest.json"),
                "files": len(file_hashes),
                "hf_cache_files": cache_file_count,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
