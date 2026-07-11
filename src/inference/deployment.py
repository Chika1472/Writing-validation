"""Pure-Python deployment configuration and artifact-contract validation.

This module intentionally does not import torch, transformers, PEFT, or joblib.
Packaging and preflight validation can therefore fail before allocating GPU memory.
"""

from __future__ import annotations

import json
import importlib.metadata
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.baselines.contracts import baseline_inference_code_contract
from src.calibration.contracts import calibration_inference_code_contract
from src.evaluation.oof_provenance import checkpoint_ensemble_signature
from src.ensemble.contracts import stacker_inference_code_contract
from src.models.contracts import SCORER_ARCHITECTURE_VERSION
from src.utils.hashing import sha256_file, sha256_json


SCHEMA_VERSION = "submission_inference_v2"
PACKAGE_SCHEMA_VERSION = 1
TRAITS = ("content", "organization", "expression")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_REVISION = re.compile(r"[0-9a-fA-F]{40}")


@dataclass(frozen=True)
class RuntimeConfig:
    offline_required: bool
    hf_home: Path | None
    package_manifest: Path | None
    requirements_lock: Path | None
    device: str
    require_cuda: bool
    require_single_visible_gpu: bool
    required_gpu_name_contains: str | None
    expected_rows: int
    batch_size: int
    scorer_precision: str
    rationale_precision: str
    seed: int
    deterministic_algorithms: bool


@dataclass(frozen=True)
class QwenConfig:
    scorer_name: str
    checkpoints: tuple[Path, ...]
    model_id: str | None
    model_revision: str | None
    calibrator: Path | None


@dataclass(frozen=True)
class StackerConfig:
    artifact: Path
    source_aliases: Mapping[str, str]


@dataclass(frozen=True)
class RationaleConfig:
    mode: str
    checkpoint: Path | None
    max_attempts: int


@dataclass(frozen=True)
class DeploymentConfig:
    path: Path
    project_root: Path
    runtime: RuntimeConfig
    scoring_mode: str
    qwen: QwenConfig
    baseline_artifact: Path | None
    anchor_artifact: Path | None
    assessment_artifact: Path | None
    stacker: StackerConfig | None
    rationale: RationaleConfig
    output_model_name: str


@dataclass(frozen=True)
class ValidatedArtifacts:
    scorer_checkpoints: tuple[Path, ...]
    scorer_files: tuple[tuple[Path, ...], ...]
    scorer_signature: str
    calibrator_files: tuple[Path, ...]
    baseline_files: tuple[Path, ...]
    baseline_scorer_name: str | None
    baseline_scorer_signature: str | None
    anchor_files: tuple[Path, ...]
    anchor_scorer_name: str | None
    anchor_scorer_signature: str | None
    assessment_files: tuple[Path, ...]
    assessment_scorer_name: str | None
    assessment_scorer_signature: str | None
    stacker_files: tuple[Path, ...]
    stacker_signature: str | None
    rationale_checkpoint: Path | None
    rationale_files: tuple[Path, ...]

    @property
    def all_files(self) -> tuple[Path, ...]:
        values: list[Path] = []
        for group in self.scorer_files:
            values.extend(group)
        values.extend(self.calibrator_files)
        values.extend(self.baseline_files)
        values.extend(self.anchor_files)
        values.extend(self.assessment_files)
        values.extend(self.stacker_files)
        values.extend(self.rationale_files)
        return tuple(dict.fromkeys(path.resolve() for path in values))

    @property
    def immutable_roots(self) -> tuple[Path, ...]:
        """Directories that output/report/package writes must never enter."""

        values: list[Path] = list(self.scorer_checkpoints)
        if self.calibrator_files:
            values.append(self.calibrator_files[0].parent)
        if self.baseline_files:
            values.append(self.baseline_files[0].parent)
        if self.anchor_files:
            values.append(self.anchor_files[0].parent)
        if self.assessment_files:
            values.append(self.assessment_files[0].parent)
        if self.stacker_files:
            values.append(self.stacker_files[0].parent)
        if self.rationale_checkpoint is not None:
            values.append(self.rationale_checkpoint)
        return tuple(dict.fromkeys(path.resolve() for path in values))


def _mapping(value: Any, *, where: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{where} keys must be exactly {sorted(keys)}; "
            f"missing={sorted(keys - actual)}, extra={sorted(actual - keys)}"
        )
    return value


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be nonempty text")
    return value.strip()


def _nullable_text(value: Any, *, where: str) -> str | None:
    if value is None:
        return None
    return _text(value, where=where)


def _bool(value: Any, *, where: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{where} must be boolean")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _path(root: Path, value: Any, *, where: str) -> Path:
    text = _text(value, where=where)
    if "PLACEHOLDER" in text.upper():
        raise ValueError(f"{where} still contains a PLACEHOLDER value")
    candidate = Path(text)
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _nullable_path(root: Path, value: Any, *, where: str) -> Path | None:
    if value is None:
        return None
    return _path(root, value, where=where)


def load_deployment_config(path: str | Path) -> DeploymentConfig:
    """Load an exact deployment schema and resolve every artifact path."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    top = _mapping(
        raw,
        where="deployment config",
        keys={"schema_version", "project_root", "runtime", "scoring", "rationale", "output"},
    )
    if top["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    root_value = _text(top["project_root"], where="project_root")
    project_root = (config_path.parent / root_value).resolve()

    runtime_raw = _mapping(
        top["runtime"],
        where="runtime",
        keys={
            "offline_required",
            "hf_home",
            "package_manifest",
            "requirements_lock",
            "device",
            "require_cuda",
            "require_single_visible_gpu",
            "required_gpu_name_contains",
            "expected_rows",
            "batch_size",
            "scorer_precision",
            "rationale_precision",
            "seed",
            "deterministic_algorithms",
        },
    )
    offline_required = _bool(runtime_raw["offline_required"], where="runtime.offline_required")
    if not offline_required:
        raise ValueError("deployment requires runtime.offline_required=true")
    scorer_precision = _text(runtime_raw["scorer_precision"], where="runtime.scorer_precision")
    rationale_precision = _text(
        runtime_raw["rationale_precision"], where="runtime.rationale_precision"
    )
    if scorer_precision not in {"4bit", "bf16"}:
        raise ValueError("runtime.scorer_precision must be '4bit' or 'bf16'")
    if rationale_precision not in {"4bit", "bf16"}:
        raise ValueError("runtime.rationale_precision must be '4bit' or 'bf16'")
    seed = runtime_raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("runtime.seed must be a nonnegative integer")
    gpu_name = _nullable_text(
        runtime_raw["required_gpu_name_contains"],
        where="runtime.required_gpu_name_contains",
    )
    runtime = RuntimeConfig(
        offline_required=offline_required,
        hf_home=_nullable_path(project_root, runtime_raw["hf_home"], where="runtime.hf_home"),
        package_manifest=_nullable_path(
            project_root,
            runtime_raw["package_manifest"],
            where="runtime.package_manifest",
        ),
        requirements_lock=_nullable_path(
            project_root,
            runtime_raw["requirements_lock"],
            where="runtime.requirements_lock",
        ),
        device=_text(runtime_raw["device"], where="runtime.device"),
        require_cuda=_bool(runtime_raw["require_cuda"], where="runtime.require_cuda"),
        require_single_visible_gpu=_bool(
            runtime_raw["require_single_visible_gpu"],
            where="runtime.require_single_visible_gpu",
        ),
        required_gpu_name_contains=gpu_name,
        expected_rows=_positive_int(runtime_raw["expected_rows"], where="runtime.expected_rows"),
        batch_size=_positive_int(runtime_raw["batch_size"], where="runtime.batch_size"),
        scorer_precision=scorer_precision,
        rationale_precision=rationale_precision,
        seed=seed,
        deterministic_algorithms=_bool(
            runtime_raw["deterministic_algorithms"],
            where="runtime.deterministic_algorithms",
        ),
    )

    scoring_raw = _mapping(
        top["scoring"],
        where="scoring",
        keys={"mode", "qwen", "baseline", "anchor", "assessment", "stacker"},
    )
    scoring_mode = _text(scoring_raw["mode"], where="scoring.mode")
    if scoring_mode not in {"qwen_ensemble", "qwen_multisource_stacker"}:
        raise ValueError(
            "scoring.mode must be qwen_ensemble or qwen_multisource_stacker"
        )
    qwen_raw = _mapping(
        scoring_raw["qwen"],
        where="scoring.qwen",
        keys={"scorer_name", "checkpoints", "model_id", "model_revision", "calibrator"},
    )
    checkpoints_raw = qwen_raw["checkpoints"]
    if not isinstance(checkpoints_raw, list) or not checkpoints_raw:
        raise ValueError("scoring.qwen.checkpoints must be a nonempty list")
    checkpoints = tuple(
        _path(project_root, value, where=f"scoring.qwen.checkpoints[{index}]")
        for index, value in enumerate(checkpoints_raw)
    )
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("scoring.qwen.checkpoints contains duplicates")
    model_revision = _text(
        qwen_raw["model_revision"], where="scoring.qwen.model_revision"
    )
    if not _REVISION.fullmatch(model_revision):
        raise ValueError("scoring.qwen.model_revision must be a pinned 40-character commit SHA")
    qwen = QwenConfig(
        scorer_name=_text(qwen_raw["scorer_name"], where="scoring.qwen.scorer_name"),
        checkpoints=checkpoints,
        model_id=_text(qwen_raw["model_id"], where="scoring.qwen.model_id"),
        model_revision=model_revision,
        calibrator=_nullable_path(
            project_root, qwen_raw["calibrator"], where="scoring.qwen.calibrator"
        ),
    )

    baseline_raw = _mapping(
        scoring_raw["baseline"], where="scoring.baseline", keys={"artifact"}
    )
    baseline_artifact = _nullable_path(
        project_root, baseline_raw["artifact"], where="scoring.baseline.artifact"
    )
    anchor_raw = _mapping(
        scoring_raw["anchor"], where="scoring.anchor", keys={"artifact"}
    )
    anchor_artifact = _nullable_path(
        project_root, anchor_raw["artifact"], where="scoring.anchor.artifact"
    )
    assessment_raw = _mapping(
        scoring_raw["assessment"], where="scoring.assessment", keys={"artifact"}
    )
    assessment_artifact = _nullable_path(
        project_root,
        assessment_raw["artifact"],
        where="scoring.assessment.artifact",
    )
    stacker_raw = _mapping(
        scoring_raw["stacker"],
        where="scoring.stacker",
        keys={"artifact", "source_aliases"},
    )
    stacker_artifact = _nullable_path(
        project_root, stacker_raw["artifact"], where="scoring.stacker.artifact"
    )
    aliases_raw = _mapping(
        stacker_raw["source_aliases"],
        where="scoring.stacker.source_aliases",
        keys={"qwen", "baseline", "anchor", "assessment"},
    )
    aliases = {
        kind: _nullable_text(value, where=f"scoring.stacker.source_aliases.{kind}")
        for kind, value in aliases_raw.items()
    }
    optional_artifacts = {
        "baseline": baseline_artifact,
        "anchor": anchor_artifact,
        "assessment": assessment_artifact,
    }
    if scoring_mode == "qwen_ensemble":
        if any(value is not None for value in optional_artifacts.values()) or stacker_artifact is not None:
            raise ValueError(
                "qwen_ensemble mode requires null baseline/anchor/assessment/stacker artifacts"
            )
        if any(aliases[kind] is not None for kind in optional_artifacts):
            raise ValueError(
                "qwen_ensemble mode requires null auxiliary source aliases"
            )
        stacker = None
    else:
        if stacker_artifact is None or not any(
            value is not None for value in optional_artifacts.values()
        ):
            raise ValueError(
                "qwen_multisource_stacker requires a stacker and at least one auxiliary source"
            )
        if aliases["qwen"] is None:
            raise ValueError("multisource stacker requires a qwen source alias")
        for kind, artifact in optional_artifacts.items():
            if (artifact is None) != (aliases[kind] is None):
                raise ValueError(
                    f"{kind} artifact and source alias must be configured together"
                )
        active_aliases = {
            kind: alias
            for kind, alias in aliases.items()
            if alias is not None
        }
        if len(set(active_aliases.values())) != len(active_aliases):
            raise ValueError("stacker source aliases must be distinct")
        stacker = StackerConfig(stacker_artifact, active_aliases)

    rationale_raw = _mapping(
        top["rationale"],
        where="rationale",
        keys={"mode", "checkpoint", "max_attempts"},
    )
    rationale_mode = _text(rationale_raw["mode"], where="rationale.mode")
    if rationale_mode not in {"deterministic", "adapter"}:
        raise ValueError("rationale.mode must be deterministic or adapter")
    rationale_checkpoint = _nullable_path(
        project_root, rationale_raw["checkpoint"], where="rationale.checkpoint"
    )
    if rationale_mode == "deterministic" and rationale_checkpoint is not None:
        raise ValueError("deterministic rationale mode requires checkpoint=null")
    if rationale_mode == "adapter" and rationale_checkpoint is None:
        raise ValueError("adapter rationale mode requires a checkpoint")
    rationale = RationaleConfig(
        mode=rationale_mode,
        checkpoint=rationale_checkpoint,
        max_attempts=_positive_int(
            rationale_raw["max_attempts"], where="rationale.max_attempts"
        ),
    )

    output_raw = _mapping(top["output"], where="output", keys={"model_name"})
    return DeploymentConfig(
        path=config_path,
        project_root=project_root,
        runtime=runtime,
        scoring_mode=scoring_mode,
        qwen=qwen,
        baseline_artifact=baseline_artifact,
        anchor_artifact=anchor_artifact,
        assessment_artifact=assessment_artifact,
        stacker=stacker,
        rationale=rationale,
        output_model_name=_text(output_raw["model_name"], where="output.model_name"),
    )


def configure_offline_environment(config: DeploymentConfig) -> None:
    """Force Hugging Face libraries offline before they are imported."""

    if not config.runtime.offline_required:
        raise ValueError("offline deployment is mandatory")
    required = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for name, value in required.items():
        current = os.environ.get(name)
        if current not in (None, value):
            raise RuntimeError(f"{name}={current!r} conflicts with mandatory offline mode")
        os.environ[name] = value
    sys.dont_write_bytecode = True
    if config.runtime.hf_home is not None:
        if not config.runtime.hf_home.is_dir():
            raise FileNotFoundError(f"configured HF_HOME is absent: {config.runtime.hf_home}")
        existing = os.environ.get("HF_HOME")
        expected = str(config.runtime.hf_home)
        if existing is not None and Path(existing).resolve() != config.runtime.hf_home:
            raise RuntimeError("existing HF_HOME conflicts with deployment config")
        os.environ["HF_HOME"] = expected


def validate_package_manifest(config: DeploymentConfig) -> dict[str, Any] | None:
    """Verify every file signed by a packaged deployment, when configured."""

    manifest_path = config.runtime.package_manifest
    if manifest_path is None:
        return None
    package_paths = [
        config.path,
        manifest_path,
        config.runtime.hf_home,
        config.runtime.requirements_lock,
        *config.qwen.checkpoints,
        config.qwen.calibrator,
        config.baseline_artifact,
        config.anchor_artifact,
        config.assessment_artifact,
        config.stacker.artifact if config.stacker is not None else None,
        config.rationale.checkpoint,
    ]
    for path in package_paths:
        if path is None:
            continue
        try:
            path.resolve().relative_to(config.project_root)
        except ValueError as error:
            raise ValueError(f"packaged deployment path escapes package root: {path}") from error
    manifest_path = _regular_file(manifest_path)
    payload = _json_object(manifest_path)
    if payload.get("artifact_type") != "submission_package_v1":
        raise ValueError("invalid submission package manifest")
    if payload.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise ValueError("unsupported submission package schema version")
    if payload.get("config_file") != config.path.name:
        raise ValueError("package manifest points to a different deployment config")
    if payload.get("config_sha256") != sha256_file(config.path):
        raise ValueError("packaged deployment config hash mismatch")
    files = payload.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("package manifest contains no signed files")
    canonical_files: dict[str, str] = {}
    for relative, expected_hash in files.items():
        relative_path = Path(relative) if isinstance(relative, str) else None
        if (
            relative_path is None
            or not relative
            or "\\" in relative
            or relative_path.is_absolute()
            or any(part in {".", ".."} for part in relative_path.parts)
        ):
            raise ValueError("package manifest contains an invalid relative path")
        if not isinstance(expected_hash, str) or not _SHA256.fullmatch(expected_hash):
            raise ValueError(f"package manifest has an invalid hash for {relative!r}")
        candidate = (config.project_root / relative).resolve()
        try:
            candidate.relative_to(config.project_root)
        except ValueError as error:
            raise ValueError("package manifest path escapes package root") from error
        _regular_file(candidate)
        actual_hash = sha256_file(candidate)
        if actual_hash != expected_hash:
            raise ValueError(f"package file hash mismatch: {relative}")
        canonical_files[relative] = expected_hash
    payload_entries = tuple(config.project_root.rglob("*"))
    symlinks = [path for path in payload_entries if path.is_symlink()]
    if symlinks:
        raise ValueError(f"submission package must not contain symlinks: {symlinks[0]}")
    actual_payload_files = {
        path.relative_to(config.project_root).as_posix()
        for path in payload_entries
        if (
            path.is_file()
            and path.resolve() != manifest_path
            and not is_runtime_cache_file(path.relative_to(config.project_root))
        )
    }
    if actual_payload_files != set(canonical_files):
        missing = sorted(set(canonical_files).difference(actual_payload_files))
        unsigned = sorted(actual_payload_files.difference(canonical_files))
        raise ValueError(
            "package payload closure differs from its manifest; "
            f"missing={missing[:5]}, unsigned={unsigned[:5]}"
        )
    if config.path.relative_to(config.project_root).as_posix() not in canonical_files:
        raise ValueError("package manifest does not sign its deployment config")
    if manifest_path.relative_to(config.project_root).as_posix() in canonical_files:
        raise ValueError("package manifest must not recursively list itself")
    signature_payload = {
        "schema_version": payload.get("schema_version"),
        "config_file": payload.get("config_file"),
        "config_sha256": payload.get("config_sha256"),
        "files": canonical_files,
    }
    if payload.get("package_signature") != sha256_json(signature_payload):
        raise ValueError("submission package signature mismatch")
    return payload


def is_runtime_cache_file(relative_path: str | Path) -> bool:
    """Identify only Python's transient ``__pycache__`` bytecode files.

    SDKs may import the signed source tree before ``from_config`` can validate
    it.  Those cache files are not package payload and are ignored by closure
    checks; every other unsigned file remains a hard failure.
    """

    path = Path(relative_path)
    return "__pycache__" in path.parts and path.suffix.lower() in {".pyc", ".pyo"}


_LOCKED_REQUIREMENT = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)"
    r"(?:\s+--hash=sha256:[0-9a-fA-F]{64})*$"
)
_REQUIRED_RUNTIME_DISTRIBUTIONS = {
    "accelerate",
    "bitsandbytes",
    "joblib",
    "numpy",
    "pandas",
    "peft",
    "pyyaml",
    "safetensors",
    "scikit-learn",
    "scipy",
    "torch",
    "transformers",
}


def _normalized_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def locked_requirements(path: str | Path) -> dict[str, tuple[str, str]]:
    """Parse a fully resolved ``name==version`` L40S lock file.

    Environment markers, URLs, editable installs, and ranges are rejected because
    they do not describe one immutable offline environment. Pip ``--hash`` tokens
    and backslash continuation are accepted.
    """

    lock_path = _regular_file(Path(path).resolve())
    logical_lines: list[str] = []
    pending = ""
    for raw in lock_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pending = f"{pending} {stripped}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical_lines.append(pending)
        pending = ""
    if pending:
        raise ValueError("requirements lock ends with an incomplete continuation")
    if not logical_lines:
        raise ValueError("requirements lock contains no pinned distributions")
    result: dict[str, tuple[str, str]] = {}
    for line in logical_lines:
        match = _LOCKED_REQUIREMENT.fullmatch(line)
        if match is None:
            raise ValueError(
                "every requirements lock entry must be an unconditional name==version "
                f"with optional sha256 hashes; invalid entry={line!r}"
            )
        original_name, version = match.groups()
        if "*" in version:
            raise ValueError(f"wildcard version is forbidden in requirements lock: {line!r}")
        name = _normalized_distribution(original_name)
        if name in result:
            raise ValueError(f"duplicate distribution in requirements lock: {original_name}")
        result[name] = (original_name, version)
    missing = sorted(_REQUIRED_RUNTIME_DISTRIBUTIONS.difference(result))
    if missing:
        raise ValueError(f"requirements lock is missing runtime distributions: {missing}")
    return result


def validate_dependency_lock(config: DeploymentConfig) -> dict[str, str] | None:
    """Require installed versions to match the packaged L40S lock exactly."""

    path = config.runtime.requirements_lock
    if path is None:
        if config.runtime.package_manifest is not None:
            raise ValueError("a packaged deployment must configure runtime.requirements_lock")
        return None
    requirements = locked_requirements(path)
    installed: dict[str, str] = {}
    for normalized, (distribution, expected) in requirements.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"locked distribution is not installed: {distribution}") from error
        if actual != expected:
            raise RuntimeError(
                f"dependency lock mismatch for {distribution}: expected {expected}, got {actual}"
            )
        installed[normalized] = actual
    return installed


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _regular_file(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"deployment artifacts must not be symlinks: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _tree_files(root: Path) -> tuple[Path, ...]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"artifact directory must be a real directory: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"deployment artifacts must not contain symlinks: {path}")
        if path.is_file():
            files.append(path.resolve())
    return tuple(files)


def resolve_scorer_checkpoint(path: Path) -> Path:
    supplied = path.resolve()
    if supplied.is_file():
        pointer = _json_object(supplied)
        relative = pointer.get("checkpoint")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"checkpoint pointer has no checkpoint field: {supplied}")
        candidate = (supplied.parent / relative).resolve()
        try:
            candidate.relative_to(supplied.parent.resolve())
        except ValueError as error:
            raise ValueError("checkpoint pointer escapes its run directory") from error
        supplied = candidate
    required = (
        supplied / "adapter" / "adapter_config.json",
        supplied / "tokenizer" / "tokenizer_config.json",
        supplied / "scoring_heads.pt",
        supplied / "scoring_head_config.json",
        supplied / "checkpoint_provenance.json",
        supplied / "oof.jsonl",
    )
    for value in required:
        _regular_file(value)
    weights = (
        supplied / "adapter" / "adapter_model.safetensors",
        supplied / "adapter" / "adapter_model.bin",
    )
    if not any(value.is_file() and not value.is_symlink() for value in weights):
        raise FileNotFoundError(f"scorer adapter weights are absent: {supplied}")
    provenance = _json_object(supplied / "checkpoint_provenance.json")
    if (
        provenance.get("artifact_type") != "qwen_scorer_fold_checkpoint"
        or provenance.get("oof_file") != "oof.jsonl"
        or provenance.get("oof_sha256") != sha256_file(supplied / "oof.jsonl")
    ):
        raise ValueError(f"invalid scorer checkpoint provenance: {supplied}")
    return supplied


def scorer_checkpoint_files(checkpoint: Path) -> tuple[Path, ...]:
    resolved = resolve_scorer_checkpoint(checkpoint)
    files = [
        resolved / "scoring_heads.pt",
        resolved / "scoring_head_config.json",
        resolved / "checkpoint_provenance.json",
        resolved / "oof.jsonl",
    ]
    files.extend(_tree_files(resolved / "adapter"))
    files.extend(_tree_files(resolved / "tokenizer"))
    return tuple(dict.fromkeys(_regular_file(path) for path in files))


def _calibrator_files(path: Path | None) -> tuple[Path, ...]:
    if path is None:
        return ()
    calibrator = _regular_file(path)
    manifest_path = _regular_file(path.with_suffix(".manifest.json"))
    manifest = _json_object(manifest_path)
    if (
        manifest.get("artifact_type") != "oof_affine_prompt_calibrator"
        or manifest.get("calibrator_file") != calibrator.name
        or manifest.get("calibrator_sha256") != sha256_file(calibrator)
    ):
        raise ValueError("calibrator does not match its adjacent manifest")
    payload = _json_object(calibrator)
    if payload.get("fit_source") != "oof" or payload.get("method") != "affine_prompt_shrinkage":
        raise ValueError("deployment calibrator must be affine_prompt_shrinkage fitted on OOF")
    if payload.get("inference_code_contract") != calibration_inference_code_contract():
        raise ValueError("calibrator inference source changed since OOF fitting")
    return (calibrator, manifest_path)


def _baseline_files(path: Path | None) -> tuple[tuple[Path, ...], str | None, str | None]:
    if path is None:
        return (), None, None
    artifact = _regular_file(path)
    payload = _json_object(artifact)
    if (
        payload.get("artifact_type") != "baseline_fold_ensemble"
        or payload.get("aggregation") != "equal_weight_mean"
    ):
        raise ValueError("invalid baseline fold-ensemble artifact")
    scorer_name = _text(payload.get("scorer_name"), where="baseline.scorer_name")
    signature = _text(payload.get("scorer_signature"), where="baseline.scorer_signature")
    signature_payload = payload.get("signature_payload")
    if not isinstance(signature_payload, Mapping) or sha256_json(signature_payload) != signature:
        raise ValueError("baseline signature payload mismatch")
    if signature_payload.get("scorer") != scorer_name:
        raise ValueError("baseline scorer name/signature mismatch")
    if (
        signature_payload.get("inference_code_contract")
        != baseline_inference_code_contract()
    ):
        raise ValueError(
            "baseline inference source has changed since the artifact was created"
        )
    fold_models = payload.get("fold_models")
    if not isinstance(fold_models, Mapping) or not fold_models:
        raise ValueError("baseline artifact contains no fold models")
    files = [artifact]
    expected_hashes: dict[str, str] = {}
    for fold_id, item in fold_models.items():
        if not isinstance(item, Mapping) or not isinstance(item.get("file"), str):
            raise ValueError(f"invalid baseline fold entry: {fold_id!r}")
        if Path(item["file"]).name != item["file"]:
            raise ValueError("baseline fold model filenames must not contain directories")
        model = (artifact.parent / item["file"]).resolve()
        try:
            model.relative_to(artifact.parent.resolve())
        except ValueError as error:
            raise ValueError("baseline fold model escapes artifact directory") from error
        _regular_file(model)
        if item.get("sha256") != sha256_file(model):
            raise ValueError(f"baseline fold hash mismatch: {model}")
        files.append(model)
        expected_hashes[str(fold_id)] = item["sha256"]
    if signature_payload.get("fold_model_hashes") != expected_hashes:
        raise ValueError("baseline signature does not bind its fold files")
    selection_name = payload.get("selection_report")
    selection_hash = payload.get("selection_report_sha256")
    if (selection_name is None) != (selection_hash is None):
        raise ValueError("baseline selection report name/hash must be present together")
    if selection_name is not None:
        if not isinstance(selection_name, str) or Path(selection_name).name != selection_name:
            raise ValueError("baseline selection report must be an adjacent filename")
        selection_path = _regular_file(artifact.parent / selection_name)
        if selection_hash != sha256_file(selection_path):
            raise ValueError("baseline selection report hash mismatch")
        if signature_payload.get("selection_report_sha256") != selection_hash:
            raise ValueError("baseline signature does not bind its selection report")
        files.append(selection_path)
    return tuple(files), scorer_name, signature


def _anchor_files(path: Path | None) -> tuple[tuple[Path, ...], str | None, str | None]:
    if path is None:
        return (), None, None
    from src.anchors.artifact import anchor_manifest_path, load_anchor_bank

    artifact = _regular_file(path)
    manifest_path = _regular_file(anchor_manifest_path(artifact))
    bank = load_anchor_bank(artifact)
    scorer_name = _text(
        bank.manifest.get("scorer_name"), where="anchor.scorer_name"
    )
    signature = _text(
        bank.manifest.get("anchor_signature"), where="anchor.anchor_signature"
    )
    return (artifact, manifest_path), scorer_name, signature


def _assessment_files(
    path: Path | None,
) -> tuple[tuple[Path, ...], str | None, str | None]:
    if path is None:
        return (), None, None
    from src.assessment.artifact import load_deployment_artifact

    artifact = _regular_file(path)
    manifest_path = _regular_file(path.with_suffix(".manifest.json"))
    payload = load_deployment_artifact(artifact)
    manifest = _json_object(manifest_path)
    scorer_name = _text(payload.get("scorer_name"), where="assessment.scorer_name")
    signature = _text(
        payload.get("artifact_signature"), where="assessment.artifact_signature"
    )
    if (
        manifest.get("artifact_type") != "assessment_ridge_deployment_manifest"
        or manifest.get("model_file") != artifact.name
        or manifest.get("model_sha256") != sha256_file(artifact)
        or manifest.get("scorer_name") != scorer_name
        or manifest.get("scorer_signature") != signature
        or manifest.get("feature_signature") != payload.get("feature_signature")
    ):
        raise ValueError("assessment artifact does not match its adjacent manifest")
    return (artifact, manifest_path), scorer_name, signature


_STACKER_SIGNED_FIELDS = (
    "artifact_version",
    "method",
    "fit_source",
    "scorer_name",
    "source_order",
    "source_contracts",
    "stacker",
    "calibrator",
    "gold_sha256",
    "folds_sha256",
    "inference_code_contract",
    "config",
)


def _stacker_files(path: Path | None) -> tuple[tuple[Path, ...], str | None]:
    if path is None:
        return (), None
    artifact = _regular_file(path)
    manifest_path = _regular_file(path.with_suffix(".manifest.json"))
    payload = _json_object(artifact)
    if payload.get("artifact_type") != "trait_simplex_stacker":
        raise ValueError("invalid trait simplex stacker artifact")
    if any(field not in payload for field in _STACKER_SIGNED_FIELDS):
        raise ValueError("stacker artifact is missing a signed field")
    signature = sha256_json({field: payload[field] for field in _STACKER_SIGNED_FIELDS})
    if payload.get("stacker_signature") != signature:
        raise ValueError("stacker signature mismatch")
    if payload.get("inference_code_contract") != stacker_inference_code_contract():
        raise ValueError("stacker inference source changed since OOF fitting")
    manifest = _json_object(manifest_path)
    if (
        manifest.get("artifact_type") != "trait_simplex_stacker_manifest"
        or manifest.get("stacker_file") != artifact.name
        or manifest.get("stacker_sha256") != sha256_file(artifact)
        or manifest.get("stacker_signature") != signature
    ):
        raise ValueError("stacker does not match its adjacent manifest")
    return (artifact, manifest_path), signature


def resolve_rationale_checkpoint(path: Path) -> Path:
    supplied = path.resolve()
    if supplied.is_file():
        pointer = _json_object(supplied)
        relative = pointer.get("checkpoint")
        if not isinstance(relative, str) or not relative:
            raise ValueError("rationale checkpoint pointer has no checkpoint field")
        candidate = (supplied.parent / relative).resolve()
        try:
            candidate.relative_to(supplied.parent.resolve())
        except ValueError as error:
            raise ValueError("rationale checkpoint pointer escapes its run directory") from error
        supplied = candidate
    required = (
        supplied / "adapter" / "adapter_config.json",
        supplied / "tokenizer" / "tokenizer_config.json",
        supplied / "checkpoint.json",
    )
    for value in required:
        _regular_file(value)
    weights = (
        supplied / "adapter" / "adapter_model.safetensors",
        supplied / "adapter" / "adapter_model.bin",
    )
    if not any(value.is_file() and not value.is_symlink() for value in weights):
        raise FileNotFoundError(f"rationale adapter weights are absent: {supplied}")
    metadata = _json_object(supplied / "checkpoint.json")
    if metadata.get("artifact_type") != "rationale_adapter_checkpoint":
        raise ValueError("invalid rationale checkpoint metadata")
    revision = metadata.get("model_revision")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise ValueError("rationale checkpoint has no pinned base revision")
    return supplied


def rationale_checkpoint_files(checkpoint: Path) -> tuple[Path, ...]:
    resolved = resolve_rationale_checkpoint(checkpoint)
    files = [resolved / "checkpoint.json"]
    files.extend(_tree_files(resolved / "adapter"))
    files.extend(_tree_files(resolved / "tokenizer"))
    return tuple(dict.fromkeys(_regular_file(path) for path in files))


def validate_artifact_contracts(config: DeploymentConfig) -> ValidatedArtifacts:
    """Validate every configured artifact and return the immutable file closure."""

    checkpoints = tuple(resolve_scorer_checkpoint(path) for path in config.qwen.checkpoints)
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("two scorer checkpoint pointers resolve to the same directory")
    for checkpoint in checkpoints:
        metadata = _json_object(checkpoint / "scoring_head_config.json")
        if metadata.get("model_id") != config.qwen.model_id:
            raise ValueError(f"scorer checkpoint model_id mismatch: {checkpoint}")
        if metadata.get("model_revision") != config.qwen.model_revision:
            raise ValueError(f"scorer checkpoint model revision mismatch: {checkpoint}")
        if metadata.get("scorer_architecture_version") != SCORER_ARCHITECTURE_VERSION:
            raise ValueError(f"scorer checkpoint architecture version mismatch: {checkpoint}")
    scorer_files = tuple(scorer_checkpoint_files(path) for path in checkpoints)
    scorer_signature = checkpoint_ensemble_signature(
        checkpoints, precision=config.runtime.scorer_precision
    )
    calibrator_files = _calibrator_files(config.qwen.calibrator)
    qwen_output_signature = scorer_signature
    if config.qwen.calibrator is not None:
        calibrator_payload = _json_object(config.qwen.calibrator)
        source = calibrator_payload.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("Qwen calibrator is missing its OOF source contract")
        if source.get("scorer_name") != config.qwen.scorer_name:
            raise ValueError("Qwen calibrator scorer_name mismatch")
        if source.get("scorer_signature") != scorer_signature:
            raise ValueError("Qwen calibrator checkpoint-ensemble signature mismatch")
        qwen_output_signature = sha256_json(
            {
                "base_scorer_signature": scorer_signature,
                "calibrator_sha256": sha256_file(config.qwen.calibrator),
                "transform": "affine_prompt_shrinkage",
            }
        )
    baseline_files, baseline_name, baseline_signature = _baseline_files(
        config.baseline_artifact
    )
    anchor_files, anchor_name, anchor_signature = _anchor_files(
        config.anchor_artifact
    )
    assessment_files, assessment_name, assessment_signature = _assessment_files(
        config.assessment_artifact
    )
    if config.anchor_artifact is not None:
        from src.anchors.artifact import load_anchor_bank
        from src.anchors.embeddings import embedding_extraction_contract_sha256

        bank = load_anchor_bank(config.anchor_artifact)
        contracts = bank.manifest.get("embedding_contracts")
        if not isinstance(contracts, Mapping):
            raise ValueError("anchor bank has no checkpoint embedding contracts")
        checkpoint_by_fold: dict[int, Path] = {}
        for checkpoint in checkpoints:
            provenance = _json_object(checkpoint / "checkpoint_provenance.json")
            fold = provenance.get("fold")
            if isinstance(fold, bool) or not isinstance(fold, int) or fold in checkpoint_by_fold:
                raise ValueError(
                    "anchor deployment requires exactly one scorer checkpoint per fold"
                )
            checkpoint_by_fold[fold] = checkpoint
        if set(checkpoint_by_fold) != set(bank.embeddings_by_fold):
            raise ValueError("anchor bank folds do not match deployment scorer folds")
        for fold, checkpoint in checkpoint_by_fold.items():
            contract = contracts.get(str(fold))
            if (
                not isinstance(contract, Mapping)
                or contract.get("scorer_signature")
                != checkpoint_ensemble_signature(
                    [checkpoint], precision=config.runtime.scorer_precision
                )
                or contract.get("extraction_contract_sha256")
                != embedding_extraction_contract_sha256()
            ):
                raise ValueError(f"anchor checkpoint contract mismatch for fold {fold}")
    if config.assessment_artifact is not None:
        from src.assessment.artifact import load_deployment_artifact

        assessment_payload = load_deployment_artifact(config.assessment_artifact)
        feature_contract = assessment_payload.get("feature_contract")
        if (
            not isinstance(feature_contract, Mapping)
            or feature_contract.get("model_id") != config.qwen.model_id
            or feature_contract.get("model_revision") != config.qwen.model_revision
            or feature_contract.get("tokenizer_revision") != config.qwen.model_revision
        ):
            raise ValueError(
                "assessment source must use the packaged Qwen model/tokenizer revision"
            )
    stacker_files, stacker_signature = _stacker_files(
        config.stacker.artifact if config.stacker else None
    )
    if config.stacker is not None:
        stacker_payload = _json_object(config.stacker.artifact)
        source_order = stacker_payload.get("source_order")
        contracts = stacker_payload.get("source_contracts")
        aliases = set(config.stacker.source_aliases.values())
        if (
            not isinstance(source_order, list)
            or len(source_order) != len(aliases)
            or set(source_order) != aliases
        ):
            raise ValueError("stacker aliases do not match deployment config")
        if not isinstance(contracts, Mapping) or set(contracts) != aliases:
            raise ValueError("stacker source contracts are absent")
        source_identities = {
            "qwen": (config.qwen.scorer_name, qwen_output_signature),
            "baseline": (baseline_name, baseline_signature),
            "anchor": (anchor_name, anchor_signature),
            "assessment": (assessment_name, assessment_signature),
        }
        expected_sources = {
            alias: source_identities[kind]
            for kind, alias in config.stacker.source_aliases.items()
        }
        for alias, (name, signature) in expected_sources.items():
            contract = contracts.get(alias)
            if (
                not isinstance(contract, Mapping)
                or contract.get("scorer_name") != name
                or contract.get("scorer_signature") != signature
            ):
                raise ValueError(f"stacker source contract mismatch for {alias}")
        calibration = stacker_payload.get("calibrator")
        if (
            not isinstance(calibration, Mapping)
            or calibration.get("method") != "affine_prompt_shrinkage"
            or calibration.get("fit_source") != "base_oof_stacked"
        ):
            raise ValueError("stacker has an invalid deployment calibrator")
    rationale_checkpoint = (
        resolve_rationale_checkpoint(config.rationale.checkpoint)
        if config.rationale.checkpoint is not None
        else None
    )
    rationale_files = (
        rationale_checkpoint_files(rationale_checkpoint)
        if rationale_checkpoint is not None
        else ()
    )
    if rationale_checkpoint is not None:
        metadata = _json_object(rationale_checkpoint / "checkpoint.json")
        if metadata.get("model_id") != config.qwen.model_id:
            raise ValueError("rationale and scorer checkpoints use different base models")
        if metadata.get("model_revision") != config.qwen.model_revision:
            raise ValueError("rationale and scorer checkpoints use different base revisions")
    for file in (
        *(path for group in scorer_files for path in group),
        *calibrator_files,
        *baseline_files,
        *anchor_files,
        *assessment_files,
        *stacker_files,
        *rationale_files,
    ):
        digest = sha256_file(file)
        if not _SHA256.fullmatch(digest):
            raise RuntimeError(f"failed to hash deployment artifact: {file}")
    return ValidatedArtifacts(
        scorer_checkpoints=checkpoints,
        scorer_files=scorer_files,
        scorer_signature=scorer_signature,
        calibrator_files=calibrator_files,
        baseline_files=baseline_files,
        baseline_scorer_name=baseline_name,
        baseline_scorer_signature=baseline_signature,
        anchor_files=anchor_files,
        anchor_scorer_name=anchor_name,
        anchor_scorer_signature=anchor_signature,
        assessment_files=assessment_files,
        assessment_scorer_name=assessment_name,
        assessment_scorer_signature=assessment_signature,
        stacker_files=stacker_files,
        stacker_signature=stacker_signature,
        rationale_checkpoint=rationale_checkpoint,
        rationale_files=rationale_files,
    )


__all__ = [
    "DeploymentConfig",
    "PACKAGE_SCHEMA_VERSION",
    "RuntimeConfig",
    "ValidatedArtifacts",
    "SCHEMA_VERSION",
    "configure_offline_environment",
    "is_runtime_cache_file",
    "load_deployment_config",
    "locked_requirements",
    "rationale_checkpoint_files",
    "resolve_rationale_checkpoint",
    "resolve_scorer_checkpoint",
    "scorer_checkpoint_files",
    "validate_artifact_contracts",
    "validate_dependency_lock",
    "validate_package_manifest",
]
