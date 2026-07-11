"""Deterministic fold/seed run registry and scorer artifact validation."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data.folds import load_folds
from src.evaluation.oof_provenance import checkpoint_fingerprint
from src.models.contracts import SCORER_ARCHITECTURE_VERSION
from src.orchestration.epoch_policy import load_epoch_policy
from src.utils.config import load_yaml
from src.utils.hashing import sha256_file, sha256_json


RUN_REGISTRY_TYPE = "qwen_scorer_training_registry"
SCHEMA_VERSION = 1
TASK_STATUSES = {"pending", "running", "completed", "failed"}
_EXPERIMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_REVISION = re.compile(r"[0-9a-fA-F]{40}")
_TASK_IDENTITY_FIELDS = (
    "task_id",
    "run_id",
    "fold",
    "seed",
    "output_dir",
    "expected_epochs",
    "fixed_epoch",
    "precision",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_reference(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _seed_value(value: Any, *, field: str = "seed") -> int:
    seed = _nonnegative_int(value, field=field)
    if seed > 2**32 - 1:
        raise ValueError(f"{field} must be at most 2**32-1 for NumPy compatibility")
    return seed


def _task_identity(task: Mapping[str, Any]) -> dict[str, Any]:
    return {field: task[field] for field in _TASK_IDENTITY_FIELDS}


def _task_signature(task: Mapping[str, Any], plan_signature: str) -> str:
    return sha256_json(
        {"plan_signature": plan_signature, "task": _task_identity(task)}
    )


def _registry_signature(plan: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json(
        {
            "plan": dict(plan),
            "tasks": [_task_identity(task) for task in tasks],
        }
    )


def build_run_registry(
    *,
    experiment_id: str,
    project_root: str | Path,
    output_root: str | Path,
    scorer_config_path: str | Path,
    data_config_path: str | Path,
    fold_path: str | Path,
    train_path: str | Path,
    epoch_policy_path: str | Path,
    source_files: Sequence[str | Path],
    model_revision: str,
    epochs: int,
    folds: Sequence[int],
    seeds: Sequence[int],
    precision: str,
    allow_download: bool,
) -> dict[str, Any]:
    if not isinstance(experiment_id, str) or not _EXPERIMENT_ID.fullmatch(experiment_id):
        raise ValueError(
            "experiment_id must use only letters, digits, dot, underscore, or hyphen"
        )
    if not isinstance(model_revision, str) or not _REVISION.fullmatch(model_revision):
        raise ValueError("model_revision must be a pinned 40-character commit SHA")
    epochs = _positive_int(epochs, field="epochs")
    if precision not in {"4bit", "bf16"}:
        raise ValueError("precision must be '4bit' or 'bf16'")
    if not isinstance(allow_download, bool):
        raise ValueError("allow_download must be boolean")

    normalized_folds = sorted({_nonnegative_int(value, field="fold") for value in folds})
    normalized_seeds = sorted({_seed_value(value) for value in seeds})
    if not normalized_folds or not normalized_seeds:
        raise ValueError("at least one fold and seed are required")
    if len(normalized_folds) != len(folds):
        raise ValueError("folds must not contain duplicates")
    if len(normalized_seeds) != len(seeds):
        raise ValueError("seeds must not contain duplicates")

    assignments = load_folds(fold_path)
    available_folds = sorted(set(assignments.values()))
    unknown_folds = sorted(set(normalized_folds).difference(available_folds))
    if unknown_folds:
        raise ValueError(
            f"requested folds are absent from the fold file: {unknown_folds}"
        )
    policy_path = Path(epoch_policy_path).resolve()
    policy = load_epoch_policy(policy_path, max_epoch=epochs)
    resolved_source_files = sorted({Path(path).resolve() for path in source_files})
    if not resolved_source_files:
        raise ValueError("at least one scorer source file must be bound to the registry")
    fixed_epoch = int(policy["fixed_epoch"])
    root = Path(project_root).resolve()
    experiment_dir = Path(output_root).resolve() / experiment_id

    plan: dict[str, Any] = {
        "experiment_id": experiment_id,
        "project_root": str(root),
        "experiment_dir": str(experiment_dir),
        "trainer_module": "src.train.train_scorer",
        "scorer_config": _file_reference(scorer_config_path),
        "data_config": _file_reference(data_config_path),
        "folds_file": _file_reference(fold_path),
        "train_data": _file_reference(train_path),
        "epoch_policy": _file_reference(policy_path),
        "source_files": [_file_reference(path) for path in resolved_source_files],
        "epoch_policy_signature": policy["policy_signature"],
        "model_revision": model_revision.lower(),
        "epochs": epochs,
        "fixed_epoch": fixed_epoch,
        "folds": normalized_folds,
        "seeds": normalized_seeds,
        "precision": precision,
        "allow_download": allow_download,
    }
    plan_signature = sha256_json(plan)
    tasks: list[dict[str, Any]] = []
    for seed in normalized_seeds:
        for fold in normalized_folds:
            task_id = f"seed_{seed}__fold_{fold}"
            task = {
                "task_id": task_id,
                "run_id": f"{experiment_id}__{task_id}",
                "fold": fold,
                "seed": seed,
                "output_dir": str(experiment_dir / task_id),
                "expected_epochs": epochs,
                "fixed_epoch": fixed_epoch,
                "precision": precision,
                "status": "pending",
                "attempts": 0,
                "last_error": None,
                "selected_checkpoint": None,
                "selected_checkpoint_fingerprint": None,
            }
            task["task_signature"] = _task_signature(task, plan_signature)
            tasks.append(task)
    return {
        "artifact_type": RUN_REGISTRY_TYPE,
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "plan": plan,
        "plan_signature": plan_signature,
        "registry_signature": _registry_signature(plan, tasks),
        "tasks": tasks,
    }


def validate_run_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(registry)
    if payload.get("artifact_type") != RUN_REGISTRY_TYPE:
        raise ValueError("not a Qwen scorer run registry")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported run registry schema")
    plan = payload.get("plan")
    tasks = payload.get("tasks")
    if not isinstance(plan, dict) or not isinstance(tasks, list) or not tasks:
        raise ValueError("run registry requires a plan and nonempty tasks")
    required_plan_fields = {
        "experiment_id",
        "project_root",
        "experiment_dir",
        "trainer_module",
        "scorer_config",
        "data_config",
        "folds_file",
        "train_data",
        "epoch_policy",
        "source_files",
        "epoch_policy_signature",
        "model_revision",
        "epochs",
        "fixed_epoch",
        "folds",
        "seeds",
        "precision",
        "allow_download",
    }
    missing_plan_fields = sorted(required_plan_fields.difference(plan))
    if missing_plan_fields:
        raise ValueError(f"run registry plan is missing fields: {missing_plan_fields}")
    experiment_id = plan["experiment_id"]
    if not isinstance(experiment_id, str) or not _EXPERIMENT_ID.fullmatch(experiment_id):
        raise ValueError("run registry experiment_id is invalid")
    if plan["trainer_module"] != "src.train.train_scorer":
        raise ValueError("run registry trainer_module is unsupported")
    if not isinstance(plan["model_revision"], str) or not _REVISION.fullmatch(
        plan["model_revision"]
    ):
        raise ValueError("run registry model_revision is not pinned")
    epochs = _positive_int(plan["epochs"], field="plan.epochs")
    fixed_epoch = _positive_int(plan["fixed_epoch"], field="plan.fixed_epoch")
    if fixed_epoch > epochs:
        raise ValueError("run registry fixed_epoch exceeds plan.epochs")
    if plan["precision"] not in {"4bit", "bf16"}:
        raise ValueError("run registry plan.precision is invalid")
    if not isinstance(plan["allow_download"], bool):
        raise ValueError("run registry plan.allow_download must be boolean")
    if not isinstance(plan["folds"], list) or not isinstance(plan["seeds"], list):
        raise ValueError("run registry plan folds/seeds must be lists")
    if not isinstance(plan["source_files"], list) or not plan["source_files"]:
        raise ValueError("run registry plan source_files must be a nonempty list")
    plan_folds = [
        _nonnegative_int(value, field="plan.fold") for value in plan["folds"]
    ]
    plan_seeds = [
        _seed_value(value, field="plan.seed") for value in plan["seeds"]
    ]
    if (
        not plan_folds
        or not plan_seeds
        or plan_folds != sorted(set(plan_folds))
        or plan_seeds != sorted(set(plan_seeds))
    ):
        raise ValueError("run registry plan folds/seeds must be nonempty sorted uniques")
    plan_signature = payload.get("plan_signature")
    if not isinstance(plan_signature, str) or plan_signature != sha256_json(plan):
        raise ValueError("run registry plan signature mismatch")
    seen_ids: set[str] = set()
    seen_outputs: set[str] = set()
    validated_tasks: list[dict[str, Any]] = []
    for index, raw_task in enumerate(tasks):
        if not isinstance(raw_task, dict):
            raise ValueError(f"registry task[{index}] must be an object")
        task = dict(raw_task)
        for field in _TASK_IDENTITY_FIELDS:
            if field not in task:
                raise ValueError(f"registry task[{index}] is missing {field}")
        task_id = task["task_id"]
        if not isinstance(task_id, str) or not task_id or task_id in seen_ids:
            raise ValueError(f"registry task_id is empty or duplicated: {task_id!r}")
        seen_ids.add(task_id)
        output_key = str(Path(task["output_dir"]).resolve()).casefold()
        if output_key in seen_outputs:
            raise ValueError("registry tasks must have distinct output directories")
        seen_outputs.add(output_key)
        _nonnegative_int(task["fold"], field=f"{task_id}.fold")
        _seed_value(task["seed"], field=f"{task_id}.seed")
        _positive_int(task["expected_epochs"], field=f"{task_id}.expected_epochs")
        _positive_int(task["fixed_epoch"], field=f"{task_id}.fixed_epoch")
        if task["precision"] not in {"4bit", "bf16"}:
            raise ValueError(f"{task_id}.precision is invalid")
        if task.get("status") not in TASK_STATUSES:
            raise ValueError(f"{task_id}.status is invalid")
        _nonnegative_int(task.get("attempts"), field=f"{task_id}.attempts")
        expected_signature = _task_signature(task, plan_signature)
        if task.get("task_signature") != expected_signature:
            raise ValueError(f"{task_id} task signature mismatch")
        expected_task_id = f"seed_{task['seed']}__fold_{task['fold']}"
        expected_output = Path(plan["experiment_dir"]).resolve() / expected_task_id
        if (
            task["fold"] not in plan_folds
            or task["seed"] not in plan_seeds
            or task_id != expected_task_id
            or task["run_id"] != f"{experiment_id}__{expected_task_id}"
            or Path(task["output_dir"]).resolve() != expected_output
            or task["expected_epochs"] != epochs
            or task["fixed_epoch"] != fixed_epoch
            or task["precision"] != plan["precision"]
        ):
            raise ValueError(f"{task_id} identity does not match the immutable plan")
        validated_tasks.append(task)
    expected_pairs = {(seed, fold) for seed in plan_seeds for fold in plan_folds}
    observed_pairs = {(task["seed"], task["fold"]) for task in validated_tasks}
    if observed_pairs != expected_pairs or len(validated_tasks) != len(expected_pairs):
        raise ValueError("run registry task grid is incomplete or duplicated")
    if payload.get("registry_signature") != _registry_signature(plan, validated_tasks):
        raise ValueError("run registry immutable task signature mismatch")
    payload["tasks"] = validated_tasks
    return payload


def load_run_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path).resolve()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"run registry must be a JSON object: {registry_path}")
    return validate_run_registry(payload)


def write_run_registry(
    path: str | Path, registry: Mapping[str, Any], *, create_only: bool = False
) -> Path:
    target = Path(path).resolve()
    if create_only and target.exists():
        raise FileExistsError(f"run registry already exists: {target}")
    payload = validate_run_registry(registry)
    payload["updated_at"] = _utc_now()
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if create_only:
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
        return target
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(target)
    return target


def validate_registry_inputs(registry: Mapping[str, Any]) -> None:
    payload = validate_run_registry(registry)
    plan = payload["plan"]
    for field in (
        "scorer_config",
        "data_config",
        "folds_file",
        "train_data",
        "epoch_policy",
    ):
        reference = plan.get(field)
        if not isinstance(reference, dict):
            raise ValueError(f"registry plan lacks file reference {field}")
        path = reference.get("path")
        digest = reference.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError(f"registry {field} reference lacks path/sha256")
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"registry input is missing: {source}")
        if sha256_file(source) != digest:
            raise ValueError(f"registry input hash drifted: {field}={source}")
    source_paths: set[Path] = set()
    for index, reference in enumerate(plan["source_files"]):
        if not isinstance(reference, dict):
            raise ValueError(f"registry source_files[{index}] must be an object")
        path = reference.get("path")
        digest = reference.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError(f"registry source_files[{index}] lacks path/sha256")
        source = Path(path).resolve()
        if source in source_paths:
            raise ValueError("registry source_files contains a duplicate path")
        source_paths.add(source)
        if not source.is_file():
            raise FileNotFoundError(f"registry source file is missing: {source}")
        if sha256_file(source) != digest:
            raise ValueError(f"registry source file hash drifted: {source}")
    policy = load_epoch_policy(
        plan["epoch_policy"]["path"], max_epoch=int(plan["epochs"])
    )
    if (
        policy["policy_signature"] != plan.get("epoch_policy_signature")
        or int(policy["fixed_epoch"]) != int(plan["fixed_epoch"])
    ):
        raise ValueError("registry epoch policy no longer matches the plan")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _finite_metric(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _validate_oof_rows(
    path: Path, *, expected_ids: set[str]
) -> int:
    observed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank OOF row at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"invalid OOF row at {path}:{line_number}")
            record_id = row["id"]
            if record_id in observed:
                raise ValueError(f"duplicate OOF id in {path}: {record_id}")
            prediction = row.get("prediction")
            if not isinstance(prediction, dict):
                raise ValueError(f"OOF row lacks prediction at {path}:{line_number}")
            for trait in ("content", "organization", "expression"):
                container = prediction.get(trait)
                score_value = container.get("score") if isinstance(container, dict) else container
                score = _finite_metric(
                    score_value, field=f"{path}:{line_number}:{trait}"
                )
                if not 1.0 <= score <= 5.0:
                    raise ValueError(f"OOF {trait} score is outside [1, 5]")
            observed.add(record_id)
    if observed != expected_ids:
        missing = sorted(expected_ids.difference(observed))
        extra = sorted(observed.difference(expected_ids))
        raise ValueError(
            f"OOF ids do not match the held-out fold; missing={missing[:5]}, extra={extra[:5]}"
        )
    return len(observed)


def _validate_checkpoint(
    checkpoint: Path,
    *,
    task: Mapping[str, Any],
    epoch: int,
    model_revision: str,
    expected_ids: set[str],
    train_sha256: str,
    folds_sha256: str,
) -> None:
    required = (
        checkpoint / "adapter" / "adapter_config.json",
        checkpoint / "tokenizer" / "tokenizer_config.json",
        checkpoint / "scoring_heads.pt",
        checkpoint / "scoring_head_config.json",
        checkpoint / "metrics.json",
        checkpoint / "checkpoint_provenance.json",
        checkpoint / "oof.jsonl",
    )
    missing = [str(path) for path in required if not path.is_file()]
    adapter_weights = (
        checkpoint / "adapter" / "adapter_model.safetensors",
        checkpoint / "adapter" / "adapter_model.bin",
    )
    if not any(path.is_file() for path in adapter_weights):
        missing.append(f"one of {[str(path) for path in adapter_weights]}")
    if missing:
        raise FileNotFoundError(f"incomplete epoch checkpoint {checkpoint}; missing={missing}")

    provenance = _read_json_object(
        checkpoint / "checkpoint_provenance.json", label="checkpoint provenance"
    )
    if provenance.get("artifact_type") != "qwen_scorer_fold_checkpoint":
        raise ValueError(f"invalid checkpoint artifact_type: {checkpoint}")
    if (
        provenance.get("fold") != task["fold"]
        or provenance.get("seed") != task["seed"]
        or provenance.get("epoch") != epoch
    ):
        raise ValueError(f"checkpoint fold/epoch mismatch: {checkpoint}")
    if provenance.get("precision") != task["precision"]:
        raise ValueError(f"checkpoint precision mismatch: {checkpoint}")
    if provenance.get("scorer_architecture_version") != SCORER_ARCHITECTURE_VERSION:
        raise ValueError(f"checkpoint architecture contract mismatch: {checkpoint}")
    if (
        provenance.get("train_sha256") != train_sha256
        or provenance.get("folds_sha256") != folds_sha256
    ):
        raise ValueError(f"checkpoint training data/fold hash mismatch: {checkpoint}")
    if provenance.get("oof_file") != "oof.jsonl":
        raise ValueError(f"checkpoint OOF path contract mismatch: {checkpoint}")
    oof_path = checkpoint / "oof.jsonl"
    if provenance.get("oof_sha256") != sha256_file(oof_path):
        raise ValueError(f"checkpoint OOF hash mismatch: {checkpoint}")
    rows = _validate_oof_rows(oof_path, expected_ids=expected_ids)
    if provenance.get("rows") != rows:
        raise ValueError(f"checkpoint OOF row count mismatch: {checkpoint}")

    head_config = _read_json_object(
        checkpoint / "scoring_head_config.json", label="scoring head config"
    )
    head_revision = head_config.get("model_revision")
    if (
        head_config.get("fold") != task["fold"]
        or head_config.get("seed") != task["seed"]
        or not isinstance(head_revision, str)
        or head_revision.lower() != model_revision.lower()
        or head_config.get("precision") != task["precision"]
        or head_config.get("scorer_architecture_version")
        != SCORER_ARCHITECTURE_VERSION
    ):
        raise ValueError(f"checkpoint head metadata mismatch: {checkpoint}")
    metrics = _read_json_object(checkpoint / "metrics.json", label="checkpoint metrics")
    if metrics.get("epoch") != epoch:
        raise ValueError(f"checkpoint metrics epoch mismatch: {checkpoint}")


def validate_task_output(
    registry: Mapping[str, Any], task: Mapping[str, Any]
) -> dict[str, Any]:
    payload = validate_run_registry(registry)
    plan = payload["plan"]
    task_id = task.get("task_id")
    registered = next(
        (candidate for candidate in payload["tasks"] if candidate["task_id"] == task_id),
        None,
    )
    if registered is None or registered.get("task_signature") != task.get("task_signature"):
        raise ValueError("task is not part of this run registry")
    output_dir = Path(registered["output_dir"]).resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"task output directory is missing: {output_dir}")
    manifest_path = output_dir / "manifest.json"
    history_path = output_dir / "history.json"
    if not manifest_path.is_file() or not history_path.is_file():
        raise FileNotFoundError(f"task output is incomplete: {output_dir}")
    manifest = _read_json_object(manifest_path, label="run manifest")
    if (
        manifest.get("run_id") != registered["run_id"]
        or manifest.get("fold") != registered["fold"]
        or manifest.get("seed") != registered["seed"]
        or str(manifest.get("model_revision", "")).lower()
        != str(plan["model_revision"]).lower()
    ):
        raise ValueError(f"run manifest identity mismatch: {manifest_path}")
    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, dict) or manifest.get("config_sha256") != sha256_json(
        manifest_config
    ):
        raise ValueError(f"run manifest config hash mismatch: {manifest_path}")
    expected_scorer_config = load_yaml(plan["scorer_config"]["path"])
    expected_data_config = load_yaml(plan["data_config"]["path"])
    expected_scorer_config["training"]["seed"] = registered["seed"]
    expected_public_config = {
        "scorer": {
            key: value
            for key, value in expected_scorer_config.items()
            if not key.startswith("_")
        },
        "data": {
            key: value
            for key, value in expected_data_config.items()
            if not key.startswith("_")
        },
    }
    if manifest_config != expected_public_config:
        raise ValueError(f"run manifest config does not match the registry plan: {manifest_path}")
    scorer_training = manifest_config.get("scorer", {}).get("training", {})
    if scorer_training.get("seed") != registered["seed"]:
        raise ValueError(f"run manifest seed override was not persisted: {manifest_path}")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"run manifest inputs are missing: {manifest_path}")
    for reference_name in ("train_data", "folds_file"):
        reference = plan[reference_name]
        if inputs.get(str(Path(reference["path"]).resolve())) != reference["sha256"]:
            raise ValueError(
                f"run manifest does not bind registry {reference_name}: {manifest_path}"
            )
    recorded_history = manifest.get("history")
    if (
        not isinstance(recorded_history, str)
        or Path(recorded_history).resolve() != history_path
        or manifest.get("history_sha256") != sha256_file(history_path)
    ):
        raise ValueError(f"run history hash mismatch: {history_path}")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list):
        raise ValueError(f"run history must be a list: {history_path}")
    expected_epochs = int(registered["expected_epochs"])
    observed_epochs: set[int] = set()
    for row in history:
        if not isinstance(row, dict):
            raise ValueError(f"run history contains a non-object: {history_path}")
        epoch = row.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch in observed_epochs:
            raise ValueError(f"run history has an invalid/duplicate epoch: {history_path}")
        validation = row.get("validation", {}).get("macro", {})
        _finite_metric(validation.get("rmse"), field=f"history epoch {epoch} rmse")
        _finite_metric(
            validation.get("spearman"), field=f"history epoch {epoch} spearman"
        )
        observed_epochs.add(epoch)
    if observed_epochs != set(range(1, expected_epochs + 1)):
        raise ValueError(
            f"run history epochs are incomplete: expected 1..{expected_epochs}, "
            f"observed={sorted(observed_epochs)}"
        )

    assignments = load_folds(plan["folds_file"]["path"])
    expected_ids = {
        record_id for record_id, fold in assignments.items() if fold == registered["fold"]
    }
    if not expected_ids:
        raise ValueError(f"registry task fold has no held-out ids: {registered['fold']}")
    for epoch in range(1, expected_epochs + 1):
        _validate_checkpoint(
            output_dir / f"epoch_{epoch}",
            task=registered,
            epoch=epoch,
            model_revision=plan["model_revision"],
            expected_ids=expected_ids,
            train_sha256=plan["train_data"]["sha256"],
            folds_sha256=plan["folds_file"]["sha256"],
        )
    selected = output_dir / f"epoch_{registered['fixed_epoch']}"
    selected_fingerprint = checkpoint_fingerprint(selected)
    if registered["status"] == "completed" and (
        registered.get("selected_checkpoint") != str(selected)
        or registered.get("selected_checkpoint_fingerprint") != selected_fingerprint
    ):
        raise ValueError(
            f"completed registry task does not bind its fixed-epoch checkpoint: {task_id}"
        )
    return {
        "task_id": registered["task_id"],
        "output_dir": str(output_dir),
        "selected_checkpoint": str(selected),
        "selected_checkpoint_fingerprint": selected_fingerprint,
        "valid": True,
    }


def validate_registry_artifacts(
    registry: Mapping[str, Any], *, require_complete: bool = False
) -> dict[str, Any]:
    payload = validate_run_registry(registry)
    input_error: str | None = None
    try:
        validate_registry_inputs(payload)
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as error:
        input_error = str(error)

    task_reports: list[dict[str, Any]] = []
    for task in payload["tasks"]:
        output_dir = Path(task["output_dir"])
        if not output_dir.exists():
            task_reports.append(
                {
                    "task_id": task["task_id"],
                    "status": task["status"],
                    "artifact_state": "missing",
                    "valid": False,
                    "error": None,
                }
            )
            continue
        try:
            report = validate_task_output(payload, task)
        except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as error:
            task_reports.append(
                {
                    "task_id": task["task_id"],
                    "status": task["status"],
                    "artifact_state": "invalid_or_incomplete",
                    "valid": False,
                    "error": str(error),
                }
            )
        else:
            task_reports.append(
                {
                    **report,
                    "status": task["status"],
                    "artifact_state": "complete",
                    "status_matches_artifact": task["status"] == "completed",
                }
            )

    valid_count = sum(bool(report["valid"]) for report in task_reports)
    missing_or_invalid = len(task_reports) - valid_count
    status_errors = [
        report
        for report in task_reports
        if (report["status"] == "completed") != bool(report["valid"])
    ]
    overall_valid = input_error is None and not status_errors
    if require_complete:
        overall_valid = overall_valid and missing_or_invalid == 0
    return {
        "valid": overall_valid,
        "require_complete": require_complete,
        "input_error": input_error,
        "tasks": task_reports,
        "summary": {
            "total": len(task_reports),
            "valid_artifacts": valid_count,
            "missing_or_invalid": missing_or_invalid,
            "status_mismatches": len(status_errors),
        },
    }


def archive_incomplete_output(task: Mapping[str, Any]) -> Path:
    """Move, never delete, one partial task directory before an explicit retry."""

    output = Path(task["output_dir"]).resolve()
    if not output.exists():
        raise FileNotFoundError(output)
    if not output.is_dir():
        raise ValueError(f"task output is not a directory: {output}")
    attempt = _nonnegative_int(task.get("attempts"), field="attempts")
    sequence = max(1, attempt)
    while True:
        candidate = output.with_name(
            f"{output.name}.incomplete.attempt_{sequence}"
        )
        if not candidate.exists():
            output.rename(candidate)
            return candidate
        sequence += 1


__all__ = [
    "RUN_REGISTRY_TYPE",
    "TASK_STATUSES",
    "archive_incomplete_output",
    "build_run_registry",
    "load_run_registry",
    "validate_registry_artifacts",
    "validate_registry_inputs",
    "validate_run_registry",
    "validate_task_output",
    "write_run_registry",
]
