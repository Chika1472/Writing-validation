from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.folds import load_folds
from src.orchestration.registry import (
    archive_incomplete_output,
    build_run_registry,
    load_run_registry,
    validate_registry_inputs,
    validate_task_output,
    write_run_registry,
)
from src.utils.config import load_yaml, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or sequentially execute a deterministic Qwen scorer fold/seed grid. "
            "Planning is the default; training starts only with --execute."
        )
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--config", default="configs/scorer_qlora.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--folds-file", required=True)
    parser.add_argument("--epoch-policy", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument(
        "--fold",
        type=int,
        action="append",
        default=None,
        help="Repeat to select folds; default is every fold present in --folds-file.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Experiment parent; defaults to <artifacts>/models.",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Defaults to <output-root>/<experiment-id>/registry.json.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--retry-partial",
        action="store_true",
        help=(
            "Move an invalid partial output to a numbered .incomplete directory, then "
            "rerun it. No artifact is deleted."
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def _training_command(registry: dict[str, Any], task: dict[str, Any]) -> list[str]:
    plan = registry["plan"]
    command = [
        sys.executable,
        "-m",
        plan["trainer_module"],
        "--config",
        plan["scorer_config"]["path"],
        "--data-config",
        plan["data_config"]["path"],
        "--folds",
        plan["folds_file"]["path"],
        "--fold",
        str(task["fold"]),
        "--seed",
        str(task["seed"]),
        "--run-id",
        task["run_id"],
        "--output-dir",
        task["output_dir"],
        "--model-revision",
        plan["model_revision"],
    ]
    if plan["allow_download"]:
        command.append("--allow-download")
    return command


def _set_completed(task: dict[str, Any], report: dict[str, Any]) -> None:
    task["status"] = "completed"
    task["last_error"] = None
    task["selected_checkpoint"] = report["selected_checkpoint"]
    task["selected_checkpoint_fingerprint"] = report[
        "selected_checkpoint_fingerprint"
    ]


@contextlib.contextmanager
def _exclusive_registry_lock(registry_path: Path):
    """Prevent two orchestrators from training the same task concurrently."""

    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(
            f"experiment lock already exists: {lock_path}. Another orchestrator may "
            "be active. After verifying no process is active, archive the stale lock "
            "manually before restarting."
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {"pid": os.getpid(), "host": socket.gethostname()},
                handle,
                ensure_ascii=False,
            )
            handle.write("\n")
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _registry_for_args(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    scorer_config_path = Path(args.config).resolve()
    data_config_path = Path(args.data_config).resolve()
    scorer_config = load_yaml(scorer_config_path)
    data_config = load_yaml(data_config_path)
    folds_path = Path(args.folds_file).resolve()
    assignments = load_folds(folds_path)
    selected_folds = args.fold or sorted(set(assignments.values()))
    artifacts = resolve_project_path(data_config, data_config["paths"]["artifacts"])
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root is not None
        else artifacts / "models"
    )
    registry_path = (
        Path(args.registry).resolve()
        if args.registry is not None
        else output_root / args.experiment_id / "registry.json"
    )
    revision = args.model_revision or scorer_config["model"].get("revision")
    if not isinstance(revision, str):
        raise ValueError(
            "a pinned model revision is required via --model-revision or model.revision"
        )
    epochs = int(scorer_config["training"]["epochs"])
    precision = (
        "4bit" if bool(scorer_config["quantization"]["load_in_4bit"]) else "bf16"
    )
    planned = build_run_registry(
        experiment_id=args.experiment_id,
        project_root=data_config["_project_root"],
        output_root=output_root,
        scorer_config_path=scorer_config_path,
        data_config_path=data_config_path,
        fold_path=folds_path,
        train_path=resolve_project_path(data_config, data_config["paths"]["train"]),
        epoch_policy_path=args.epoch_policy,
        source_files=[
            data_config["_project_root"] / relative
            for relative in (
                "pyproject.toml",
                "scripts/orchestrate_scorer_training.py",
                "src/data/folds.py",
                "src/data/load.py",
                "src/data/normalize.py",
                "src/data/schema.py",
                "src/evaluation/metrics.py",
                "src/evaluation/predictions.py",
                "src/inference/serializer.py",
                "src/models/losses.py",
                "src/models/ordinal_heads.py",
                "src/models/qwen_scorer.py",
                "src/orchestration/epoch_policy.py",
                "src/orchestration/registry.py",
                "src/train/dataset.py",
                "src/train/prompting.py",
                "src/train/train_scorer.py",
                "src/utils/config.py",
                "src/utils/hashing.py",
                "src/utils/manifest.py",
                "src/utils/reproducibility.py",
            )
        ],
        model_revision=revision,
        epochs=epochs,
        folds=selected_folds,
        seeds=args.seed,
        precision=precision,
        allow_download=args.allow_download,
    )
    if registry_path.exists():
        existing = load_run_registry(registry_path)
        if existing["registry_signature"] != planned["registry_signature"]:
            raise ValueError(
                "existing registry has a different immutable experiment plan; use a "
                "new experiment-id instead of mutating an experiment in place"
            )
        return existing, registry_path
    write_run_registry(registry_path, planned, create_only=True)
    return planned, registry_path


def _execute(
    registry: dict[str, Any],
    registry_path: Path,
    *,
    retry_partial: bool,
    continue_on_error: bool,
) -> None:
    validate_registry_inputs(registry)
    project_root = Path(registry["plan"]["project_root"])
    failures: list[str] = []
    for task in registry["tasks"]:
        validate_registry_inputs(registry)
        output_dir = Path(task["output_dir"])
        if output_dir.exists():
            try:
                recovered = validate_task_output(registry, task)
            except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as error:
                if task["status"] == "completed":
                    raise RuntimeError(
                        f"completed task artifact is invalid; refusing overwrite: "
                        f"{task['task_id']}: {error}"
                    ) from error
                if not retry_partial:
                    task["status"] = "failed"
                    task["last_error"] = (
                        "partial output requires --retry-partial: " + str(error)
                    )
                    write_run_registry(registry_path, registry)
                    raise RuntimeError(task["last_error"]) from error
                archived = archive_incomplete_output(task)
                task.setdefault("archived_partial_outputs", []).append(str(archived))
                task["last_error"] = f"archived incomplete output at {archived}"
                write_run_registry(registry_path, registry)
            else:
                _set_completed(task, recovered)
                write_run_registry(registry_path, registry)
                continue
        elif task["status"] == "completed":
            raise RuntimeError(
                f"completed task output is missing; refusing silent retraining: "
                f"{task['task_id']}"
            )

        command = _training_command(registry, task)
        task["status"] = "running"
        task["attempts"] = int(task["attempts"]) + 1
        task["last_command"] = command
        write_run_registry(registry_path, registry)
        completed = subprocess.run(command, cwd=project_root, check=False)
        if completed.returncode != 0:
            message = f"trainer exited with code {completed.returncode}"
            task["status"] = "failed"
            task["last_error"] = message
            write_run_registry(registry_path, registry)
            failures.append(f"{task['task_id']}: {message}")
            if continue_on_error:
                continue
            raise RuntimeError(failures[-1])
        try:
            report = validate_task_output(registry, task)
        except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as error:
            task["status"] = "failed"
            task["last_error"] = f"post-training artifact validation failed: {error}"
            write_run_registry(registry_path, registry)
            failures.append(f"{task['task_id']}: {task['last_error']}")
            if continue_on_error:
                continue
            raise RuntimeError(failures[-1]) from error
        _set_completed(task, report)
        write_run_registry(registry_path, registry)

    if failures:
        raise RuntimeError("one or more training tasks failed: " + "; ".join(failures))


def main() -> None:
    args = parse_args()
    registry, registry_path = _registry_for_args(args)
    commands = [
        {"task_id": task["task_id"], "command": _training_command(registry, task)}
        for task in registry["tasks"]
        if task["status"] != "completed"
    ]
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "plan_only",
                    "registry": str(registry_path),
                    "fixed_epoch": registry["plan"]["fixed_epoch"],
                    "commands": commands,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    with _exclusive_registry_lock(registry_path):
        _execute(
            registry,
            registry_path,
            retry_partial=args.retry_partial,
            continue_on_error=args.continue_on_error,
        )
    print(
        json.dumps(
            {
                "mode": "executed",
                "registry": str(registry_path),
                "tasks": len(registry["tasks"]),
                "fixed_epoch": registry["plan"]["fixed_epoch"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
