from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.orchestration.epoch_policy import (
    create_prespecified_policy,
    write_epoch_policy,
)
from src.orchestration.registry import (
    archive_incomplete_output,
    build_run_registry,
    validate_registry_artifacts,
    validate_registry_inputs,
    validate_run_registry,
    validate_task_output,
)
from src.models.contracts import SCORER_ARCHITECTURE_VERSION
from src.utils.hashing import sha256_file, sha256_json


def _registry_fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    scorer_config = tmp_path / "scorer.yaml"
    data_config = tmp_path / "data.yaml"
    folds = tmp_path / "folds.jsonl"
    train = tmp_path / "train.jsonl"
    policy = tmp_path / "epoch_policy.json"
    scorer_config.write_text(
        "model: {}\nquantization: {}\ntraining:\n  epochs: 1\n  seed: 0\nloss: {}\n",
        encoding="utf-8",
    )
    data_config.write_text("paths: {}\n", encoding="utf-8")
    folds.write_text('{"id":"essay-1","fold":0}\n', encoding="utf-8")
    train.write_text('{"id":"essay-1"}\n', encoding="utf-8")
    write_epoch_policy(
        policy,
        create_prespecified_policy(1, reason="frozen before the outer run"),
    )
    registry = build_run_registry(
        experiment_id="static-test",
        project_root=tmp_path,
        output_root=tmp_path / "models",
        scorer_config_path=scorer_config,
        data_config_path=data_config,
        fold_path=folds,
        train_path=train,
        epoch_policy_path=policy,
        source_files=[scorer_config],
        model_revision="a" * 40,
        epochs=1,
        folds=[0],
        seeds=[42],
        precision="4bit",
        allow_download=False,
    )
    return registry, folds, train


def _write_complete_output(registry: dict, folds: Path, train: Path) -> None:
    task = registry["tasks"][0]
    output = Path(task["output_dir"])
    checkpoint = output / "epoch_1"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "tokenizer").mkdir()
    (checkpoint / "adapter" / "adapter_config.json").write_text(
        "{}", encoding="utf-8"
    )
    (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "tokenizer" / "tokenizer_config.json").write_text(
        "{}", encoding="utf-8"
    )
    (checkpoint / "scoring_heads.pt").write_bytes(b"heads")
    (checkpoint / "scoring_head_config.json").write_text(
        json.dumps(
            {
                "fold": 0,
                "seed": 42,
                "model_revision": "a" * 40,
                "precision": "4bit",
                "scorer_architecture_version": SCORER_ARCHITECTURE_VERSION,
                "train_sha256": sha256_file(train),
                "folds_sha256": sha256_file(folds),
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "metrics.json").write_text(
        json.dumps({"epoch": 1}), encoding="utf-8"
    )
    oof = checkpoint / "oof.jsonl"
    oof.write_text(
        json.dumps(
            {
                "id": "essay-1",
                "prompt_num": "Q1",
                "prediction": {
                    trait: 3.0
                    for trait in ("content", "organization", "expression")
                },
                "model": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (checkpoint / "checkpoint_provenance.json").write_text(
        json.dumps(
            {
                "artifact_type": "qwen_scorer_fold_checkpoint",
                "fold": 0,
                "seed": 42,
                "epoch": 1,
                "oof_file": "oof.jsonl",
                "oof_sha256": sha256_file(oof),
                "rows": 1,
                "precision": "4bit",
                "train_sha256": sha256_file(train),
                "folds_sha256": sha256_file(folds),
                "scorer_architecture_version": SCORER_ARCHITECTURE_VERSION,
            }
        ),
        encoding="utf-8",
    )
    history = output / "history.json"
    history.write_text(
        json.dumps(
            [
                {
                    "epoch": 1,
                    "validation": {"macro": {"rmse": 0.7, "spearman": 0.6}},
                }
            ]
        ),
        encoding="utf-8",
    )
    config = {
        "scorer": {
            "model": {},
            "quantization": {},
            "training": {"epochs": 1, "seed": 42},
            "loss": {},
        },
        "data": {"paths": {}},
    }
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": task["run_id"],
                "fold": 0,
                "seed": 42,
                "model_revision": "a" * 40,
                "config": config,
                "config_sha256": sha256_json(config),
                "inputs": {
                    str(train.resolve()): sha256_file(train),
                    str(folds.resolve()): sha256_file(folds),
                },
                "history": str(history.resolve()),
                "history_sha256": sha256_file(history),
            }
        ),
        encoding="utf-8",
    )


def test_registry_signatures_reject_task_plan_mutation(tmp_path: Path) -> None:
    registry, _, _ = _registry_fixture(tmp_path)
    assert validate_run_registry(registry)["tasks"][0]["seed"] == 42
    registry["tasks"][0]["seed"] = 43
    with pytest.raises(ValueError, match="task signature mismatch"):
        validate_run_registry(registry)


def test_registry_input_hashes_are_immutable(tmp_path: Path) -> None:
    registry, _, train = _registry_fixture(tmp_path)
    validate_registry_inputs(registry)
    train.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash drifted"):
        validate_registry_inputs(registry)


def test_complete_task_output_and_selected_epoch_are_validated(tmp_path: Path) -> None:
    registry, folds, train = _registry_fixture(tmp_path)
    _write_complete_output(registry, folds, train)
    task = registry["tasks"][0]
    report = validate_task_output(registry, task)
    task["status"] = "completed"
    task["selected_checkpoint"] = report["selected_checkpoint"]
    task["selected_checkpoint_fingerprint"] = report[
        "selected_checkpoint_fingerprint"
    ]

    full = validate_registry_artifacts(registry, require_complete=True)
    assert full["valid"] is True
    assert full["summary"]["valid_artifacts"] == 1


def test_partial_retry_archives_without_deleting(tmp_path: Path) -> None:
    registry, _, _ = _registry_fixture(tmp_path)
    task = registry["tasks"][0]
    output = Path(task["output_dir"])
    output.mkdir(parents=True)
    (output / "partial.txt").write_text("keep", encoding="utf-8")

    archived = archive_incomplete_output(task)
    assert not output.exists()
    assert (archived / "partial.txt").read_text(encoding="utf-8") == "keep"
