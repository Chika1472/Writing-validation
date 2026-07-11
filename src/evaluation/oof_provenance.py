"""Tamper-evident provenance contract for out-of-fold prediction artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.hashing import sha256_file, sha256_json


OOF_ARTIFACT_TYPE = "out_of_fold_predictions"


def oof_manifest_path(prediction_path: str | Path) -> Path:
    return Path(prediction_path).resolve().with_suffix(".manifest.json")


def oof_provenance_fields(
    *,
    prediction_path: str | Path,
    gold_path: str | Path,
    fold_path: str | Path,
    rows: int,
    scorer_name: str,
    scorer_signature: str,
    oof_level: str = "base_model_oof",
) -> dict[str, Any]:
    prediction = Path(prediction_path).resolve()
    gold = Path(gold_path).resolve()
    folds = Path(fold_path).resolve()
    return {
        "artifact_type": OOF_ARTIFACT_TYPE,
        "oof_prediction": str(prediction),
        "oof_sha256": sha256_file(prediction),
        "gold": str(gold),
        "gold_sha256": sha256_file(gold),
        "folds": str(folds),
        "folds_sha256": sha256_file(folds),
        "rows": int(rows),
        "scorer_name": str(scorer_name),
        "scorer_signature": str(scorer_signature),
        "oof_level": str(oof_level),
    }


def checkpoint_fingerprint(checkpoint_dir: str | Path) -> str:
    """Hash every adapter, tokenizer, and scoring-head file in one checkpoint."""

    checkpoint = Path(checkpoint_dir).resolve()
    fixed = (
        checkpoint / "scoring_heads.pt",
        checkpoint / "scoring_head_config.json",
        checkpoint / "checkpoint_provenance.json",
    )
    files = list(fixed)
    for directory_name in ("adapter", "tokenizer"):
        directory = checkpoint / directory_name
        files.extend(path for path in directory.rglob("*") if path.is_file())
    missing = [str(path) for path in fixed if not path.is_file()]
    if not (checkpoint / "adapter" / "adapter_config.json").is_file():
        missing.append(str(checkpoint / "adapter" / "adapter_config.json"))
    adapter_weights = (
        checkpoint / "adapter" / "adapter_model.safetensors",
        checkpoint / "adapter" / "adapter_model.bin",
    )
    if not any(path.is_file() for path in adapter_weights):
        missing.append(f"one of {[str(path) for path in adapter_weights]}")
    if not (checkpoint / "tokenizer" / "tokenizer_config.json").is_file():
        missing.append(str(checkpoint / "tokenizer" / "tokenizer_config.json"))
    if missing:
        raise FileNotFoundError(
            f"cannot fingerprint incomplete scorer checkpoint {checkpoint}; missing={missing}"
        )
    hashes = {
        path.relative_to(checkpoint).as_posix(): sha256_file(path)
        for path in sorted(set(files))
    }
    return sha256_json(hashes)


def checkpoint_ensemble_signature(
    checkpoint_dirs: list[str | Path] | tuple[str | Path, ...],
    *,
    precision: str,
) -> str:
    if precision not in {"4bit", "bf16"}:
        raise ValueError("precision must be '4bit' or 'bf16'")
    fingerprints = sorted(checkpoint_fingerprint(path) for path in checkpoint_dirs)
    if not fingerprints:
        raise ValueError("at least one checkpoint is required for an ensemble signature")
    return sha256_json({"checkpoint_fingerprints": fingerprints, "precision": precision})


def checkpoint_set_signature(
    checkpoint_dirs: list[str | Path] | tuple[str | Path, ...],
) -> str:
    """Identify the exact fold checkpoints independently of inference precision."""

    fingerprints = sorted(checkpoint_fingerprint(path) for path in checkpoint_dirs)
    if not fingerprints:
        raise ValueError("at least one checkpoint is required for a checkpoint-set signature")
    return sha256_json({"checkpoint_fingerprints": fingerprints})


def validate_oof_provenance(
    *,
    prediction_path: str | Path,
    gold_path: str | Path,
    fold_path: str | Path,
) -> dict[str, Any]:
    """Verify that an OOF sidecar binds the exact prediction, gold, and folds."""

    prediction = Path(prediction_path).resolve()
    gold = Path(gold_path).resolve()
    folds = Path(fold_path).resolve()
    manifest_path = oof_manifest_path(prediction)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"OOF provenance sidecar is required: {manifest_path}. "
            "Create the artifact with run_baselines.py or build_oof.py."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"OOF manifest must contain a JSON object: {manifest_path}")
    if payload.get("artifact_type") != OOF_ARTIFACT_TYPE:
        raise ValueError(f"not a recognized OOF artifact manifest: {manifest_path}")

    expected_paths = {
        "oof_prediction": prediction,
        "gold": gold,
        "folds": folds,
    }
    for field, expected in expected_paths.items():
        recorded = payload.get(field)
        if not isinstance(recorded, str) or Path(recorded).resolve() != expected:
            raise ValueError(
                f"OOF provenance {field} mismatch: expected {expected}, got {recorded!r}"
            )
    expected_hashes = {
        "oof_sha256": sha256_file(prediction),
        "gold_sha256": sha256_file(gold),
        "folds_sha256": sha256_file(folds),
    }
    for field, expected in expected_hashes.items():
        if payload.get(field) != expected:
            raise ValueError(f"OOF provenance hash mismatch for {field}")
    rows = payload.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError("OOF provenance rows must be a positive integer")
    for field in ("scorer_name", "scorer_signature"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"OOF provenance {field} must be a nonempty string")
    if payload.get("oof_level") not in {
        "base_model_oof",
        "level1_meta_crossfit_not_fully_nested",
    }:
        raise ValueError("OOF provenance has an unknown oof_level")
    return payload
