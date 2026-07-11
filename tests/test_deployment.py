from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.inference.deployment import (
    load_deployment_config,
    locked_requirements,
    validate_package_manifest,
)
from src.utils.hashing import sha256_file, sha256_json


def _config(*, package_manifest: str | None = None) -> dict:
    return {
        "schema_version": "submission_inference_v2",
        "project_root": ".",
        "runtime": {
            "offline_required": True,
            "hf_home": None,
            "package_manifest": package_manifest,
            "requirements_lock": None,
            "device": "cuda:0",
            "require_cuda": True,
            "require_single_visible_gpu": True,
            "required_gpu_name_contains": "L40S",
            "expected_rows": 400,
            "batch_size": 1,
            "scorer_precision": "4bit",
            "rationale_precision": "4bit",
            "seed": 42,
            "deterministic_algorithms": True,
        },
        "scoring": {
            "mode": "qwen_ensemble",
            "qwen": {
                "scorer_name": "qwen",
                "checkpoints": ["checkpoint"],
                "model_id": "Qwen/Qwen3-14B",
                "model_revision": "a" * 40,
                "calibrator": None,
            },
            "baseline": {"artifact": None},
            "anchor": {"artifact": None},
            "assessment": {"artifact": None},
            "stacker": {
                "artifact": None,
                "source_aliases": {
                    "qwen": "qwen",
                    "baseline": None,
                    "anchor": None,
                    "assessment": None,
                },
            },
        },
        "rationale": {"mode": "deterministic", "checkpoint": None, "max_attempts": 2},
        "output": {"model_name": "submission"},
    }


def _write_config(path: Path, value: dict) -> None:
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def test_deployment_config_rejects_unknown_keys_and_unpinned_revision(tmp_path: Path) -> None:
    path = tmp_path / "inference.yaml"
    value = _config()
    value["runtime"]["unexpected"] = True
    _write_config(path, value)
    with pytest.raises(ValueError, match="keys must be exactly"):
        load_deployment_config(path)

    value = _config()
    value["scoring"]["qwen"]["model_revision"] = None
    _write_config(path, value)
    with pytest.raises(ValueError, match="model_revision"):
        load_deployment_config(path)


def test_deployment_config_accepts_explicit_multisource_aliases(tmp_path: Path) -> None:
    value = _config()
    value["scoring"]["mode"] = "qwen_multisource_stacker"
    value["scoring"]["baseline"]["artifact"] = "baseline.json"
    value["scoring"]["stacker"]["artifact"] = "stacker.json"
    value["scoring"]["stacker"]["source_aliases"]["baseline"] = "tfidf"
    path = tmp_path / "inference.yaml"
    _write_config(path, value)
    config = load_deployment_config(path)
    assert config.stacker is not None
    assert dict(config.stacker.source_aliases) == {"qwen": "qwen", "baseline": "tfidf"}

    value["scoring"]["stacker"]["source_aliases"]["anchor"] = "anchor"
    _write_config(path, value)
    with pytest.raises(ValueError, match="configured together"):
        load_deployment_config(path)

def test_requirements_lock_requires_exact_complete_pins(tmp_path: Path) -> None:
    required = (
        "accelerate",
        "bitsandbytes",
        "joblib",
        "numpy",
        "pandas",
        "peft",
        "PyYAML",
        "safetensors",
        "scikit-learn",
        "scipy",
        "torch",
        "transformers",
    )
    path = tmp_path / "requirements-lock.txt"
    path.write_text("\n".join(f"{name}==1.2.3" for name in required) + "\n", encoding="utf-8")
    assert len(locked_requirements(path)) == len(required)
    path.write_text("torch>=2.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="name==version"):
        locked_requirements(path)


def test_package_manifest_rejects_path_escape(tmp_path: Path) -> None:
    config_path = tmp_path / "inference.yaml"
    _write_config(config_path, _config(package_manifest="package.manifest.json"))
    payload = {
        "artifact_type": "submission_package_v1",
        "schema_version": 1,
        "config_file": config_path.name,
        "config_sha256": sha256_file(config_path),
        "files": {"../escape": "0" * 64},
    }
    signature_payload = {
        key: payload[key]
        for key in ("schema_version", "config_file", "config_sha256", "files")
    }
    payload["package_signature"] = sha256_json(signature_payload)
    (tmp_path / "package.manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    config = load_deployment_config(config_path)
    with pytest.raises(ValueError, match="invalid relative path|escapes package root"):
        validate_package_manifest(config)
