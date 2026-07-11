from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.inference.deployment import (
    load_deployment_config,
    locked_requirements,
    validate_dependency_lock,
    validate_package_manifest,
)
from src.inference.finalize import finalize_prediction
from src.utils.hashing import sha256_file, sha256_json


RUNTIME_PACKAGES = (
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


def _config(*, revision: str | None = None, checkpoint: str = "checkpoint") -> dict:
    return {
        "schema_version": "submission_inference_v2",
        "project_root": ".",
        "runtime": {
            "offline_required": True,
            "hf_home": None,
            "package_manifest": None,
            "requirements_lock": None,
            "device": "cpu",
            "require_cuda": False,
            "require_single_visible_gpu": False,
            "required_gpu_name_contains": None,
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
                "checkpoints": [checkpoint],
                "model_id": "Qwen/Qwen3-14B",
                "model_revision": revision if revision is not None else "a" * 40,
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
        "rationale": {
            "mode": "deterministic",
            "checkpoint": None,
            "max_attempts": 2,
        },
        "output": {"model_name": "submission"},
    }


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _signed_package_manifest(config_path: Path, extra: dict[str, str]) -> dict:
    files = {config_path.name: sha256_file(config_path), **extra}
    signature_payload = {
        "schema_version": 1,
        "config_file": config_path.name,
        "config_sha256": sha256_file(config_path),
        "files": files,
    }
    return {
        "artifact_type": "submission_package_v1",
        **signature_payload,
        "package_signature": sha256_json(signature_payload),
    }


def test_deployment_config_rejects_unknown_key(tmp_path: Path) -> None:
    value = _config()
    value["runtime"]["unexpected"] = True
    path = tmp_path / "inference.yaml"
    _write_yaml(path, value)
    with pytest.raises(ValueError, match="keys must be exactly"):
        load_deployment_config(path)


def test_deployment_config_rejects_unpinned_revision(tmp_path: Path) -> None:
    value = _config()
    value["scoring"]["qwen"]["model_revision"] = None
    path = tmp_path / "inference.yaml"
    _write_yaml(path, value)
    with pytest.raises(ValueError, match="model_revision"):
        load_deployment_config(path)


def test_deployment_config_rejects_placeholder_path(tmp_path: Path) -> None:
    path = tmp_path / "inference.yaml"
    _write_yaml(path, _config(checkpoint="artifacts/PLACEHOLDER_CHECKPOINT"))
    with pytest.raises(ValueError, match="PLACEHOLDER"):
        load_deployment_config(path)


def test_requirements_lock_accepts_only_exact_pins(tmp_path: Path) -> None:
    lock = tmp_path / "requirements-lock.txt"
    lock.write_text(
        "\n".join(f"{name}==1.2.3" for name in RUNTIME_PACKAGES) + "\n",
        encoding="utf-8",
    )
    parsed = locked_requirements(lock)
    assert parsed["torch"] == ("torch", "1.2.3")
    assert parsed["pyyaml"] == ("PyYAML", "1.2.3")

    lock.write_text(
        "\n".join(
            "torch>=1.2" if name == "torch" else f"{name}==1.2.3"
            for name in RUNTIME_PACKAGES
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="name==version"):
        locked_requirements(lock)


def test_packaged_config_requires_dependency_lock(tmp_path: Path) -> None:
    value = _config()
    value["runtime"]["package_manifest"] = "package.manifest.json"
    path = tmp_path / "inference.yaml"
    _write_yaml(path, value)
    config = load_deployment_config(path)
    with pytest.raises(ValueError, match="requirements_lock"):
        validate_dependency_lock(config)


def test_package_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    value = _config()
    value["runtime"]["package_manifest"] = "package.manifest.json"
    config_path = tmp_path / "inference.yaml"
    _write_yaml(config_path, value)
    escaped = tmp_path.parent / f"{tmp_path.name}_escaped.txt"
    escaped.write_text("outside", encoding="utf-8")
    manifest = _signed_package_manifest(
        config_path,
        {f"../{escaped.name}": sha256_file(escaped)},
    )
    (tmp_path / "package.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    config = load_deployment_config(config_path)
    with pytest.raises(ValueError, match="invalid relative path"):
        validate_package_manifest(config)


def test_package_manifest_rejects_payload_hash_tampering(tmp_path: Path) -> None:
    value = _config()
    value["runtime"]["package_manifest"] = "package.manifest.json"
    config_path = tmp_path / "inference.yaml"
    payload = tmp_path / "payload.bin"
    _write_yaml(config_path, value)
    payload.write_bytes(b"original")
    manifest = _signed_package_manifest(
        config_path,
        {payload.name: sha256_file(payload)},
    )
    (tmp_path / "package.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    config = load_deployment_config(config_path)
    assert validate_package_manifest(config) == manifest
    payload.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_package_manifest(config)


def test_package_manifest_ignores_only_python_runtime_cache(tmp_path: Path) -> None:
    value = _config()
    value["runtime"]["package_manifest"] = "package.manifest.json"
    config_path = tmp_path / "inference.yaml"
    _write_yaml(config_path, value)
    manifest = _signed_package_manifest(config_path, {})
    (tmp_path / "package.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    cache = tmp_path / "src" / "__pycache__" / "module.cpython-311.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"transient")
    config = load_deployment_config(config_path)
    assert validate_package_manifest(config) == manifest

    (tmp_path / "unsigned.txt").write_text("not a cache", encoding="utf-8")
    with pytest.raises(ValueError, match="unsigned"):
        validate_package_manifest(config)


def test_finalizer_preserves_score_floats_exactly() -> None:
    scores = {
        "content": 3.123456789012345,
        "organization": 2.500000000000001,
        "expression": 4.875000000000001,
    }
    rationales = {
        "content": "본문의 주장과 근거를 직접 확인한 내용 평가 근거입니다.",
        "organization": "글의 도입과 전개 및 결론 연결을 확인한 구성 평가 근거입니다.",
        "expression": "문장과 어휘의 명료성 및 자연스러움을 확인한 표현 평가 근거입니다.",
    }
    finalized = finalize_prediction(scores, rationales)
    assert {
        trait: finalized[trait]["score"] for trait in scores
    } == scores
