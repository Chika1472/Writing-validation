from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.baselines.contracts import (
    BASELINE_INFERENCE_FILES,
    baseline_inference_code_contract,
)
from src.inference.deployment import _baseline_files
from src.utils.hashing import sha256_file, sha256_json


def _artifact(tmp_path: Path, *, filename: str = "fold0.joblib") -> Path:
    model = tmp_path / "fold0.joblib"
    model.write_bytes(b"serialized-model-placeholder")
    signature_payload = {
        "scorer": "tfidf",
        "fold_model_hashes": {"0": sha256_file(model)},
        "inference_code_contract": baseline_inference_code_contract(),
    }
    artifact = tmp_path / "tfidf_ensemble.json"
    artifact.write_text(
        json.dumps(
            {
                "artifact_type": "baseline_fold_ensemble",
                "aggregation": "equal_weight_mean",
                "scorer_name": "tfidf",
                "scorer_signature": sha256_json(signature_payload),
                "signature_payload": signature_payload,
                "fold_models": {
                    "0": {"file": filename, "sha256": sha256_file(model)}
                },
            }
        ),
        encoding="utf-8",
    )
    return artifact


def test_baseline_contract_covers_transitive_project_inference_sources() -> None:
    contract = baseline_inference_code_contract()
    assert tuple(contract) == BASELINE_INFERENCE_FILES
    assert "src/data/schema.py" in contract
    assert "src/evaluation/metrics.py" in contract
    assert all(len(value) == 64 for value in contract.values())


def test_deployment_rejects_changed_baseline_source_contract(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["signature_payload"]["inference_code_contract"][
        "src/data/schema.py"
    ] = "0" * 64
    payload["scorer_signature"] = sha256_json(payload["signature_payload"])
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source has changed"):
        _baseline_files(artifact)


def test_deployment_accepts_only_adjacent_baseline_model_names(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, filename="nested/fold0.joblib")
    with pytest.raises(ValueError, match="must not contain directories"):
        _baseline_files(artifact)
