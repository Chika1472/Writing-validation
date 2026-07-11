"""Portable provenance contract for non-OOF scorer prediction files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.hashing import sha256_file


PREDICTION_ARTIFACT_TYPE = "scorer_predictions"


def prediction_manifest_path(prediction_path: str | Path) -> Path:
    return Path(prediction_path).resolve().with_suffix(".manifest.json")


def prediction_provenance_fields(
    *,
    prediction_path: str | Path,
    input_path: str | Path,
    rows: int,
    scorer_name: str,
    scorer_signature: str,
    model_artifact: str | Path | None = None,
) -> dict[str, Any]:
    prediction = Path(prediction_path).resolve()
    source = Path(input_path).resolve()
    payload: dict[str, Any] = {
        "artifact_type": PREDICTION_ARTIFACT_TYPE,
        "prediction_file": prediction.name,
        "prediction_creation_path": str(prediction),
        "prediction_sha256": sha256_file(prediction),
        "input_creation_path": str(source),
        "input_sha256": sha256_file(source),
        "rows": int(rows),
        "scorer_name": str(scorer_name),
        "scorer_signature": str(scorer_signature),
    }
    if model_artifact is not None:
        artifact = Path(model_artifact).resolve()
        payload.update(
            {
                "model_artifact_file": artifact.name,
                "model_artifact_creation_path": str(artifact),
                "model_artifact_sha256": sha256_file(artifact),
            }
        )
    return payload


def validate_prediction_provenance(
    prediction_path: str | Path,
    *,
    expected_scorer_name: str | None = None,
    expected_scorer_signature: str | None = None,
) -> dict[str, Any]:
    prediction = Path(prediction_path).resolve()
    manifest_path = prediction_manifest_path(prediction)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prediction manifest is required: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("artifact_type") != PREDICTION_ARTIFACT_TYPE:
        raise ValueError(f"not a scorer prediction manifest: {manifest_path}")
    if (
        payload.get("prediction_file") != prediction.name
        or payload.get("prediction_sha256") != sha256_file(prediction)
    ):
        raise ValueError("prediction file does not match its adjacent manifest")
    rows = payload.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError("prediction manifest rows must be a positive integer")
    for field in ("scorer_name", "scorer_signature"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"prediction manifest {field} must be nonempty")
    if expected_scorer_name is not None and payload["scorer_name"] != expected_scorer_name:
        raise ValueError("prediction scorer_name mismatch")
    if (
        expected_scorer_signature is not None
        and payload["scorer_signature"] != expected_scorer_signature
    ):
        raise ValueError("prediction scorer_signature mismatch")
    return payload
