"""Strict storage contract for scorer hidden-state embedding artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.hashing import sha256_file, sha256_json


EMBEDDING_ARTIFACT_TYPE = "qwen_scorer_shared_hidden_embeddings"


def embedding_extraction_contract_sha256() -> str:
    """Bind every project source file that defines the shared-hidden space."""

    project_root = Path(__file__).resolve().parents[2]
    return sha256_json(
        {
            "embedding_contract": sha256_file(
                project_root / "src" / "anchors" / "embeddings.py"
            ),
            "scorer_inference": sha256_file(
                project_root / "src" / "inference" / "scorer.py"
            ),
            "scorer_model": sha256_file(
                project_root / "src" / "models" / "qwen_scorer.py"
            ),
            "ordinal_heads": sha256_file(
                project_root / "src" / "models" / "ordinal_heads.py"
            ),
            "model_contract": sha256_file(
                project_root / "src" / "models" / "contracts.py"
            ),
            "inference_dataset": sha256_file(
                project_root / "src" / "inference" / "dataset.py"
            ),
            "scoring_prompt": sha256_file(
                project_root / "src" / "train" / "prompting.py"
            ),
            "data_schema": sha256_file(
                project_root / "src" / "data" / "schema.py"
            ),
            "representation": "shared_hidden_post_projection",
        }
    )


def embedding_manifest_path(path: str | Path) -> Path:
    return Path(path).resolve().with_suffix(".manifest.json")


@dataclass(frozen=True)
class EmbeddingMatrix:
    ids: np.ndarray
    prompt_hashes: np.ndarray
    embeddings: np.ndarray
    manifest: dict[str, Any]


def save_embedding_matrix(
    path: str | Path,
    *,
    ids: list[str],
    prompt_hashes: list[str],
    embeddings: np.ndarray,
) -> Path:
    target = Path(path).resolve()
    if target.suffix.lower() != ".npz":
        raise ValueError("embedding output must use the .npz suffix")
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids):
        raise ValueError("embeddings must have shape (len(ids), hidden_size)")
    if len(ids) != len(prompt_hashes) or len(set(ids)) != len(ids):
        raise ValueError("ids/prompt_hashes must align and ids must be unique")
    if not ids or not np.isfinite(matrix).all():
        raise ValueError("embedding matrix must be non-empty and finite")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("zero-norm embeddings are not valid anchor features")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        ids=np.asarray(ids, dtype=str),
        prompt_hashes=np.asarray(prompt_hashes, dtype=str),
        embeddings=matrix.astype(np.float16),
    )
    return target


def load_embedding_matrix(
    path: str | Path,
    *,
    expected_input_sha256: str | None = None,
    expected_scorer_signature: str | None = None,
) -> EmbeddingMatrix:
    source = Path(path).resolve()
    manifest_path = embedding_manifest_path(source)
    if not source.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"embedding artifact or manifest is missing: {source}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("artifact_type") != EMBEDDING_ARTIFACT_TYPE:
        raise ValueError(f"invalid embedding manifest type: {manifest_path}")
    if manifest.get("embedding_sha256") != sha256_file(source):
        raise ValueError(f"embedding hash mismatch: {source}")
    if manifest.get("embedding_role") not in {"reference", "query"}:
        raise ValueError("embedding manifest has an invalid role")
    contract = manifest.get("extraction_contract_sha256")
    if not isinstance(contract, str) or re.fullmatch(r"[0-9a-f]{64}", contract) is None:
        raise ValueError("embedding manifest has no valid extraction contract hash")
    if manifest.get("prompt_identity") != "sha256_exact_prompt_text":
        raise ValueError("embedding prompt identity contract changed")
    manifest_config = manifest.get("config")
    if (
        not isinstance(manifest_config, dict)
        or manifest_config.get("representation") != "shared_hidden_post_projection"
        or manifest_config.get("embedding_role") != manifest.get("embedding_role")
        or manifest_config.get("extraction_contract_sha256") != contract
        or manifest_config.get("precision") != manifest.get("precision")
    ):
        raise ValueError("embedding config and manifest contract differ")
    if expected_input_sha256 and manifest.get("input_sha256") != expected_input_sha256:
        raise ValueError("embedding artifact belongs to a different essay input")
    if (
        expected_scorer_signature
        and manifest.get("scorer_signature") != expected_scorer_signature
    ):
        raise ValueError("embedding artifact belongs to a different scorer checkpoint")
    with np.load(source, allow_pickle=False) as payload:
        if set(payload.files) != {"ids", "prompt_hashes", "embeddings"}:
            raise ValueError(f"unexpected embedding NPZ schema: {payload.files}")
        ids = np.asarray(payload["ids"], dtype=str)
        prompt_hashes = np.asarray(payload["prompt_hashes"], dtype=str)
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if ids.ndim != 1 or prompt_hashes.shape != ids.shape:
        raise ValueError("embedding ids and prompt hashes must be aligned vectors")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ids):
        raise ValueError("embedding matrix row count does not match ids")
    if len(ids) == 0 or len(set(ids.tolist())) != len(ids):
        raise ValueError("embedding artifact must have unique, non-empty ids")
    if not np.isfinite(embeddings).all() or np.any(np.linalg.norm(embeddings, axis=1) <= 0):
        raise ValueError("embedding artifact contains invalid vectors")
    if int(manifest.get("rows", -1)) != len(ids):
        raise ValueError("embedding row count disagrees with manifest")
    if int(manifest.get("hidden_size", -1)) != embeddings.shape[1]:
        raise ValueError("embedding hidden size disagrees with manifest")
    return EmbeddingMatrix(ids, prompt_hashes, embeddings, manifest)
