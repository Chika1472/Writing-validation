"""Immutable multi-checkpoint anchor-bank artifact used for deployment inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.utils.hashing import sha256_file, sha256_json


ANCHOR_BANK_ARTIFACT_TYPE = "prompt_aware_knn_anchor_bank"


def anchor_manifest_path(path: str | Path) -> Path:
    return Path(path).resolve().with_suffix(".manifest.json")


def anchor_scorer_signature(
    *,
    anchor_bank_sha256: str,
    k: int,
    temperature: float,
    unknown_prompt_policy: str,
    folds: list[int],
    embedding_contracts: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    payload = {
        "artifact_version": 1,
        "method": "prompt_aware_cosine_knn",
        "anchor_bank_sha256": anchor_bank_sha256,
        "k": int(k),
        "temperature": float(temperature),
        "unknown_prompt_policy": str(unknown_prompt_policy),
        "folds": [int(value) for value in folds],
        "embedding_contracts": dict(embedding_contracts),
        "prompt_identity": "sha256_exact_prompt_text",
        "representation": "shared_hidden_post_projection",
        "aggregation": "arithmetic_mean_across_fold_embedding_spaces",
    }
    return sha256_json(payload), payload


@dataclass(frozen=True)
class AnchorBank:
    ids: np.ndarray
    prompt_hashes: np.ndarray
    scores: np.ndarray
    embeddings_by_fold: dict[int, np.ndarray]
    manifest: dict[str, Any]


def save_anchor_bank(
    path: str | Path,
    *,
    ids: list[str],
    prompt_hashes: list[str],
    scores: np.ndarray,
    embeddings_by_fold: Mapping[int, np.ndarray],
) -> Path:
    target = Path(path).resolve()
    if target.suffix.lower() != ".npz":
        raise ValueError("anchor bank must use the .npz suffix")
    score_matrix = np.asarray(scores, dtype=np.float32)
    if score_matrix.shape != (len(ids), 3):
        raise ValueError("anchor scores must have shape (len(ids), 3)")
    if len(ids) == 0 or len(ids) != len(prompt_hashes) or len(set(ids)) != len(ids):
        raise ValueError("anchor metadata must be aligned, unique, and non-empty")
    folds = sorted(embeddings_by_fold)
    if not folds:
        raise ValueError("at least one fold embedding matrix is required")
    arrays: dict[str, np.ndarray] = {
        "folds": np.asarray(folds, dtype=np.int16),
        "ids": np.asarray(ids, dtype=str),
        "prompt_hashes": np.asarray(prompt_hashes, dtype=str),
        "scores": score_matrix,
    }
    hidden_size: int | None = None
    for fold in folds:
        matrix = np.asarray(embeddings_by_fold[fold], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] <= 0:
            raise ValueError(f"fold {fold} embeddings do not align with anchor ids")
        hidden_size = matrix.shape[1] if hidden_size is None else hidden_size
        if (
            matrix.shape[1] != hidden_size
            or not np.isfinite(matrix).all()
            or np.any(np.linalg.norm(matrix, axis=1) <= 0.0)
        ):
            raise ValueError("all fold embeddings must share a finite hidden size")
        arrays[f"embeddings_fold_{fold}"] = matrix.astype(np.float16)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    return target


def load_anchor_bank(path: str | Path) -> AnchorBank:
    source = Path(path).resolve()
    manifest_path = anchor_manifest_path(source)
    if not source.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"anchor bank or manifest missing: {source}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("artifact_type") != ANCHOR_BANK_ARTIFACT_TYPE:
        raise ValueError(f"invalid anchor bank manifest: {manifest_path}")
    if not isinstance(manifest.get("scorer_name"), str) or not manifest["scorer_name"].strip():
        raise ValueError("anchor bank manifest has no scorer_name")
    if manifest.get("anchor_bank_sha256") != sha256_file(source):
        raise ValueError("anchor bank hash disagrees with manifest")
    signature_payload = manifest.get("anchor_signature_payload")
    if not isinstance(signature_payload, dict):
        raise ValueError("anchor bank has no scorer-signature payload")
    expected_signature, expected_payload = anchor_scorer_signature(
        anchor_bank_sha256=manifest["anchor_bank_sha256"],
        k=int(manifest.get("config", {}).get("k", 0)),
        temperature=float(manifest.get("config", {}).get("temperature", 0.0)),
        unknown_prompt_policy=str(
            manifest.get("config", {}).get("unknown_prompt_policy", "")
        ),
        folds=[int(value) for value in manifest.get("folds", [])],
        embedding_contracts=manifest.get("embedding_contracts", {}),
    )
    if signature_payload != expected_payload or manifest.get("anchor_signature") != expected_signature:
        raise ValueError("anchor bank scorer signature/config contract is invalid")
    with np.load(source, allow_pickle=False) as payload:
        folds = np.asarray(payload["folds"], dtype=int)
        expected = {"folds", "ids", "prompt_hashes", "scores"} | {
            f"embeddings_fold_{fold}" for fold in folds.tolist()
        }
        if set(payload.files) != expected:
            raise ValueError(f"unexpected anchor NPZ schema: {payload.files}")
        ids = np.asarray(payload["ids"], dtype=str)
        prompt_hashes = np.asarray(payload["prompt_hashes"], dtype=str)
        scores = np.asarray(payload["scores"], dtype=np.float32)
        embeddings = {
            int(fold): np.asarray(payload[f"embeddings_fold_{fold}"], dtype=np.float32)
            for fold in folds.tolist()
        }
    if folds.ndim != 1 or len(folds) == 0 or len(set(folds.tolist())) != len(folds):
        raise ValueError("anchor folds must be a unique non-empty vector")
    if ids.ndim != 1 or len(ids) == 0 or len(set(ids.tolist())) != len(ids):
        raise ValueError("anchor ids must be unique and non-empty")
    if prompt_hashes.shape != ids.shape or scores.shape != (len(ids), 3):
        raise ValueError("anchor metadata and scores do not align")
    if not np.isfinite(scores).all() or np.any((scores < 1.0) | (scores > 5.0)):
        raise ValueError("anchor scores must be finite and in [1, 5]")
    hidden_sizes = set()
    for fold, matrix in embeddings.items():
        if (
            matrix.ndim != 2
            or matrix.shape[0] != len(ids)
            or matrix.shape[1] <= 0
            or not np.isfinite(matrix).all()
            or np.any(np.linalg.norm(matrix, axis=1) <= 0.0)
        ):
            raise ValueError(f"invalid embeddings for fold {fold}")
        hidden_sizes.add(matrix.shape[1])
    if len(hidden_sizes) != 1:
        raise ValueError("anchor fold embeddings have inconsistent hidden sizes")
    if int(manifest.get("rows", -1)) != len(ids):
        raise ValueError("anchor row count disagrees with manifest")
    if sorted(manifest.get("folds", [])) != sorted(folds.tolist()):
        raise ValueError("anchor folds disagree with manifest")
    return AnchorBank(ids, prompt_hashes, scores, embeddings, manifest)
