"""Deterministic prompt-aware cosine KNN for reference-essay scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KNNResult:
    predictions: np.ndarray
    mean_cosine_distance: np.ndarray
    neighbor_score_variance: np.ndarray
    used_global_fallback: np.ndarray
    neighbor_ids: tuple[tuple[str, ...], ...]


def _validated_matrix(value: np.ndarray, *, name: str, columns: int | None = None) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or (columns is not None and matrix.shape[1] != columns):
        expected = f"(*, {columns})" if columns is not None else "two-dimensional"
        raise ValueError(f"{name} must have shape {expected}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    return matrix


def _unit_rows(matrix: np.ndarray, *, name: str) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError(f"{name} contains a zero-norm row")
    return matrix / norms


def prompt_aware_knn_predict(
    *,
    query_ids: list[str],
    query_prompt_hashes: list[str],
    query_embeddings: np.ndarray,
    bank_ids: list[str],
    bank_prompt_hashes: list[str],
    bank_embeddings: np.ndarray,
    bank_scores: np.ndarray,
    k: int,
    temperature: float = 0.07,
    unknown_prompt_policy: str = "global",
) -> KNNResult:
    """Score queries from reference essays without ever admitting a same-ID neighbor."""

    if k <= 0 or temperature <= 0.0:
        raise ValueError("k and temperature must be positive")
    if unknown_prompt_policy not in {"global", "error"}:
        raise ValueError("unknown_prompt_policy must be 'global' or 'error'")
    query = _validated_matrix(query_embeddings, name="query_embeddings")
    bank = _validated_matrix(bank_embeddings, name="bank_embeddings")
    scores = _validated_matrix(bank_scores, name="bank_scores", columns=3)
    if query.shape[1] != bank.shape[1] or bank.shape[0] != scores.shape[0]:
        raise ValueError("query, bank, and score dimensions do not align")
    if len(query_ids) != len(query) or len(query_prompt_hashes) != len(query):
        raise ValueError("query metadata does not align with query embeddings")
    if len(bank_ids) != len(bank) or len(bank_prompt_hashes) != len(bank):
        raise ValueError("bank metadata does not align with bank embeddings")
    if not query_ids or not bank_ids or len(set(bank_ids)) != len(bank_ids):
        raise ValueError("query/bank must be non-empty and bank ids unique")
    if np.any((scores < 1.0) | (scores > 5.0)):
        raise ValueError("anchor scores must be in [1, 5]")

    query_unit = _unit_rows(query, name="query_embeddings")
    bank_unit = _unit_rows(bank, name="bank_embeddings")
    bank_ids_array = np.asarray(bank_ids, dtype=str)
    bank_prompts_array = np.asarray(bank_prompt_hashes, dtype=str)
    predictions = np.empty((len(query), 3), dtype=np.float32)
    mean_distance = np.empty(len(query), dtype=np.float32)
    score_variance = np.empty((len(query), 3), dtype=np.float32)
    global_fallback = np.zeros(len(query), dtype=bool)
    neighbors: list[tuple[str, ...]] = []

    for row_index, (record_id, prompt_hash) in enumerate(
        zip(query_ids, query_prompt_hashes, strict=True)
    ):
        eligible = (bank_prompts_array == prompt_hash) & (bank_ids_array != record_id)
        if not np.any(eligible):
            if unknown_prompt_policy == "error":
                raise ValueError(f"no same-prompt anchors for query {record_id}")
            eligible = bank_ids_array != record_id
            global_fallback[row_index] = True
        candidate_indices = np.flatnonzero(eligible)
        if len(candidate_indices) == 0:
            raise ValueError(f"anchor bank has no non-self candidate for {record_id}")
        similarities = bank_unit[candidate_indices] @ query_unit[row_index]
        # Stable lexical tie-breaking makes equal-similarity results reproducible.
        order = np.lexsort((bank_ids_array[candidate_indices], -similarities))
        selected = candidate_indices[order[: min(k, len(order))]]
        selected_similarity = bank_unit[selected] @ query_unit[row_index]
        logits = (selected_similarity - selected_similarity.max()) / temperature
        weights = np.exp(logits.astype(np.float64))
        weights /= weights.sum()
        selected_scores = scores[selected]
        predictions[row_index] = weights @ selected_scores
        mean_distance[row_index] = float(weights @ (1.0 - selected_similarity))
        centered = selected_scores - predictions[row_index]
        score_variance[row_index] = weights @ (centered * centered)
        neighbors.append(tuple(bank_ids_array[selected].tolist()))

    return KNNResult(
        predictions=np.clip(predictions, 1.0, 5.0),
        mean_cosine_distance=mean_distance,
        neighbor_score_variance=score_variance,
        used_global_fallback=global_fallback,
        neighbor_ids=tuple(neighbors),
    )
