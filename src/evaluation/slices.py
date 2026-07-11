"""Prompt and essay-length slice diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from .metrics import TRAITS, evaluate_predictions, get_field, gold_matrix, prediction_matrix


def _evaluate_indices(
    truth: np.ndarray, predictions: np.ndarray, indices: Sequence[int]
) -> dict[str, Any]:
    selected = np.asarray(indices, dtype=int)
    gold_records = [
        {"scores": {trait: float(truth[row, column]) for column, trait in enumerate(TRAITS)}}
        for row in selected
    ]
    return evaluate_predictions(gold_records, predictions[selected])


def prompt_slice_report(
    gold_records: Sequence[Any], predictions: Any
) -> dict[str, dict[str, Any]]:
    """Evaluate each ``prompt_num`` independently."""

    rows = list(gold_records)
    truth = gold_matrix(rows)
    pred = prediction_matrix(predictions, rows)
    if truth.shape != pred.shape:
        raise ValueError("gold and prediction row counts differ")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(rows):
        groups[str(get_field(record, "prompt_num"))].append(index)
    return {
        prompt: _evaluate_indices(truth, pred, indices)
        for prompt, indices in sorted(groups.items())
    }


def assign_length_quantiles(
    gold_records: Sequence[Any], *, n_quantiles: int = 4
) -> np.ndarray:
    """Assign deterministic, approximately balanced essay-length quantiles.

    Stable rank-based assignment guarantees useful non-empty diagnostic groups when
    at least ``n_quantiles`` records are present.  Equal lengths retain input order.
    """

    if n_quantiles < 1:
        raise ValueError("n_quantiles must be positive")
    rows = list(gold_records)
    if not rows:
        return np.empty(0, dtype=int)
    lengths = np.asarray([len(str(get_field(row, "essay", ""))) for row in rows])
    order = np.argsort(lengths, kind="stable")
    quantiles = np.empty(len(rows), dtype=int)
    quantiles[order] = np.minimum(
        (np.arange(len(rows), dtype=int) * n_quantiles) // len(rows),
        n_quantiles - 1,
    )
    return quantiles


def length_slice_report(
    gold_records: Sequence[Any],
    predictions: Any,
    *,
    n_quantiles: int = 4,
) -> dict[str, dict[str, Any]]:
    """Evaluate rank-balanced essay-length slices named ``Q1`` ... ``Qk``."""

    rows = list(gold_records)
    truth = gold_matrix(rows)
    pred = prediction_matrix(predictions, rows)
    if truth.shape != pred.shape:
        raise ValueError("gold and prediction row counts differ")
    assignments = assign_length_quantiles(rows, n_quantiles=n_quantiles)
    lengths = np.asarray([len(str(get_field(row, "essay", ""))) for row in rows])

    report: dict[str, dict[str, Any]] = {}
    for quantile in range(n_quantiles):
        indices = np.flatnonzero(assignments == quantile)
        if indices.size == 0:
            continue
        metrics = _evaluate_indices(truth, pred, indices)
        metrics["length_min"] = int(lengths[indices].min())
        metrics["length_max"] = int(lengths[indices].max())
        report[f"Q{quantile + 1}"] = metrics
    return report


def evaluation_report(
    gold_records: Sequence[Any],
    predictions: Any,
    *,
    n_length_quantiles: int = 4,
) -> dict[str, Any]:
    """Return overall metrics plus the required prompt and length slices."""

    rows = list(gold_records)
    # Materialize and align once so generators and id-reordered artifacts work.
    pred = prediction_matrix(predictions, rows)
    return {
        "overall": evaluate_predictions(rows, pred),
        "by_prompt": prompt_slice_report(rows, pred),
        "by_length_quantile": length_slice_report(
            rows, pred, n_quantiles=n_length_quantiles
        ),
    }


__all__ = [
    "assign_length_quantiles",
    "evaluation_report",
    "length_slice_report",
    "prompt_slice_report",
]
