"""Numeric evaluation for the three essay-scoring traits.

The public helpers deliberately accept both mappings and record-like objects.  This
keeps the evaluator independent from the concrete dataset dataclass and also makes
it usable on JSONL records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


TRAITS: tuple[str, ...] = ("content", "organization", "expression")
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.25, 0.5, 1.0)
_NO_DEFAULT = object()
_MISSING = object()


def get_field(record: Any, name: str, default: Any = _NO_DEFAULT) -> Any:
    """Read *name* from a mapping or an attribute-bearing object."""

    if isinstance(record, Mapping):
        if name in record:
            return record[name]
    elif hasattr(record, name):
        return getattr(record, name)
    if default is _NO_DEFAULT:
        raise KeyError(f"missing field {name!r}")
    return default


def _trait_value(container: Any, trait: str) -> float:
    value = get_field(container, trait)
    if isinstance(value, Mapping) or hasattr(value, "score"):
        value = get_field(value, "score")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{trait} score is not numeric: {value!r}") from exc
    if not np.isfinite(number):
        raise ValueError(f"{trait} score must be finite, got {number!r}")
    return number


def extract_scores(record: Any) -> np.ndarray:
    """Extract gold scores in ``TRAITS`` order.

    Accepted score containers are ``scores`` (the canonical in-memory API),
    ``score`` (the supplied dataset JSON), and ``gold_score`` (legacy artifacts).
    A mapping/object whose top level directly contains all traits is also accepted.
    """

    container = get_field(record, "scores", _MISSING)
    if container is _MISSING:
        container = get_field(record, "score", _MISSING)
    if container is _MISSING:
        container = get_field(record, "gold_score", _MISSING)
    if container is _MISSING:
        if all(
            (trait in record if isinstance(record, Mapping) else hasattr(record, trait))
            for trait in TRAITS
        ):
            container = record
        else:
            raise KeyError("record has no scores, score, or gold_score field")
    return np.asarray([_trait_value(container, trait) for trait in TRAITS], dtype=float)


def extract_prediction(record: Any) -> np.ndarray:
    """Extract a prediction vector from canonical or legacy nested records."""

    if isinstance(record, np.ndarray):
        vector = np.asarray(record, dtype=float)
        if vector.shape != (len(TRAITS),):
            raise ValueError(f"prediction vector must have shape (3,), got {vector.shape}")
        if not np.isfinite(vector).all():
            raise ValueError("prediction vector contains a non-finite value")
        return vector

    if isinstance(record, Sequence) and not isinstance(record, (str, bytes, Mapping)):
        vector = np.asarray(record, dtype=float)
        if vector.shape == (len(TRAITS),):
            if not np.isfinite(vector).all():
                raise ValueError("prediction vector contains a non-finite value")
            return vector

    container = get_field(record, "prediction", _MISSING)
    if container is _MISSING:
        container = get_field(record, "parsed", _MISSING)
    if container is _MISSING:
        container = get_field(record, "scores", _MISSING)
    if container is _MISSING:
        if all(
            (trait in record if isinstance(record, Mapping) else hasattr(record, trait))
            for trait in TRAITS
        ):
            container = record
        else:
            raise KeyError("record has no prediction/parsed score container")
    return np.asarray([_trait_value(container, trait) for trait in TRAITS], dtype=float)


def gold_matrix(records: Sequence[Any]) -> np.ndarray:
    """Return an ``(n, 3)`` gold-score matrix."""

    rows = [extract_scores(record) for record in records]
    return np.vstack(rows) if rows else np.empty((0, len(TRAITS)), dtype=float)


def prediction_matrix(predictions: Any, gold_records: Sequence[Any] | None = None) -> np.ndarray:
    """Return predictions as an ``(n, 3)`` array, aligning by id when possible."""

    if isinstance(predictions, np.ndarray):
        matrix = np.asarray(predictions, dtype=float)
        if matrix.ndim == 1 and matrix.shape[0] == len(TRAITS):
            matrix = matrix.reshape(1, -1)
        _validate_matrix(matrix, "predictions")
        return matrix

    rows = list(predictions)
    if gold_records is not None and rows:
        pred_ids = [get_field(row, "id", None) for row in rows]
        gold_ids = [get_field(row, "id", None) for row in gold_records]
        if all(value is not None for value in pred_ids) and all(value is not None for value in gold_ids):
            if len(set(pred_ids)) != len(pred_ids):
                raise ValueError("prediction ids must be unique")
            by_id = dict(zip(pred_ids, rows, strict=True))
            missing = [value for value in gold_ids if value not in by_id]
            extra = sorted(set(pred_ids).difference(gold_ids))
            if missing or extra:
                raise ValueError(f"prediction id mismatch: missing={missing[:5]}, extra={extra[:5]}")
            rows = [by_id[value] for value in gold_ids]
    matrix = np.vstack([extract_prediction(row) for row in rows]) if rows else np.empty((0, 3))
    _validate_matrix(matrix, "predictions")
    return matrix


def _validate_matrix(matrix: np.ndarray, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[1] != len(TRAITS):
        raise ValueError(f"{name} must have shape (n, 3), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains a non-finite value")


def _safe_correlation(y_true: np.ndarray, y_pred: np.ndarray, kind: str) -> float:
    # A constant input has no mathematical correlation.  Returning zero is more
    # useful for automated reports and mirrors the score assigned to a model with
    # no ranking signal.
    if len(y_true) < 2 or np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return 0.0
    result = pearsonr(y_true, y_pred) if kind == "pearson" else spearmanr(y_true, y_pred)
    value = float(result.statistic)
    return value if np.isfinite(value) else 0.0


def regression_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Compute all required metrics for one trait.

    Standard deviations use the sample convention (``ddof=1``), matching pandas
    diagnostics in the strategy document.  Correlations are tie-aware through
    SciPy; undefined constant-vector correlations are reported as ``0.0``.
    """

    truth = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true={truth.shape}, y_pred={pred.shape}")
    if truth.size == 0:
        raise ValueError("metrics require at least one observation")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("metrics require finite values")

    error = pred - truth
    absolute_error = np.abs(error)
    within = {
        _threshold_label(threshold): float(np.mean(absolute_error <= float(threshold)))
        for threshold in thresholds
    }
    result: dict[str, Any] = {
        "n": int(truth.size),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(absolute_error)),
        "pearson": _safe_correlation(truth, pred, "pearson"),
        "spearman": _safe_correlation(truth, pred, "spearman"),
        "bias": float(np.mean(error)),
        "pred_mean": float(np.mean(pred)),
        "gold_mean": float(np.mean(truth)),
        "pred_sd": float(np.std(pred, ddof=1)) if truth.size > 1 else 0.0,
        "gold_sd": float(np.std(truth, ddof=1)) if truth.size > 1 else 0.0,
        "within": within,
    }
    # Flat aliases are convenient in CSV reports and stable for callers that do
    # not want to unpack the nested threshold mapping.
    result.update({f"within_{label}": value for label, value in within.items()})
    return result


def _threshold_label(value: float) -> str:
    return str(float(value))


def evaluate_predictions(
    gold_records: Sequence[Any],
    predictions: Any,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Evaluate three-trait predictions, returning trait, macro, and micro metrics."""

    gold_rows = list(gold_records)
    truth = gold_matrix(gold_rows)
    pred = prediction_matrix(predictions, gold_rows)
    if truth.shape != pred.shape:
        raise ValueError(f"row count mismatch: gold={truth.shape[0]}, predictions={pred.shape[0]}")
    if truth.shape[0] == 0:
        raise ValueError("evaluation requires at least one record")

    by_trait = {
        trait: regression_metrics(truth[:, column], pred[:, column], thresholds=thresholds)
        for column, trait in enumerate(TRAITS)
    }
    scalar_fields = (
        "rmse",
        "mae",
        "pearson",
        "spearman",
        "bias",
        "pred_mean",
        "gold_mean",
        "pred_sd",
        "gold_sd",
    )
    macro: dict[str, Any] = {
        field: float(np.mean([by_trait[trait][field] for trait in TRAITS]))
        for field in scalar_fields
    }
    macro["n"] = int(truth.shape[0])
    macro["within"] = {
        _threshold_label(threshold): float(
            np.mean(
                [
                    by_trait[trait]["within"][_threshold_label(threshold)]
                    for trait in TRAITS
                ]
            )
        )
        for threshold in thresholds
    }
    macro.update({f"within_{label}": value for label, value in macro["within"].items()})

    return {
        "n": int(truth.shape[0]),
        "traits": by_trait,
        "macro": macro,
        "micro": regression_metrics(truth.ravel(), pred.ravel(), thresholds=thresholds),
    }


# Concise aliases used by experiment scripts.
compute_metrics = evaluate_predictions
compute_regression_metrics = regression_metrics


__all__ = [
    "DEFAULT_THRESHOLDS",
    "TRAITS",
    "compute_metrics",
    "compute_regression_metrics",
    "evaluate_predictions",
    "extract_prediction",
    "extract_scores",
    "get_field",
    "gold_matrix",
    "prediction_matrix",
    "regression_metrics",
]
