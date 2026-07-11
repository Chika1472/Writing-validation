"""Global-mean and prompt-mean scoring baselines."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

from src.evaluation.metrics import TRAITS, get_field, gold_matrix
from src.evaluation.predictions import prediction_records


class MeanBaseline(RegressorMixin, BaseEstimator):
    """Predict training means, optionally using a prompt-specific mean.

    Unseen prompts deliberately fall back to the global training mean.
    """

    def __init__(self, *, by_prompt: bool = False) -> None:
        self.by_prompt = by_prompt

    def fit(self, records: Sequence[Any], y: Any = None) -> "MeanBaseline":
        rows = list(records)
        targets = gold_matrix(rows) if y is None else _target_matrix(y, len(rows))
        if not rows:
            raise ValueError("cannot fit a mean baseline on no records")
        self.global_mean_ = targets.mean(axis=0)
        self.prompt_means_: dict[str, np.ndarray] = {}
        if self.by_prompt:
            grouped: dict[str, list[np.ndarray]] = defaultdict(list)
            for record, target in zip(rows, targets, strict=True):
                grouped[str(get_field(record, "prompt_num"))].append(target)
            self.prompt_means_ = {
                prompt: np.mean(values, axis=0) for prompt, values in grouped.items()
            }
        self.n_features_in_ = 0
        return self

    def predict(self, records: Sequence[Any]) -> np.ndarray:
        if not hasattr(self, "global_mean_"):
            raise RuntimeError("MeanBaseline must be fitted before predict")
        rows = list(records)
        result = np.empty((len(rows), len(TRAITS)), dtype=float)
        for index, record in enumerate(rows):
            if self.by_prompt:
                prompt = str(get_field(record, "prompt_num"))
                result[index] = self.prompt_means_.get(prompt, self.global_mean_)
            else:
                result[index] = self.global_mean_
        return result

    def predict_records(
        self, records: Sequence[Any], *, model: str | None = None
    ) -> list[dict[str, Any]]:
        rows = list(records)
        name = model or ("prompt_mean" if self.by_prompt else "global_mean")
        return prediction_records(rows, self.predict(rows), model=name)


class PromptMeanBaseline(MeanBaseline):
    """Convenience class equivalent to ``MeanBaseline(by_prompt=True)``."""

    def __init__(self) -> None:
        super().__init__(by_prompt=True)


def _target_matrix(y: Any, n_rows: int) -> np.ndarray:
    targets = np.asarray(y, dtype=float)
    if targets.shape != (n_rows, len(TRAITS)):
        raise ValueError(f"targets must have shape ({n_rows}, 3), got {targets.shape}")
    if not np.isfinite(targets).all():
        raise ValueError("targets contain a non-finite value")
    return targets


def _fold_splits(
    folds: Any, n_rows: int
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    if hasattr(folds, "split"):
        yield from (
            (np.asarray(train, dtype=int), np.asarray(validation, dtype=int))
            for train, validation in folds.split(np.arange(n_rows))
        )
        return

    values = list(folds)
    if len(values) == n_rows and all(np.isscalar(value) for value in values):
        fold_ids = np.asarray(values)
        for fold_id in np.unique(fold_ids):
            validation = np.flatnonzero(fold_ids == fold_id)
            train = np.flatnonzero(fold_ids != fold_id)
            yield train, validation
        return

    for pair in values:
        if len(pair) != 2:
            raise ValueError("each fold split must contain train and validation indices")
        train, validation = pair
        yield np.asarray(train, dtype=int), np.asarray(validation, dtype=int)


def oof_predict(
    estimator: Any,
    records: Sequence[Any],
    folds: Any,
) -> np.ndarray:
    """Generate leakage-free OOF predictions while preserving original row order."""

    rows = list(records)
    if not rows:
        raise ValueError("OOF prediction requires at least one record")
    if isinstance(folds, Mapping):
        missing_ids = [
            get_field(record, "id")
            for record in rows
            if get_field(record, "id") not in folds
        ]
        if missing_ids:
            raise ValueError(f"fold mapping is missing ids: {missing_ids[:5]}")
        folds = [folds[get_field(record, "id")] for record in rows]
    predictions = np.full((len(rows), len(TRAITS)), np.nan, dtype=float)
    seen = np.zeros(len(rows), dtype=int)

    for train_indices, validation_indices in _fold_splits(folds, len(rows)):
        if train_indices.size == 0 or validation_indices.size == 0:
            raise ValueError("every fold needs non-empty train and validation sets")
        if (
            (train_indices < 0).any()
            or (validation_indices < 0).any()
            or (train_indices >= len(rows)).any()
            or (validation_indices >= len(rows)).any()
        ):
            raise ValueError("fold index is out of range")
        if np.intersect1d(train_indices, validation_indices).size:
            raise ValueError("train and validation indices overlap")
        model = clone(estimator)
        train_rows = [rows[index] for index in train_indices]
        validation_rows = [rows[index] for index in validation_indices]
        model.fit(train_rows)
        fold_predictions = np.asarray(model.predict(validation_rows), dtype=float)
        if fold_predictions.shape != (len(validation_indices), len(TRAITS)):
            raise ValueError("estimator returned an invalid prediction shape")
        predictions[validation_indices] = fold_predictions
        seen[validation_indices] += 1

    if not np.all(seen == 1):
        missing = np.flatnonzero(seen == 0).tolist()
        repeated = np.flatnonzero(seen > 1).tolist()
        raise ValueError(
            f"folds must predict every row exactly once; missing={missing[:5]}, repeated={repeated[:5]}"
        )
    return predictions


def oof_predict_with_models(
    estimator: Any,
    records: Sequence[Any],
    fold_ids: Sequence[Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create OOF predictions and retain the exact fold estimators for target ensembling."""

    rows = list(records)
    values = np.asarray(list(fold_ids), dtype=object)
    if not rows or values.shape != (len(rows),):
        raise ValueError("fold_ids must contain one value for every nonempty record row")
    predictions = np.full((len(rows), len(TRAITS)), np.nan, dtype=float)
    models: dict[str, Any] = {}
    seen = np.zeros(len(rows), dtype=int)
    for fold_id in np.unique(values):
        validation_indices = np.flatnonzero(values == fold_id)
        train_indices = np.flatnonzero(values != fold_id)
        if train_indices.size == 0 or validation_indices.size == 0:
            raise ValueError("every fold needs non-empty train and validation sets")
        model = clone(estimator)
        model.fit([rows[index] for index in train_indices])
        fold_predictions = np.asarray(
            model.predict([rows[index] for index in validation_indices]),
            dtype=float,
        )
        if fold_predictions.shape != (len(validation_indices), len(TRAITS)):
            raise ValueError("estimator returned an invalid prediction shape")
        key = str(fold_id)
        if key in models:
            raise ValueError(f"fold id string collision: {fold_id!r}")
        models[key] = model
        predictions[validation_indices] = fold_predictions
        seen[validation_indices] += 1
    if not np.all(seen == 1) or not np.isfinite(predictions).all():
        raise RuntimeError("OOF fold estimators did not predict every row exactly once")
    return predictions, models


def fit_predict(
    estimator: Any,
    train_records: Sequence[Any],
    validation_records: Sequence[Any],
) -> tuple[Any, np.ndarray]:
    """Fit a fresh clone on train and predict validation."""

    model = clone(estimator)
    model.fit(list(train_records))
    return model, np.asarray(model.predict(list(validation_records)), dtype=float)


__all__ = [
    "MeanBaseline",
    "PromptMeanBaseline",
    "fit_predict",
    "oof_predict",
    "oof_predict_with_models",
]
