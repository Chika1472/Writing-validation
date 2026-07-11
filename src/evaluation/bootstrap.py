"""Paired, prompt-stratified bootstrap comparisons."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata

from .metrics import TRAITS, get_field, gold_matrix, prediction_matrix


def stratified_resample_indices(
    strata: Sequence[Any], rng: np.random.Generator
) -> np.ndarray:
    """Sample with replacement inside every stratum, preserving stratum counts."""

    values = list(strata)
    groups: dict[Any, list[int]] = {}
    for index, value in enumerate(values):
        try:
            groups.setdefault(value, []).append(index)
        except TypeError as exc:
            raise ValueError("strata values must be hashable") from exc
    sampled = [
        rng.choice(np.asarray(indices), size=len(indices), replace=True)
        for indices in groups.values()
    ]
    return np.concatenate(sampled).astype(int, copy=False) if sampled else np.empty(0, dtype=int)


def bootstrap_metric_difference(
    y_true: Sequence[float] | np.ndarray,
    candidate: Sequence[float] | np.ndarray,
    baseline: Sequence[float] | np.ndarray,
    strata: Sequence[Any],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 42,
    higher_is_better: bool = False,
) -> dict[str, float | int]:
    """Bootstrap ``metric(candidate) - metric(baseline)`` on paired rows."""

    truth = np.asarray(y_true, dtype=float).reshape(-1)
    candidate_array = np.asarray(candidate, dtype=float).reshape(-1)
    baseline_array = np.asarray(baseline, dtype=float).reshape(-1)
    if truth.shape != candidate_array.shape or truth.shape != baseline_array.shape:
        raise ValueError("truth, candidate, and baseline must have identical shapes")
    _validate_bootstrap_inputs(len(truth), strata, n_resamples, confidence)

    point_candidate = float(metric(truth, candidate_array))
    point_baseline = float(metric(truth, baseline_array))
    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=float)
    for iteration in range(n_resamples):
        indices = stratified_resample_indices(strata, rng)
        differences[iteration] = float(
            metric(truth[indices], candidate_array[indices])
            - metric(truth[indices], baseline_array[indices])
        )
    return _summarize_differences(
        point_candidate,
        point_baseline,
        differences,
        confidence=confidence,
        higher_is_better=higher_is_better,
    )


def _validate_bootstrap_inputs(
    n_rows: int, strata: Sequence[Any], n_resamples: int, confidence: float
) -> None:
    if n_rows == 0:
        raise ValueError("bootstrap requires at least one row")
    if len(strata) != n_rows:
        raise ValueError("strata length must equal the number of rows")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")


def _summarize_differences(
    point_candidate: float,
    point_baseline: float,
    differences: np.ndarray,
    *,
    confidence: float,
    higher_is_better: bool,
) -> dict[str, float | int]:
    tail = (1.0 - confidence) / 2.0
    if higher_is_better:
        probability_better = float(np.mean(differences > 0.0) + 0.5 * np.mean(differences == 0.0))
    else:
        probability_better = float(np.mean(differences < 0.0) + 0.5 * np.mean(differences == 0.0))
    return {
        "candidate": point_candidate,
        "baseline": point_baseline,
        "delta": point_candidate - point_baseline,
        "bootstrap_mean_delta": float(np.mean(differences)),
        "bootstrap_sd": float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0,
        "ci_low": float(np.quantile(differences, tail)),
        "ci_high": float(np.quantile(differences, 1.0 - tail)),
        "probability_candidate_better": probability_better,
        "n_resamples": int(len(differences)),
    }


def _rmse_by_trait(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(prediction - truth), axis=0))


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return 0.0
    true_rank = rankdata(y_true, method="average")
    pred_rank = rankdata(y_pred, method="average")
    value = float(np.corrcoef(true_rank, pred_rank)[0, 1])
    return value if np.isfinite(value) else 0.0


def _spearman_by_trait(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.asarray(
        [_spearman(truth[:, column], prediction[:, column]) for column in range(len(TRAITS))]
    )


def paired_stratified_bootstrap(
    gold_records: Sequence[Any],
    candidate_predictions: Any,
    baseline_predictions: Any,
    *,
    strata: Sequence[Any] | str | Callable[[Any], Any] | None = None,
    n_resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare two three-trait systems with paired stratified resampling.

    By default rows are stratified by ``prompt_num``.  The report includes macro
    and trait-level RMSE/Spearman deltas.  Delta is always candidate minus baseline;
    therefore negative RMSE and positive Spearman deltas favor the candidate.
    """

    rows = list(gold_records)
    truth = gold_matrix(rows)
    candidate = prediction_matrix(candidate_predictions, rows)
    baseline = prediction_matrix(baseline_predictions, rows)
    if truth.shape != candidate.shape or truth.shape != baseline.shape:
        raise ValueError("gold, candidate, and baseline row counts must match")

    if strata is None:
        stratum_values = [str(get_field(row, "prompt_num")) for row in rows]
    elif isinstance(strata, str):
        stratum_values = [get_field(row, strata) for row in rows]
    elif callable(strata):
        stratum_values = [strata(row) for row in rows]
    else:
        stratum_values = list(strata)
    _validate_bootstrap_inputs(len(rows), stratum_values, n_resamples, confidence)

    candidate_rmse = _rmse_by_trait(truth, candidate)
    baseline_rmse = _rmse_by_trait(truth, baseline)
    candidate_rho = _spearman_by_trait(truth, candidate)
    baseline_rho = _spearman_by_trait(truth, baseline)

    metric_names = [f"{trait}_rmse" for trait in TRAITS]
    metric_names += [f"{trait}_spearman" for trait in TRAITS]
    metric_names += ["macro_rmse", "macro_spearman"]
    point_candidate = np.concatenate(
        [candidate_rmse, candidate_rho, [candidate_rmse.mean(), candidate_rho.mean()]]
    )
    point_baseline = np.concatenate(
        [baseline_rmse, baseline_rho, [baseline_rmse.mean(), baseline_rho.mean()]]
    )

    rng = np.random.default_rng(seed)
    differences = np.empty((n_resamples, len(metric_names)), dtype=float)
    for iteration in range(n_resamples):
        indices = stratified_resample_indices(stratum_values, rng)
        sampled_truth = truth[indices]
        sampled_candidate = candidate[indices]
        sampled_baseline = baseline[indices]
        candidate_rmse_sample = _rmse_by_trait(sampled_truth, sampled_candidate)
        baseline_rmse_sample = _rmse_by_trait(sampled_truth, sampled_baseline)
        candidate_rho_sample = _spearman_by_trait(sampled_truth, sampled_candidate)
        baseline_rho_sample = _spearman_by_trait(sampled_truth, sampled_baseline)
        differences[iteration] = np.concatenate(
            [
                candidate_rmse_sample - baseline_rmse_sample,
                candidate_rho_sample - baseline_rho_sample,
                [
                    candidate_rmse_sample.mean() - baseline_rmse_sample.mean(),
                    candidate_rho_sample.mean() - baseline_rho_sample.mean(),
                ],
            ]
        )

    summaries: dict[str, Any] = {}
    for column, name in enumerate(metric_names):
        summaries[name] = _summarize_differences(
            float(point_candidate[column]),
            float(point_baseline[column]),
            differences[:, column],
            confidence=confidence,
            higher_is_better=name.endswith("spearman"),
        )
    return {
        "n": len(rows),
        "n_resamples": n_resamples,
        "confidence": confidence,
        "seed": seed,
        "strata_counts": dict(Counter(str(value) for value in stratum_values)),
        "metrics": summaries,
    }


paired_bootstrap = paired_stratified_bootstrap


__all__ = [
    "bootstrap_metric_difference",
    "paired_bootstrap",
    "paired_stratified_bootstrap",
    "stratified_resample_indices",
]
