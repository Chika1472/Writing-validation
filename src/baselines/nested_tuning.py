"""Leakage-safe nested-CV selection for the TF-IDF Ridge baseline."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from src.baselines.mean_baseline import oof_predict
from src.baselines.tfidf_ridge import TfidfRidgeBaseline
from src.data.folds import make_folds
from src.evaluation.metrics import TRAITS, get_field, gold_matrix
from src.utils.hashing import sha256_file, sha256_json


NESTED_TUNING_CODE_FILES = (
    "src/baselines/nested_tuning.py",
    "src/data/folds.py",
    "src/evaluation/metrics.py",
    "scripts/run_nested_tfidf.py",
)


@dataclass(frozen=True)
class NestedTfidfResult:
    """Complete outer OOF predictions and target-time outer-fold models."""

    oof_predictions: np.ndarray
    fold_models: dict[str, TfidfRidgeBaseline]
    outer_reports: list[dict[str, Any]]
    normalized_config: dict[str, Any]


def nested_tuning_code_contract() -> dict[str, str]:
    """Hash source files that determine inner selection and outer refitting."""

    project_root = Path(__file__).resolve().parents[2]
    return {
        relative: sha256_file(project_root / relative)
        for relative in NESTED_TUNING_CODE_FILES
    }


def _exact_keys(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping")
    keys = set(value)
    missing = sorted(expected.difference(keys))
    extra = sorted(keys.difference(expected))
    if missing or extra:
        raise ValueError(f"{where} keys mismatch; missing={missing}, extra={extra}")
    return dict(value)


def _positive_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return int(value)


def _min_df(value: Any, *, where: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be an integer count or fraction")
    number = float(value)
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"{where} integer value must be at least 1")
        return int(value)
    if not math.isfinite(number) or not 0.0 < number <= 1.0:
        raise ValueError(f"{where} fractional value must lie in (0, 1]")
    return number


def _ngram_range(value: Any, *, where: str) -> list[int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"{where} must contain exactly two integers")
    lower = _positive_int(value[0], where=f"{where}[0]")
    upper = _positive_int(value[1], where=f"{where}[1]")
    if lower > upper:
        raise ValueError(f"{where} lower bound must not exceed its upper bound")
    return [lower, upper]


def _bool(value: Any, *, where: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{where} must be boolean")
    return value


def _unique_sequence(value: Any, *, where: str) -> list[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise ValueError(f"{where} must be a nonempty sequence")
    result = list(value)
    if len({sha256_json(item) for item in result}) != len(result):
        raise ValueError(f"{where} must not contain duplicates")
    return result


def normalize_nested_tfidf_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate configuration strictly and return a JSON-serializable form."""

    if not isinstance(config, Mapping):
        raise ValueError("nested TF-IDF configuration must be a mapping")
    public = {
        key: value
        for key, value in config.items()
        if not str(key).startswith("_") and key != "project_root"
    }
    top = _exact_keys(
        public,
        {"model", "inner_cv", "search", "selection"},
        where="nested TF-IDF configuration",
    )
    model = _exact_keys(
        top["model"],
        {
            "char_ngram_range",
            "word_ngram_range",
            "char_min_df",
            "word_min_df",
            "prompt_min_df",
            "include_prompt",
            "include_prompt_text",
            "include_surface",
            "clip",
        },
        where="model",
    )
    clip = model["clip"]
    if (
        not isinstance(clip, Sequence)
        or isinstance(clip, (str, bytes))
        or len(clip) != 2
    ):
        raise ValueError("model.clip must contain exactly two finite bounds")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in clip):
        raise ValueError("model.clip values must be real numbers")
    clip_values = [float(value) for value in clip]
    if not all(math.isfinite(value) for value in clip_values) or not (
        clip_values[0] < clip_values[1]
    ):
        raise ValueError("model.clip must contain increasing finite bounds")

    inner = _exact_keys(
        top["inner_cv"],
        {"n_splits", "seed", "score_bins"},
        where="inner_cv",
    )
    n_splits = _positive_int(inner["n_splits"], where="inner_cv.n_splits")
    if n_splits < 2:
        raise ValueError("inner_cv.n_splits must be at least 2")
    if (
        isinstance(inner["seed"], bool)
        or not isinstance(inner["seed"], int)
        or inner["seed"] < 0
    ):
        raise ValueError("inner_cv.seed must be a non-negative integer")

    search = _exact_keys(
        top["search"],
        {"alpha", "max_char_features", "max_word_features"},
        where="search",
    )
    raw_alphas = _unique_sequence(search["alpha"], where="search.alpha")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_alphas):
        raise ValueError("search.alpha values must be real numbers")
    alphas = [float(value) for value in raw_alphas]
    if any(not math.isfinite(value) or value <= 0.0 for value in alphas):
        raise ValueError("search.alpha values must be finite and positive")
    if len(set(alphas)) != len(alphas):
        raise ValueError("search.alpha must not contain numeric duplicates")

    def feature_grid(name: str) -> list[int | None]:
        values = _unique_sequence(search[name], where=f"search.{name}")
        normalized: list[int | None] = []
        for index, value in enumerate(values):
            if value is None:
                normalized.append(None)
            else:
                normalized.append(
                    _positive_int(value, where=f"search.{name}[{index}]")
                )
        return normalized

    selection = _exact_keys(
        top["selection"],
        {"metric", "tie_break"},
        where="selection",
    )
    if selection["metric"] != "trait_mean_rmse":
        raise ValueError("selection.metric must be 'trait_mean_rmse'")
    if selection["tie_break"] != "config_order":
        raise ValueError("selection.tie_break must be 'config_order'")

    return {
        "model": {
            "char_ngram_range": _ngram_range(
                model["char_ngram_range"], where="model.char_ngram_range"
            ),
            "word_ngram_range": _ngram_range(
                model["word_ngram_range"], where="model.word_ngram_range"
            ),
            "char_min_df": _min_df(model["char_min_df"], where="model.char_min_df"),
            "word_min_df": _min_df(model["word_min_df"], where="model.word_min_df"),
            "prompt_min_df": _min_df(
                model["prompt_min_df"], where="model.prompt_min_df"
            ),
            "include_prompt": _bool(
                model["include_prompt"], where="model.include_prompt"
            ),
            "include_prompt_text": _bool(
                model["include_prompt_text"], where="model.include_prompt_text"
            ),
            "include_surface": _bool(
                model["include_surface"], where="model.include_surface"
            ),
            "clip": clip_values,
        },
        "inner_cv": {
            "n_splits": n_splits,
            "seed": int(inner["seed"]),
            "score_bins": _positive_int(
                inner["score_bins"], where="inner_cv.score_bins"
            ),
        },
        "search": {
            "alpha": alphas,
            "max_char_features": feature_grid("max_char_features"),
            "max_word_features": feature_grid("max_word_features"),
        },
        "selection": {
            "metric": "trait_mean_rmse",
            "tie_break": "config_order",
        },
    }


def _candidate_grid(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    search = config["search"]
    return [
        {
            "alpha": float(alpha),
            "max_char_features": max_char,
            "max_word_features": max_word,
        }
        for alpha, max_char, max_word in product(
            search["alpha"],
            search["max_char_features"],
            search["max_word_features"],
        )
    ]


def _estimator(
    config: Mapping[str, Any], candidate: Mapping[str, Any]
) -> TfidfRidgeBaseline:
    model = config["model"]
    return TfidfRidgeBaseline(
        alpha=float(candidate["alpha"]),
        char_ngram_range=tuple(model["char_ngram_range"]),
        word_ngram_range=tuple(model["word_ngram_range"]),
        char_min_df=model["char_min_df"],
        word_min_df=model["word_min_df"],
        prompt_min_df=model["prompt_min_df"],
        max_char_features=candidate["max_char_features"],
        max_word_features=candidate["max_word_features"],
        include_prompt=model["include_prompt"],
        include_prompt_text=model["include_prompt_text"],
        include_surface=model["include_surface"],
        clip=tuple(model["clip"]),
    )


def _record_ids(records: Sequence[Any]) -> list[str]:
    ids = [str(get_field(record, "id")) for record in records]
    if any(not value.strip() for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("records must have unique nonempty ids")
    return ids


def _inner_seed(base_seed: int, outer_fold: int) -> int:
    fold_offset = int(sha256_json({"outer_fold": outer_fold})[:8], 16)
    return (base_seed + fold_offset) % 2_147_483_647


def _trait_rmse(truth: np.ndarray, predictions: np.ndarray) -> list[float]:
    if truth.shape != predictions.shape or truth.ndim != 2 or truth.shape[1] != len(TRAITS):
        raise ValueError("truth and predictions must have aligned (rows, 3) shapes")
    if not np.isfinite(truth).all() or not np.isfinite(predictions).all():
        raise ValueError("RMSE selection requires finite values")
    return [
        float(np.sqrt(np.mean(np.square(predictions[:, index] - truth[:, index]))))
        for index in range(len(TRAITS))
    ]


def _select_tfidf_candidate(
    records: Sequence[Any],
    normalized_config: Mapping[str, Any],
    *,
    outer_fold: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = list(records)
    ids = _record_ids(rows)
    inner_config = normalized_config["inner_cv"]
    if len(rows) < inner_config["n_splits"]:
        raise ValueError("outer-training rows are fewer than inner_cv.n_splits")
    inner_seed = _inner_seed(inner_config["seed"], outer_fold)
    inner_assignments = make_folds(
        rows,
        n_splits=inner_config["n_splits"],
        seed=inner_seed,
        score_bins=inner_config["score_bins"],
    )
    inner_fold_ids = [inner_assignments[record_id] for record_id in ids]
    truth = gold_matrix(rows)
    candidates: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(_candidate_grid(normalized_config)):
        predictions = oof_predict(
            _estimator(normalized_config, candidate), rows, inner_fold_ids
        )
        trait_rmse = _trait_rmse(truth, predictions)
        candidates.append(
            {
                "candidate_index": candidate_index,
                "parameters": candidate,
                "trait_rmse": dict(zip(TRAITS, trait_rmse, strict=True)),
                "trait_mean_rmse": float(np.mean(trait_rmse)),
            }
        )
    selected = min(
        candidates,
        key=lambda item: (item["trait_mean_rmse"], item["candidate_index"]),
    )
    report = {
        "inner_seed": inner_seed,
        "inner_fold_generation": "outer_train_prompt_cohort_score_band",
        "inner_folds": {
            record_id: int(inner_assignments[record_id]) for record_id in sorted(ids)
        },
        "inner_folds_sha256": sha256_json(inner_assignments),
        "inner_rows": len(rows),
        "selection_scope": "outer_train_only",
        "outer_held_labels_used_for_selection": False,
        "selection_metric": "trait_mean_rmse",
        "tie_break": "config_order",
        "candidate_count": len(candidates),
        "selected": dict(selected),
        "candidates": candidates,
    }
    return dict(selected["parameters"]), report


def select_tfidf_candidate(
    records: Sequence[Any], config: Mapping[str, Any], *, outer_fold: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select only from records supplied by an outer-training partition."""

    if isinstance(outer_fold, bool) or not isinstance(outer_fold, int) or outer_fold < 0:
        raise ValueError("outer_fold must be a non-negative integer")
    normalized = normalize_nested_tfidf_config(config)
    return _select_tfidf_candidate(records, normalized, outer_fold=outer_fold)


def _outer_values(records: Sequence[Any], folds: Any) -> np.ndarray:
    ids = _record_ids(records)
    if isinstance(folds, Mapping):
        missing = sorted(set(ids).difference(folds))
        extra = sorted(set(folds).difference(ids))
        if missing or extra:
            raise ValueError(
                f"outer folds must match record ids exactly; missing={missing[:5]}, "
                f"extra={extra[:5]}"
            )
        values = [folds[record_id] for record_id in ids]
    else:
        values = list(folds)
        if len(values) != len(ids):
            raise ValueError("outer folds must contain one value per record")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0
        for value in values
    ):
        raise ValueError("outer folds must be non-negative integers")
    result = np.asarray(values, dtype=int)
    if np.unique(result).size < 2:
        raise ValueError("nested TF-IDF tuning requires at least two outer folds")
    return result


def nested_tfidf_oof(
    records: Sequence[Any], folds: Any, config: Mapping[str, Any]
) -> NestedTfidfResult:
    """Tune inside each outer-training partition, then predict its held fold once."""

    rows = list(records)
    if not rows:
        raise ValueError("nested TF-IDF tuning requires nonempty records")
    ids = _record_ids(rows)
    outer_values = _outer_values(rows, folds)
    normalized = normalize_nested_tfidf_config(config)
    predictions = np.full((len(rows), len(TRAITS)), np.nan, dtype=float)
    seen = np.zeros(len(rows), dtype=int)
    fold_models: dict[str, TfidfRidgeBaseline] = {}
    outer_reports: list[dict[str, Any]] = []

    for outer_fold in sorted(int(value) for value in np.unique(outer_values)):
        held_indices = np.flatnonzero(outer_values == outer_fold)
        fit_indices = np.flatnonzero(outer_values != outer_fold)
        if held_indices.size == 0 or fit_indices.size == 0:
            raise ValueError("every outer fold needs nonempty fit and held partitions")
        fit_rows = [rows[index] for index in fit_indices]
        held_rows = [rows[index] for index in held_indices]
        selected, selection = _select_tfidf_candidate(
            fit_rows, normalized, outer_fold=outer_fold
        )
        model = _estimator(normalized, selected)
        model.fit(fit_rows)
        fold_prediction = np.asarray(model.predict(held_rows), dtype=float)
        if fold_prediction.shape != (len(held_rows), len(TRAITS)) or not np.isfinite(
            fold_prediction
        ).all():
            raise RuntimeError("outer TF-IDF model returned invalid held-fold predictions")
        predictions[held_indices] = fold_prediction
        seen[held_indices] += 1
        key = str(outer_fold)
        fold_models[key] = model
        outer_reports.append(
            {
                "outer_fold": outer_fold,
                "fit_rows": int(fit_indices.size),
                "held_rows": int(held_indices.size),
                "fit_ids": [ids[index] for index in fit_indices],
                "held_ids": [ids[index] for index in held_indices],
                "fit_ids_sha256": sha256_json(
                    [ids[index] for index in fit_indices]
                ),
                "held_ids_sha256": sha256_json(
                    [ids[index] for index in held_indices]
                ),
                **selection,
            }
        )

    if not np.all(seen == 1) or not np.isfinite(predictions).all():
        raise RuntimeError("nested TF-IDF outer folds did not predict every row exactly once")
    return NestedTfidfResult(
        oof_predictions=predictions,
        fold_models=fold_models,
        outer_reports=outer_reports,
        normalized_config=normalized,
    )


__all__ = [
    "NESTED_TUNING_CODE_FILES",
    "NestedTfidfResult",
    "nested_tfidf_oof",
    "nested_tuning_code_contract",
    "normalize_nested_tfidf_config",
    "select_tfidf_candidate",
]
