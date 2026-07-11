"""Leakage-safe nested-CV Ridge for assessment-question probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import Ridge

from src.assessment.questions import QUESTIONS, TRAITS, questions_for_trait
from src.utils.hashing import sha256_json


@dataclass(frozen=True)
class NestedAssessmentResult:
    oof_predictions: np.ndarray
    outer_reports: list[dict[str, Any]]
    fold_models: dict[str, dict[str, dict[str, Any]]]


def trait_feature_matrix(
    probabilities: np.ndarray,
    trait: str,
    question_count: int,
) -> np.ndarray:
    tensor = np.asarray(probabilities, dtype=np.float64)
    if tensor.ndim != 3 or tensor.shape[1:] != (18, 5):
        raise ValueError("assessment probabilities must have shape (rows, 18, 5)")
    trait_questions = questions_for_trait(trait)
    if question_count < 1 or question_count > len(trait_questions):
        raise ValueError(f"invalid question_count for {trait}: {question_count}")
    all_ids = [question.question_id for question in QUESTIONS]
    selected_ids = {question.question_id for question in trait_questions[:question_count]}
    selected_indices = [
        index for index, question_id in enumerate(all_ids) if question_id in selected_ids
    ]
    return tensor[:, selected_indices, :].reshape(len(tensor), -1)


def _fit_standardized_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
    trait: str,
    question_count: int,
) -> dict[str, Any]:
    matrix = np.asarray(features, dtype=np.float64)
    values = np.asarray(targets, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != len(values) or len(values) < 2:
        raise ValueError("Ridge fitting requires aligned features and at least two rows")
    if not np.isfinite(matrix).all() or not np.isfinite(values).all():
        raise ValueError("Ridge fitting requires finite values")
    if alpha <= 0.0:
        raise ValueError("Ridge alpha must be positive")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (matrix - mean) / scale
    regressor = Ridge(alpha=float(alpha), solver="lsqr", fit_intercept=True)
    regressor.fit(normalized, values)
    selected_questions = questions_for_trait(trait)[:question_count]
    return {
        "trait": trait,
        "question_count": int(question_count),
        "question_ids": [question.question_id for question in selected_questions],
        "alpha": float(alpha),
        "feature_dimension": int(matrix.shape[1]),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": np.asarray(regressor.coef_, dtype=float).reshape(-1).tolist(),
        "intercept": float(regressor.intercept_),
    }


def _predict_model(model: dict[str, Any], probabilities: np.ndarray) -> np.ndarray:
    trait = model.get("trait")
    question_count = model.get("question_count")
    if trait not in TRAITS or isinstance(question_count, bool) or not isinstance(question_count, int):
        raise ValueError("assessment Ridge model has an invalid trait/question_count")
    expected_questions = [
        question.question_id for question in questions_for_trait(trait)[:question_count]
    ]
    if model.get("question_ids") != expected_questions:
        raise ValueError("assessment Ridge model question layout mismatch")
    features = trait_feature_matrix(probabilities, trait, question_count)
    if model.get("feature_dimension") != features.shape[1]:
        raise ValueError("assessment Ridge feature_dimension is invalid")
    alpha = float(model.get("alpha"))
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("assessment Ridge alpha must be finite and positive")
    mean = np.asarray(model.get("mean"), dtype=np.float64)
    scale = np.asarray(model.get("scale"), dtype=np.float64)
    coefficients = np.asarray(model.get("coefficients"), dtype=np.float64)
    if mean.shape != (features.shape[1],) or scale.shape != mean.shape or coefficients.shape != mean.shape:
        raise ValueError("assessment Ridge model parameter dimensions are invalid")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not np.isfinite(coefficients).all():
        raise ValueError("assessment Ridge model contains non-finite parameters")
    if (scale <= 0.0).any():
        raise ValueError("assessment Ridge model scales must be positive")
    intercept = float(model.get("intercept"))
    if not np.isfinite(intercept):
        raise ValueError("assessment Ridge model intercept must be finite")
    return ((features - mean) / scale) @ coefficients + intercept


def _select_hyperparameters(
    probabilities: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    *,
    trait: str,
    question_counts: tuple[int, ...],
    alphas: tuple[float, ...],
) -> dict[str, Any]:
    unique_folds = sorted(int(value) for value in np.unique(folds))
    if len(unique_folds) < 2:
        raise ValueError("inner CV requires at least two training-only folds")
    candidates: list[dict[str, Any]] = []
    for question_count in question_counts:
        features = trait_feature_matrix(probabilities, trait, question_count)
        for alpha in alphas:
            squared_errors: list[np.ndarray] = []
            for held_fold in unique_folds:
                held = folds == held_fold
                fit = ~held
                if int(held.sum()) == 0 or int(fit.sum()) < 2:
                    raise ValueError("inner CV produced an empty held or fit partition")
                model = _fit_standardized_ridge(
                    features[fit],
                    targets[fit],
                    alpha=alpha,
                    trait=trait,
                    question_count=question_count,
                )
                prediction = _predict_model(model, probabilities[held])
                squared_errors.append(np.square(prediction - targets[held]))
            rmse = float(np.sqrt(np.mean(np.concatenate(squared_errors))))
            candidates.append(
                {
                    "question_count": int(question_count),
                    "alpha": float(alpha),
                    "inner_rmse": rmse,
                }
            )
    best = min(
        candidates,
        key=lambda item: (
            item["inner_rmse"],
            item["question_count"],
            -item["alpha"],
        ),
    )
    return {"selected": dict(best), "candidates": candidates}


def nested_oof_ridge(
    probabilities: np.ndarray,
    targets: np.ndarray,
    fold_ids: Sequence[int],
    record_ids: Sequence[str],
    *,
    question_counts: Sequence[int],
    alphas: Sequence[float],
    clip_min: float = 1.0,
    clip_max: float = 5.0,
) -> NestedAssessmentResult:
    """Nested CV where each outer fold is absent from all feature/model selection.

    The versioned question text and order are fixed before seeing labels. Inner
    CV may select only a prefix length (3--6), Ridge alpha, scaler, and weights.
    """

    tensor = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    folds = np.asarray(fold_ids, dtype=int)
    ids = tuple(str(value) for value in record_ids)
    if tensor.shape != (len(ids), 18, 5) or truth.shape != (len(ids), 3):
        raise ValueError("nested assessment inputs have incompatible shapes")
    if folds.shape != (len(ids),) or len(set(ids)) != len(ids):
        raise ValueError("fold ids and unique record ids must match assessment rows")
    if not np.isfinite(tensor).all() or not np.isfinite(truth).all():
        raise ValueError("nested assessment inputs must be finite")
    if (tensor < 0.0).any() or (tensor > 1.0).any() or not np.allclose(
        tensor.sum(axis=2), 1.0, atol=1e-5, rtol=1e-5
    ):
        raise ValueError("nested assessment features must be answer probabilities")
    outer_folds = sorted(int(value) for value in np.unique(folds))
    if len(outer_folds) < 3:
        raise ValueError("nested assessment CV requires at least three outer folds")
    raw_counts = tuple(question_counts)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in raw_counts):
        raise ValueError("question_counts must contain integers")
    counts = tuple(sorted(set(int(value) for value in raw_counts)))
    alpha_grid = tuple(sorted(set(float(value) for value in alphas)))
    if not counts or any(value < 3 or value > 6 for value in counts):
        raise ValueError("question_counts must be nonempty values in [3, 6]")
    if not alpha_grid or any(not np.isfinite(value) or value <= 0.0 for value in alpha_grid):
        raise ValueError("alphas must be nonempty positive values")
    if not clip_min < clip_max:
        raise ValueError("clip_min must be smaller than clip_max")

    oof = np.full((len(ids), 3), np.nan, dtype=np.float64)
    outer_reports: list[dict[str, Any]] = []
    fold_models: dict[str, dict[str, dict[str, Any]]] = {}
    for outer_fold in outer_folds:
        held = folds == outer_fold
        fit = ~held
        report: dict[str, Any] = {
            "outer_fold": outer_fold,
            "fit_ids_sha256": sha256_json([ids[index] for index in np.flatnonzero(fit)]),
            "held_ids_sha256": sha256_json([ids[index] for index in np.flatnonzero(held)]),
            "fit_rows": int(fit.sum()),
            "held_rows": int(held.sum()),
            "selection_scope": "outer_train_only",
            "traits": {},
        }
        fold_models[str(outer_fold)] = {}
        for trait_index, trait in enumerate(TRAITS):
            selection = _select_hyperparameters(
                tensor[fit],
                truth[fit, trait_index],
                folds[fit],
                trait=trait,
                question_counts=counts,
                alphas=alpha_grid,
            )
            selected = selection["selected"]
            model = _fit_standardized_ridge(
                trait_feature_matrix(tensor[fit], trait, selected["question_count"]),
                truth[fit, trait_index],
                alpha=selected["alpha"],
                trait=trait,
                question_count=selected["question_count"],
            )
            oof[held, trait_index] = _predict_model(model, tensor[held])
            fold_models[str(outer_fold)][trait] = model
            report["traits"][trait] = selection
        outer_reports.append(report)
    if not np.isfinite(oof).all():
        raise RuntimeError("nested assessment CV left one or more OOF rows unset")
    oof = np.clip(oof, clip_min, clip_max)

    return NestedAssessmentResult(
        oof_predictions=oof,
        outer_reports=outer_reports,
        fold_models=fold_models,
    )


def predict_assessment_ridge(
    probabilities: np.ndarray,
    models: dict[str, dict[str, Any]],
    *,
    clip_min: float = 1.0,
    clip_max: float = 5.0,
) -> np.ndarray:
    if not clip_min < clip_max:
        raise ValueError("clip_min must be smaller than clip_max")
    if set(models) != set(TRAITS):
        raise ValueError(f"assessment deployment models must contain exactly {TRAITS}")
    matrix = np.column_stack(
        [_predict_model(models[trait], probabilities) for trait in TRAITS]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("assessment deployment predictions are non-finite")
    return np.clip(matrix, clip_min, clip_max)


def predict_assessment_fold_ensemble(
    probabilities: np.ndarray,
    fold_models: dict[str, dict[str, dict[str, Any]]],
    *,
    clip_min: float = 1.0,
    clip_max: float = 5.0,
) -> np.ndarray:
    if not isinstance(fold_models, dict) or not fold_models:
        raise ValueError("assessment deployment requires at least one fold model")
    if not clip_min < clip_max:
        raise ValueError("clip_min must be smaller than clip_max")
    predictions: list[np.ndarray] = []
    for models in fold_models.values():
        if set(models) != set(TRAITS):
            raise ValueError(f"assessment fold models must contain exactly {TRAITS}")
        predictions.append(
            np.column_stack(
                [_predict_model(models[trait], probabilities) for trait in TRAITS]
            )
        )
    matrix = np.mean(np.stack(predictions, axis=0), axis=0)
    if not np.isfinite(matrix).all():
        raise ValueError("assessment fold ensemble predictions are non-finite")
    return np.clip(matrix, clip_min, clip_max)


__all__ = [
    "NestedAssessmentResult",
    "nested_oof_ridge",
    "predict_assessment_fold_ensemble",
    "predict_assessment_ridge",
    "trait_feature_matrix",
]
