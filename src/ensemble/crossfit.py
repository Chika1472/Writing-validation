from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.calibration.affine import AffinePromptCalibrator
from src.evaluation.metrics import TRAITS
from src.ensemble.simplex import TraitSimplexStacker, _matrix


def _by_trait(matrix: np.ndarray) -> dict[str, np.ndarray]:
    return {trait: matrix[:, index] for index, trait in enumerate(TRAITS)}


def _from_trait(values: Mapping[str, Sequence[float]]) -> np.ndarray:
    return np.column_stack([np.asarray(values[trait], dtype=float) for trait in TRAITS])


@dataclass(frozen=True)
class StackerCrossFitResult:
    cross_fitted_predictions: np.ndarray
    final_stacker: TraitSimplexStacker
    final_calibrator: AffinePromptCalibrator
    fold_reports: tuple[dict[str, Any], ...]


def cross_fit_simplex_stacker(
    gold: Any,
    source_predictions: Mapping[str, Any],
    fold_ids: Sequence[Any],
    prompts: Sequence[str],
    *,
    source_order: Sequence[str],
    epsilon: float = 1e-12,
    prompt_shrinkage: float = 20.0,
    clip_min: float = 1.0,
    clip_max: float = 5.0,
) -> StackerCrossFitResult:
    """Level-1 cross-fit of simplex weights and affine/prompt calibration.

    Base inputs must already be genuine OOF predictions. This estimates the
    meta learner without fitting a row's weight/calibration on that row, but is
    not a fully nested re-training of every base scorer.
    """

    truth = _matrix(gold, name="gold")
    names = tuple(source_order)
    if len(names) < 2 or len(set(names)) != len(names) or set(names) != set(source_predictions):
        raise ValueError("source_order must contain at least two unique source aliases exactly")
    sources = {
        name: _matrix(source_predictions[name], rows=len(truth), name=name)
        for name in names
    }
    folds = np.asarray(list(fold_ids), dtype=object)
    prompt_array = np.asarray(list(prompts), dtype=str)
    if folds.shape != (len(truth),) or prompt_array.shape != (len(truth),):
        raise ValueError("fold_ids and prompts must align with gold rows")
    unique_folds = np.unique(folds)
    if len(unique_folds) < 2:
        raise ValueError("cross-fitting requires at least two folds")

    cross_fitted = np.full_like(truth, np.nan, dtype=float)
    seen = np.zeros(len(truth), dtype=int)
    reports: list[dict[str, Any]] = []
    for fold_id in unique_folds:
        held_out = folds == fold_id
        fit_rows = ~held_out
        if int(held_out.sum()) == 0 or int(fit_rows.sum()) < 2:
            raise ValueError(f"invalid meta fold: {fold_id!r}")
        stacker = TraitSimplexStacker.fit(
            truth[fit_rows],
            {name: matrix[fit_rows] for name, matrix in sources.items()},
            source_order=names,
            epsilon=epsilon,
        )
        stacked_fit = stacker.transform(
            {name: matrix[fit_rows] for name, matrix in sources.items()}
        )
        calibrator = AffinePromptCalibrator.fit(
            _by_trait(truth[fit_rows]),
            _by_trait(stacked_fit),
            prompt_array[fit_rows].tolist(),
            prompt_shrinkage=prompt_shrinkage,
            clip_min=clip_min,
            clip_max=clip_max,
            fit_source="level1_meta_train_oof",
        )
        stacked_held_out = stacker.transform(
            {name: matrix[held_out] for name, matrix in sources.items()}
        )
        calibrated = calibrator.transform(
            _by_trait(stacked_held_out),
            prompt_array[held_out].tolist(),
        )
        cross_fitted[held_out] = _from_trait(calibrated)
        seen[held_out] += 1
        reports.append(
            {
                "fold": str(fold_id),
                "fit_rows": int(fit_rows.sum()),
                "held_out_rows": int(held_out.sum()),
                "stacker": stacker.to_dict(),
                "calibrator": calibrator.to_dict(),
            }
        )
    if not np.all(seen == 1) or not np.isfinite(cross_fitted).all():
        raise RuntimeError("meta cross-fit did not produce every row exactly once")

    final_stacker = TraitSimplexStacker.fit(
        truth,
        sources,
        source_order=names,
        epsilon=epsilon,
    )
    final_raw = final_stacker.transform(sources)
    final_calibrator = AffinePromptCalibrator.fit(
        _by_trait(truth),
        _by_trait(final_raw),
        prompt_array.tolist(),
        prompt_shrinkage=prompt_shrinkage,
        clip_min=clip_min,
        clip_max=clip_max,
        fit_source="base_oof_stacked",
    )
    return StackerCrossFitResult(
        cross_fitted_predictions=cross_fitted,
        final_stacker=final_stacker,
        final_calibrator=final_calibrator,
        fold_reports=tuple(reports),
    )
