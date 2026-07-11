from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.baselines.nested_tuning import (
    nested_tfidf_oof,
    normalize_nested_tfidf_config,
)


def _config() -> dict:
    return {
        "model": {
            "char_ngram_range": [2, 3],
            "word_ngram_range": [1, 2],
            "char_min_df": 1,
            "word_min_df": 1,
            "prompt_min_df": 1,
            "include_prompt": True,
            "include_prompt_text": True,
            "include_surface": True,
            "clip": [1.0, 5.0],
        },
        "inner_cv": {"n_splits": 2, "seed": 17, "score_bins": 2},
        "search": {
            "alpha": [1.0, 5.0],
            "max_char_features": [80],
            "max_word_features": [40],
        },
        "selection": {
            "metric": "trait_mean_rmse",
            "tie_break": "config_order",
        },
    }


def _records() -> list[dict]:
    rows: list[dict] = []
    for index in range(18):
        base = 1.0 + float(index % 5)
        scores = [base, 1.0 + float((index + 1) % 5), 1.0 + float((index + 2) % 5)]
        rows.append(
            {
                "id": f"id-{index:02d}",
                "document_id": f"doc-{index:02d}",
                "prompt_num": f"Q{1 + index % 2}",
                "prompt": "이 제도의 도입 여부에 관한 의견을 논리적으로 쓰시오.",
                "essay": (
                    f"사례 {index}를 근거로 입장을 제시한다. "
                    "반론을 검토한 뒤 구체적인 대안을 제안한다."
                ),
                "score": {
                    "content": scores[0],
                    "organization": scores[1],
                    "expression": scores[2],
                    "average": float(np.mean(scores)),
                },
            }
        )
    return rows


def test_nested_tfidf_predicts_every_outer_row_and_audits_inner_scope() -> None:
    records = _records()
    folds = {record["id"]: index % 3 for index, record in enumerate(records)}

    result = nested_tfidf_oof(records, folds, _config())

    assert result.oof_predictions.shape == (len(records), 3)
    assert np.isfinite(result.oof_predictions).all()
    assert set(result.fold_models) == {"0", "1", "2"}
    assert len(result.outer_reports) == 3
    for report in result.outer_reports:
        assert report["selection_scope"] == "outer_train_only"
        assert report["outer_held_labels_used_for_selection"] is False
        assert set(report["inner_folds"]) == set(report["fit_ids"])
        assert set(report["inner_folds"]).isdisjoint(report["held_ids"])
        assert report["selected"] in report["candidates"]
        assert report["candidate_count"] == 2
        assert len(report["candidates"]) == 2


def test_outer_held_labels_cannot_change_their_oof_model_or_selection() -> None:
    records = _records()
    folds = {record["id"]: index % 3 for index, record in enumerate(records)}
    original = nested_tfidf_oof(records, folds, _config())

    changed_records = deepcopy(records)
    for record in changed_records:
        if folds[record["id"]] == 0:
            for trait in ("content", "organization", "expression"):
                record["score"][trait] = 6.0 - record["score"][trait]
            record["score"]["average"] = float(
                np.mean(
                    [
                        record["score"][trait]
                        for trait in ("content", "organization", "expression")
                    ]
                )
            )
    changed = nested_tfidf_oof(changed_records, folds, _config())
    held_zero = np.asarray([folds[record["id"]] == 0 for record in records])

    np.testing.assert_allclose(
        original.oof_predictions[held_zero], changed.oof_predictions[held_zero]
    )
    original_report = next(
        report for report in original.outer_reports if report["outer_fold"] == 0
    )
    changed_report = next(
        report for report in changed.outer_reports if report["outer_fold"] == 0
    )
    assert original_report == changed_report


def test_nested_config_rejects_silent_metric_or_grid_drift() -> None:
    config = _config()
    config["selection"]["metric"] = "micro_rmse"
    with pytest.raises(ValueError, match="trait_mean_rmse"):
        normalize_nested_tfidf_config(config)

    config = _config()
    config["search"]["alpha"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="duplicates"):
        normalize_nested_tfidf_config(config)
