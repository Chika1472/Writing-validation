from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.baselines.mean_baseline import (
    MeanBaseline,
    PromptMeanBaseline,
    fit_predict,
    oof_predict,
    oof_predict_with_models,
)
from src.baselines.tfidf_ridge import SurfaceOLSBaseline, TfidfRidgeBaseline


@dataclass
class Record:
    id: str
    prompt_num: str
    prompt: str
    essay: str
    scores: dict[str, float]


TRAITS = ("content", "organization", "expression")


def _record(
    index: int,
    prompt_num: str,
    essay: str,
    scores: tuple[float, float, float],
) -> Record:
    return Record(
        id=f"id-{index}",
        prompt_num=prompt_num,
        prompt=f"{prompt_num}에 대한 논증적 글을 작성하시오.",
        essay=essay,
        scores=dict(zip(TRAITS, scores, strict=True)),
    )


def _training_records() -> list[Record]:
    return [
        _record(0, "Q1", "첫째 근거를 들어 찬성한다. 따라서 제도를 도입해야 한다.", (2.0, 2.0, 2.5)),
        _record(1, "Q1", "둘째 구체적 사례가 있다. 그러므로 제도의 장점이 크다.", (3.0, 3.0, 3.5)),
        _record(2, "Q1", "반론도 가능하지만 통계와 사례를 고려하면 찬성한다.", (4.0, 4.0, 4.5)),
        _record(3, "Q1", "주장과 근거가 명확하다. 결론적으로 반드시 도입해야 한다.", (4.5, 4.5, 4.75)),
        _record(4, "Q2", "첫째 부작용 때문에 반대한다. 대안을 먼저 마련해야 한다.", (1.5, 2.0, 2.0)),
        _record(5, "Q2", "둘째 피해 사례가 존재한다. 그러나 절충안은 가능하다.", (2.5, 3.0, 3.0)),
        _record(6, "Q2", "반론을 검토하고 구체적인 해결책을 제시하며 반대한다.", (3.5, 4.0, 4.0)),
        _record(7, "Q2", "근거와 사례를 체계적으로 연결했다. 따라서 반대 입장이다.", (4.0, 4.5, 4.5)),
    ]


def test_global_mean_baseline() -> None:
    train = _training_records()
    model = MeanBaseline().fit(train)

    predictions = model.predict(train[:2])
    expected = np.mean([[row.scores[trait] for trait in TRAITS] for row in train], axis=0)

    assert predictions.shape == (2, 3)
    assert np.allclose(predictions[0], expected)
    assert np.allclose(predictions[1], expected)


def test_prompt_mean_uses_prompt_and_falls_back_for_unknown_prompt() -> None:
    train = _training_records()
    model = PromptMeanBaseline().fit(train)
    unknown = _record(99, "Q9", "새로운 문항에 대한 충분히 긴 글이다.", (3, 3, 3))

    predictions = model.predict([train[0], train[4], unknown])
    q1_expected = np.mean(
        [[row.scores[trait] for trait in TRAITS] for row in train[:4]], axis=0
    )
    q2_expected = np.mean(
        [[row.scores[trait] for trait in TRAITS] for row in train[4:]], axis=0
    )
    global_expected = np.mean(
        [[row.scores[trait] for trait in TRAITS] for row in train], axis=0
    )

    assert np.allclose(predictions[0], q1_expected)
    assert np.allclose(predictions[1], q2_expected)
    assert np.allclose(predictions[2], global_expected)


def test_oof_mean_never_uses_validation_fold_labels() -> None:
    records = [
        _record(0, "Q1", "충분히 긴 첫 번째 글입니다.", (1, 1, 1)),
        _record(1, "Q1", "충분히 긴 두 번째 글입니다.", (1, 1, 1)),
        _record(2, "Q1", "충분히 긴 세 번째 글입니다.", (5, 5, 5)),
        _record(3, "Q1", "충분히 긴 네 번째 글입니다.", (5, 5, 5)),
    ]

    predictions = oof_predict(MeanBaseline(), records, folds=[0, 0, 1, 1])

    assert np.allclose(predictions[:2], 5.0)
    assert np.allclose(predictions[2:], 1.0)

    fold_mapping = {record.id: fold for record, fold in zip(records, [0, 0, 1, 1])}
    assert np.allclose(oof_predict(MeanBaseline(), records, fold_mapping), predictions)

    retained_predictions, fold_models = oof_predict_with_models(
        MeanBaseline(), records, [0, 0, 1, 1]
    )
    assert np.allclose(retained_predictions, predictions)
    assert set(fold_models) == {"0", "1"}


def test_fit_predict_returns_fitted_clone_and_validation_predictions() -> None:
    records = _training_records()
    estimator = MeanBaseline()

    fitted, predictions = fit_predict(estimator, records[:6], records[6:])

    assert not hasattr(estimator, "global_mean_")
    assert hasattr(fitted, "global_mean_")
    assert predictions.shape == (2, 3)


def test_surface_ols_train_validation_and_unknown_prompt() -> None:
    records = _training_records()
    model = SurfaceOLSBaseline().fit(records[:6])

    predictions = model.predict(records[6:])

    assert predictions.shape == (2, 3)
    assert np.isfinite(predictions).all()
    assert ((predictions >= 1.0) & (predictions <= 5.0)).all()


def test_tfidf_ridge_uses_requested_ngram_ranges_and_train_only_vocabulary() -> None:
    train = _training_records()[:6]
    validation = [
        _record(
            99,
            "Q9",
            "검증전용비밀어휘는 학습 자료에 절대로 등장하지 않는 표현이다.",
            (3, 3, 3),
        )
    ]
    model = TfidfRidgeBaseline(
        alpha=1.0,
        char_min_df=1,
        word_min_df=1,
        prompt_min_df=1,
    ).fit(train)

    predictions = model.predict(validation)

    assert model.char_vectorizer_.ngram_range == (3, 5)
    assert model.word_vectorizer_.ngram_range == (1, 2)
    assert "검증전용비밀어휘는" not in model.word_vectorizer_.vocabulary_
    assert predictions.shape == (1, 3)
    assert np.isfinite(predictions).all()


def test_tfidf_oof_predicts_every_row_once_without_nonfinite_values() -> None:
    records = _training_records()
    model = TfidfRidgeBaseline(
        alpha=2.0,
        char_min_df=1,
        word_min_df=1,
        prompt_min_df=1,
        max_char_features=200,
        max_word_features=100,
    )

    predictions = model.oof_predict(records, folds=[0, 1, 0, 1, 0, 1, 0, 1])

    assert predictions.shape == (len(records), 3)
    assert np.isfinite(predictions).all()


def test_oof_rejects_overlapping_or_incomplete_splits() -> None:
    records = _training_records()[:4]
    with pytest.raises(ValueError, match="overlap"):
        oof_predict(MeanBaseline(), records, [([0, 1, 2], [2, 3])])

    with pytest.raises(ValueError, match="exactly once"):
        oof_predict(MeanBaseline(), records, [([2, 3], [0, 1])])
