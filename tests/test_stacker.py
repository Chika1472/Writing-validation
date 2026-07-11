from __future__ import annotations

import numpy as np

from src.ensemble.crossfit import cross_fit_simplex_stacker
from src.ensemble.simplex import TraitSimplexStacker


def test_two_source_simplex_recovers_known_trait_weights() -> None:
    left = np.asarray(
        [[1.5, 2.0, 2.5], [2.5, 3.0, 3.5], [3.5, 4.0, 4.5], [4.5, 3.5, 2.0]]
    )
    right = np.asarray(
        [[2.5, 1.5, 3.0], [3.5, 2.5, 4.0], [4.5, 3.5, 4.0], [3.5, 4.5, 3.0]]
    )
    expected_left = (0.2, 0.6, 0.8)
    gold = np.column_stack(
        [
            expected_left[index] * left[:, index]
            + (1.0 - expected_left[index]) * right[:, index]
            for index in range(3)
        ]
    )
    model = TraitSimplexStacker.fit(
        gold,
        {"qwen": left, "tfidf": right},
        source_order=("qwen", "tfidf"),
    )

    actual = tuple(model.weights[trait][0] for trait in ("content", "organization", "expression"))
    np.testing.assert_allclose(actual, expected_left, atol=1e-12)
    np.testing.assert_allclose(
        TraitSimplexStacker.from_dict(model.to_dict()).transform(
            {"qwen": left, "tfidf": right}
        ),
        gold,
        atol=1e-12,
    )


def test_level1_crossfit_produces_one_finite_prediction_per_row() -> None:
    truth = np.tile(np.linspace(1.5, 4.5, 12)[:, None], (1, 3))
    qwen = np.clip(truth + 0.2, 1.0, 5.0)
    tfidf = np.clip(truth - 0.1, 1.0, 5.0)
    result = cross_fit_simplex_stacker(
        truth,
        {"qwen": qwen, "tfidf": tfidf},
        [index % 3 for index in range(len(truth))],
        ["Q1" if index < 6 else "Q2" for index in range(len(truth))],
        source_order=("qwen", "tfidf"),
    )

    assert result.cross_fitted_predictions.shape == truth.shape
    assert np.isfinite(result.cross_fitted_predictions).all()
    assert len(result.fold_reports) == 3


def test_three_source_simplex_uses_all_sources_and_roundtrips() -> None:
    base = np.tile(np.linspace(1.2, 4.8, 20)[:, None], (1, 3))
    first = np.clip(base + np.asarray([0.3, -0.2, 0.1]), 1.0, 5.0)
    second = np.clip(base + np.asarray([-0.2, 0.3, -0.1]), 1.0, 5.0)
    third = np.clip(base + np.asarray([0.1, -0.1, 0.2]), 1.0, 5.0)
    sources = {"qwen": first, "tfidf": second, "anchor": third}
    gold = 0.5 * first + 0.3 * second + 0.2 * third
    model = TraitSimplexStacker.fit(
        gold,
        sources,
        source_order=("qwen", "tfidf", "anchor"),
    )
    restored = TraitSimplexStacker.from_dict(model.to_dict())
    assert restored.source_names == ("qwen", "tfidf", "anchor")
    np.testing.assert_allclose(restored.transform(sources), gold, atol=1e-6)
