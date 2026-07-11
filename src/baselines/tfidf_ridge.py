"""Leakage-safe surface OLS and TF-IDF Ridge baselines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.features import build_surface_features
from src.evaluation.metrics import TRAITS, get_field, gold_matrix
from src.evaluation.predictions import prediction_records

from .mean_baseline import oof_predict as cross_validated_predict


def _texts(records: Sequence[Any], field: str) -> list[str]:
    return [str(get_field(record, field, "")) for record in records]


def _prompt_column(records: Sequence[Any]) -> np.ndarray:
    return np.asarray(
        [[str(get_field(record, "prompt_num"))] for record in records], dtype=object
    )


def _surface_matrix(records: Sequence[Any]) -> np.ndarray:
    # build_surface_features intentionally requires the full canonical data schema.
    # Predictions, however, must not depend on labels, so adapters below insert a
    # fixed unused score object rather than copying the true targets into features.
    feature_records = [
        {
            "id": str(get_field(record, "id", f"row-{index}")),
            "document_id": str(
                get_field(record, "document_id", get_field(record, "id", f"row-{index}"))
            ),
            "prompt_num": str(get_field(record, "prompt_num")),
            "prompt": str(get_field(record, "prompt", "")),
            "essay": str(get_field(record, "essay", "")),
            "score": {
                "content": 3.0,
                "organization": 3.0,
                "expression": 3.0,
                "average": 3.0,
            },
        }
        for index, record in enumerate(records)
    ]
    return build_surface_features(feature_records).to_numpy(dtype=np.float64, copy=True)


def _clip(predictions: np.ndarray, bounds: tuple[float, float] | None) -> np.ndarray:
    if bounds is None:
        return predictions
    lower, upper = bounds
    if lower >= upper:
        raise ValueError("clip bounds must be increasing")
    return np.clip(predictions, lower, upper)


class SurfaceOLSBaseline(RegressorMixin, BaseEstimator):
    """OLS on train-fitted surface scaling and optional prompt one-hot features."""

    def __init__(
        self,
        *,
        include_prompt: bool = True,
        fit_intercept: bool = True,
        clip: tuple[float, float] | None = (1.0, 5.0),
    ) -> None:
        self.include_prompt = include_prompt
        self.fit_intercept = fit_intercept
        self.clip = clip

    def _fit_features(self, records: Sequence[Any]) -> sparse.csr_matrix:
        surface = _surface_matrix(records)
        self.surface_scaler_ = StandardScaler()
        parts: list[sparse.spmatrix] = [
            sparse.csr_matrix(self.surface_scaler_.fit_transform(surface))
        ]
        self.prompt_encoder_ = None
        if self.include_prompt:
            self.prompt_encoder_ = OneHotEncoder(
                handle_unknown="ignore", sparse_output=True, dtype=np.float64
            )
            parts.append(self.prompt_encoder_.fit_transform(_prompt_column(records)))
        return sparse.hstack(parts, format="csr")

    def _transform_features(self, records: Sequence[Any]) -> sparse.csr_matrix:
        surface = _surface_matrix(records)
        parts: list[sparse.spmatrix] = [
            sparse.csr_matrix(self.surface_scaler_.transform(surface))
        ]
        if self.prompt_encoder_ is not None:
            parts.append(self.prompt_encoder_.transform(_prompt_column(records)))
        return sparse.hstack(parts, format="csr")

    def fit(self, records: Sequence[Any], y: Any = None) -> "SurfaceOLSBaseline":
        rows = list(records)
        if not rows:
            raise ValueError("cannot fit surface OLS on no records")
        targets = gold_matrix(rows) if y is None else _target_matrix(y, len(rows))
        features = self._fit_features(rows)
        self.regressor_ = LinearRegression(fit_intercept=self.fit_intercept)
        self.regressor_.fit(features, targets)
        self.n_features_in_ = int(features.shape[1])
        return self

    def predict(self, records: Sequence[Any]) -> np.ndarray:
        if not hasattr(self, "regressor_"):
            raise RuntimeError("SurfaceOLSBaseline must be fitted before predict")
        rows = list(records)
        predictions = np.asarray(
            self.regressor_.predict(self._transform_features(rows)), dtype=float
        ).reshape(len(rows), len(TRAITS))
        return _clip(predictions, self.clip)

    def predict_records(
        self, records: Sequence[Any], *, model: str = "surface_ols"
    ) -> list[dict[str, Any]]:
        rows = list(records)
        return prediction_records(rows, self.predict(rows), model=model)

    def oof_predict(self, records: Sequence[Any], folds: Any) -> np.ndarray:
        return cross_validated_predict(self, records, folds)


class TfidfRidgeBaseline(RegressorMixin, BaseEstimator):
    """Character 3--5 + word 1--2 TF-IDF with prompt/surface features.

    Every vocabulary, IDF, one-hot category, and scaling statistic is learned only
    in :meth:`fit`.  Calling this model through :func:`oof_predict` consequently
    prevents validation-fold vocabulary and distribution leakage.
    """

    def __init__(
        self,
        *,
        alpha: float = 10.0,
        char_ngram_range: tuple[int, int] = (3, 5),
        word_ngram_range: tuple[int, int] = (1, 2),
        char_min_df: int | float = 2,
        word_min_df: int | float = 2,
        prompt_min_df: int | float = 2,
        max_char_features: int | None = None,
        max_word_features: int | None = None,
        include_prompt: bool = True,
        include_prompt_text: bool = True,
        include_surface: bool = True,
        clip: tuple[float, float] | None = (1.0, 5.0),
    ) -> None:
        self.alpha = alpha
        self.char_ngram_range = char_ngram_range
        self.word_ngram_range = word_ngram_range
        self.char_min_df = char_min_df
        self.word_min_df = word_min_df
        self.prompt_min_df = prompt_min_df
        self.max_char_features = max_char_features
        self.max_word_features = max_word_features
        self.include_prompt = include_prompt
        self.include_prompt_text = include_prompt_text
        self.include_surface = include_surface
        self.clip = clip

    @staticmethod
    def _fit_vectorizer(
        vectorizer: TfidfVectorizer, texts: list[str]
    ) -> tuple[TfidfVectorizer | None, sparse.spmatrix | None]:
        try:
            return vectorizer, vectorizer.fit_transform(texts)
        except ValueError as exc:
            message = str(exc).lower()
            if "empty vocabulary" not in message and "no terms remain" not in message:
                raise
            return None, None

    def _fit_features(self, records: Sequence[Any]) -> sparse.csr_matrix:
        essays = _texts(records, "essay")
        prompts = _texts(records, "prompt")
        parts: list[sparse.spmatrix] = []

        char = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.char_ngram_range,
            min_df=self.char_min_df,
            sublinear_tf=True,
            max_features=self.max_char_features,
            dtype=np.float32,
        )
        self.char_vectorizer_, matrix = self._fit_vectorizer(char, essays)
        if matrix is not None:
            parts.append(matrix)

        word = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.word_ngram_range,
            min_df=self.word_min_df,
            sublinear_tf=True,
            max_features=self.max_word_features,
            dtype=np.float32,
        )
        self.word_vectorizer_, matrix = self._fit_vectorizer(word, essays)
        if matrix is not None:
            parts.append(matrix)

        self.prompt_text_vectorizer_ = None
        if self.include_prompt_text:
            prompt_vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=self.char_ngram_range,
                min_df=self.prompt_min_df,
                sublinear_tf=True,
                dtype=np.float32,
            )
            self.prompt_text_vectorizer_, matrix = self._fit_vectorizer(
                prompt_vectorizer, prompts
            )
            if matrix is not None:
                parts.append(matrix)

        self.prompt_encoder_ = None
        if self.include_prompt:
            self.prompt_encoder_ = OneHotEncoder(
                handle_unknown="ignore", sparse_output=True, dtype=np.float32
            )
            parts.append(self.prompt_encoder_.fit_transform(_prompt_column(records)))

        self.surface_scaler_ = None
        if self.include_surface:
            self.surface_scaler_ = StandardScaler()
            surface = self.surface_scaler_.fit_transform(_surface_matrix(records))
            parts.append(sparse.csr_matrix(surface, dtype=np.float32))

        if not parts:
            raise ValueError("TF-IDF configuration produced no feature blocks")
        return sparse.hstack(parts, format="csr", dtype=np.float32)

    def _transform_features(self, records: Sequence[Any]) -> sparse.csr_matrix:
        essays = _texts(records, "essay")
        prompts = _texts(records, "prompt")
        parts: list[sparse.spmatrix] = []
        if self.char_vectorizer_ is not None:
            parts.append(self.char_vectorizer_.transform(essays))
        if self.word_vectorizer_ is not None:
            parts.append(self.word_vectorizer_.transform(essays))
        if self.prompt_text_vectorizer_ is not None:
            parts.append(self.prompt_text_vectorizer_.transform(prompts))
        if self.prompt_encoder_ is not None:
            parts.append(self.prompt_encoder_.transform(_prompt_column(records)))
        if self.surface_scaler_ is not None:
            surface = self.surface_scaler_.transform(_surface_matrix(records))
            parts.append(sparse.csr_matrix(surface, dtype=np.float32))
        return sparse.hstack(parts, format="csr", dtype=np.float32)

    def fit(self, records: Sequence[Any], y: Any = None) -> "TfidfRidgeBaseline":
        rows = list(records)
        if not rows:
            raise ValueError("cannot fit TF-IDF Ridge on no records")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        targets = gold_matrix(rows) if y is None else _target_matrix(y, len(rows))
        features = self._fit_features(rows)
        self.regressor_ = Ridge(alpha=self.alpha, solver="lsqr")
        self.regressor_.fit(features, targets)
        self.n_features_in_ = int(features.shape[1])
        return self

    def predict(self, records: Sequence[Any]) -> np.ndarray:
        if not hasattr(self, "regressor_"):
            raise RuntimeError("TfidfRidgeBaseline must be fitted before predict")
        rows = list(records)
        predictions = np.asarray(
            self.regressor_.predict(self._transform_features(rows)), dtype=float
        ).reshape(len(rows), len(TRAITS))
        return _clip(predictions, self.clip)

    def predict_records(
        self, records: Sequence[Any], *, model: str = "tfidf_ridge"
    ) -> list[dict[str, Any]]:
        rows = list(records)
        return prediction_records(rows, self.predict(rows), model=model)

    def oof_predict(self, records: Sequence[Any], folds: Any) -> np.ndarray:
        return cross_validated_predict(self, records, folds)


def _target_matrix(y: Any, n_rows: int) -> np.ndarray:
    targets = np.asarray(y, dtype=float)
    if targets.shape != (n_rows, len(TRAITS)):
        raise ValueError(f"targets must have shape ({n_rows}, 3), got {targets.shape}")
    if not np.isfinite(targets).all():
        raise ValueError("targets contain a non-finite value")
    return targets


SurfaceOLS = SurfaceOLSBaseline
TfidfRidge = TfidfRidgeBaseline


__all__ = [
    "SurfaceOLS",
    "SurfaceOLSBaseline",
    "TfidfRidge",
    "TfidfRidgeBaseline",
]
