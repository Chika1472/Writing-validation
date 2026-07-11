from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from src.data.normalize import build_model_view
from src.data.schema import EssayRecord, ensure_record
from src.data.sentence_split import split_sentences


_TOKEN_RE = re.compile(r"[가-힣]+|[A-Za-z]+|\d+(?:[.,]\d+)?")
_HANGUL_RE = re.compile(r"[가-힣]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"\d")
_PUNCTUATION_RE = re.compile(r"[^\w\s가-힣]", re.UNICODE)
_QUOTE_RE = re.compile(r"[\"'“”‘’]")
_PARENTHESIS_RE = re.compile(r"[()\[\]{}]")
_DOUBLE_SPACE_RE = re.compile(r" {2,}")
_PARAGRAPH_RE = re.compile(r"(?:\r\n?|\n)\s*(?:\r\n?|\n)+")

_CONNECTIVES = (
    "따라서",
    "그러므로",
    "그러나",
    "하지만",
    "또한",
    "반면",
    "왜냐하면",
    "한편",
    "결국",
)
_ENUMERATION_MARKERS = ("첫째", "둘째", "셋째", "첫 번째", "두 번째", "세 번째", "마지막으로")
_STANCE_MARKERS = ("찬성", "반대", "입장", "생각한다", "생각합니다", "주장한다", "주장합니다")
_COUNTERARGUMENT_MARKERS = ("물론", "반론", "반대 측", "일각에서는", "그럼에도")
_EXAMPLE_MARKERS = ("예를 들어", "예컨대", "사례", "경험", "통계", "따르면")
_CONCLUSION_MARKERS = ("결론", "결국", "따라서", "그러므로", "이상으로")


SURFACE_FEATURE_COLUMNS = (
    "essay_char_count",
    "essay_char_count_no_space",
    "essay_word_count",
    "essay_unique_word_ratio",
    "essay_sentence_count",
    "essay_mean_sentence_chars",
    "essay_std_sentence_chars",
    "essay_max_sentence_chars",
    "essay_hangul_ratio",
    "essay_latin_ratio",
    "essay_digit_ratio",
    "essay_whitespace_ratio",
    "essay_punctuation_ratio",
    "essay_comma_count",
    "essay_question_count",
    "essay_exclamation_count",
    "essay_quote_count",
    "essay_parenthesis_count",
    "essay_double_space_count",
    "essay_newline_count",
    "essay_paragraph_count",
    "essay_connective_count",
    "essay_enumeration_count",
    "essay_stance_marker_count",
    "essay_counterargument_count",
    "essay_example_marker_count",
    "essay_conclusion_marker_count",
    "prompt_char_count",
    "prompt_essay_token_jaccard",
    "prompt_token_coverage",
)


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


def _marker_count(text: str, markers: tuple[str, ...]) -> int:
    return sum(text.count(marker) for marker in markers)


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _population_std(values: list[int]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def surface_feature_row(record: EssayRecord | Mapping[str, Any]) -> dict[str, float]:
    """Build numeric text-only features, deliberately excluding IDs and cohort."""

    canonical = ensure_record(record)
    view = build_model_view(canonical)
    essay = view.essay
    raw_essay = canonical.raw_essay
    prompt = view.prompt

    essay_tokens = _tokens(essay)
    prompt_tokens = _tokens(prompt)
    essay_token_set = set(essay_tokens)
    prompt_token_set = set(prompt_tokens)
    token_union = essay_token_set | prompt_token_set
    token_intersection = essay_token_set & prompt_token_set

    sentences = split_sentences(essay)
    sentence_lengths = [len(sentence) for sentence in sentences]
    char_count = len(essay)
    non_space_count = sum(not character.isspace() for character in essay)
    whitespace_count = sum(character.isspace() for character in essay)

    values: dict[str, int | float] = {
        "essay_char_count": char_count,
        "essay_char_count_no_space": non_space_count,
        "essay_word_count": len(essay_tokens),
        "essay_unique_word_ratio": _ratio(len(essay_token_set), len(essay_tokens)),
        "essay_sentence_count": len(sentences),
        "essay_mean_sentence_chars": _ratio(sum(sentence_lengths), len(sentence_lengths)),
        "essay_std_sentence_chars": _population_std(sentence_lengths),
        "essay_max_sentence_chars": max(sentence_lengths, default=0),
        "essay_hangul_ratio": _ratio(len(_HANGUL_RE.findall(essay)), char_count),
        "essay_latin_ratio": _ratio(len(_LATIN_RE.findall(essay)), char_count),
        "essay_digit_ratio": _ratio(len(_DIGIT_RE.findall(essay)), char_count),
        "essay_whitespace_ratio": _ratio(whitespace_count, char_count),
        "essay_punctuation_ratio": _ratio(len(_PUNCTUATION_RE.findall(essay)), char_count),
        "essay_comma_count": essay.count(",") + essay.count("，"),
        "essay_question_count": essay.count("?"),
        "essay_exclamation_count": essay.count("!"),
        "essay_quote_count": len(_QUOTE_RE.findall(essay)),
        "essay_parenthesis_count": len(_PARENTHESIS_RE.findall(essay)),
        "essay_double_space_count": len(_DOUBLE_SPACE_RE.findall(raw_essay)),
        "essay_newline_count": raw_essay.count("\n"),
        "essay_paragraph_count": len(_PARAGRAPH_RE.findall(raw_essay)) + 1,
        "essay_connective_count": _marker_count(essay, _CONNECTIVES),
        "essay_enumeration_count": _marker_count(essay, _ENUMERATION_MARKERS),
        "essay_stance_marker_count": _marker_count(essay, _STANCE_MARKERS),
        "essay_counterargument_count": _marker_count(essay, _COUNTERARGUMENT_MARKERS),
        "essay_example_marker_count": _marker_count(essay, _EXAMPLE_MARKERS),
        "essay_conclusion_marker_count": _marker_count(essay, _CONCLUSION_MARKERS),
        "prompt_char_count": len(prompt),
        "prompt_essay_token_jaccard": _ratio(len(token_intersection), len(token_union)),
        "prompt_token_coverage": _ratio(len(token_intersection), len(prompt_token_set)),
    }
    return {column: float(values[column]) for column in SURFACE_FEATURE_COLUMNS}


def build_surface_features(
    records: Sequence[EssayRecord | Mapping[str, Any]],
) -> pd.DataFrame:
    """Return one ordered, all-numeric feature row per input record."""

    rows = [surface_feature_row(record) for record in records]
    return pd.DataFrame(rows, columns=SURFACE_FEATURE_COLUMNS, dtype=float)
