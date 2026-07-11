from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any


TRAITS = ("content", "organization", "expression")
SCORE_FIELDS = (*TRAITS, "average")
INPUT_FIELDS = ("id", "document_id", "prompt_num", "prompt", "essay")
RECORD_FIELDS = (*INPUT_FIELDS, "score")
MIN_SCORE = 1.0
MAX_SCORE = 5.0


class SchemaError(ValueError):
    """Raised when a dataset row does not match the canonical schema."""


def _require_fields(data: Mapping[str, Any], fields: tuple[str, ...], *, where: str) -> None:
    missing = [field for field in fields if field not in data]
    if missing:
        raise SchemaError(f"{where} is missing required field(s): {', '.join(missing)}")


def _required_text(data: Mapping[str, Any], field: str) -> str:
    value = data[field]
    if not isinstance(value, str):
        raise SchemaError(f"{field} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise SchemaError(f"{field} must not be empty")
    return value


def _bounded_score(data: Mapping[str, Any], field: str) -> float:
    value = data[field]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SchemaError(f"score.{field} must be a real number")
    score = float(value)
    if not math.isfinite(score):
        raise SchemaError(f"score.{field} must be finite")
    if not MIN_SCORE <= score <= MAX_SCORE:
        raise SchemaError(
            f"score.{field} must be in [{MIN_SCORE}, {MAX_SCORE}], got {score}"
        )
    return score


@dataclass(frozen=True, slots=True)
class EssayScores:
    content: float
    organization: float
    expression: float
    average: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> EssayScores:
        if not isinstance(data, Mapping):
            raise SchemaError(f"score must be an object, got {type(data).__name__}")
        _require_fields(data, SCORE_FIELDS, where="score")
        return cls(**{field: _bounded_score(data, field) for field in SCORE_FIELDS})

    @property
    def trait_values(self) -> tuple[float, float, float]:
        return (self.content, self.organization, self.expression)

    @property
    def computed_average(self) -> float:
        """Mean of the three official traits; ``average`` is diagnostic only."""

        return sum(self.trait_values) / len(self.trait_values)

    def to_dict(self) -> dict[str, float]:
        return {field: getattr(self, field) for field in SCORE_FIELDS}


@dataclass(frozen=True, slots=True)
class EssayInput:
    """An essay that can be scored, without requiring unavailable test labels."""

    id: str
    document_id: str
    prompt_num: str
    prompt: str
    essay: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EssayInput":
        if not isinstance(data, Mapping):
            raise SchemaError(f"dataset row must be an object, got {type(data).__name__}")
        _require_fields(data, INPUT_FIELDS, where="dataset row")
        return cls(**{field: _required_text(data, field) for field in INPUT_FIELDS})

    def to_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in INPUT_FIELDS}


@dataclass(frozen=True, slots=True)
class EssayRecord:
    id: str
    document_id: str
    prompt_num: str
    prompt: str
    essay: str
    score: EssayScores

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> EssayRecord:
        if not isinstance(data, Mapping):
            raise SchemaError(f"dataset row must be an object, got {type(data).__name__}")
        _require_fields(data, RECORD_FIELDS, where="dataset row")
        return cls(
            id=_required_text(data, "id"),
            document_id=_required_text(data, "document_id"),
            prompt_num=_required_text(data, "prompt_num"),
            prompt=_required_text(data, "prompt"),
            essay=_required_text(data, "essay"),
            score=EssayScores.from_mapping(data["score"]),
        )

    @property
    def raw_essay(self) -> str:
        """The essay exactly as decoded from JSON, including surrounding spaces."""

        return self.essay

    @property
    def raw_prompt(self) -> str:
        return self.prompt

    @property
    def scoring_input(self) -> EssayInput:
        return EssayInput(
            id=self.id,
            document_id=self.document_id,
            prompt_num=self.prompt_num,
            prompt=self.prompt,
            essay=self.essay,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "prompt_num": self.prompt_num,
            "prompt": self.prompt,
            "essay": self.essay,
            "score": self.score.to_dict(),
        }


def ensure_record(value: EssayRecord | Mapping[str, Any]) -> EssayRecord:
    if isinstance(value, EssayRecord):
        return value
    return EssayRecord.from_mapping(value)


def ensure_essay_input(
    value: EssayInput | EssayRecord | Mapping[str, Any],
) -> EssayInput:
    if isinstance(value, EssayInput):
        return value
    if isinstance(value, EssayRecord):
        return value.scoring_input
    return EssayInput.from_mapping(value)
