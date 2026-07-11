"""Strict construction and parsing of challenge prediction JSON."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any


TRAITS = ("content", "organization", "expression")
FIELDS = ("score", "rationale")


class PredictionValidationError(ValueError):
    """Raised when a prediction does not match the exact submission schema."""


def validate_prediction(payload: Any) -> dict[str, dict[str, float | str]]:
    """Validate and return a canonical, JSON-serializable prediction dictionary."""

    if not isinstance(payload, Mapping):
        raise PredictionValidationError("Prediction must be a JSON object.")
    if set(payload) != set(TRAITS):
        raise PredictionValidationError(f"Prediction keys must be exactly {TRAITS}.")

    canonical: dict[str, dict[str, float | str]] = {}
    for trait in TRAITS:
        item = payload[trait]
        if not isinstance(item, Mapping) or set(item) != set(FIELDS):
            raise PredictionValidationError(
                f"{trait} must be an object with exactly the keys {FIELDS}."
            )
        score = item["score"]
        if isinstance(score, bool) or not isinstance(score, Real):
            raise PredictionValidationError(f"{trait}.score must be a real number, not {type(score).__name__}.")
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise PredictionValidationError(f"{trait}.score must be finite.")
        if not 1.0 <= numeric_score <= 5.0:
            raise PredictionValidationError(f"{trait}.score must be between 1 and 5.")
        rationale = item["rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise PredictionValidationError(f"{trait}.rationale must be a nonempty string.")
        canonical[trait] = {"score": numeric_score, "rationale": rationale.strip()}
    return canonical


def build_prediction(
    scores: Mapping[str, Real],
    rationales: Mapping[str, str],
    *,
    decimals: int | None = None,
) -> dict[str, dict[str, float | str]]:
    """Assemble the exact outer schema from independently produced scores and rationales."""

    if set(scores) != set(TRAITS) or set(rationales) != set(TRAITS):
        raise PredictionValidationError(f"scores and rationales must contain exactly {TRAITS}.")
    if decimals is not None and (
        isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0
    ):
        raise ValueError("decimals must be None or a non-negative integer.")
    unrounded = {
        trait: {
            "score": scores[trait],
            "rationale": rationales[trait],
        }
        for trait in TRAITS
    }
    canonical = validate_prediction(unrounded)
    if decimals is not None:
        for item in canonical.values():
            item["score"] = round(float(item["score"]), decimals)
    return canonical


def serialize_prediction(payload: Any) -> str:
    """Serialize only a valid prediction, with Korean text preserved and no wrapper."""

    canonical = validate_prediction(payload)
    return json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def serialize_scores(
    scores: Mapping[str, Real],
    rationales: Mapping[str, str],
    *,
    decimals: int | None = None,
) -> str:
    """Build and serialize a prediction in one call."""

    return serialize_prediction(build_prediction(scores, rationales, decimals=decimals))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PredictionValidationError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def _reject_nonstandard_number(value: str) -> None:
    raise PredictionValidationError(f"Invalid JSON numeric constant: {value}.")


def strict_parse_prediction(text: str) -> dict[str, dict[str, float | str]]:
    """Parse standards-compliant JSON and enforce the exact prediction schema."""

    if not isinstance(text, str):
        raise PredictionValidationError("Serialized prediction must be text.")
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_number,
        )
    except PredictionValidationError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise PredictionValidationError(f"Invalid prediction JSON: {error}.") from error
    return validate_prediction(payload)


def is_valid_prediction_json(text: str) -> bool:
    """Return whether ``text`` passes strict parsing and schema validation."""

    try:
        strict_parse_prediction(text)
    except PredictionValidationError:
        return False
    return True


def build_fallback_rationale(strength_evidence: str, limitation_evidence: str) -> str:
    """Build a deterministic rationale from two already-grounded evidence summaries."""

    if not isinstance(strength_evidence, str) or not strength_evidence.strip():
        raise PredictionValidationError("strength_evidence must be a nonempty string.")
    if not isinstance(limitation_evidence, str) or not limitation_evidence.strip():
        raise PredictionValidationError("limitation_evidence must be a nonempty string.")
    strength = strength_evidence.strip().rstrip(".!?")
    limitation = limitation_evidence.strip().rstrip(".!?")
    return f"{strength}. 다만 {limitation}."
