from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

from src.inference.serializer import TRAITS, serialize_prediction, strict_parse_prediction


def finalize_prediction(
    scores: Mapping[str, Real],
    rationales: Mapping[str, str],
) -> dict[str, dict[str, float | str]]:
    """Attach rationales without rounding, regenerating, or otherwise changing scores."""

    if set(scores) != set(TRAITS) or set(rationales) != set(TRAITS):
        raise ValueError(f"scores and rationales must contain exactly {TRAITS}")
    payload = {
        trait: {
            "score": scores[trait],
            "rationale": rationales[trait],
        }
        for trait in TRAITS
    }
    serialized = serialize_prediction(payload)
    canonical = strict_parse_prediction(serialized)
    for trait in TRAITS:
        if float(canonical[trait]["score"]) != float(scores[trait]):
            raise RuntimeError(f"finalization changed the {trait} score")
    return canonical


def final_prediction_row(
    *,
    record_id: str,
    prompt_num: str,
    model: str,
    scores: Mapping[str, Real],
    rationales: Mapping[str, str],
) -> dict[str, Any]:
    if not record_id or not prompt_num or not model:
        raise ValueError("record_id, prompt_num, and model must be nonempty")
    return {
        "id": str(record_id),
        "prompt_num": str(prompt_num),
        "prediction": finalize_prediction(scores, rationales),
        "model": str(model),
    }
