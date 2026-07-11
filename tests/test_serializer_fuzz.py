import json
import math
import random

import pytest

from src.inference.serializer import (
    PredictionValidationError,
    build_fallback_rationale,
    build_prediction,
    is_valid_prediction_json,
    serialize_prediction,
    serialize_scores,
    strict_parse_prediction,
    validate_prediction,
)


TRAITS = ("content", "organization", "expression")


def valid_payload(score: float = 3.0) -> dict:
    return {
        trait: {"score": score, "rationale": f"{trait}에 대한 실제 글 근거"}
        for trait in TRAITS
    }


def test_serializer_is_compact_unicode_and_exact_schema() -> None:
    text = serialize_prediction(valid_payload())
    assert "글 근거" in text
    assert "\\u" not in text
    assert " " not in text.split('"rationale":"', 1)[0]
    assert text.startswith('{"content":{"score":3.0,"rationale":')
    assert strict_parse_prediction(text) == validate_prediction(valid_payload())


def test_builder_preserves_scores_by_default_and_rounds_only_when_requested() -> None:
    scores = {trait: 3.14159 for trait in TRAITS}
    rationales = {trait: "  구체적인 근거  " for trait in TRAITS}
    payload = build_prediction(scores, rationales)
    assert all(item["score"] == 3.14159 for item in payload.values())
    assert all(item["rationale"] == "구체적인 근거" for item in payload.values())
    assert strict_parse_prediction(serialize_scores(scores, rationales)) == payload
    rounded = build_prediction(scores, rationales, decimals=3)
    assert all(item["score"] == 3.142 for item in rounded.values())
    with pytest.raises(PredictionValidationError):
        build_prediction({trait: "3" for trait in TRAITS}, rationales)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf"), 0.999, 5.001, True, "3"])
def test_invalid_scores_are_rejected(score) -> None:
    with pytest.raises(PredictionValidationError):
        validate_prediction(valid_payload(score))


@pytest.mark.parametrize("rationale", ["", "   ", None, 123])
def test_invalid_rationales_are_rejected(rationale) -> None:
    payload = valid_payload()
    payload["content"]["rationale"] = rationale
    with pytest.raises(PredictionValidationError):
        validate_prediction(payload)


def test_extra_missing_duplicate_and_nonstandard_json_are_rejected() -> None:
    extra = valid_payload()
    extra["total"] = {"score": 3.0, "rationale": "불필요"}
    assert not is_valid_prediction_json(json.dumps(extra, ensure_ascii=False))
    assert not is_valid_prediction_json('{"content":{},"content":{},"organization":{},"expression":{}}')
    assert not is_valid_prediction_json(serialize_prediction(valid_payload()).replace("3.0", "NaN", 1))
    assert not is_valid_prediction_json("```json\n{}\n```")


def test_score_and_unicode_fuzz_round_trip() -> None:
    generator = random.Random(42)
    fragments = ["한글", '따옴표 "', "역슬래시 \\", "줄바꿈\n", "이모지 🙂"]
    for _ in range(1000):
        scores = {trait: generator.uniform(1.0, 5.0) for trait in TRAITS}
        rationales = {
            trait: f"{generator.choice(fragments)} 실제 근거 {generator.randrange(10000)}"
            for trait in TRAITS
        }
        text = serialize_scores(scores, rationales)
        parsed = strict_parse_prediction(text)
        assert set(parsed) == set(TRAITS)
        assert all(math.isfinite(item["score"]) and 1.0 <= item["score"] <= 5.0 for item in parsed.values())
        assert all(item["rationale"] for item in parsed.values())


def test_deterministic_fallback_uses_only_supplied_evidence() -> None:
    rationale = build_fallback_rationale("입장이 첫 문장에 명시되어 있다.", "사례가 제시되지 않았다!")
    assert rationale == "입장이 첫 문장에 명시되어 있다. 다만 사례가 제시되지 않았다."
    with pytest.raises(PredictionValidationError):
        build_fallback_rationale("", "한계")
