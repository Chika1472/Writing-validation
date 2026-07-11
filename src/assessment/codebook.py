"""Fail-closed validation of ordinal answer codes in exact assistant context."""

from __future__ import annotations

from typing import Any, Sequence

from src.assessment.prompting import render_assessment_prompt
from src.assessment.questions import QUESTIONS
from src.data.schema import EssayInput
from src.utils.hashing import sha256_json, sha256_text


DEFAULT_ANSWER_CODES = ("A", "B", "C", "D", "E")
ANSWER_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0)

_SENTINEL = EssayInput(
    id="assessment-codebook-sentinel",
    document_id="assessment-codebook-sentinel",
    prompt_num="SENTINEL",
    prompt="한 가지 사회적 쟁점에 대한 자신의 입장을 논리적으로 쓰시오.",
    essay="나는 이 쟁점에 찬성한다. 그 이유는 사회적 편익이 비용보다 크기 때문이다.",
)


def _input_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("tokenizer returned an unexpected batched encoding")
        ids = ids[0]
    return [int(value) for value in ids]


def single_token_code_ids(
    tokenizer: Any,
    prefix: str,
    answer_codes: Sequence[str] = DEFAULT_ANSWER_CODES,
) -> tuple[int, ...]:
    """Resolve codes only when each appends exactly one distinct token to prefix."""

    codes = tuple(answer_codes)
    if len(codes) != 5 or any(not isinstance(code, str) or not code for code in codes):
        raise ValueError("answer_codes must contain five nonempty strings")
    if len(set(codes)) != len(codes):
        raise ValueError("answer_codes must be unique")
    prefix_ids = _input_ids(tokenizer, prefix)
    resolved: list[int] = []
    for code in codes:
        combined = _input_ids(tokenizer, prefix + code)
        if combined[: len(prefix_ids)] != prefix_ids or len(combined) != len(prefix_ids) + 1:
            raise ValueError(
                f"answer code {code!r} is not exactly one appended token in assistant context"
            )
        resolved.append(combined[-1])
    if len(set(resolved)) != len(resolved):
        raise ValueError("answer codes must resolve to five distinct token ids")
    return tuple(resolved)


def validate_codebook(
    tokenizer: Any,
    answer_codes: Sequence[str] = DEFAULT_ANSWER_CODES,
) -> dict:
    """Validate every versioned question against the exact Qwen chat prefix."""

    expected_ids: tuple[int, ...] | None = None
    context_hashes: dict[str, str] = {}
    for question in QUESTIONS:
        prefix = render_assessment_prompt(tokenizer, _SENTINEL, question)
        token_ids = single_token_code_ids(tokenizer, prefix, answer_codes)
        if expected_ids is None:
            expected_ids = token_ids
        elif token_ids != expected_ids:
            raise ValueError(
                "answer-code token ids changed across assessment question contexts"
            )
        context_hashes[question.question_id] = sha256_text(prefix)
    assert expected_ids is not None
    payload = {
        "answer_codes": list(answer_codes),
        "answer_values": list(ANSWER_VALUES),
        "answer_token_ids": list(expected_ids),
        "context_hashes": context_hashes,
    }
    return {**payload, "codebook_sha256": sha256_json(payload)}


__all__ = [
    "ANSWER_VALUES",
    "DEFAULT_ANSWER_CODES",
    "single_token_code_ids",
    "validate_codebook",
]
