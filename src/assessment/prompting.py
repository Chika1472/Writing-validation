"""Qwen3 non-thinking prompt contract for one ordinal assessment question."""

from __future__ import annotations

from typing import Any

from src.assessment.questions import AssessmentQuestion, QUESTION_VERSION
from src.data.schema import EssayInput, EssayRecord, ensure_essay_input
from src.utils.hashing import sha256_text


SYSTEM_PROMPT = """너는 한국어 논증적 글을 분석하는 엄격한 평가 보조자이다.
한 번에 하나의 평가 질문에만 답하라. 점수나 설명 문장을 생성하지 말고 지정된 응답 코드 하나만 선택하라."""

USER_TEMPLATE = """[논제]
{prompt}

[학생 글]
{essay}

[평가 영역]
{trait}

[평가 질문]
{question}

[응답 척도]
A: 전혀 그렇지 않다
B: 대체로 그렇지 않다
C: 보통이다
D: 대체로 그렇다
E: 매우 그렇다

[응답 규칙]
A, B, C, D, E 중 하나만 출력하라."""

CHAT_TEMPLATE_FLAGS = "add_generation_prompt=true;enable_thinking=false"
TRAIT_LABELS = {
    "content": "내용(Content)",
    "organization": "구성(Organization)",
    "expression": "표현(Expression)",
}
ASSESSMENT_QUERY_CONTRACT = (
    QUESTION_VERSION
    + "\n"
    + SYSTEM_PROMPT
    + "\n"
    + USER_TEMPLATE
    + "\n"
    + repr(TRAIT_LABELS)
    + "\n"
    + CHAT_TEMPLATE_FLAGS
)
ASSESSMENT_QUERY_SHA256 = sha256_text(ASSESSMENT_QUERY_CONTRACT)


def build_assessment_messages(
    record: EssayInput | EssayRecord | dict[str, Any],
    question: AssessmentQuestion,
) -> list[dict[str, str]]:
    canonical = ensure_essay_input(record)
    user = USER_TEMPLATE.format(
        prompt=canonical.prompt,
        essay=canonical.essay,
        trait=TRAIT_LABELS[question.trait],
        question=question.text,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def render_assessment_prompt(
    tokenizer: Any,
    record: EssayInput | EssayRecord | dict[str, Any],
    question: AssessmentQuestion,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            build_assessment_messages(record, question),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as error:
        raise RuntimeError(
            "The tokenizer must expose Qwen3's enable_thinking chat-template switch."
        ) from error


__all__ = [
    "ASSESSMENT_QUERY_CONTRACT",
    "ASSESSMENT_QUERY_SHA256",
    "build_assessment_messages",
    "render_assessment_prompt",
]
