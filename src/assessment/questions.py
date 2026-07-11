"""Versioned Korean rubric questions for ordinal-logit feature extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.utils.hashing import sha256_json


QUESTION_VERSION = "korean_aq_v1"
TRAITS = ("content", "organization", "expression")


@dataclass(frozen=True, slots=True)
class AssessmentQuestion:
    question_id: str
    trait: str
    text: str


QUESTIONS: tuple[AssessmentQuestion, ...] = (
    AssessmentQuestion("content_1", "content", "글이 논제에 직접 답하며 필자의 입장이 분명하게 식별되는가?"),
    AssessmentQuestion("content_2", "content", "핵심 주장과 세부 논점이 논제와 직접 관련되는가?"),
    AssessmentQuestion("content_3", "content", "근거가 주장의 반복을 넘어 이유, 사례 또는 자료로 충분히 발전하는가?"),
    AssessmentQuestion("content_4", "content", "제시한 근거와 최종 결론 사이의 인과적·정책적 연결이 타당한가?"),
    AssessmentQuestion("content_5", "content", "예상 가능한 반론, 조건 또는 한계를 적절히 다루는가?"),
    AssessmentQuestion("content_6", "content", "논제의 배경 설명을 되풀이하는 데 그치지 않고 독립적인 내용을 제시하는가?"),
    AssessmentQuestion("organization_1", "organization", "도입에서 글의 쟁점과 입장이 효과적으로 설정되는가?"),
    AssessmentQuestion("organization_2", "organization", "본론의 논점이 명확히 구분되고 자연스러운 순서로 전개되는가?"),
    AssessmentQuestion("organization_3", "organization", "각 문단과 근거가 하나의 중심 주장에 일관되게 기여하는가?"),
    AssessmentQuestion("organization_4", "organization", "전환·지시·인과 표현이 실제 논리 관계에 맞게 사용되는가?"),
    AssessmentQuestion("organization_5", "organization", "결론이 앞선 논증을 적절히 회수하며 새로운 논점을 갑자기 추가하지 않는가?"),
    AssessmentQuestion("organization_6", "organization", "불필요한 반복, 논리적 비약, 역순 전개 또는 주제 이탈 없이 글이 응집되어 있는가?"),
    AssessmentQuestion("expression_1", "expression", "조사·어미·호응·문장성분이 정확하여 의미를 쉽게 이해할 수 있는가?"),
    AssessmentQuestion("expression_2", "expression", "맞춤법·띄어쓰기·오탈자가 적고 글의 의미를 방해하지 않는가?"),
    AssessmentQuestion("expression_3", "expression", "문장 길이와 중첩 정도가 적절하여 읽기 쉬운가?"),
    AssessmentQuestion("expression_4", "expression", "어휘 선택이 정확하고 글 전체의 문체가 일관되는가?"),
    AssessmentQuestion("expression_5", "expression", "동일한 표현·접속어·문장 틀의 불필요한 반복이 적은가?"),
    AssessmentQuestion("expression_6", "expression", "글 전체의 문장이 자연스럽고 명료한가?"),
)


def questions_for_trait(trait: str) -> tuple[AssessmentQuestion, ...]:
    if trait not in TRAITS:
        raise ValueError(f"unknown assessment trait: {trait!r}")
    return tuple(question for question in QUESTIONS if question.trait == trait)


def question_contract() -> dict:
    return {
        "version": QUESTION_VERSION,
        "questions": [asdict(question) for question in QUESTIONS],
    }


QUESTIONS_SHA256 = sha256_json(question_contract())
QUESTION_IDS = tuple(question.question_id for question in QUESTIONS)


if len(set(QUESTION_IDS)) != len(QUESTION_IDS):
    raise RuntimeError("assessment question ids must be unique")
if any(len(questions_for_trait(trait)) != 6 for trait in TRAITS):
    raise RuntimeError("assessment question v1 must contain six questions per trait")


__all__ = [
    "AssessmentQuestion",
    "QUESTION_IDS",
    "QUESTION_VERSION",
    "QUESTIONS",
    "QUESTIONS_SHA256",
    "TRAITS",
    "question_contract",
    "questions_for_trait",
]
