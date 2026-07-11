from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

from src.evaluation.metrics import TRAITS
from src.rationale.evidence import EvidenceLedger, EvidenceSpan


RATIONALE_TEMPLATE_VERSION = "grounded_fallback_v1"


def _quote(span: EvidenceSpan | None, *, limit: int = 64) -> str | None:
    if span is None:
        return None
    text = span.text.strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return f"‘{text}’"


def _scores(values: Mapping[str, Real]) -> dict[str, float]:
    if set(values) != set(TRAITS):
        raise ValueError(f"scores must contain exactly {TRAITS}")
    result = {}
    for trait in TRAITS:
        value = values[trait]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{trait} score must be numeric")
        score = float(value)
        if not math.isfinite(score) or not 1.0 <= score <= 5.0:
            raise ValueError(f"{trait} score must be finite and within [1, 5]")
        result[trait] = score
    return result


def _content(ledger: EvidenceLedger, score: float) -> str:
    stance = _quote(ledger.spans["stance"])
    support = _quote(ledger.spans["support"])
    counterpoint = _quote(ledger.spans["counterpoint"])
    first = _quote(ledger.spans["first_sentence"])
    if stance and support:
        strength = f"{stance}에서 입장을 드러내고, {support}에서 근거 전개의 단서를 제시한다"
    elif stance:
        strength = f"{stance}에서 중심 입장은 확인된다"
    elif first:
        strength = f"첫 문장 {first}에서 글의 논점을 시작한다"
    else:
        strength = "평가할 수 있는 본문 내용이 매우 제한적이다"

    if counterpoint:
        development = f"{counterpoint}처럼 다른 관점을 언급해 논의를 확장한다"
    elif ledger.support_marker_count >= 2:
        development = "둘 이상의 근거 표지가 나타나 주장 전개를 구분한다"
    else:
        development = "반론 검토나 구체 사례의 발전은 본문에서 뚜렷하게 확인되지 않는다"

    if score >= 4.0:
        return f"{strength}. {development}."
    if score >= 3.0:
        return f"{strength}. 다만 {development}."
    return (
        f"{strength}. 그러나 근거 표지가 제한적으로 확인되어, "
        "주장과 근거의 구체적 연결을 더 분명히 할 필요가 있다."
    )


def _organization(ledger: EvidenceLedger, score: float) -> str:
    first = _quote(ledger.spans["first_sentence"])
    last = _quote(ledger.spans["last_sentence"])
    conclusion = _quote(ledger.spans["conclusion"])
    opening = f"{first}로 시작해" if first else "도입부가 뚜렷하지 않은 채"
    if conclusion:
        closing = f"{conclusion}에서 결론 방향을 회수한다"
    elif last:
        closing = f"{last}로 끝나지만 명시적인 결론 표지는 확인되지 않는다"
    else:
        closing = "결말을 확인하기 어렵다"
    structure = "본문의 문단·문장 배열과 연결 표현을 통해 전개 단계를 확인할 수 있다"
    if score >= 4.0:
        return f"{opening} {closing}. {structure}, 논점의 흐름을 따라가기 쉽다."
    if score >= 3.0:
        return f"{opening} {closing}. {structure}; 일부 전환의 관계는 더 명시할 수 있다."
    return (
        f"{opening} {closing}. {structure}에 그쳐, 논점의 순서와 문단별 역할을 "
        "더 분명히 나눌 필요가 있다."
    )


def _expression(ledger: EvidenceLedger, score: float) -> str:
    sample = _quote(ledger.spans["first_sentence"])
    sample_text = f"첫 문장 {sample}은 의미를 파악할 수 있다" if sample else "문장 표본이 부족하다"
    surface_issues = ledger.repeated_space_count + ledger.repeated_punctuation_count
    if surface_issues:
        issue = "연속 공백 또는 반복 문장부호가 보여 표기 정리가 필요하다"
    elif ledger.long_sentence_count:
        issue = "길게 이어지는 문장이 있어 적절히 나누면 명료성이 높아질 수 있다"
    else:
        issue = "연속 공백·반복 문장부호·과도하게 긴 문장은 자동 점검에서 두드러지지 않는다"
    if score >= 4.0:
        return f"{sample_text}. {issue}."
    if score >= 3.0:
        if surface_issues or ledger.long_sentence_count:
            return f"{sample_text}. 다만 {issue}."
        return (
            f"{sample_text}. 표면적인 공백·문장부호 문제는 두드러지지 않지만, "
            "어휘 선택과 문장 연결의 정교함은 더 다듬을 수 있다."
        )
    return (
        f"{sample_text}. {issue}; 어휘 선택과 문장 연결을 한 차례 더 다듬어 "
        "표현의 정확성과 자연스러움을 높일 필요가 있다."
    )


def generate_grounded_rationales(
    ledger: EvidenceLedger,
    scores: Mapping[str, Real],
) -> dict[str, str]:
    canonical = _scores(scores)
    return {
        "content": _content(ledger, canonical["content"]),
        "organization": _organization(ledger, canonical["organization"]),
        "expression": _expression(ledger, canonical["expression"]),
    }
