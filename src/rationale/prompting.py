from __future__ import annotations

import json
from typing import Any, Mapping

from src.data.schema import EssayInput, EssayRecord, ensure_essay_input
from src.evaluation.metrics import TRAITS
from src.rationale.evidence import EvidenceLedger


RATIONALE_SYSTEM_PROMPT = """너는 한국어 논증문 채점 근거 작성기이다.
입력의 세 점수는 이미 확정되어 있으며 절대로 수정하거나 다시 판단하지 않는다.
essay와 evidence에 실제로 있는 내용만 사용하고, 존재하지 않는 사례·통계·오류를 만들지 않는다.
각 영역은 서로 다른 관찰을 제공하며, 정확히 지정된 JSON 객체만 출력한다."""

RATIONALE_USER_TEMPLATE = """[고정 점수]
{scores_json}

[평가 영역]
content: 과제 수행, 입장, 주장·근거의 관련성·타당성·구체성·발전
organization: 서론·본론·결론의 역할, 논점 순서, 통일성, 연결, 결론 회수
expression: 문장 정확성·명료성, 문법·맞춤법·띄어쓰기, 어휘, 자연스러움

[논제]
{prompt}

[본문]
{essay}

[사용 가능한 exact evidence]
{evidence_json}

[출력 계약]
{{"content":"근거", "organization":"근거", "expression":"근거"}}
점수를 출력하지 말고 세 근거 문자열만 출력한다. 각 근거에는 evidence의 실제 문구를 짧게 인용한다.
signals의 횟수는 내부 판단에만 쓰고, 그 숫자가 본문에 직접 적혀 있지 않으면 근거 문장에 숫자로 쓰지 않는다."""

RATIONALE_PROMPT_CONTRACT = (
    RATIONALE_SYSTEM_PROMPT
    + "\n"
    + RATIONALE_USER_TEMPLATE
    + "\n[CHAT_TEMPLATE_FLAGS]\nadd_generation_prompt=true;enable_thinking=false"
)


def _score_payload(scores: Mapping[str, float]) -> dict[str, float]:
    if set(scores) != set(TRAITS):
        raise ValueError(f"scores must contain exactly {TRAITS}")
    return {trait: float(scores[trait]) for trait in TRAITS}


def _evidence_payload(ledger: EvidenceLedger) -> dict[str, Any]:
    return {
        "spans": {
            key: span.text if span is not None else None
            for key, span in ledger.spans.items()
        },
        "signals": {
            "sentence_count": ledger.sentence_count,
            "paragraph_count": ledger.paragraph_count,
            "connective_count": ledger.connective_count,
            "support_marker_count": ledger.support_marker_count,
            "repeated_space_count": ledger.repeated_space_count,
            "repeated_punctuation_count": ledger.repeated_punctuation_count,
            "long_sentence_count": ledger.long_sentence_count,
        },
    }


def build_rationale_messages(
    value: EssayInput | EssayRecord | dict[str, Any],
    scores: Mapping[str, float],
    ledger: EvidenceLedger,
) -> list[dict[str, str]]:
    record = ensure_essay_input(value)
    if ledger.record_id != record.id or ledger.prompt_num != record.prompt_num:
        raise ValueError("evidence ledger does not belong to the supplied essay")
    user = RATIONALE_USER_TEMPLATE.format(
        scores_json=json.dumps(_score_payload(scores), ensure_ascii=False, separators=(",", ":")),
        prompt=record.prompt,
        essay=record.essay,
        evidence_json=json.dumps(
            _evidence_payload(ledger), ensure_ascii=False, separators=(",", ":")
        ),
    )
    return [
        {"role": "system", "content": RATIONALE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def render_rationale_prompt(
    tokenizer: Any,
    value: EssayInput | EssayRecord | dict[str, Any],
    scores: Mapping[str, float],
    ledger: EvidenceLedger,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            build_rationale_messages(value, scores, ledger),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as error:
        raise RuntimeError(
            "The tokenizer must support Qwen3 enable_thinking=False."
        ) from error
