from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.predictions import read_final_predictions
from src.inference.finalize import finalize_prediction
from src.inference.serializer import serialize_prediction, strict_parse_prediction
from src.rationale.deterministic import generate_grounded_rationales
from src.rationale.evidence import build_evidence_ledger
from src.rationale.parsing import assess_grounding
from src.train.rationale_dataset import RationaleSFTDataset


def _input() -> dict:
    return {
        "id": "essay-1",
        "document_id": "document-1",
        "prompt_num": "Q1",
        "prompt": "제도 도입에 대한 의견을 쓰시오.",
        "essay": (
            "저는 이 제도에 찬성합니다. 첫째, 피해를 줄일 수 있기 때문입니다. "
            "하지만 비용 문제도 검토해야 합니다. 따라서 단계적으로 도입해야 합니다."
        ),
    }


def test_evidence_spans_roundtrip_and_rationales_are_nonempty() -> None:
    source = _input()
    ledger = build_evidence_ledger(source)
    for span in ledger.spans.values():
        if span is not None:
            assert source["essay"][span.start : span.end] == span.text

    rationales = generate_grounded_rationales(
        ledger,
        {"content": 4.1, "organization": 3.8, "expression": 3.6},
    )
    assert set(rationales) == {"content", "organization", "expression"}
    assert all(value.strip() for value in rationales.values())
    grounding = assess_grounding(rationales, essay=source["essay"], ledger=ledger)
    assert grounding.accepted, grounding.reasons


def test_finalization_preserves_scores_and_strict_schema() -> None:
    scores = {"content": 3.123456, "organization": 3.25, "expression": 4.0}
    rationales = {
        "content": "본문의 입장과 근거를 확인했다.",
        "organization": "문장 간 연결을 확인했다.",
        "expression": "표현의 명료성을 확인했다.",
    }
    payload = finalize_prediction(scores, rationales)
    restored = strict_parse_prediction(serialize_prediction(payload))

    assert {trait: restored[trait]["score"] for trait in scores} == scores


def test_exact_final_row_reader(tmp_path: Path) -> None:
    scores = {"content": 3.1, "organization": 3.2, "expression": 3.3}
    rationales = {trait: f"{trait} 본문 근거" for trait in scores}
    path = tmp_path / "final.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "essay-1",
                "prompt_num": "Q1",
                "prediction": finalize_prediction(scores, rationales),
                "model": "final-v1",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = read_final_predictions(path)
    assert rows[0]["prediction"]["content"]["score"] == 3.1


def test_grounding_requires_exact_evidence_for_every_trait() -> None:
    source = _input()
    ledger = build_evidence_ledger(source)
    generic = {
        "content": "입장이 분명하고 근거가 충분하여 내용 전개가 매우 타당하다.",
        "organization": "서론과 본론과 결론이 자연스럽게 연결되어 흐름이 좋다.",
        "expression": "문장이 명료하며 어휘와 문법 사용이 전체적으로 자연스럽다.",
    }
    report = assess_grounding(generic, essay=source["essay"], ledger=ledger)
    assert not report.accepted
    assert set(report.exact_evidence_hits) == {
        "content",
        "organization",
        "expression",
    }


class _RationaleTokenizer:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        assert tokenize is True
        assert enable_thinking is False
        prompt_messages = [message for message in messages if message["role"] != "assistant"]
        prefix = "|".join(message["content"] for message in prompt_messages) + "|ASSISTANT|"
        if add_generation_prompt:
            rendered = prefix
        else:
            rendered = prefix + messages[-1]["content"]
        return [ord(character) for character in rendered]


def test_rationale_dataset_adds_deterministic_small_score_jitter() -> None:
    source = _input()
    ledger = build_evidence_ledger(source)
    scores = {"content": 3.2, "organization": 3.3, "expression": 3.4}
    rows = [
        {
            "id": source["id"],
            "prompt_num": source["prompt_num"],
            "conditioned_scores": scores,
            "rationales": generate_grounded_rationales(ledger, scores),
            "evidence": ledger.to_dict(),
        }
    ]
    dataset = RationaleSFTDataset(
        {source["id"]: source},
        rows,
        _RationaleTokenizer(),
        max_length=20_000,
        score_jitter=0.08,
        score_jitter_copies=1,
        jitter_seed=42,
    )
    assert len(dataset) == 2
    assert dataset[0]["id"] == source["id"]
    assert dataset[1]["id"] == source["id"] + "#score_jitter_1"
