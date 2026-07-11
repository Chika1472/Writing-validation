from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from src.data.schema import EssayInput, EssayRecord, ensure_essay_input


STANCE_MARKERS = (
    "찬성",
    "반대",
    "생각한다",
    "생각합니다",
    "주장한다",
    "필요하다",
    "필요합니다",
    "해야 한다",
    "해야 합니다",
)
SUPPORT_MARKERS = (
    "왜냐하면",
    "예를 들어",
    "예를 들면",
    "사례",
    "근거",
    "첫째",
    "둘째",
    "셋째",
)
CONNECTIVE_MARKERS = (
    "따라서",
    "그러므로",
    "하지만",
    "그러나",
    "반면",
    "또한",
    "그리고",
    "우선",
    "다음으로",
    "결론적으로",
)
COUNTER_MARKERS = ("하지만", "그러나", "반면", "물론", "그럼에도")
CONCLUSION_MARKERS = ("결론적으로", "따라서", "그러므로", "이상으로", "결국")


@dataclass(frozen=True)
class EvidenceSpan:
    kind: str
    start: int
    end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceLedger:
    record_id: str
    prompt_num: str
    essay_char_count: int
    sentence_count: int
    paragraph_count: int
    token_count: int
    lexical_diversity: float
    connective_count: int
    support_marker_count: int
    repeated_space_count: int
    repeated_punctuation_count: int
    long_sentence_count: int
    spans: dict[str, EvidenceSpan | None]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spans"] = {
            key: span.to_dict() if span is not None else None
            for key, span in self.spans.items()
        }
        return payload


def _sentence_spans(text: str) -> list[EvidenceSpan]:
    spans: list[EvidenceSpan] = []
    for match in re.finditer(r"[^.!?。！？\n]+(?:[.!?。！？]+|(?=\n)|$)", text):
        raw = match.group(0)
        leading = len(raw) - len(raw.lstrip())
        trailing_text = raw.rstrip()
        start = match.start() + leading
        end = match.start() + len(trailing_text)
        if start < end:
            spans.append(
                EvidenceSpan(
                    kind="sentence",
                    start=start,
                    end=end,
                    text=text[start:end],
                )
            )
    if not spans and text.strip():
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())
        spans.append(EvidenceSpan("sentence", start, end, text[start:end]))
    return spans


def _first_with_marker(
    sentences: list[EvidenceSpan],
    markers: tuple[str, ...],
    *,
    kind: str,
) -> EvidenceSpan | None:
    for sentence in sentences:
        if any(marker in sentence.text for marker in markers):
            return EvidenceSpan(kind, sentence.start, sentence.end, sentence.text)
    return None


def build_evidence_ledger(
    value: EssayInput | EssayRecord | dict[str, Any],
) -> EvidenceLedger:
    record = ensure_essay_input(value)
    essay = record.essay
    sentences = _sentence_spans(essay)
    paragraphs = [part for part in re.split(r"\n\s*\n", essay) if part.strip()]
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", essay)
    unique_tokens = len(set(tokens))
    marker_count = lambda markers: sum(essay.count(marker) for marker in markers)
    spans: dict[str, EvidenceSpan | None] = {
        "first_sentence": sentences[0] if sentences else None,
        "last_sentence": sentences[-1] if sentences else None,
        "stance": _first_with_marker(sentences, STANCE_MARKERS, kind="stance"),
        "support": _first_with_marker(sentences, SUPPORT_MARKERS, kind="support"),
        "counterpoint": _first_with_marker(
            sentences, COUNTER_MARKERS, kind="counterpoint"
        ),
        "conclusion": _first_with_marker(
            sentences, CONCLUSION_MARKERS, kind="conclusion"
        ),
    }
    for span in spans.values():
        if span is not None and essay[span.start : span.end] != span.text:
            raise RuntimeError("evidence span does not round-trip to the source essay")
    return EvidenceLedger(
        record_id=record.id,
        prompt_num=record.prompt_num,
        essay_char_count=len(essay),
        sentence_count=len(sentences),
        paragraph_count=max(len(paragraphs), 1 if essay.strip() else 0),
        token_count=len(tokens),
        lexical_diversity=(unique_tokens / len(tokens)) if tokens else 0.0,
        connective_count=marker_count(CONNECTIVE_MARKERS),
        support_marker_count=marker_count(SUPPORT_MARKERS),
        repeated_space_count=len(re.findall(r" {2,}", essay)),
        repeated_punctuation_count=len(re.findall(r"[!?。！？]{2,}|\.{2,}", essay)),
        long_sentence_count=sum(len(span.text) >= 120 for span in sentences),
        spans=spans,
    )
