from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.evaluation.metrics import TRAITS
from src.rationale.evidence import EvidenceLedger


class RationaleValidationError(ValueError):
    pass


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output = {}
    for key, value in pairs:
        if key in output:
            raise RationaleValidationError(f"duplicate rationale key: {key}")
        output[key] = value
    return output


def validate_rationales(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(TRAITS):
        raise RationaleValidationError(f"rationale keys must be exactly {TRAITS}")
    output = {}
    for trait in TRAITS:
        rationale = value[trait]
        if not isinstance(rationale, str) or not rationale.strip():
            raise RationaleValidationError(f"{trait} rationale must be nonempty text")
        text = rationale.strip()
        if not 20 <= len(text) <= 600:
            raise RationaleValidationError(
                f"{trait} rationale length must be between 20 and 600 characters"
            )
        output[trait] = text
    if len(set(output.values())) != len(TRAITS):
        raise RationaleValidationError("the three trait rationales must be distinct")
    return output


def parse_rationales(text: str) -> dict[str, str]:
    if not isinstance(text, str):
        raise RationaleValidationError("rationale output must be text")
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicates)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise RationaleValidationError(f"invalid rationale JSON: {error}") from error
    return validate_rationales(value)


def serialize_rationales(value: Mapping[str, str]) -> str:
    return json.dumps(
        validate_rationales(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class GroundingReport:
    accepted: bool
    reasons: tuple[str, ...]
    exact_evidence_hits: dict[str, int]


_UNSUPPORTED_FACT_PATTERNS = (
    "통계에 따르면",
    "연구에 따르면",
    "조사 결과",
    "전문가들은",
    "자료에 따르면",
)


def _quoted_texts(text: str) -> list[str]:
    values = []
    for match in re.finditer(
        r"‘([^’]+)’|“([^”]+)”|\"([^\"]+)\"|'([^']+)'", text
    ):
        values.append(next(group for group in match.groups() if group is not None))
    return values


def _trait_evidence(ledger: EvidenceLedger, trait: str) -> list[str]:
    keys = {
        "content": (
            "stance",
            "support",
            "counterpoint",
            "first_sentence",
            "last_sentence",
        ),
        "organization": ("first_sentence", "last_sentence", "conclusion"),
        "expression": ("first_sentence", "last_sentence"),
    }[trait]
    return [
        ledger.spans[key].text
        for key in keys
        if ledger.spans.get(key) is not None
    ]


def assess_grounding(
    rationales: Mapping[str, str],
    *,
    essay: str,
    ledger: EvidenceLedger,
) -> GroundingReport:
    canonical = validate_rationales(rationales)
    reasons: list[str] = []
    hits: dict[str, int] = {}
    for trait, text in canonical.items():
        for phrase in _UNSUPPORTED_FACT_PATTERNS:
            if phrase in text and phrase not in essay:
                reasons.append(f"{trait}: unsupported factual phrase {phrase!r}")
        for number in re.findall(r"\d+(?:\.\d+)?", text):
            if number not in essay:
                reasons.append(f"{trait}: numeric claim absent from essay: {number}")
        for quote in _quoted_texts(text):
            source = quote[:-1].rstrip() if quote.endswith("…") else quote
            if source and source not in essay:
                reasons.append(f"{trait}: quoted text absent from essay")
        evidence = _trait_evidence(ledger, trait)
        trait_hits = sum(
            1
            for span in evidence
            if (span in text) or (len(span) >= 12 and span[:12] in text)
        )
        hits[trait] = trait_hits
        if essay.strip() and trait_hits == 0:
            reasons.append(f"{trait}: no exact evidence overlap")
    return GroundingReport(not reasons, tuple(reasons), hits)
