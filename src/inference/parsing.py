from __future__ import annotations

import re

from src.inference.serializer import strict_parse_prediction


def parse_strict_json(text: str) -> dict[str, dict[str, float | str]]:
    """Apply the same exact schema contract used by final serialization."""

    return strict_parse_prediction(text)


def repair_common_qwen_json(text: str) -> str:
    """Repair only the two observed, deterministic JSON defects.

    The repaired output remains a separate artifact; callers must never report
    it as strict parsing success.
    """

    first = text.find("{")
    last = text.rfind("}")
    candidate = text[first : last + 1] if first >= 0 and last > first else text
    candidate = re.sub(
        r'("rationale"\s*:\s*"(?:\\.|[^"\\])*")\s*\)',
        r"\1}",
        candidate,
        flags=re.DOTALL,
    )
    return re.sub(r",\s*([}\]])", r"\1", candidate)


def parse_repaired_json(text: str) -> dict[str, dict[str, float | str]]:
    """Repair observed syntax defects, then apply the unchanged strict contract."""

    return strict_parse_prediction(repair_common_qwen_json(text))
