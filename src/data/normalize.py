from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.data.schema import EssayRecord, ensure_record


@dataclass(frozen=True, slots=True)
class ModelTextView:
    prompt: str
    essay: str


def conservative_model_view(text: str) -> str:
    """Remove only surrounding whitespace; retain spelling and internal spacing."""

    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    return text.strip()


def build_model_view(record: EssayRecord | Mapping[str, Any]) -> ModelTextView:
    canonical = ensure_record(record)
    return ModelTextView(
        prompt=conservative_model_view(canonical.prompt),
        essay=conservative_model_view(canonical.essay),
    )
