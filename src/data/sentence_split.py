from __future__ import annotations

import re


_BOUNDARY_RE = re.compile(
    r"(?:\r\n?|\n)+|(?:(?<=[.!?])|(?<=[.!?][\"'”’\)\]]))[\t ]+"
)


def split_sentences(text: str) -> list[str]:
    """Conservatively split Korean prose at explicit punctuation or line breaks."""

    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    stripped = text.strip()
    if not stripped:
        return []
    return [fragment.strip() for fragment in _BOUNDARY_RE.split(stripped) if fragment.strip()]


def count_sentences(text: str) -> int:
    return len(split_sentences(text))
