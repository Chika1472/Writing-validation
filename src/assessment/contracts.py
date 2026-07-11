"""Lightweight source contract for restricted assessment feature extraction."""

from __future__ import annotations

from pathlib import Path

from src.utils.hashing import sha256_file, sha256_json


ASSESSMENT_EXTRACTION_FILES = (
    "src/assessment/contracts.py",
    "src/assessment/extraction.py",
    "src/assessment/prompting.py",
    "src/assessment/questions.py",
    "src/assessment/codebook.py",
)


def assessment_extraction_code_sha256() -> str:
    project_root = Path(__file__).resolve().parents[2]
    return sha256_json(
        {
            relative: sha256_file(project_root / relative)
            for relative in ASSESSMENT_EXTRACTION_FILES
        }
    )


__all__ = ["ASSESSMENT_EXTRACTION_FILES", "assessment_extraction_code_sha256"]
