"""Immutable code contract for serialized CPU baseline inference.

The joblib estimators call project feature/normalization helpers at prediction
time.  Hashing only the serialized estimators would therefore allow those
helpers to change silently after an OOF artifact was selected.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.hashing import sha256_file


BASELINE_INFERENCE_FILES = (
    "src/baselines/contracts.py",
    "src/baselines/mean_baseline.py",
    "src/baselines/tfidf_ridge.py",
    "src/data/features.py",
    "src/data/normalize.py",
    "src/data/schema.py",
    "src/data/sentence_split.py",
    "src/evaluation/metrics.py",
    "src/evaluation/predictions.py",
    "src/inference/serializer.py",
)


def baseline_inference_code_contract() -> dict[str, str]:
    """Return the exact source hashes required to reproduce inference."""

    project_root = Path(__file__).resolve().parents[2]
    return {
        relative: sha256_file(project_root / relative)
        for relative in BASELINE_INFERENCE_FILES
    }
