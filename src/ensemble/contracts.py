"""Source contract for deployment-time stacking and calibration."""

from __future__ import annotations

from pathlib import Path

from src.utils.hashing import sha256_file


STACKER_INFERENCE_FILES = (
    "src/ensemble/contracts.py",
    "src/ensemble/simplex.py",
    "src/calibration/affine.py",
    "src/evaluation/metrics.py",
)


def stacker_inference_code_contract() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[2]
    return {
        relative: sha256_file(project_root / relative)
        for relative in STACKER_INFERENCE_FILES
    }


__all__ = ["STACKER_INFERENCE_FILES", "stacker_inference_code_contract"]
