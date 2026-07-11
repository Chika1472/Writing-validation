"""Source contract for standalone affine/prompt calibration inference."""

from __future__ import annotations

from pathlib import Path

from src.utils.hashing import sha256_file


CALIBRATION_INFERENCE_FILES = (
    "src/calibration/contracts.py",
    "src/calibration/affine.py",
)


def calibration_inference_code_contract() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[2]
    return {
        relative: sha256_file(project_root / relative)
        for relative in CALIBRATION_INFERENCE_FILES
    }


__all__ = [
    "CALIBRATION_INFERENCE_FILES",
    "calibration_inference_code_contract",
]
