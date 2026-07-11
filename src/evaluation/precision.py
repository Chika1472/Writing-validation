"""Validation and promotion policy for precision-controlled OOF comparisons."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from src.evaluation.oof_provenance import validate_oof_provenance


PRECISIONS = frozenset({"4bit", "bf16"})
_SHA256 = re.compile(r"[0-9a-f]{64}")


def validate_precision_oof(
    *,
    prediction_path: str | Path,
    gold_path: str | Path,
    fold_path: str | Path,
    expected_precision: str | None = None,
) -> dict[str, Any]:
    """Validate an OOF artifact and its precision-independent checkpoint identity."""

    payload = validate_oof_provenance(
        prediction_path=prediction_path,
        gold_path=gold_path,
        fold_path=fold_path,
    )
    if payload.get("oof_level") != "base_model_oof":
        raise ValueError("precision comparison accepts only base_model_oof artifacts")
    precision = payload.get("precision")
    if precision not in PRECISIONS:
        raise ValueError(f"OOF manifest has invalid precision: {precision!r}")
    if expected_precision is not None and precision != expected_precision:
        raise ValueError(
            f"OOF precision mismatch: expected {expected_precision!r}, got {precision!r}"
        )
    signature = payload.get("checkpoint_set_signature")
    if not isinstance(signature, str) or _SHA256.fullmatch(signature) is None:
        raise ValueError("OOF manifest lacks a valid checkpoint_set_signature")
    return payload


def validate_precision_pair(
    *,
    candidate_path: str | Path,
    baseline_path: str | Path,
    gold_path: str | Path,
    fold_path: str | Path,
    candidate_precision: str,
    baseline_precision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require that two OOF files differ in precision, not checkpoints or rows."""

    if candidate_precision not in PRECISIONS or baseline_precision not in PRECISIONS:
        raise ValueError("candidate and baseline precision must be '4bit' or 'bf16'")
    if candidate_precision == baseline_precision:
        raise ValueError("precision comparison requires two different precisions")
    candidate = validate_precision_oof(
        prediction_path=candidate_path,
        gold_path=gold_path,
        fold_path=fold_path,
        expected_precision=candidate_precision,
    )
    baseline = validate_precision_oof(
        prediction_path=baseline_path,
        gold_path=gold_path,
        fold_path=fold_path,
        expected_precision=baseline_precision,
    )
    for field in ("gold_sha256", "folds_sha256", "rows", "checkpoint_set_signature"):
        if candidate.get(field) != baseline.get(field):
            raise ValueError(f"precision OOF provenance differs for {field}")
    return candidate, baseline


def precision_promotion_gate(
    bootstrap_report: dict[str, Any],
    *,
    max_rmse_increase: float,
    max_spearman_drop: float,
    min_rmse_improvement: float,
    min_spearman_improvement: float,
    min_probability: float,
) -> dict[str, Any]:
    """Promote a costlier precision only with non-inferiority and positive evidence."""

    thresholds = (
        max_rmse_increase,
        max_spearman_drop,
        min_rmse_improvement,
        min_spearman_improvement,
    )
    if any(value < 0.0 for value in thresholds):
        raise ValueError("precision gate thresholds must be non-negative")
    if not 0.5 <= min_probability <= 1.0:
        raise ValueError("min_probability must lie in [0.5, 1.0]")
    try:
        rmse = bootstrap_report["metrics"]["macro_rmse"]
        spearman = bootstrap_report["metrics"]["macro_spearman"]
        rmse_delta = float(rmse["delta"])
        spearman_delta = float(spearman["delta"])
        rmse_probability = float(rmse["probability_candidate_better"])
        spearman_probability = float(spearman["probability_candidate_better"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("bootstrap report lacks precision-gate macro metrics") from error
    observed = (
        rmse_delta,
        spearman_delta,
        rmse_probability,
        spearman_probability,
    )
    if not all(math.isfinite(value) for value in observed):
        raise ValueError("precision-gate metrics must be finite")
    if not 0.0 <= rmse_probability <= 1.0 or not 0.0 <= spearman_probability <= 1.0:
        raise ValueError("candidate-better probabilities must lie in [0, 1]")

    rmse_noninferior = rmse_delta <= max_rmse_increase
    spearman_noninferior = spearman_delta >= -max_spearman_drop
    rmse_evidence = (
        rmse_delta <= -min_rmse_improvement
        and rmse_probability >= min_probability
    )
    spearman_evidence = (
        spearman_delta >= min_spearman_improvement
        and spearman_probability >= min_probability
    )
    promote = bool(
        rmse_noninferior
        and spearman_noninferior
        and (rmse_evidence or spearman_evidence)
    )
    return {
        "promote_candidate": promote,
        "policy": "noninferior_on_both_and_supported_improvement_on_at_least_one",
        "checks": {
            "rmse_noninferior": rmse_noninferior,
            "spearman_noninferior": spearman_noninferior,
            "rmse_improvement_supported": rmse_evidence,
            "spearman_improvement_supported": spearman_evidence,
        },
        "observed": {
            "macro_rmse_delta": rmse_delta,
            "macro_spearman_delta": spearman_delta,
            "rmse_probability_candidate_better": rmse_probability,
            "spearman_probability_candidate_better": spearman_probability,
        },
        "thresholds": {
            "max_rmse_increase": max_rmse_increase,
            "max_spearman_drop": max_spearman_drop,
            "min_rmse_improvement": min_rmse_improvement,
            "min_spearman_improvement": min_spearman_improvement,
            "min_probability": min_probability,
        },
    }


__all__ = [
    "PRECISIONS",
    "precision_promotion_gate",
    "validate_precision_oof",
    "validate_precision_pair",
]
