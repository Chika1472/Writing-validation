import json
from pathlib import Path

import pytest

from src.evaluation.oof_provenance import oof_manifest_path, oof_provenance_fields
from src.evaluation.precision import (
    precision_promotion_gate,
    validate_precision_pair,
)


def _write_precision_oof(
    path: Path,
    *,
    gold: Path,
    folds: Path,
    precision: str,
    checkpoint_set_signature: str,
) -> None:
    path.write_text("{}\n", encoding="utf-8")
    payload = {
        **oof_provenance_fields(
            prediction_path=path,
            gold_path=gold,
            fold_path=folds,
            rows=1,
            scorer_name=f"scorer:{precision}",
            scorer_signature="a" * 64,
        ),
        "precision": precision,
        "checkpoint_set_signature": checkpoint_set_signature,
    }
    oof_manifest_path(path).write_text(json.dumps(payload), encoding="utf-8")


def test_precision_pair_requires_the_same_checkpoint_set(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    folds = tmp_path / "folds.jsonl"
    gold.write_text("{}\n", encoding="utf-8")
    folds.write_text("{}\n", encoding="utf-8")
    candidate = tmp_path / "bf16.jsonl"
    baseline = tmp_path / "4bit.jsonl"
    _write_precision_oof(
        candidate,
        gold=gold,
        folds=folds,
        precision="bf16",
        checkpoint_set_signature="b" * 64,
    )
    _write_precision_oof(
        baseline,
        gold=gold,
        folds=folds,
        precision="4bit",
        checkpoint_set_signature="b" * 64,
    )
    candidate_manifest, baseline_manifest = validate_precision_pair(
        candidate_path=candidate,
        baseline_path=baseline,
        gold_path=gold,
        fold_path=folds,
        candidate_precision="bf16",
        baseline_precision="4bit",
    )
    assert candidate_manifest["checkpoint_set_signature"] == baseline_manifest[
        "checkpoint_set_signature"
    ]

    baseline_payload = json.loads(oof_manifest_path(baseline).read_text(encoding="utf-8"))
    baseline_payload["checkpoint_set_signature"] = "c" * 64
    oof_manifest_path(baseline).write_text(json.dumps(baseline_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_set_signature"):
        validate_precision_pair(
            candidate_path=candidate,
            baseline_path=baseline,
            gold_path=gold,
            fold_path=folds,
            candidate_precision="bf16",
            baseline_precision="4bit",
        )


def test_precision_gate_requires_noninferiority_and_supported_gain() -> None:
    report = {
        "metrics": {
            "macro_rmse": {
                "delta": -0.003,
                "probability_candidate_better": 0.91,
            },
            "macro_spearman": {
                "delta": -0.0005,
                "probability_candidate_better": 0.45,
            },
        }
    }
    decision = precision_promotion_gate(
        report,
        max_rmse_increase=0.002,
        max_spearman_drop=0.001,
        min_rmse_improvement=0.001,
        min_spearman_improvement=0.001,
        min_probability=0.8,
    )
    assert decision["promote_candidate"] is True

    report["metrics"]["macro_spearman"]["delta"] = -0.002
    decision = precision_promotion_gate(
        report,
        max_rmse_increase=0.002,
        max_spearman_drop=0.001,
        min_rmse_improvement=0.001,
        min_spearman_improvement=0.001,
        min_probability=0.8,
    )
    assert decision["promote_candidate"] is False
