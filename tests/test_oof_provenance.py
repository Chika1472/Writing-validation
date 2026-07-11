from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.oof_provenance import (
    oof_manifest_path,
    oof_provenance_fields,
    validate_oof_provenance,
)
from src.utils.paths import (
    require_distinct_paths,
    require_new_paths,
    require_outside_roots,
)


def test_oof_provenance_binds_prediction_gold_and_folds(tmp_path: Path) -> None:
    prediction = tmp_path / "oof.jsonl"
    gold = tmp_path / "gold.jsonl"
    folds = tmp_path / "folds.jsonl"
    prediction.write_text("{}\n", encoding="utf-8")
    gold.write_text("{}\n", encoding="utf-8")
    folds.write_text("{}\n", encoding="utf-8")
    payload = {
        **oof_provenance_fields(
            prediction_path=prediction,
            gold_path=gold,
            fold_path=folds,
            rows=1,
            scorer_name="model_v1",
            scorer_signature="abc123",
        )
    }
    oof_manifest_path(prediction).write_text(
        json.dumps(payload), encoding="utf-8"
    )

    restored = validate_oof_provenance(
        prediction_path=prediction,
        gold_path=gold,
        fold_path=folds,
    )
    assert restored["scorer_signature"] == "abc123"

    prediction.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_oof_provenance(
            prediction_path=prediction,
            gold_path=gold,
            fold_path=folds,
        )


def test_path_guards_reject_aliases_and_existing_outputs(tmp_path: Path) -> None:
    existing = tmp_path / "existing.jsonl"
    existing.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="must be distinct"):
        require_distinct_paths(input=existing, output=existing)
    with pytest.raises(FileExistsError, match="already exists"):
        require_new_paths(output=existing)
    with pytest.raises(ValueError, match="must not be inside"):
        require_outside_roots(
            {"checkpoint": tmp_path / "checkpoint"},
            output=tmp_path / "checkpoint" / "adapter" / "new.json",
        )
