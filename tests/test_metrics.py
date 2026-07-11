from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from src.evaluation.bootstrap import (
    paired_stratified_bootstrap,
    stratified_resample_indices,
)
from src.evaluation.metrics import evaluate_predictions, regression_metrics
from src.evaluation.predictions import read_predictions, write_predictions
from src.evaluation.slices import evaluation_report


@dataclass
class Record:
    id: str
    prompt_num: str
    prompt: str
    essay: str
    scores: dict[str, float]


def _records() -> list[Record]:
    return [
        Record("a", "Q1", "주제 하나", "짧은 글", {"content": 1, "organization": 2, "expression": 3}),
        Record("b", "Q1", "주제 하나", "조금 더 긴 글입니다", {"content": 3, "organization": 4, "expression": 5}),
        Record("c", "Q2", "주제 둘", "중간 길이 글입니다", {"content": 2, "organization": 3, "expression": 4}),
        Record("d", "Q2", "주제 둘", "가장 길게 작성한 논증적인 글입니다", {"content": 4, "organization": 5, "expression": 2}),
    ]


def test_regression_metrics_exact_values_and_ties() -> None:
    result = regression_metrics([1, 3], [2, 2])

    assert result["rmse"] == pytest.approx(1.0)
    assert result["mae"] == pytest.approx(1.0)
    assert result["bias"] == pytest.approx(0.0)
    assert result["pred_sd"] == pytest.approx(0.0)
    assert result["gold_sd"] == pytest.approx(np.sqrt(2.0))
    assert result["pearson"] == 0.0
    assert result["spearman"] == 0.0
    assert result["within_0.5"] == 0.0
    assert result["within_1.0"] == 1.0


def test_evaluate_predictions_aligns_records_by_id() -> None:
    records = _records()[:2]
    predictions = [
        {
            "id": "b",
            "prompt_num": "Q1",
            "prediction": {"content": 3, "organization": 4, "expression": 5},
            "model": "perfect",
        },
        {
            "id": "a",
            "prompt_num": "Q1",
            "prediction": {"content": 1, "organization": 2, "expression": 3},
            "model": "perfect",
        },
    ]

    report = evaluate_predictions(records, predictions)

    assert report["n"] == 2
    assert report["macro"]["rmse"] == 0.0
    assert report["macro"]["spearman"] == 1.0
    assert set(report["traits"]) == {"content", "organization", "expression"}


def test_prediction_reader_accepts_official_nested_and_legacy_parsed(tmp_path) -> None:
    path = tmp_path / "legacy.jsonl"
    rows = [
        {
            "id": "a",
            "prompt_num": "Q1",
            "model_id": "legacy-model",
            "content": {"score": 2.5, "rationale": "근거"},
            "organization": {"score": 3.0, "rationale": "근거"},
            "expression": {"score": 3.5, "rationale": "근거"},
        },
        {
            "id": "b",
            "prompt_num": "Q2",
            "run_tag": "old-run",
            "parsed": {
                "content": {"score": 4.0},
                "organization": {"score": 4.25},
                "expression": {"score": 4.5},
            },
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    normalized = read_predictions(path)

    assert normalized[0] == {
        "id": "a",
        "prompt_num": "Q1",
        "prediction": {"content": 2.5, "organization": 3.0, "expression": 3.5},
        "model": "legacy-model",
    }
    assert normalized[1]["model"] == "old-run"
    assert normalized[1]["prediction"]["organization"] == 4.25


def test_prediction_writer_emits_only_canonical_schema(tmp_path) -> None:
    output = tmp_path / "predictions.jsonl"
    source = [
        {
            "id": "a",
            "prompt_num": "Q1",
            "content": {"score": 2.5, "rationale": "제외할 값"},
            "organization": {"score": 3.0},
            "expression": {"score": 3.5},
        }
    ]

    write_predictions(output, source, model="unit-test")
    row = json.loads(output.read_text(encoding="utf-8"))

    assert list(row) == ["id", "prompt_num", "prediction", "model"]
    assert row["model"] == "unit-test"
    assert row["prediction"] == {
        "content": 2.5,
        "organization": 3.0,
        "expression": 3.5,
    }


def test_prediction_reader_rejects_duplicate_ids_and_out_of_range() -> None:
    duplicated = [
        {"id": "a", "prompt_num": "Q1", "prediction": {"content": 2, "organization": 3, "expression": 4}, "model": "m"},
        {"id": "a", "prompt_num": "Q1", "prediction": {"content": 2, "organization": 3, "expression": 4}, "model": "m"},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        read_predictions(duplicated)

    invalid = [
        {"id": "a", "prompt_num": "Q1", "prediction": {"content": 0, "organization": 3, "expression": 4}, "model": "m"}
    ]
    with pytest.raises(ValueError, match=r"\[1, 5\]"):
        read_predictions(invalid)


def test_prompt_and_length_slice_report_covers_all_rows() -> None:
    records = _records()
    predictions = np.asarray(
        [[record.scores[trait] for trait in ("content", "organization", "expression")] for record in records]
    )

    report = evaluation_report(records, predictions)

    assert report["overall"]["macro"]["rmse"] == 0.0
    assert {key: value["n"] for key, value in report["by_prompt"].items()} == {
        "Q1": 2,
        "Q2": 2,
    }
    assert sum(value["n"] for value in report["by_length_quantile"].values()) == 4
    assert set(report["by_length_quantile"]) == {"Q1", "Q2", "Q3", "Q4"}


def test_stratified_resampling_preserves_counts() -> None:
    strata = ["Q1", "Q1", "Q2", "Q2", "Q2"]
    indices = stratified_resample_indices(strata, np.random.default_rng(7))
    sampled = [strata[index] for index in indices]

    assert sampled.count("Q1") == 2
    assert sampled.count("Q2") == 3


def test_paired_bootstrap_identical_predictions_has_zero_delta() -> None:
    records = _records()
    predictions = np.asarray(
        [[record.scores[trait] for trait in ("content", "organization", "expression")] for record in records]
    )

    report = paired_stratified_bootstrap(
        records,
        predictions,
        predictions.copy(),
        n_resamples=25,
        seed=11,
    )

    for metric in report["metrics"].values():
        assert metric["delta"] == 0.0
        assert metric["ci_low"] == 0.0
        assert metric["ci_high"] == 0.0
        assert metric["probability_candidate_better"] == 0.5
