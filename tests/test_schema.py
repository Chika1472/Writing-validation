from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.features import build_surface_features
from src.data.load import (
    DatasetLoadError,
    load_inference_jsonl,
    load_jsonl,
    load_train_validation,
)
from src.data.normalize import build_model_view
from src.data.schema import EssayRecord, SchemaError
from src.data.sentence_split import split_sentences


def _row(*, record_id: str = "GWGR2300000001", essay: str = "  문장입니다.  ") -> dict:
    return {
        "id": record_id,
        "document_id": record_id.replace("GWGR", "GWRW") + ".1",
        "prompt_num": "Q1",
        "prompt": " 논제입니다. ",
        "essay": essay,
        "score": {
            "content": 3.5,
            "organization": 3.25,
            "expression": 4.0,
            "average": 3.58,
        },
    }


def test_schema_preserves_raw_text_and_model_view_only_strips() -> None:
    payload = _row(essay="  맞 춤법은  고치지 않는다.  다음 문장입니다!  ")
    record = EssayRecord.from_mapping(payload)

    assert record.raw_essay == payload["essay"]
    assert record.to_dict() == payload
    view = build_model_view(record)
    assert view.prompt == "논제입니다."
    assert view.essay == "맞 춤법은  고치지 않는다.  다음 문장입니다!"
    assert split_sentences(view.essay) == ["맞 춤법은  고치지 않는다.", "다음 문장입니다!"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.pop("essay"), "missing required field"),
        (lambda row: row["score"].pop("content"), "missing required field"),
        (lambda row: row["score"].update(content=5.1), "must be in"),
        (lambda row: row["score"].update(expression=float("nan")), "must be finite"),
        (lambda row: row["score"].update(organization=True), "real number"),
        (lambda row: row.update(prompt="   "), "must not be empty"),
    ],
)
def test_schema_rejects_missing_or_invalid_fields(mutation, message: str) -> None:
    payload = _row()
    mutation(payload)
    with pytest.raises(SchemaError, match=message):
        EssayRecord.from_mapping(payload)


def test_utf8_jsonl_loader_and_cross_split_guard(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_text(
        json.dumps(_row(essay=" 한글 원문 "), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    validation_path.write_text(
        json.dumps(_row(record_id="GWGR2400000002"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    train, validation = load_train_validation(train_path, validation_path)
    assert train[0].essay == " 한글 원문 "
    assert validation[0].id == "GWGR2400000002"

    validation_path.write_text(train_path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(DatasetLoadError, match="share id"):
        load_train_validation(train_path, validation_path)


def test_loader_reports_line_and_duplicate_id(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    line = json.dumps(_row(), ensure_ascii=False)
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match=r":2: duplicate id"):
        load_jsonl(path)


def test_loader_rejects_duplicate_keys_extra_fields_and_nonstandard_json(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate-key.jsonl"
    duplicate.write_text(
        '{"id":"a","id":"b","document_id":"d","prompt_num":"Q1",'
        '"prompt":"p","essay":"e","score":{"content":3,'
        '"organization":3,"expression":3,"average":3}}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetLoadError, match="duplicate JSON key"):
        load_jsonl(duplicate)

    extra = tmp_path / "extra.jsonl"
    payload = _row()
    payload["extra"] = "forbidden"
    extra.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(DatasetLoadError, match="fields must be exactly"):
        load_jsonl(extra)

    nonstandard = tmp_path / "nan.jsonl"
    nonstandard.write_text(
        json.dumps(_row(), ensure_ascii=False).replace("3.5", "NaN", 1) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetLoadError, match="non-standard JSON constant"):
        load_jsonl(nonstandard)


def test_inference_loader_does_not_require_score(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    payload = _row()
    payload.pop("score")
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    records = load_inference_jsonl(path)
    assert records[0].essay == payload["essay"]
    assert not hasattr(records[0], "score")


def test_surface_features_are_numeric_text_only_and_keep_order() -> None:
    records = [
        EssayRecord.from_mapping(_row(essay="첫째, 찬성한다. 따라서 필요하다.")),
        EssayRecord.from_mapping(
            _row(record_id="GWGR2400000002", essay="짧다.")
        ),
    ]
    features = build_surface_features(records)

    forbidden = {"id", "document_id", "prompt_num", "year", "cohort"}
    assert forbidden.isdisjoint(features.columns)
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes)
    assert features.loc[0, "essay_char_count"] > features.loc[1, "essay_char_count"]
    assert features.loc[0, "essay_stance_marker_count"] == 1.0
    assert features.loc[0, "essay_connective_count"] == 1.0
