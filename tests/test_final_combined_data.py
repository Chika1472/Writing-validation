from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.build_final_combined_data import main
from src.data.final_combined import (
    ArtifactExistsError,
    FinalCombinedDataError,
    RulesAcknowledgementError,
    build_final_combined_dataset,
    require_validation_label_training_acknowledgement,
)
from src.utils.config import load_yaml
from src.utils.hashing import sha256_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _row(
    record_id: str,
    *,
    document_id: str | None = None,
    essay: str = " 원문 공백을 보존한다. ",
) -> dict:
    return {
        "id": record_id,
        "document_id": document_id or record_id.replace("GWGR", "GWRW") + ".1",
        "prompt_num": "Q1",
        "prompt": " 논제 원문 ",
        "essay": essay,
        "score": {
            "content": 3.0,
            "organization": 3.5,
            "expression": 4.0,
            "average": 3.5,
        },
    }


def _line(payload: dict, *, spaced: bool = False) -> bytes:
    separators = (", ", ": ") if spaced else (",", ":")
    return json.dumps(payload, ensure_ascii=False, separators=separators).encode("utf-8")


def _write_source(path: Path, rows: list[bytes], *, crlf: bool = False) -> bytes:
    separator = b"\r\n" if crlf else b"\n"
    content = separator.join(rows) + separator
    path.write_bytes(content)
    return content


def _build(
    tmp_path: Path,
    train_rows: list[bytes],
    validation_rows: list[bytes],
) -> tuple[dict, Path, Path, Path, Path, bytes, bytes]:
    train = tmp_path / "source_train.jsonl"
    validation = tmp_path / "source_validation.jsonl"
    output = tmp_path / "final" / "combined.jsonl"
    manifest = tmp_path / "final" / "combined.manifest.json"
    config = tmp_path / "config.yaml"
    config.write_text("project_root: .\n", encoding="utf-8")
    train_before = _write_source(train, train_rows, crlf=True)
    validation_before = _write_source(validation, validation_rows)
    payload = build_final_combined_dataset(
        train_source=train,
        validation_source=validation,
        output_path=output,
        manifest_path=manifest,
        project_root=PROJECT_ROOT,
        config_path=config,
        validation_label_training_acknowledged=True,
    )
    return (
        payload,
        train,
        validation,
        output,
        manifest,
        train_before,
        validation_before,
    )


def test_acknowledgement_is_fail_closed_and_precedes_config_read(tmp_path: Path) -> None:
    with pytest.raises(RulesAcknowledgementError, match="explicitly allow"):
        require_validation_label_training_acknowledgement(False)

    missing_config = tmp_path / "must_not_be_read.yaml"
    with pytest.raises(RulesAcknowledgementError, match="explicitly allow"):
        main(["--config", str(missing_config)])


def test_loaded_config_hash_must_still_match_at_build_start(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    config = tmp_path / "config.yaml"
    _write_source(train, [_line(_row("GWGR2300000001"))])
    _write_source(
        validation,
        [_line(_row("GWGR2400000002", essay="다른 본문"))],
    )
    config.write_text("project_root: .\n", encoding="utf-8")

    with pytest.raises(FinalCombinedDataError, match="changed after it was loaded"):
        build_final_combined_dataset(
            train_source=train,
            validation_source=validation,
            output_path=tmp_path / "combined.jsonl",
            manifest_path=tmp_path / "combined.manifest.json",
            project_root=PROJECT_ROOT,
            config_path=config,
            validation_label_training_acknowledged=True,
            expected_config_sha256="0" * 64,
        )


def test_cli_rejects_destinations_outside_artifact_namespace(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            (
                "project_root: .",
                "paths:",
                "  train: ../outside.jsonl",
                "  artifacts: artifacts/final_train_validation",
                "final_combined:",
                "  train_source: missing_train.jsonl",
                "  validation_source: missing_validation.jsonl",
                "  manifest: artifacts/final_train_validation/data/combined.manifest.json",
                "",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(FinalCombinedDataError, match="below paths.artifacts"):
        main(
            [
                "--config",
                str(config),
                "--acknowledge-rules-allow-validation-label-training",
            ]
        )


def test_combination_preserves_raw_rows_source_order_and_complete_provenance(
    tmp_path: Path,
) -> None:
    train_rows = [
        _line(_row("GWGR2300000001", essay="  첫 원문  "), spaced=True),
        _line(_row("GWGR2400000002", essay="둘째 원문")),
    ]
    validation_rows = [_line(_row("GWGR2500000003", essay=" 검증 원문 "), spaced=True)]
    (
        payload,
        train,
        validation,
        output,
        manifest,
        train_before,
        validation_before,
    ) = _build(tmp_path, train_rows, validation_rows)

    expected_output = b"\n".join([*train_rows, *validation_rows]) + b"\n"
    assert output.read_bytes() == expected_output
    assert train.read_bytes() == train_before
    assert validation.read_bytes() == validation_before
    assert json.loads(manifest.read_text(encoding="utf-8")) == payload

    assert payload["authorization"]["validation_label_training_acknowledged"] is True
    assert payload["sources"]["train"]["sha256"] == hashlib.sha256(
        train_before
    ).hexdigest()
    assert payload["sources"]["validation"]["sha256"] == hashlib.sha256(
        validation_before
    ).hexdigest()
    assert payload["sources"]["train"]["ordered_ids"] == [
        "GWGR2300000001",
        "GWGR2400000002",
    ]
    assert payload["combined"]["ordered_ids"] == [
        "GWGR2300000001",
        "GWGR2400000002",
        "GWGR2500000003",
    ]
    assert payload["combined"]["ordered_document_ids"] == [
        "GWRW2300000001.1",
        "GWRW2400000002.1",
        "GWRW2500000003.1",
    ]
    assert payload["combined"]["source_order"] == ["train", "validation"]
    assert payload["combined"]["serialization"] == {
        "format": "jsonl",
        "encoding": "utf-8",
        "row_bytes": "source JSON bytes excluding the line terminator",
        "line_ending": "LF",
        "terminal_newline": True,
    }
    assert payload["combined"]["cohort_counts"] == {
        "2023": 1,
        "2024": 1,
        "2025": 1,
    }
    assert payload["combined"]["sha256"] == hashlib.sha256(expected_output).hexdigest()
    assert payload["code"]["sha256"] == sha256_json(payload["code"]["files"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update(extra="forbidden"), "non-exact schema"),
        (lambda row: row["score"].update(note="forbidden"), "non-exact schema"),
        (lambda row: row.pop("essay"), "non-exact schema"),
    ],
)
def test_exact_schema_rejects_missing_and_extra_fields(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    invalid = _row("GWGR2300000001")
    mutate(invalid)
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    config = tmp_path / "config.yaml"
    _write_source(train, [_line(invalid)])
    _write_source(validation, [_line(_row("GWGR2400000002"))])
    config.write_text("project_root: .\n", encoding="utf-8")

    with pytest.raises(FinalCombinedDataError, match=message):
        build_final_combined_dataset(
            train_source=train,
            validation_source=validation,
            output_path=tmp_path / "combined.jsonl",
            manifest_path=tmp_path / "combined.manifest.json",
            project_root=PROJECT_ROOT,
            config_path=config,
            validation_label_training_acknowledged=True,
        )


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    valid = _line(_row("GWGR2400000002"))
    duplicate_key = (
        b'{"id":"GWGR2300000001","id":"GWGR2300000099",'
        b'"document_id":"GWRW2300000001.1","prompt_num":"Q1",'
        b'"prompt":"p","essay":"e","score":{"content":3,'
        b'"organization":3,"expression":3,"average":3}}'
    )
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    config = tmp_path / "config.yaml"
    _write_source(train, [duplicate_key])
    _write_source(validation, [valid])
    config.write_text("project_root: .\n", encoding="utf-8")

    with pytest.raises(FinalCombinedDataError, match="duplicate JSON object key"):
        build_final_combined_dataset(
            train_source=train,
            validation_source=validation,
            output_path=tmp_path / "combined.jsonl",
            manifest_path=tmp_path / "combined.manifest.json",
            project_root=PROJECT_ROOT,
            config_path=config,
            validation_label_training_acknowledged=True,
        )


@pytest.mark.parametrize(
    ("train_row", "validation_row", "message"),
    [
        (
            _row("GWGR2300000001"),
            _row("GWGR2300000001"),
            "duplicate id",
        ),
        (
            _row("GWGR2300000001"),
            _row("GWGR2300000001", essay="다른 본문"),
            "conflicting id",
        ),
        (
            _row("GWGR2300000001", document_id="GWRW-SHARED.1"),
            _row("GWGR2400000002", document_id="GWRW-SHARED.1"),
            "conflicting document_id",
        ),
    ],
)
def test_id_and_document_id_duplicates_or_conflicts_are_rejected(
    tmp_path: Path,
    train_row: dict,
    validation_row: dict,
    message: str,
) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    config = tmp_path / "config.yaml"
    _write_source(train, [_line(train_row)])
    _write_source(validation, [_line(validation_row)])
    config.write_text("project_root: .\n", encoding="utf-8")

    with pytest.raises(FinalCombinedDataError, match=message):
        build_final_combined_dataset(
            train_source=train,
            validation_source=validation,
            output_path=tmp_path / "combined.jsonl",
            manifest_path=tmp_path / "combined.manifest.json",
            project_root=PROJECT_ROOT,
            config_path=config,
            validation_label_training_acknowledged=True,
        )


def test_cross_source_duplicate_essay_text_is_rejected(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    config = tmp_path / "config.yaml"
    _write_source(
        train,
        [_line(_row("GWGR2300000001", essay="동일한 본문"))],
    )
    _write_source(
        validation,
        [_line(_row("GWGR2400000002", essay="동일한 본문"))],
    )
    config.write_text("project_root: .\n", encoding="utf-8")

    with pytest.raises(FinalCombinedDataError, match="duplicate essay text"):
        build_final_combined_dataset(
            train_source=train,
            validation_source=validation,
            output_path=tmp_path / "combined.jsonl",
            manifest_path=tmp_path / "combined.manifest.json",
            project_root=PROJECT_ROOT,
            config_path=config,
            validation_label_training_acknowledged=True,
        )


def test_outputs_are_create_only_and_never_overwritten(tmp_path: Path) -> None:
    payload, train, validation, output, manifest, _, _ = _build(
        tmp_path,
        [_line(_row("GWGR2300000001"))],
        [_line(_row("GWGR2400000002"))],
    )
    output_before = output.read_bytes()
    manifest_before = manifest.read_bytes()

    with pytest.raises(ArtifactExistsError, match="refusing to overwrite"):
        build_final_combined_dataset(
            train_source=train,
            validation_source=validation,
            output_path=output,
            manifest_path=manifest,
            project_root=PROJECT_ROOT,
            config_path=payload["configuration"]["path"],
            validation_label_training_acknowledged=True,
        )

    assert output.read_bytes() == output_before
    assert manifest.read_bytes() == manifest_before


def test_second_exclusive_create_race_rolls_back_first_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    output = tmp_path / "final" / "combined.jsonl"
    manifest = tmp_path / "final" / "combined.manifest.json"
    config = tmp_path / "config.yaml"
    _write_source(train, [_line(_row("GWGR2300000001"))])
    _write_source(validation, [_line(_row("GWGR2400000002", essay="다른 본문"))])
    config.write_text("project_root: .\n", encoding="utf-8")

    real_open = os.open
    call_count = 0

    def racing_open(path, flags, mode=0o777):
        nonlocal call_count
        call_count += 1
        assert flags & os.O_EXCL
        if call_count == 2:
            raise FileExistsError(17, "simulated destination race", str(path))
        return real_open(path, flags, mode)

    monkeypatch.setattr("src.data.final_combined.os.open", racing_open)
    with pytest.raises(ArtifactExistsError, match="refusing to overwrite"):
        build_final_combined_dataset(
            train_source=train,
            validation_source=validation,
            output_path=output,
            manifest_path=manifest,
            project_root=PROJECT_ROOT,
            config_path=config,
            validation_label_training_acknowledged=True,
        )

    assert call_count == 2
    assert not output.exists()
    assert not manifest.exists()


def test_output_may_not_alias_either_read_only_source(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    config = tmp_path / "config.yaml"
    train_before = _write_source(train, [_line(_row("GWGR2300000001"))])
    validation_before = _write_source(
        validation,
        [_line(_row("GWGR2400000002"))],
    )
    config.write_text("project_root: .\n", encoding="utf-8")

    with pytest.raises(FinalCombinedDataError, match="read-only source"):
        build_final_combined_dataset(
            train_source=train,
            validation_source=validation,
            output_path=train,
            manifest_path=tmp_path / "combined.manifest.json",
            project_root=PROJECT_ROOT,
            config_path=config,
            validation_label_training_acknowledged=True,
        )

    assert train.read_bytes() == train_before
    assert validation.read_bytes() == validation_before


def test_final_data_config_is_fold_and_orchestrator_compatible() -> None:
    config = load_yaml(
        PROJECT_ROOT / "configs" / "data_final_combined.yaml"
    )
    assert config["paths"]["train"] == (
        "artifacts/final_train_validation/data/train_validation_combined.jsonl"
    )
    assert config["paths"]["artifacts"] == "artifacts/final_train_validation"
    assert "validation" not in config["paths"]
    assert config["folds"] == {"n_splits": 5, "seed": 42, "score_bins": 5}
