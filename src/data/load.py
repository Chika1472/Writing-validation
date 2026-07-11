from __future__ import annotations

import json
from pathlib import Path

from src.data.schema import (
    INPUT_FIELDS,
    RECORD_FIELDS,
    SCORE_FIELDS,
    EssayInput,
    EssayRecord,
    SchemaError,
)


class DatasetLoadError(ValueError):
    """Raised when a JSONL dataset cannot be loaded canonically."""


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise DatasetLoadError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise DatasetLoadError(f"non-standard JSON constant: {value}")


def _strict_json_object(line: str, *, source: Path, line_number: int) -> dict:
    try:
        payload = json.loads(
            line,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except DatasetLoadError as exc:
        raise DatasetLoadError(f"{source}:{line_number}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetLoadError(
            f"{source}:{line_number}: invalid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise DatasetLoadError(
            f"{source}:{line_number}: dataset row must be a JSON object"
        )
    return payload


def _require_exact_fields(
    payload: dict,
    expected: tuple[str, ...],
    *,
    source: Path,
    line_number: int,
    where: str,
) -> None:
    actual = set(payload)
    required = set(expected)
    if actual != required:
        raise DatasetLoadError(
            f"{source}:{line_number}: {where} fields must be exactly "
            f"{sorted(required)}; missing={sorted(required - actual)}, "
            f"extra={sorted(actual - required)}"
        )


def load_jsonl(path: str | Path) -> list[EssayRecord]:
    """Load a strict UTF-8 JSONL file and validate every labeled essay row."""

    dataset_path = Path(path)
    records: list[EssayRecord] = []
    seen_ids: set[str] = set()

    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise DatasetLoadError(
                        f"{dataset_path}:{line_number}: blank lines are not valid JSONL rows"
                    )
                payload = _strict_json_object(
                    line, source=dataset_path, line_number=line_number
                )
                _require_exact_fields(
                    payload,
                    RECORD_FIELDS,
                    source=dataset_path,
                    line_number=line_number,
                    where="dataset row",
                )
                if not isinstance(payload["score"], dict):
                    raise DatasetLoadError(
                        f"{dataset_path}:{line_number}: score must be a JSON object"
                    )
                _require_exact_fields(
                    payload["score"],
                    SCORE_FIELDS,
                    source=dataset_path,
                    line_number=line_number,
                    where="score",
                )
                try:
                    record = EssayRecord.from_mapping(payload)
                except SchemaError as exc:
                    raise DatasetLoadError(f"{dataset_path}:{line_number}: {exc}") from exc
                if record.id in seen_ids:
                    raise DatasetLoadError(
                        f"{dataset_path}:{line_number}: duplicate id {record.id!r}"
                    )
                seen_ids.add(record.id)
                records.append(record)
    except UnicodeDecodeError as exc:
        raise DatasetLoadError(f"{dataset_path}: file is not valid UTF-8") from exc

    if not records:
        raise DatasetLoadError(f"{dataset_path}: dataset is empty")
    return records


def load_inference_jsonl(path: str | Path) -> list[EssayInput]:
    """Load score inputs while accepting labeled rows without reading their labels."""

    dataset_path = Path(path)
    records: list[EssayInput] = []
    seen_ids: set[str] = set()
    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise DatasetLoadError(
                        f"{dataset_path}:{line_number}: blank lines are not valid JSONL rows"
                    )
                payload = _strict_json_object(
                    line, source=dataset_path, line_number=line_number
                )
                allowed_fields = (
                    set(INPUT_FIELDS),
                    set(RECORD_FIELDS),
                )
                if set(payload) not in allowed_fields:
                    raise DatasetLoadError(
                        f"{dataset_path}:{line_number}: inference row fields must be "
                        "exactly the unlabeled or labeled dataset schema"
                    )
                try:
                    record = EssayInput.from_mapping(payload)
                except SchemaError as exc:
                    raise DatasetLoadError(f"{dataset_path}:{line_number}: {exc}") from exc
                if record.id in seen_ids:
                    raise DatasetLoadError(
                        f"{dataset_path}:{line_number}: duplicate id {record.id!r}"
                    )
                seen_ids.add(record.id)
                records.append(record)
    except UnicodeDecodeError as exc:
        raise DatasetLoadError(f"{dataset_path}: file is not valid UTF-8") from exc
    if not records:
        raise DatasetLoadError(f"{dataset_path}: dataset is empty")
    return records


def load_train_validation(
    train_path: str | Path, validation_path: str | Path
) -> tuple[list[EssayRecord], list[EssayRecord]]:
    """Load both splits and reject cross-split ID, document, or exact-text leakage."""

    train = load_jsonl(train_path)
    validation = load_jsonl(validation_path)
    overlap = {record.id for record in train}.intersection(record.id for record in validation)
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise DatasetLoadError(f"train and validation share id(s): {examples}")
    document_overlap = {record.document_id for record in train}.intersection(
        record.document_id for record in validation
    )
    if document_overlap:
        examples = ", ".join(sorted(document_overlap)[:5])
        raise DatasetLoadError(f"train and validation share document_id(s): {examples}")
    train_essay_to_id = {record.essay: record.id for record in train}
    duplicate_essays = [
        (train_essay_to_id[record.essay], record.id)
        for record in validation
        if record.essay in train_essay_to_id
    ]
    if duplicate_essays:
        raise DatasetLoadError(
            "train and validation contain exact duplicate essay text; "
            f"first pair={duplicate_essays[0]}"
        )
    return train, validation
