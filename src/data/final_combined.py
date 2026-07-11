from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.folds import infer_cohort
from src.data.schema import RECORD_FIELDS, SCORE_FIELDS, EssayRecord, SchemaError
from src.utils.hashing import sha256_file, sha256_json


FINAL_COMBINED_FORMAT_VERSION = 1
FINAL_COMBINED_CODE_FILES = (
    "scripts/build_final_combined_data.py",
    "src/data/final_combined.py",
    "src/data/folds.py",
    "src/data/schema.py",
    "src/utils/config.py",
    "src/utils/hashing.py",
)


class FinalCombinedDataError(ValueError):
    """Raised when a final train+validation artifact cannot be built safely."""


class RulesAcknowledgementError(FinalCombinedDataError):
    """Raised unless validation-label training has been explicitly authorized."""


class ArtifactExistsError(FinalCombinedDataError):
    """Raised when either requested output already exists."""


class _DuplicateJsonKeyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _SourceRow:
    source: str
    line_number: int
    raw_json: bytes
    record: EssayRecord

    @property
    def location(self) -> str:
        return f"{self.source}:{self.line_number}"


@dataclass(frozen=True, slots=True)
class _SourceDataset:
    name: str
    path: Path
    sha256: str
    rows: tuple[_SourceRow, ...]


def require_validation_label_training_acknowledgement(acknowledged: bool) -> None:
    """Fail closed before reading any supplied data or configuration."""

    if acknowledged is not True:
        raise RulesAcknowledgementError(
            "validation labels may be used for final training only if the competition "
            "rules explicitly allow it; rerun with "
            "--acknowledge-rules-allow-validation-label-training only after confirming "
            "that authorization"
        )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _require_exact_keys(
    value: Any,
    expected: Iterable[str],
    *,
    where: str,
) -> None:
    if not isinstance(value, dict):
        raise FinalCombinedDataError(
            f"{where} must be a JSON object, got {type(value).__name__}"
        )
    expected_set = set(expected)
    actual_set = set(value)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise FinalCombinedDataError(f"{where} has non-exact schema: {'; '.join(details)}")


def _decode_row(raw_json: bytes, *, source: str, line_number: int) -> _SourceRow:
    location = f"{source}:{line_number}"
    try:
        text = raw_json.decode("utf-8")
    except UnicodeDecodeError as error:
        raise FinalCombinedDataError(f"{location}: row is not valid UTF-8") from error
    try:
        payload = json.loads(text, object_pairs_hook=_unique_object)
    except (_DuplicateJsonKeyError, json.JSONDecodeError) as error:
        raise FinalCombinedDataError(f"{location}: invalid strict JSON: {error}") from error

    _require_exact_keys(payload, RECORD_FIELDS, where=f"{location}: dataset row")
    _require_exact_keys(payload["score"], SCORE_FIELDS, where=f"{location}: score")
    try:
        record = EssayRecord.from_mapping(payload)
    except SchemaError as error:
        raise FinalCombinedDataError(f"{location}: {error}") from error
    return _SourceRow(
        source=source,
        line_number=line_number,
        raw_json=raw_json,
        record=record,
    )


def _load_source(name: str, path: Path) -> _SourceDataset:
    try:
        source_bytes = path.read_bytes()
    except OSError as error:
        raise FinalCombinedDataError(f"cannot read {name} source {path}: {error}") from error
    if not source_bytes:
        raise FinalCombinedDataError(f"{name} source is empty: {path}")

    raw_rows = source_bytes.split(b"\n")
    if raw_rows[-1] == b"":
        raw_rows.pop()
    rows: list[_SourceRow] = []
    for line_number, raw_row in enumerate(raw_rows, start=1):
        if raw_row.endswith(b"\r"):
            raw_row = raw_row[:-1]
        if not raw_row.strip():
            raise FinalCombinedDataError(f"{name}:{line_number}: blank JSONL row")
        rows.append(_decode_row(raw_row, source=name, line_number=line_number))
    if not rows:
        raise FinalCombinedDataError(f"{name} source is empty: {path}")
    return _SourceDataset(
        name=name,
        path=path,
        sha256=hashlib.sha256(source_bytes).hexdigest(),
        rows=tuple(rows),
    )


def _collision_kind(previous: _SourceRow, current: _SourceRow) -> str:
    return "duplicate" if previous.record == current.record else "conflicting"


def _validate_unique_identifiers(sources: Iterable[_SourceDataset]) -> None:
    ids: dict[str, _SourceRow] = {}
    document_ids: dict[str, _SourceRow] = {}
    for source in sources:
        for row in source.rows:
            previous_id = ids.get(row.record.id)
            if previous_id is not None:
                kind = _collision_kind(previous_id, row)
                raise FinalCombinedDataError(
                    f"{kind} id {row.record.id!r}: "
                    f"{previous_id.location} and {row.location}"
                )
            ids[row.record.id] = row

            previous_document = document_ids.get(row.record.document_id)
            if previous_document is not None:
                kind = _collision_kind(previous_document, row)
                raise FinalCombinedDataError(
                    f"{kind} document_id {row.record.document_id!r}: "
                    f"{previous_document.location} and {row.location}"
                )
            document_ids[row.record.document_id] = row


def _validate_no_cross_source_duplicate_essays(
    train: _SourceDataset,
    validation: _SourceDataset,
) -> None:
    """Preserve the canonical train/validation duplicate-text leakage guard."""

    train_by_essay: dict[str, _SourceRow] = {}
    for row in train.rows:
        train_by_essay.setdefault(row.record.essay, row)
    for row in validation.rows:
        previous = train_by_essay.get(row.record.essay)
        if previous is not None:
            raise FinalCombinedDataError(
                "train and validation contain exact duplicate essay text: "
                f"{previous.location} and {row.location}"
            )


def _cohort_counts(rows: Iterable[_SourceRow]) -> dict[str, int]:
    counts = Counter(infer_cohort(row.record.id) for row in rows)
    return dict(sorted(counts.items()))


def _source_manifest(source: _SourceDataset) -> dict[str, Any]:
    return {
        "path": str(source.path),
        "sha256": source.sha256,
        "record_count": len(source.rows),
        "ordered_ids": [row.record.id for row in source.rows],
        "ordered_document_ids": [row.record.document_id for row in source.rows],
        "cohort_counts": _cohort_counts(source.rows),
    }


def final_combined_code_contract(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    files = {
        relative: sha256_file(root / relative)
        for relative in FINAL_COMBINED_CODE_FILES
    }
    return {"files": files, "sha256": sha256_json(files)}


def _require_unchanged_files(expected_sha256: dict[Path, str]) -> None:
    for path, expected in expected_sha256.items():
        if sha256_file(path) != expected:
            raise FinalCombinedDataError(
                f"immutable final-training input changed during the run: {path}"
            )


def _resolve_and_validate_paths(
    *,
    train_source: str | Path,
    validation_source: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
) -> tuple[Path, Path, Path, Path]:
    train = Path(train_source).resolve()
    validation = Path(validation_source).resolve()
    output = Path(output_path).resolve()
    manifest = Path(manifest_path).resolve()
    if train == validation:
        raise FinalCombinedDataError("train and validation sources must be different files")
    if output == manifest:
        raise FinalCombinedDataError("combined JSONL and manifest paths must differ")
    for destination in (output, manifest):
        if destination in (train, validation):
            raise FinalCombinedDataError(
                f"output must not alias a read-only source: {destination}"
            )
        if destination.exists():
            raise ArtifactExistsError(f"refusing to overwrite existing artifact: {destination}")
    return train, validation, output, manifest


def _write_exclusive_pair(
    *,
    output_path: Path,
    output_bytes: bytes,
    manifest_path: Path,
    manifest_bytes: bytes,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_fd: int | None = None
    manifest_fd: int | None = None
    created: list[Path] = []
    completed = False
    exclusive_flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    )
    try:
        output_fd = os.open(output_path, exclusive_flags, 0o644)
        created.append(output_path)
        manifest_fd = os.open(
            manifest_path,
            exclusive_flags,
            0o644,
        )
        created.append(manifest_path)

        output_handle = os.fdopen(output_fd, "wb")
        output_fd = None
        with output_handle:
            output_handle.write(output_bytes)
            output_handle.flush()
            os.fsync(output_handle.fileno())

        manifest_handle = os.fdopen(manifest_fd, "wb")
        manifest_fd = None
        with manifest_handle:
            manifest_handle.write(manifest_bytes)
            manifest_handle.flush()
            os.fsync(manifest_handle.fileno())
        completed = True
    except FileExistsError as error:
        raise ArtifactExistsError(
            f"refusing to overwrite existing artifact: {error.filename}"
        ) from error
    finally:
        if output_fd is not None:
            os.close(output_fd)
        if manifest_fd is not None:
            os.close(manifest_fd)
        if not completed:
            for created_path in reversed(created):
                created_path.unlink(missing_ok=True)


def build_final_combined_dataset(
    *,
    train_source: str | Path,
    validation_source: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    project_root: str | Path,
    config_path: str | Path,
    validation_label_training_acknowledged: bool,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a new immutable train→validation JSONL and provenance manifest."""

    require_validation_label_training_acknowledgement(
        validation_label_training_acknowledged
    )
    train_path, validation_path, output, manifest = _resolve_and_validate_paths(
        train_source=train_source,
        validation_source=validation_source,
        output_path=output_path,
        manifest_path=manifest_path,
    )
    config = Path(config_path).resolve()
    config_sha256 = sha256_file(config)
    if (
        expected_config_sha256 is not None
        and config_sha256 != expected_config_sha256
    ):
        raise FinalCombinedDataError(
            f"configuration changed after it was loaded: {config}"
        )
    code_contract = final_combined_code_contract(project_root)
    train = _load_source("train", train_path)
    validation = _load_source("validation", validation_path)
    _validate_unique_identifiers((train, validation))
    _validate_no_cross_source_duplicate_essays(train, validation)
    immutable_inputs = {
        config: config_sha256,
        train_path: train.sha256,
        validation_path: validation.sha256,
    }

    combined_rows = (*train.rows, *validation.rows)
    output_bytes = b"\n".join(row.raw_json for row in combined_rows) + b"\n"
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    payload: dict[str, Any] = {
        "artifact_type": "final_combined_labeled_dataset",
        "format_version": FINAL_COMBINED_FORMAT_VERSION,
        "authorization": {
            "validation_label_training_acknowledged": True,
            "condition": "competition_rules_explicitly_allow_validation_label_training",
        },
        "configuration": {
            "path": str(config),
            "sha256": config_sha256,
        },
        "sources": {
            "train": _source_manifest(train),
            "validation": _source_manifest(validation),
        },
        "combined": {
            "path": str(output),
            "sha256": output_sha256,
            "record_count": len(combined_rows),
            "serialization": {
                "format": "jsonl",
                "encoding": "utf-8",
                "row_bytes": "source JSON bytes excluding the line terminator",
                "line_ending": "LF",
                "terminal_newline": True,
            },
            "ordered_ids": [row.record.id for row in combined_rows],
            "ordered_document_ids": [
                row.record.document_id for row in combined_rows
            ],
            "cohort_counts": _cohort_counts(combined_rows),
            "source_order": ["train", "validation"],
        },
        "code": code_contract,
    }
    manifest_bytes = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")
    _require_unchanged_files(immutable_inputs)
    if final_combined_code_contract(project_root) != code_contract:
        raise FinalCombinedDataError(
            "final-training combination source code changed during the run"
        )
    _write_exclusive_pair(
        output_path=output,
        output_bytes=output_bytes,
        manifest_path=manifest,
        manifest_bytes=manifest_bytes,
    )
    return payload
