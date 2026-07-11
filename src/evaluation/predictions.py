"""Canonical JSONL prediction I/O.

Canonical rows have exactly this shape::

    {"id": ..., "prompt_num": ...,
     "prediction": {"content": 3.1, "organization": 3.2, "expression": 3.7},
     "model": ...}

Readers also understand the official/natural LLM form where each trait is a
``{"score": ..., "rationale": ...}`` object, including legacy ``parsed`` wrappers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from .metrics import TRAITS, extract_prediction, get_field
from src.inference.serializer import validate_prediction as validate_final_prediction


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def canonical_prediction(
    record_id: str,
    prompt_num: str,
    scores: Any,
    model: str,
    *,
    validate_range: bool = True,
) -> dict[str, Any]:
    """Build and validate one canonical prediction row."""

    if record_id is None or not str(record_id).strip():
        raise ValueError("prediction id must be non-empty")
    if prompt_num is None or not str(prompt_num).strip():
        raise ValueError("prediction prompt_num must be non-empty")
    if model is None or not str(model).strip():
        raise ValueError("prediction model must be non-empty")

    vector = extract_prediction(scores)
    if validate_range and ((vector < 1.0).any() or (vector > 5.0).any()):
        raise ValueError(f"prediction scores must lie in [1, 5], got {vector.tolist()}")
    return {
        "id": str(record_id),
        "prompt_num": str(prompt_num),
        "prediction": {
            trait: float(vector[index]) for index, trait in enumerate(TRAITS)
        },
        "model": str(model),
    }


def normalize_prediction(
    record: Any,
    *,
    model: str | None = None,
    validate_range: bool = True,
) -> dict[str, Any]:
    """Normalize a canonical, official nested, or legacy prediction record."""

    record_id = get_field(record, "id")
    prompt_num = get_field(record, "prompt_num")
    resolved_model = model
    if resolved_model is None:
        resolved_model = get_field(record, "model", None)
    if resolved_model is None:
        resolved_model = get_field(record, "model_id", None)
    if resolved_model is None:
        resolved_model = get_field(record, "run_tag", "unknown")
    return canonical_prediction(
        record_id,
        prompt_num,
        record,
        resolved_model,
        validate_range=validate_range,
    )


def _iter_json_source(source: str | Path | TextIO | Iterable[Any]) -> Iterable[Any]:
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        return

    for line_number, item in enumerate(source, start=1):
        if isinstance(item, str):
            if not item.strip():
                continue
            try:
                yield json.loads(item)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        else:
            yield item


def read_predictions(
    source: str | Path | TextIO | Iterable[Any],
    *,
    model: str | None = None,
    validate_range: bool = True,
    require_unique_ids: bool = True,
) -> list[dict[str, Any]]:
    """Read and normalize prediction JSONL or an iterable of prediction objects."""

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, record in enumerate(_iter_json_source(source), start=1):
        try:
            normalized = normalize_prediction(
                record, model=model, validate_range=validate_range
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid prediction on row {line_number}: {exc}") from exc
        if require_unique_ids and normalized["id"] in seen:
            raise ValueError(f"duplicate prediction id: {normalized['id']!r}")
        seen.add(normalized["id"])
        result.append(normalized)
    return result


def read_canonical_predictions(path: str | Path) -> list[dict[str, Any]]:
    """Read only the exact internal canonical JSONL contract, without legacy coercion."""

    source = Path(path)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank line is forbidden in strict JSONL: {line_number}")
            try:
                row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid strict JSON on line {line_number}: {error}") from error
            if not isinstance(row, Mapping) or set(row) != {
                "id",
                "prompt_num",
                "prediction",
                "model",
            }:
                raise ValueError(
                    f"strict row {line_number} must contain exactly "
                    "id, prompt_num, prediction, model"
                )
            if any(
                not isinstance(row[field], str) or not row[field].strip()
                for field in ("id", "prompt_num", "model")
            ):
                raise ValueError(f"strict row {line_number} has an invalid text field")
            prediction = row["prediction"]
            if not isinstance(prediction, Mapping) or set(prediction) != set(TRAITS):
                raise ValueError(
                    f"strict row {line_number} prediction keys must be exactly {TRAITS}"
                )
            for trait in TRAITS:
                score = prediction[trait]
                if isinstance(score, bool) or not isinstance(score, Real):
                    raise ValueError(
                        f"strict row {line_number} {trait} score must be a JSON number"
                    )
            normalized = canonical_prediction(
                row["id"], row["prompt_num"], prediction, row["model"]
            )
            if normalized["id"] in seen:
                raise ValueError(f"duplicate prediction id: {normalized['id']!r}")
            seen.add(normalized["id"])
            result.append(normalized)
    if not result:
        raise ValueError(f"prediction file is empty: {source}")
    return result


def read_final_predictions(path: str | Path) -> list[dict[str, Any]]:
    """Read the exact ID-bearing final `score+rationale` JSONL contract."""

    source = Path(path)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank line is forbidden in final JSONL: {line_number}")
            try:
                row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid final JSON on line {line_number}: {error}") from error
            if not isinstance(row, Mapping) or set(row) != {
                "id",
                "prompt_num",
                "prediction",
                "model",
            }:
                raise ValueError(
                    f"final row {line_number} must contain exactly "
                    "id, prompt_num, prediction, model"
                )
            for field in ("id", "prompt_num", "model"):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError(f"final row {line_number} has invalid {field}")
            prediction = validate_final_prediction(row["prediction"])
            record_id = row["id"]
            if record_id in seen:
                raise ValueError(f"duplicate final prediction id: {record_id!r}")
            seen.add(record_id)
            result.append(
                {
                    "id": record_id,
                    "prompt_num": row["prompt_num"],
                    "prediction": prediction,
                    "model": row["model"],
                }
            )
    if not result:
        raise ValueError(f"final prediction file is empty: {source}")
    return result


def write_predictions(
    path: str | Path,
    records: Iterable[Any],
    *,
    model: str | None = None,
    validate_range: bool = True,
) -> Path:
    """Validate and write canonical UTF-8 JSONL, one compact object per line."""

    output_path = Path(path)
    normalized_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_number, record in enumerate(records, start=1):
        try:
            normalized = normalize_prediction(
                record, model=model, validate_range=validate_range
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid prediction on row {row_number}: {exc}") from exc
        if normalized["id"] in seen:
            raise ValueError(f"duplicate prediction id: {normalized['id']!r}")
        seen.add(normalized["id"])
        normalized_rows.append(normalized)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for normalized in normalized_rows:
            handle.write(
                json.dumps(
                    normalized,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
    return output_path


def prediction_records(
    records: Sequence[Any],
    scores: Sequence[Sequence[float]] | np.ndarray,
    *,
    model: str,
    validate_range: bool = True,
) -> list[dict[str, Any]]:
    """Pair an ``(n, 3)`` score matrix with source ids and prompt numbers."""

    matrix = np.asarray(scores, dtype=float)
    if matrix.shape != (len(records), len(TRAITS)):
        raise ValueError(
            f"scores must have shape ({len(records)}, 3), got {matrix.shape}"
        )
    return [
        canonical_prediction(
            get_field(record, "id"),
            get_field(record, "prompt_num"),
            matrix[index],
            model,
            validate_range=validate_range,
        )
        for index, record in enumerate(records)
    ]


__all__ = [
    "canonical_prediction",
    "normalize_prediction",
    "prediction_records",
    "read_canonical_predictions",
    "read_final_predictions",
    "read_predictions",
    "write_predictions",
]
