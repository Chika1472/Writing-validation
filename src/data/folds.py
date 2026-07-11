from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.data.schema import EssayRecord, ensure_record


_FULL_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_SHORT_COHORT_RE = re.compile(r"^[^\d]*(\d{2})")


class FoldError(ValueError):
    """Raised when fold assignments are invalid."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise FoldError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def infer_cohort(record_id: str) -> str:
    """Infer collection year for split balancing only, never for model features."""

    if not isinstance(record_id, str):
        raise TypeError("record_id must be a string")
    full_year = _FULL_YEAR_RE.search(record_id)
    if full_year:
        return full_year.group(1)
    short_year = _SHORT_COHORT_RE.match(record_id)
    if short_year:
        return f"20{short_year.group(1)}"
    return "unknown"


def _stable_hash(seed: int, value: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}\0{value}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big", signed=False)


def _score_bands(records: Sequence[EssayRecord], score_bins: int) -> dict[str, int]:
    if score_bins < 1:
        raise FoldError("score_bins must be at least 1")
    values = np.asarray([record.score.computed_average for record in records], dtype=float)
    if score_bins == 1:
        return {record.id: 0 for record in records}
    cut_points = np.quantile(
        values, np.linspace(0.0, 1.0, score_bins + 1)[1:-1], method="linear"
    )
    bands = np.searchsorted(cut_points, values, side="right")
    return {record.id: int(band) for record, band in zip(records, bands, strict=True)}


def make_folds(
    records: Sequence[EssayRecord | Mapping[str, Any]],
    *,
    n_splits: int = 5,
    seed: int = 42,
    score_bins: int = 5,
) -> dict[str, int]:
    """Create deterministic prompt × cohort × score-band folds.

    A greedy stratifier is used instead of ``StratifiedKFold`` so a composite
    stratum with fewer than ``n_splits`` rows is safely spread over as many
    folds as possible rather than raising or silently dropping records.
    """

    if n_splits < 2:
        raise FoldError("n_splits must be at least 2")
    canonical = [ensure_record(record) for record in records]
    if len(canonical) < n_splits:
        raise FoldError("number of records must be at least n_splits")

    ids = [record.id for record in canonical]
    duplicate_ids = sorted(record_id for record_id, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        raise FoldError(f"duplicate record id(s): {', '.join(duplicate_ids[:5])}")

    bands = _score_bands(canonical, score_bins)
    strata: dict[tuple[str, str, int], list[EssayRecord]] = defaultdict(list)
    for record in canonical:
        key = (record.prompt_num, infer_cohort(record.id), bands[record.id])
        strata[key].append(record)

    fold_sizes = [0] * n_splits
    assignments: dict[str, int] = {}
    ordered_strata = sorted(
        strata.items(),
        key=lambda item: (-len(item[1]), _stable_hash(seed, repr(item[0]))),
    )
    for stratum, members in ordered_strata:
        stratum_counts = [0] * n_splits
        ordered_members = sorted(
            members, key=lambda record: (_stable_hash(seed, record.id), record.id)
        )
        for record in ordered_members:
            fold = min(
                range(n_splits),
                key=lambda candidate: (
                    stratum_counts[candidate],
                    fold_sizes[candidate],
                    _stable_hash(seed, f"{stratum!r}\0{candidate}"),
                ),
            )
            assignments[record.id] = fold
            stratum_counts[fold] += 1
            fold_sizes[fold] += 1

    if set(assignments) != set(ids):
        raise FoldError("internal error: every record must receive exactly one fold")
    return assignments


def validate_folds(
    assignments: Mapping[str, int], *, n_splits: int | None = None
) -> dict[str, int]:
    validated: dict[str, int] = {}
    for record_id, fold in assignments.items():
        if not isinstance(record_id, str) or not record_id.strip():
            raise FoldError("fold assignment id must be a non-empty string")
        if isinstance(fold, bool) or not isinstance(fold, int):
            raise FoldError(f"fold for {record_id!r} must be an integer")
        if fold < 0 or (n_splits is not None and fold >= n_splits):
            expected = f"[0, {n_splits})" if n_splits is not None else "non-negative"
            raise FoldError(f"fold for {record_id!r} must be {expected}, got {fold}")
        validated[record_id] = fold
    if not validated:
        raise FoldError("fold assignments must not be empty")
    return validated


def save_folds(
    assignments: Mapping[str, int],
    path: str | Path,
    *,
    create_only: bool = False,
) -> Path:
    """Save canonical fold JSONL ordered by ID.

    ``create_only=True`` is the immutable-artifact mode used by fold generation
    CLIs.  The legacy default remains available to callers that intentionally
    manage replacement themselves.
    """

    validated = validate_folds(assignments)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        handle = output_path.open(
            "x" if create_only else "w",
            encoding="utf-8",
            newline="\n",
        )
        created = create_only
        with handle:
            for record_id in sorted(validated):
                row = {"id": record_id, "fold": validated[record_id]}
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
    except Exception:
        if created:
            output_path.unlink(missing_ok=True)
        raise
    return output_path


def load_folds(path: str | Path, *, n_splits: int | None = None) -> dict[str, int]:
    fold_path = Path(path)
    assignments: dict[str, int] = {}
    with fold_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise FoldError(f"{fold_path}:{line_number}: blank JSONL row")
            try:
                row = json.loads(line, object_pairs_hook=_unique_json_object)
            except FoldError as exc:
                raise FoldError(f"{fold_path}:{line_number}: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise FoldError(
                    f"{fold_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict) or set(row) != {"id", "fold"}:
                raise FoldError(
                    f"{fold_path}:{line_number}: fields must be exactly id and fold"
                )
            record_id = row["id"]
            if isinstance(record_id, str) and record_id in assignments:
                raise FoldError(f"{fold_path}:{line_number}: duplicate id {record_id!r}")
            single = validate_folds({record_id: row["fold"]}, n_splits=n_splits)
            assignments.update(single)
    return validate_folds(assignments, n_splits=n_splits)
