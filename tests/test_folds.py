from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest

from src.data.folds import FoldError, infer_cohort, load_folds, make_folds, save_folds
from src.data.schema import EssayRecord
from src.utils.manifest import write_manifest


def _record(index: int, *, prompt: int, year: int, score: float) -> EssayRecord:
    record_id = f"GWGR{year % 100:02d}{index:08d}"
    return EssayRecord.from_mapping(
        {
            "id": record_id,
            "document_id": f"GWRW{year % 100:02d}{index:08d}.1",
            "prompt_num": f"Q{prompt}",
            "prompt": f"논제 {prompt}",
            "essay": f"본문 {index}입니다.",
            "score": {
                "content": score,
                "organization": score,
                "expression": score,
                "average": score,
            },
        }
    )


def _records() -> list[EssayRecord]:
    records = []
    for index in range(53):
        records.append(
            _record(
                index,
                prompt=1 + (index % 3),
                year=2023 + (index % 2),
                score=1.0 + (index % 17) * 0.25,
            )
        )
    return records


def test_cohort_is_derived_for_stratification() -> None:
    assert infer_cohort("GWGR2300001070") == "2023"
    assert infer_cohort("essay-2024-x") == "2024"
    assert infer_cohort("no-year") == "unknown"


def test_folds_are_deterministic_complete_balanced_and_order_invariant() -> None:
    records = _records()
    first = make_folds(records, n_splits=5, seed=42, score_bins=5)
    second = make_folds(list(reversed(records)), n_splits=5, seed=42, score_bins=5)

    assert first == second
    assert set(first) == {record.id for record in records}
    assert set(first.values()) == set(range(5))
    sizes = Counter(first.values())
    assert max(sizes.values()) - min(sizes.values()) <= 1


def test_composite_strata_are_even_when_large_and_safe_when_rare() -> None:
    records = _records()
    assignments = make_folds(records, n_splits=5, seed=1337, score_bins=1)
    counts: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    for record in records:
        key = (record.prompt_num, infer_cohort(record.id))
        counts[key][assignments[record.id]] += 1

    for fold_counts in counts.values():
        values = [fold_counts[fold] for fold in range(5)]
        assert max(values) - min(values) <= 1


def test_fold_jsonl_round_trip_is_canonical(tmp_path: Path) -> None:
    assignments = make_folds(_records(), n_splits=5, seed=42)
    path = save_folds(assignments, tmp_path / "nested" / "folds.jsonl")

    assert load_folds(path, n_splits=5) == assignments
    ids_on_disk = [line.split('"')[3] for line in path.read_text(encoding="utf-8").splitlines()]
    assert ids_on_disk == sorted(assignments)

    with pytest.raises(FileExistsError):
        save_folds(assignments, path, create_only=True)


def test_manifest_create_only_never_replaces_an_existing_artifact(tmp_path: Path) -> None:
    path = tmp_path / "folds.manifest.json"
    write_manifest(path, {"version": 1}, create_only=True)
    before = path.read_bytes()

    with pytest.raises(FileExistsError):
        write_manifest(path, {"version": 2}, create_only=True)

    assert path.read_bytes() == before


def test_manifest_rejects_nonstandard_nan_without_leaving_an_artifact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.manifest.json"
    with pytest.raises(ValueError):
        write_manifest(path, {"invalid": float("nan")}, create_only=True)
    assert not path.exists()


def test_fold_validation_rejects_duplicates_and_bad_ranges(tmp_path: Path) -> None:
    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate_path.write_text(
        '{"id":"a","fold":0}\n{"id":"a","fold":1}\n', encoding="utf-8"
    )
    with pytest.raises(FoldError, match="duplicate id"):
        load_folds(duplicate_path, n_splits=5)

    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text('{"id":"a","fold":5}\n', encoding="utf-8")
    with pytest.raises(FoldError, match=r"\[0, 5\)"):
        load_folds(bad_path, n_splits=5)

    duplicate_key_path = tmp_path / "duplicate-key.jsonl"
    duplicate_key_path.write_text(
        '{"id":"a","id":"b","fold":0}\n', encoding="utf-8"
    )
    with pytest.raises(FoldError, match="duplicate JSON key"):
        load_folds(duplicate_key_path, n_splits=5)

    extra_field_path = tmp_path / "extra-field.jsonl"
    extra_field_path.write_text(
        '{"id":"a","fold":0,"extra":true}\n', encoding="utf-8"
    )
    with pytest.raises(FoldError, match="exactly id and fold"):
        load_folds(extra_field_path, n_splits=5)


def test_make_folds_rejects_duplicate_ids() -> None:
    record = _record(1, prompt=1, year=2023, score=3.0)
    with pytest.raises(FoldError, match="duplicate record id"):
        make_folds([record] * 5)
