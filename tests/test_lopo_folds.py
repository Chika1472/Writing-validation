from __future__ import annotations

from collections import defaultdict

import pytest

from src.data.folds import FoldError
from src.data.lopo import (
    LOPO_CONTRACT_VERSION,
    LOPO_PROMPT_ORDERING,
    build_lopo_contract,
    make_lopo_folds,
)


def _row(record_id: str, prompt_num: str) -> dict[str, object]:
    return {
        "id": record_id,
        "document_id": f"document-{record_id}",
        "prompt_num": prompt_num,
        "prompt": f"{prompt_num} 논제",
        "essay": f"{record_id} 본문",
        # Deliberately invalid as a label: LOPO must never parse or use it.
        "score": {"content": "unused"},
    }


def _records() -> list[dict[str, object]]:
    return [
        _row("essay-4", "Q2"),
        _row("essay-1", "Q1"),
        _row("essay-3", "Q10"),
        _row("essay-2", "Q1"),
        _row("essay-5", "Q2"),
    ]


def test_lopo_is_label_free_deterministic_complete_and_order_invariant() -> None:
    records = _records()
    first = make_lopo_folds(records)
    second = make_lopo_folds(list(reversed(records)))

    assert first == second
    assert set(first) == {str(row["id"]) for row in records}
    assert len(first) == len(records)

    prompts_by_fold: dict[int, set[str]] = defaultdict(set)
    for row in records:
        prompts_by_fold[first[str(row["id"])]].add(str(row["prompt_num"]))
    assert all(len(prompts) == 1 for prompts in prompts_by_fold.values())
    assert set(prompts_by_fold) == {0, 1, 2}


def test_lopo_contract_records_the_exact_prompt_fold_bijection() -> None:
    records = _records()
    assignments = make_lopo_folds(records)
    contract = build_lopo_contract(records, assignments)

    assert contract == {
        "version": LOPO_CONTRACT_VERSION,
        "ordering": LOPO_PROMPT_ORDERING,
        "n_splits": 3,
        "prompt_to_fold": {"Q1": 0, "Q10": 1, "Q2": 2},
        "fold_to_prompt": {"0": "Q1", "1": "Q10", "2": "Q2"},
    }


def test_lopo_rejects_duplicate_ids_and_a_single_prompt() -> None:
    duplicate = _row("same", "Q1")
    with pytest.raises(FoldError, match="duplicate record id"):
        make_lopo_folds([duplicate, duplicate, _row("other", "Q2")])

    with pytest.raises(FoldError, match="at least two distinct"):
        make_lopo_folds([_row("a", "Q1"), _row("b", "Q1")])


def test_lopo_contract_rejects_missing_ids_and_split_prompts() -> None:
    records = _records()
    assignments = make_lopo_folds(records)

    missing = dict(assignments)
    missing.pop("essay-1")
    with pytest.raises(FoldError, match="every input id exactly once"):
        build_lopo_contract(records, missing)

    split_prompt = dict(assignments)
    split_prompt["essay-2"] = 2
    with pytest.raises(FoldError, match="fold mismatch"):
        build_lopo_contract(records, split_prompt)
