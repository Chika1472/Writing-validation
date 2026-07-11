from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from src.data.folds import FoldError, validate_folds
from src.data.schema import EssayInput, EssayRecord, ensure_essay_input


LOPO_CONTRACT_VERSION = "leave_one_prompt_out_v1"
LOPO_PROMPT_ORDERING = "prompt_num_unicode_codepoint_ascending_v1"


def _canonical_inputs(
    records: Sequence[EssayInput | EssayRecord | Mapping[str, Any]],
) -> list[EssayInput]:
    canonical = [ensure_essay_input(record) for record in records]
    if not canonical:
        raise FoldError("LOPO records must not be empty")

    ids = [record.id for record in canonical]
    duplicates = sorted(
        record_id for record_id, count in Counter(ids).items() if count > 1
    )
    if duplicates:
        raise FoldError(f"duplicate record id(s): {', '.join(duplicates[:5])}")

    prompts = {record.prompt_num for record in canonical}
    if len(prompts) < 2:
        raise FoldError("LOPO requires at least two distinct prompt_num values")
    return canonical


def _expected_prompt_mapping(records: Sequence[EssayInput]) -> dict[str, int]:
    ordered_prompts = sorted({record.prompt_num for record in records})
    return {prompt_num: fold for fold, prompt_num in enumerate(ordered_prompts)}


def make_lopo_folds(
    records: Sequence[EssayInput | EssayRecord | Mapping[str, Any]],
) -> dict[str, int]:
    """Assign every prompt to one deterministic validation fold.

    Only score-free :class:`EssayInput` fields are canonicalized.  In
    particular, score values are neither inspected nor used to choose folds.
    Fold numbers follow the Unicode code-point ordering of ``prompt_num``.
    """

    canonical = _canonical_inputs(records)
    prompt_to_fold = _expected_prompt_mapping(canonical)
    assignments = {
        record.id: prompt_to_fold[record.prompt_num] for record in canonical
    }
    build_lopo_contract(canonical, assignments)
    return assignments


def build_lopo_contract(
    records: Sequence[EssayInput | EssayRecord | Mapping[str, Any]],
    assignments: Mapping[str, int],
) -> dict[str, Any]:
    """Validate assignments and return the prompt-to-fold bijection contract."""

    canonical = _canonical_inputs(records)
    prompt_to_fold = _expected_prompt_mapping(canonical)
    validated = validate_folds(assignments, n_splits=len(prompt_to_fold))

    expected_ids = {record.id for record in canonical}
    observed_ids = set(validated)
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise FoldError(
            "LOPO assignments must contain every input id exactly once; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    for record in canonical:
        expected_fold = prompt_to_fold[record.prompt_num]
        if validated[record.id] != expected_fold:
            raise FoldError(
                f"LOPO fold mismatch for {record.id!r}: prompt_num "
                f"{record.prompt_num!r} must map to fold {expected_fold}"
            )

    fold_to_prompt = {
        str(fold): prompt_num for prompt_num, fold in prompt_to_fold.items()
    }
    return {
        "version": LOPO_CONTRACT_VERSION,
        "ordering": LOPO_PROMPT_ORDERING,
        "n_splits": len(prompt_to_fold),
        "prompt_to_fold": prompt_to_fold,
        "fold_to_prompt": fold_to_prompt,
    }
