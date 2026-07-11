from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.rationale_judge import (
    DECISION_FIELDS,
    RationaleJudgeValidationError,
    load_verified_review_pack,
    normalize_judgment,
    paired_attempt_seed,
    parse_judgment_json,
    reconcile_order_judgments,
    summarize_keyed_results,
    validate_judge_config,
    validate_key_rows,
    validate_result_rows,
)
from src.utils.hashing import sha256_file


def _judgment(decision: str, reason: str = "글의 실제 내용과 점수의 부합도를 비교했다.") -> dict:
    return {**{field: decision for field in DECISION_FIELDS}, "reason": reason}


def _review_row() -> dict:
    return {
        "review_id": "R0001",
        "prompt_num": "Q1",
        "prompt": "제도 도입에 관한 의견을 쓰시오.",
        "essay": "저는 제도 도입에 찬성합니다. 다만 비용은 검토해야 합니다.",
        "fixed_scores": {
            "content": 3.5,
            "organization": 3.0,
            "expression": 3.5,
        },
        "option_a": {
            "content": "찬성 입장과 비용 검토를 함께 짚었다.",
            "organization": "입장 뒤에 제한점을 제시했다.",
            "expression": "문장은 대체로 명료하다.",
        },
        "option_b": {
            "content": "입장은 있으나 근거가 짧다.",
            "organization": "전개가 단순하다.",
            "expression": "표현은 이해할 수 있다.",
        },
        "review": {
            "grounded_in_essay": None,
            "specific_and_helpful": None,
            "trait_separation": None,
            "consistent_with_fixed_scores": None,
            "overall_preference": None,
            "notes": "",
        },
    }


def _order(raw_decision: str, *, reverse: bool) -> dict:
    judgment = _judgment(raw_decision)
    return {
        "presented_order": (
            {"A": "option_b", "B": "option_a"}
            if reverse
            else {"A": "option_a", "B": "option_b"}
        ),
        "attempts": [
            {
                "attempt": 1,
                "seed": 7,
                "prompt_sha256": "1" * 64,
                "response_sha256": "2" * 64,
                "valid": True,
            }
        ],
        "judgment": judgment,
        "normalized_judgment": normalize_judgment(
            judgment,
            reverse=reverse,
            reason_max_chars=1000,
        ),
    }


def _stable_result(review_id: str, decision: str) -> dict:
    reverse_raw = {"A": "B", "B": "A", "TIE": "TIE"}[decision]
    ab = _order(decision, reverse=False)
    ba = _order(reverse_raw, reverse=True)
    consensus, unstable = reconcile_order_judgments(
        ab["normalized_judgment"],
        ba["normalized_judgment"],
        reason_max_chars=1000,
    )
    return {
        "review_id": review_id,
        "review_row_sha256": "0" * 64,
        "orders": {"ab": ab, "ba": ba},
        "consensus": consensus,
        "unstable": unstable,
    }


def _judge_config() -> dict:
    return {
        "seed": 2026,
        "reason_max_chars": 220,
        "runtime": {
            "n_ctx": 8192,
            "n_batch": 512,
            "n_gpu_layers": -1,
            "n_threads": None,
            "use_mmap": True,
            "use_mlock": False,
            "verbose": False,
        },
        "generation": {
            "max_tokens": 384,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 40,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "stop": [],
            "max_attempts": 2,
        },
    }


def test_strict_judgment_json_rejects_wrappers_and_duplicate_keys() -> None:
    valid = json.dumps(_judgment("A"), ensure_ascii=False)
    assert parse_judgment_json(valid, reason_max_chars=220)["overall"] == "A"
    with pytest.raises(RationaleJudgeValidationError):
        parse_judgment_json(f"```json\n{valid}\n```", reason_max_chars=220)
    duplicate = valid[:-1] + ',"overall":"B"}'
    with pytest.raises(RationaleJudgeValidationError):
        parse_judgment_json(duplicate, reason_max_chars=220)


def test_swapped_order_is_normalized_to_original_option_labels() -> None:
    normalized = normalize_judgment(
        _judgment("A"),
        reverse=True,
        reason_max_chars=220,
    )
    assert all(normalized[field] == "B" for field in DECISION_FIELDS)


def test_order_disagreement_becomes_unstable_tie_per_field() -> None:
    forward = _judgment("A")
    reverse = _judgment("A")
    reverse["overall"] = "B"
    consensus, unstable = reconcile_order_judgments(
        forward,
        reverse,
        reason_max_chars=220,
    )
    assert consensus["grounding"] == {"decision": "A", "unstable": False}
    assert consensus["overall"] == {"decision": "TIE", "unstable": True}
    assert unstable is True


def test_paired_seed_is_identical_for_both_orders_and_changes_by_attempt() -> None:
    first = paired_attempt_seed(2026, "R0001", 1)
    assert first == paired_attempt_seed(2026, "R0001", 1)
    assert first != paired_attempt_seed(2026, "R0001", 2)


def test_result_validation_enforces_configured_seed_attempt_and_reason_contract() -> None:
    result = _stable_result("R0001", "A")
    expected_seed = paired_attempt_seed(2026, "R0001", 1)
    for order in result["orders"].values():
        order["attempts"][0]["seed"] = expected_seed
    validate_result_rows(
        [result],
        reason_max_chars=220,
        base_seed=2026,
        max_attempts=2,
    )

    bad_seed = json.loads(json.dumps(result, ensure_ascii=False))
    bad_seed["orders"]["ba"]["attempts"][0]["seed"] += 1
    with pytest.raises(RationaleJudgeValidationError):
        validate_result_rows(
            [bad_seed],
            reason_max_chars=220,
            base_seed=2026,
            max_attempts=2,
        )

    bad_reason = json.loads(json.dumps(result, ensure_ascii=False))
    bad_reason["orders"]["ab"]["judgment"]["reason"] = "가" * 221
    with pytest.raises(RationaleJudgeValidationError):
        validate_result_rows(
            [bad_reason],
            reason_max_chars=220,
            base_seed=2026,
            max_attempts=2,
        )


def test_judge_config_rejects_unknown_top_level_keys() -> None:
    config = _judge_config()
    config["assignment_key"] = "must-never-be-accepted.jsonl"
    with pytest.raises(ValueError, match="unknown"):
        validate_judge_config(config)


def test_key_validation_rejects_duplicate_source_ids() -> None:
    rows = [
        {
            "review_id": f"R{index:04d}",
            "id": "same-source-id",
            "option_a": "candidate",
            "option_b": "baseline",
        }
        for index in (1, 2)
    ]
    with pytest.raises(RationaleJudgeValidationError, match="duplicate key source id"):
        validate_key_rows(rows)


def test_review_manifest_is_verified_without_opening_key_file(tmp_path: Path) -> None:
    review_path = tmp_path / "review.jsonl"
    review_path.write_text(
        json.dumps(_review_row(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "review.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "blind_rationale_review_pack",
                "review_file": review_path.name,
                "review_sha256": sha256_file(review_path),
                "key_file": "does-not-exist.jsonl",
                "key_sha256": "a" * 64,
                "rows": 1,
                "score_equality_verified": True,
                "assignment_hidden_from_review_file": True,
            }
        ),
        encoding="utf-8",
    )
    rows, manifest = load_verified_review_pack(review_path, manifest_path)
    assert rows[0]["review_id"] == "R0001"
    assert manifest["key_sha256"] == "a" * 64


def test_keyed_summary_counts_candidate_only_after_blind_judging() -> None:
    result = _stable_result("R0001", "A")
    summary = summarize_keyed_results(
        [result],
        {"R0001": {"option_a": "candidate", "option_b": "baseline"}},
    )
    for field in DECISION_FIELDS:
        assert summary["criteria"][field]["candidate_wins"] == 1
        assert summary["criteria"][field]["baseline_wins"] == 0
        assert summary["criteria"][field]["ties"] == 0


def test_unstable_order_result_is_counted_as_tie_not_candidate_win() -> None:
    ab = _order("A", reverse=False)
    ba = _order("A", reverse=True)  # Displayed A normalizes to original option B.
    consensus, unstable = reconcile_order_judgments(
        ab["normalized_judgment"],
        ba["normalized_judgment"],
        reason_max_chars=1000,
    )
    result = {
        "review_id": "R0001",
        "review_row_sha256": "0" * 64,
        "orders": {"ab": ab, "ba": ba},
        "consensus": consensus,
        "unstable": unstable,
    }
    summary = summarize_keyed_results(
        [result],
        {"R0001": {"option_a": "candidate", "option_b": "baseline"}},
    )
    assert summary["unstable_rows"] == 1
    assert summary["criteria"]["overall"]["ties"] == 1
    assert summary["criteria"]["overall"]["candidate_wins"] == 0
