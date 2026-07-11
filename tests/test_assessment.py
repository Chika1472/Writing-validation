import json
from pathlib import Path

import numpy as np
import pytest

from src.assessment.artifact import (
    build_deployment_artifact,
    load_deployment_artifact,
)
from src.assessment.cache import (
    ASSESSMENT_CACHE_ARTIFACT,
    assessment_cache_manifest_path,
    load_assessment_cache,
    write_assessment_npz,
)
from src.assessment.codebook import (
    ANSWER_VALUES,
    DEFAULT_ANSWER_CODES,
    single_token_code_ids,
    validate_codebook,
)
from src.assessment.contracts import assessment_extraction_code_sha256
from src.assessment.prompting import ASSESSMENT_QUERY_SHA256
from src.assessment.questions import (
    QUESTION_IDS,
    QUESTION_VERSION,
    QUESTIONS,
    QUESTIONS_SHA256,
    TRAITS,
    question_contract,
    questions_for_trait,
)
from src.assessment.ridge import nested_oof_ridge
from src.utils.hashing import sha256_file, sha256_json


class _CharacterTokenizer:
    def __call__(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return json.dumps(messages, ensure_ascii=False) + "\nASSISTANT\n"


def _probabilities(rows: int) -> np.ndarray:
    values = np.empty((rows, 18, 5), dtype=float)
    for row in range(rows):
        for question in range(18):
            raw = np.asarray(
                [1 + ((row * 3 + question + code) % 7) for code in range(5)],
                dtype=float,
            )
            values[row, question] = raw / raw.sum()
    return values


def test_question_and_codebook_contract_is_fixed_and_single_token() -> None:
    assert QUESTION_VERSION == "korean_aq_v1"
    assert len(QUESTIONS) == 18
    assert len(set(QUESTION_IDS)) == 18
    assert all(len(questions_for_trait(trait)) == 6 for trait in TRAITS)

    tokenizer = _CharacterTokenizer()
    assert single_token_code_ids(tokenizer, "prefix", DEFAULT_ANSWER_CODES) == tuple(
        ord(code) for code in DEFAULT_ANSWER_CODES
    )
    codebook = validate_codebook(tokenizer)
    assert codebook["answer_token_ids"] == [ord(code) for code in DEFAULT_ANSWER_CODES]
    with pytest.raises(ValueError, match="not exactly one appended token"):
        single_token_code_ids(tokenizer, "prefix", ("AA", "B", "C", "D", "E"))


def test_assessment_cache_is_hash_and_question_bound(tmp_path: Path) -> None:
    cache_path = tmp_path / "features.npz"
    probabilities = _probabilities(2).astype(np.float32)
    write_assessment_npz(
        cache_path,
        ids=["a", "b"],
        prompt_nums=["Q1", "Q2"],
        logits=np.log(probabilities),
        probabilities=probabilities,
    )
    context_hashes = {question_id: "c" * 64 for question_id in QUESTION_IDS}
    token_ids = [65, 66, 67, 68, 69]
    codebook_payload = {
        "answer_codes": list(DEFAULT_ANSWER_CODES),
        "answer_values": list(ANSWER_VALUES),
        "answer_token_ids": token_ids,
        "context_hashes": context_hashes,
    }
    feature_payload = {
        "artifact_version": 1,
        "model_id": "Qwen/Qwen3-14B",
        "model_revision": "a" * 40,
        "tokenizer_revision": "b" * 40,
        "precision": "4bit",
        "max_length": 3072,
        "batch_size": 4,
        "seed": 42,
        "deterministic_torch": True,
        "extractor_code_sha256": "d" * 64,
        "reproducibility_code_sha256": "e" * 64,
        "quantization": {
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "compute_dtype": "bfloat16",
        },
        "feature_type": "restricted_answer_probabilities",
        "question_version": QUESTION_VERSION,
        "questions_sha256": QUESTIONS_SHA256,
        "assessment_query_sha256": ASSESSMENT_QUERY_SHA256,
        "answer_codes": list(DEFAULT_ANSWER_CODES),
        "answer_token_ids": token_ids,
        "codebook_sha256": sha256_json(codebook_payload),
    }
    manifest = {
        "artifact_type": ASSESSMENT_CACHE_ARTIFACT,
        "artifact_version": 1,
        "candidate_branch": True,
        "auto_promoted": False,
        "label_free_feature_extraction": True,
        "cache_file": cache_path.name,
        "cache_sha256": sha256_file(cache_path),
        "rows": 2,
        "logit_shape": [2, 18, 5],
        "probability_shape": [2, 18, 5],
        "question_version": QUESTION_VERSION,
        "questions_sha256": QUESTIONS_SHA256,
        "question_ids": list(QUESTION_IDS),
        "question_contract": question_contract(),
        "assessment_query_sha256": ASSESSMENT_QUERY_SHA256,
        "model_id": "Qwen/Qwen3-14B",
        "model_revision": "a" * 40,
        "tokenizer_revision": "b" * 40,
        "precision": "4bit",
        "max_length": 3072,
        "batch_size": 4,
        "seed": 42,
        "deterministic_torch": True,
        "extractor_code_sha256": feature_payload["extractor_code_sha256"],
        "reproducibility_code_sha256": feature_payload[
            "reproducibility_code_sha256"
        ],
        "quantization": feature_payload["quantization"],
        "feature_type": "restricted_answer_probabilities",
        "answer_codes": list(DEFAULT_ANSWER_CODES),
        "answer_values": list(ANSWER_VALUES),
        "answer_token_ids": token_ids,
        "codebook_sha256": sha256_json(codebook_payload),
        "codebook_context_hashes": context_hashes,
        "feature_signature_payload": feature_payload,
        "feature_signature": sha256_json(feature_payload),
    }
    assessment_cache_manifest_path(cache_path).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    restored = load_assessment_cache(cache_path)
    np.testing.assert_allclose(restored.probabilities, probabilities)

    manifest["questions_sha256"] = "0" * 64
    assessment_cache_manifest_path(cache_path).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="question contract"):
        load_assessment_cache(cache_path)


def test_outer_held_labels_cannot_change_their_assessment_oof_model() -> None:
    rows = 16
    probabilities = _probabilities(rows)
    folds = np.asarray([index % 4 for index in range(rows)])
    targets = np.column_stack(
        [
            1.0 + 4.0 * probabilities[:, 0, 4],
            1.0 + 4.0 * probabilities[:, 6, 4],
            1.0 + 4.0 * probabilities[:, 12, 4],
        ]
    )
    ids = [f"row-{index}" for index in range(rows)]
    original = nested_oof_ridge(
        probabilities,
        targets,
        folds,
        ids,
        question_counts=(3, 6),
        alphas=(0.1, 10.0),
    )
    changed_targets = targets.copy()
    changed_targets[folds == 0] = 5.0 - changed_targets[folds == 0] + 1.0
    changed = nested_oof_ridge(
        probabilities,
        changed_targets,
        folds,
        ids,
        question_counts=(3, 6),
        alphas=(0.1, 10.0),
    )
    np.testing.assert_allclose(
        original.oof_predictions[folds == 0],
        changed.oof_predictions[folds == 0],
    )
    assert original.outer_reports[0] == changed.outer_reports[0]
    assert original.fold_models["0"] == changed.fold_models["0"]


def test_deployment_artifact_is_candidate_only_and_self_signed(tmp_path: Path) -> None:
    probabilities = _probabilities(12)
    folds = [index % 3 for index in range(12)]
    targets = np.column_stack(
        [
            1.0 + 4.0 * probabilities[:, 0, 4],
            1.0 + 4.0 * probabilities[:, 6, 4],
            1.0 + 4.0 * probabilities[:, 12, 4],
        ]
    )
    result = nested_oof_ridge(
        probabilities,
        targets,
        folds,
        [f"id-{index}" for index in range(12)],
        question_counts=(3,),
        alphas=(1.0,),
    )
    project_root = Path(__file__).resolve().parents[1]
    feature_contract = {
        "artifact_version": 1,
        "model_id": "Qwen/Qwen3-14B",
        "model_revision": "a" * 40,
        "tokenizer_revision": "b" * 40,
        "precision": "4bit",
        "max_length": 3072,
        "batch_size": 4,
        "seed": 42,
        "deterministic_torch": True,
        "extractor_code_sha256": assessment_extraction_code_sha256(),
        "reproducibility_code_sha256": sha256_file(
            project_root / "src" / "utils" / "reproducibility.py"
        ),
        "quantization": {
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "compute_dtype": "bfloat16",
        },
        "feature_type": "restricted_answer_probabilities",
        "question_version": QUESTION_VERSION,
        "questions_sha256": QUESTIONS_SHA256,
        "assessment_query_sha256": ASSESSMENT_QUERY_SHA256,
        "answer_codes": list(DEFAULT_ANSWER_CODES),
        "answer_token_ids": [1, 2, 3, 4, 5],
        "codebook_sha256": "c" * 64,
    }
    feature_signature = sha256_json(feature_contract)
    artifact = build_deployment_artifact(
        scorer_name="assessment_candidate",
        feature_signature=feature_signature,
        feature_contract=feature_contract,
        fold_models=result.fold_models,
        outer_selection=result.outer_reports,
        clip_min=1.0,
        clip_max=5.0,
        training_source={"gold_sha256": "b" * 64},
        selection_config={
            "held_fold_exclusion": "strict",
            "normalization": "inner_train_standardization",
            "selection_metric": "rmse",
            "question_count_candidates": [3],
            "alpha_grid": [1.0],
        },
    )
    path = tmp_path / "assessment.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    restored = load_deployment_artifact(path)
    assert restored["auto_promoted"] is False

    artifact["auto_promoted"] = True
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="signature/contract"):
        load_deployment_artifact(path)
