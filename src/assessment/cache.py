"""Strict NPZ + JSON provenance contract for assessment probabilities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.assessment.codebook import ANSWER_VALUES, DEFAULT_ANSWER_CODES
from src.assessment.prompting import ASSESSMENT_QUERY_SHA256
from src.assessment.questions import (
    QUESTION_IDS,
    QUESTION_VERSION,
    QUESTIONS_SHA256,
    question_contract,
)
from src.data.schema import EssayInput, EssayRecord, ensure_essay_input
from src.utils.hashing import sha256_file, sha256_json, sha256_text


ASSESSMENT_CACHE_ARTIFACT = "assessment_question_probability_cache"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class AssessmentCache:
    path: Path
    ids: tuple[str, ...]
    prompt_nums: tuple[str, ...]
    logits: np.ndarray
    probabilities: np.ndarray
    manifest: dict[str, Any]

    @property
    def feature_signature(self) -> str:
        return str(self.manifest["feature_signature"])


def assessment_cache_manifest_path(cache_path: str | Path) -> Path:
    return Path(cache_path).resolve().with_suffix(".manifest.json")


def write_assessment_npz(
    path: str | Path,
    *,
    ids: Sequence[str],
    prompt_nums: Sequence[str],
    logits: np.ndarray,
    probabilities: np.ndarray,
) -> Path:
    output = Path(path).resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("assessment cache output must use the .npz extension")
    matrix = np.asarray(probabilities, dtype=np.float32)
    logit_matrix = np.asarray(logits, dtype=np.float32)
    expected_shape = (len(ids), len(QUESTION_IDS), 5)
    if matrix.shape != expected_shape or logit_matrix.shape != expected_shape:
        raise ValueError(f"assessment logits/probabilities must have shape {expected_shape}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            ids=np.asarray([str(value) for value in ids], dtype=str),
            prompt_nums=np.asarray([str(value) for value in prompt_nums], dtype=str),
            question_ids=np.asarray(QUESTION_IDS, dtype=str),
            logits=logit_matrix,
            probabilities=matrix,
        )
    return output


def load_assessment_cache(path: str | Path) -> AssessmentCache:
    cache_path = Path(path).resolve()
    manifest_path = assessment_cache_manifest_path(cache_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"assessment cache manifest is required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("artifact_type") != ASSESSMENT_CACHE_ARTIFACT:
        raise ValueError(f"not an assessment probability cache: {manifest_path}")
    if (
        manifest.get("artifact_version") != 1
        or manifest.get("candidate_branch") is not True
        or manifest.get("auto_promoted") is not False
        or manifest.get("label_free_feature_extraction") is not True
    ):
        raise ValueError("assessment cache safety contract is invalid")
    if manifest.get("cache_file") != cache_path.name:
        raise ValueError("assessment cache manifest points to a different filename")
    if manifest.get("cache_sha256") != sha256_file(cache_path):
        raise ValueError("assessment cache hash mismatch")
    if (
        manifest.get("question_version") != QUESTION_VERSION
        or manifest.get("questions_sha256") != QUESTIONS_SHA256
    ):
        raise ValueError("assessment question contract changed since cache creation")
    if manifest.get("question_ids") != list(QUESTION_IDS):
        raise ValueError("assessment cache question order mismatch")
    if manifest.get("question_contract") != question_contract():
        raise ValueError("assessment cache embedded question contract mismatch")
    feature_payload = manifest.get("feature_signature_payload")
    feature_signature = manifest.get("feature_signature")
    if (
        not isinstance(feature_payload, dict)
        or not isinstance(feature_signature, str)
        or _SHA256.fullmatch(feature_signature) is None
        or sha256_json(feature_payload) != feature_signature
    ):
        raise ValueError("assessment cache feature signature is invalid")
    bound_fields = (
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "precision",
        "max_length",
        "batch_size",
        "seed",
        "deterministic_torch",
        "extractor_code_sha256",
        "reproducibility_code_sha256",
        "quantization",
        "feature_type",
        "question_version",
        "questions_sha256",
        "assessment_query_sha256",
        "answer_codes",
        "answer_token_ids",
        "codebook_sha256",
    )
    for field in bound_fields:
        if feature_payload.get(field) != manifest.get(field):
            raise ValueError(f"assessment feature signature does not bind {field}")
    if manifest.get("model_id") != "Qwen/Qwen3-14B":
        raise ValueError("assessment cache model id mismatch")
    if manifest.get("assessment_query_sha256") != ASSESSMENT_QUERY_SHA256:
        raise ValueError("assessment cache query template changed")
    for field in ("model_revision", "tokenizer_revision"):
        value = manifest.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
            raise ValueError(f"assessment cache {field} is not a pinned commit SHA")
    if manifest.get("precision") not in {"4bit", "bf16"}:
        raise ValueError("assessment cache precision is invalid")
    if manifest.get("feature_type") != "restricted_answer_probabilities":
        raise ValueError("assessment cache feature type is invalid")
    max_length = manifest.get("max_length")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("assessment cache max_length is invalid")
    if (
        isinstance(manifest.get("batch_size"), bool)
        or not isinstance(manifest.get("batch_size"), int)
        or manifest["batch_size"] <= 0
        or isinstance(manifest.get("seed"), bool)
        or not isinstance(manifest.get("seed"), int)
        or manifest.get("deterministic_torch") is not True
    ):
        raise ValueError("assessment cache deterministic extraction contract is invalid")
    for field in ("extractor_code_sha256", "reproducibility_code_sha256"):
        value = manifest.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"assessment cache {field} is invalid")
    quantization = manifest.get("quantization")
    if (
        not isinstance(quantization, dict)
        or set(quantization)
        != {"bnb_4bit_quant_type", "bnb_4bit_use_double_quant", "compute_dtype"}
        or quantization.get("compute_dtype") != "bfloat16"
        or quantization.get("bnb_4bit_quant_type") != "nf4"
        or not isinstance(quantization.get("bnb_4bit_use_double_quant"), bool)
    ):
        raise ValueError("assessment cache quantization contract is invalid")
    if manifest.get("answer_codes") != list(DEFAULT_ANSWER_CODES) or manifest.get(
        "answer_values"
    ) != list(ANSWER_VALUES):
        raise ValueError("assessment cache answer-code scale mismatch")
    token_ids = manifest.get("answer_token_ids")
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != 5
        or len(set(token_ids)) != 5
        or any(isinstance(value, bool) or not isinstance(value, int) for value in token_ids)
    ):
        raise ValueError("assessment cache answer token ids are invalid")
    context_hashes = manifest.get("codebook_context_hashes")
    if (
        not isinstance(context_hashes, dict)
        or set(context_hashes) != set(QUESTION_IDS)
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in context_hashes.values()
        )
    ):
        raise ValueError("assessment cache codebook contexts are invalid")
    codebook_payload = {
        "answer_codes": manifest["answer_codes"],
        "answer_values": manifest["answer_values"],
        "answer_token_ids": token_ids,
        "context_hashes": context_hashes,
    }
    if manifest.get("codebook_sha256") != sha256_json(codebook_payload):
        raise ValueError("assessment cache codebook hash is invalid")

    with np.load(cache_path, allow_pickle=False) as archive:
        required = {"ids", "prompt_nums", "question_ids", "logits", "probabilities"}
        if set(archive.files) != required:
            raise ValueError(f"assessment cache NPZ keys must be exactly {sorted(required)}")
        ids = tuple(str(value) for value in archive["ids"].tolist())
        prompt_nums = tuple(str(value) for value in archive["prompt_nums"].tolist())
        question_ids = tuple(str(value) for value in archive["question_ids"].tolist())
        logits = np.asarray(archive["logits"], dtype=np.float32)
        probabilities = np.asarray(archive["probabilities"], dtype=np.float32)

    if question_ids != QUESTION_IDS:
        raise ValueError("assessment NPZ question ids do not match the code contract")
    if not ids or len(ids) != len(prompt_nums) or len(set(ids)) != len(ids):
        raise ValueError("assessment cache ids must be nonempty and unique")
    if any(not value.strip() for value in ids + prompt_nums):
        raise ValueError("assessment cache ids and prompt numbers must be nonempty")
    expected_shape = (len(ids), len(QUESTION_IDS), 5)
    if probabilities.shape != expected_shape or logits.shape != expected_shape:
        raise ValueError("assessment logit/probability tensors have an invalid shape")
    if not np.isfinite(probabilities).all() or not np.isfinite(logits).all():
        raise ValueError("assessment logits/probabilities contain non-finite values")
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("assessment probabilities must lie in [0, 1]")
    if not np.allclose(probabilities.sum(axis=2), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("assessment answer probabilities must sum to one")
    shifted = logits - logits.max(axis=2, keepdims=True)
    expected_probabilities = np.exp(shifted)
    expected_probabilities /= expected_probabilities.sum(axis=2, keepdims=True)
    if not np.allclose(probabilities, expected_probabilities, atol=1e-5, rtol=1e-5):
        raise ValueError("assessment probabilities do not match cached restricted logits")
    if manifest.get("rows") != len(ids):
        raise ValueError("assessment cache row count does not match its manifest")
    if manifest.get("logit_shape") != list(expected_shape) or manifest.get(
        "probability_shape"
    ) != list(expected_shape):
        raise ValueError("assessment cache tensor shapes do not match its manifest")
    return AssessmentCache(cache_path, ids, prompt_nums, logits, probabilities, manifest)


def prompt_hashes(records: Sequence[EssayInput | EssayRecord | dict]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for value in records:
        record = ensure_essay_input(value)
        if record.id in hashes:
            raise ValueError(f"duplicate assessment source id: {record.id!r}")
        hashes[record.id] = sha256_text(record.prompt)
    return dict(sorted(hashes.items()))


def validate_cache_source(
    cache: AssessmentCache,
    records: Sequence[EssayInput | EssayRecord | dict],
    source_path: str | Path,
) -> None:
    rows = [ensure_essay_input(value) for value in records]
    if cache.ids != tuple(record.id for record in rows):
        raise ValueError("assessment cache ids/order do not match the source dataset")
    if cache.prompt_nums != tuple(record.prompt_num for record in rows):
        raise ValueError("assessment cache prompt numbers do not match the source dataset")
    if cache.manifest.get("source_data_sha256") != sha256_file(source_path):
        raise ValueError("assessment cache was not extracted from this exact source file")
    if cache.manifest.get("prompt_hashes") != prompt_hashes(rows):
        raise ValueError("assessment cache prompt hashes do not match the source dataset")
    expected_ids_hash = sha256_json([record.id for record in rows])
    if cache.manifest.get("ordered_ids_sha256") != expected_ids_hash:
        raise ValueError("assessment cache ordered-id hash mismatch")


__all__ = [
    "ASSESSMENT_CACHE_ARTIFACT",
    "AssessmentCache",
    "assessment_cache_manifest_path",
    "load_assessment_cache",
    "prompt_hashes",
    "validate_cache_source",
    "write_assessment_npz",
]
