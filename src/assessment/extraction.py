"""Shared restricted-logit extraction used by offline caching and deployment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.assessment.codebook import DEFAULT_ANSWER_CODES, validate_codebook
from src.assessment.contracts import assessment_extraction_code_sha256
from src.assessment.prompting import ASSESSMENT_QUERY_SHA256, render_assessment_prompt
from src.assessment.questions import QUESTION_VERSION, QUESTIONS, QUESTIONS_SHA256
from src.utils.hashing import sha256_file, sha256_json
from src.utils.reproducibility import seed_everything


@dataclass(frozen=True)
class LoadedAssessmentExtractor:
    model: Any
    tokenizer: Any
    answer_token_ids: tuple[int, ...]
    batch_size: int
    max_length: int
    feature_payload: dict[str, Any]
    feature_signature: str
    codebook: dict[str, Any]


def _optional_qwen_imports() -> tuple[Any, ...]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise ImportError(
            "Assessment-logit extraction requires `pip install -e .[qwen]`."
        ) from error
    return AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _revision(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise ValueError(f"{where} must be a pinned 40-character commit SHA")
    return value


def load_assessment_extractor(
    *,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    precision: str,
    batch_size: int,
    max_length: int,
    seed: int,
    quantization: dict[str, Any],
    answer_codes: Sequence[str] = DEFAULT_ANSWER_CODES,
    device: str = "cuda:0",
    allow_download: bool = False,
) -> LoadedAssessmentExtractor:
    """Load the pinned causal LM and construct its immutable feature contract."""

    if model_id != "Qwen/Qwen3-14B":
        raise ValueError("assessment question v1 is fixed to Qwen/Qwen3-14B")
    model_revision = _revision(model_revision, where="assessment model revision")
    tokenizer_revision = _revision(
        tokenizer_revision, where="assessment tokenizer revision"
    )
    if precision not in {"4bit", "bf16"}:
        raise ValueError("assessment precision must be '4bit' or 'bf16'")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
        or isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or max_length <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError("assessment batch_size/max_length/seed contract is invalid")
    expected_quantization = {
        "bnb_4bit_quant_type",
        "bnb_4bit_use_double_quant",
        "compute_dtype",
    }
    if (
        not isinstance(quantization, dict)
        or set(quantization) != expected_quantization
        or quantization.get("compute_dtype") != "bfloat16"
        or quantization.get("bnb_4bit_quant_type") != "nf4"
        or not isinstance(quantization.get("bnb_4bit_use_double_quant"), bool)
    ):
        raise ValueError("assessment quantization contract is invalid")
    configured_codes = tuple(str(value) for value in answer_codes)
    if configured_codes != DEFAULT_ANSWER_CODES:
        raise ValueError("assessment answer codes must be exactly A, B, C, D, E")
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("Qwen3-14B assessment extraction requires a CUDA device")

    seed_everything(seed, deterministic_torch=True)
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = _optional_qwen_imports()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=tokenizer_revision,
        trust_remote_code=False,
        local_files_only=not allow_download,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    codebook = validate_codebook(tokenizer, configured_codes)

    bnb_config = None
    if precision == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quantization["bnb_4bit_quant_type"]),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=bool(
                quantization["bnb_4bit_use_double_quant"]
            ),
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        trust_remote_code=False,
        local_files_only=not allow_download,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map={"": device},
    )
    model.eval()
    source_root = Path(__file__).resolve().parents[2]
    payload = {
        "artifact_version": 1,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "precision": precision,
        "max_length": max_length,
        "batch_size": batch_size,
        "seed": seed,
        "deterministic_torch": True,
        "extractor_code_sha256": assessment_extraction_code_sha256(),
        "reproducibility_code_sha256": sha256_file(
            source_root / "src" / "utils" / "reproducibility.py"
        ),
        "quantization": dict(quantization),
        "feature_type": "restricted_answer_probabilities",
        "question_version": QUESTION_VERSION,
        "questions_sha256": QUESTIONS_SHA256,
        "assessment_query_sha256": ASSESSMENT_QUERY_SHA256,
        "answer_codes": codebook["answer_codes"],
        "answer_token_ids": codebook["answer_token_ids"],
        "codebook_sha256": codebook["codebook_sha256"],
    }
    return LoadedAssessmentExtractor(
        model=model,
        tokenizer=tokenizer,
        answer_token_ids=tuple(int(value) for value in codebook["answer_token_ids"]),
        batch_size=batch_size,
        max_length=max_length,
        feature_payload=payload,
        feature_signature=sha256_json(payload),
        codebook=dict(codebook),
    )


@torch.inference_mode()
def extract_assessment_probabilities(
    loaded: LoadedAssessmentExtractor,
    records: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return restricted `(rows, questions, A..E)` logits and probabilities."""

    logits = np.empty((len(records), len(QUESTIONS), 5), dtype=np.float32)
    probabilities = np.empty_like(logits)
    device = next(loaded.model.parameters()).device
    allowed = torch.tensor(
        loaded.answer_token_ids, dtype=torch.long, device=device
    )
    for question_index, question in enumerate(QUESTIONS):
        for start in range(0, len(records), loaded.batch_size):
            stop = min(start + loaded.batch_size, len(records))
            rendered = [
                render_assessment_prompt(loaded.tokenizer, record, question)
                for record in records[start:stop]
            ]
            encoded = loaded.tokenizer(
                rendered,
                add_special_tokens=False,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            lengths = encoded["attention_mask"].sum(dim=1)
            if int(lengths.max().item()) > loaded.max_length:
                offending = int(torch.argmax(lengths).item()) + start
                raise ValueError(
                    f"assessment query for {records[offending].id} exceeds "
                    f"max_length={loaded.max_length}; truncation is forbidden"
                )
            outputs = loaded.model(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
                use_cache=False,
                logits_to_keep=1,
            )
            restricted = outputs.logits[:, -1, :].float().index_select(1, allowed)
            batch_probabilities = torch.softmax(restricted, dim=1)
            logits[start:stop, question_index, :] = restricted.cpu().numpy()
            probabilities[start:stop, question_index, :] = (
                batch_probabilities.cpu().numpy()
            )
    if not np.isfinite(probabilities).all() or not np.isfinite(logits).all():
        raise RuntimeError("Qwen returned non-finite assessment features")
    return logits, probabilities


__all__ = [
    "LoadedAssessmentExtractor",
    "extract_assessment_probabilities",
    "load_assessment_extractor",
]
