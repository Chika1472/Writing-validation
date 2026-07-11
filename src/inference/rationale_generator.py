from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from src.data.schema import EssayInput, EssayRecord, ensure_essay_input
from src.rationale.deterministic import generate_grounded_rationales
from src.rationale.evidence import build_evidence_ledger
from src.rationale.parsing import assess_grounding, parse_rationales
from src.rationale.prompting import RATIONALE_PROMPT_CONTRACT, build_rationale_messages
from src.utils.hashing import sha256_file, sha256_json, sha256_text


@dataclass(frozen=True)
class LoadedRationaleGenerator:
    model: Any
    tokenizer: Any
    checkpoint_dir: Path
    model_id: str
    model_revision: str
    precision: str
    max_input_length: int
    max_new_tokens: int
    generator_signature: str


@dataclass(frozen=True)
class RationaleGenerationResult:
    rationales: dict[str, str]
    fallback_used: bool
    attempts: tuple[dict[str, Any], ...]
    evidence: dict[str, Any]


def _imports() -> tuple[Any, ...]:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise ImportError("Install the pinned qwen dependencies with `pip install -e .[qwen]`.") from error
    return AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PeftModel


def resolve_rationale_checkpoint(path: str | Path) -> Path:
    supplied = Path(path).resolve()
    if supplied.is_file():
        pointer = json.loads(supplied.read_text(encoding="utf-8"))
        relative = pointer.get("checkpoint")
        if not isinstance(relative, str) or not relative:
            raise ValueError("rationale checkpoint pointer has no checkpoint field")
        candidate = (supplied.parent / relative).resolve()
        try:
            candidate.relative_to(supplied.parent.resolve())
        except ValueError as error:
            raise ValueError("rationale pointer must stay within its run directory") from error
        supplied = candidate
    required = (
        supplied / "adapter" / "adapter_config.json",
        supplied / "tokenizer" / "tokenizer_config.json",
        supplied / "checkpoint.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    weights = (
        supplied / "adapter" / "adapter_model.safetensors",
        supplied / "adapter" / "adapter_model.bin",
    )
    if not any(path.is_file() for path in weights):
        missing.append(f"one of {[str(path) for path in weights]}")
    if missing:
        raise FileNotFoundError(f"incomplete rationale checkpoint; missing={missing}")
    return supplied


def rationale_checkpoint_files(checkpoint: str | Path) -> tuple[Path, ...]:
    resolved = resolve_rationale_checkpoint(checkpoint)
    files = [resolved / "checkpoint.json"]
    files.extend(path for path in (resolved / "adapter").rglob("*") if path.is_file())
    files.extend(path for path in (resolved / "tokenizer").rglob("*") if path.is_file())
    return tuple(sorted(files))


def _checkpoint_signature(checkpoint: Path, *, precision: str) -> str:
    files = rationale_checkpoint_files(checkpoint)
    hashes = {
        path.relative_to(checkpoint).as_posix(): sha256_file(path)
        for path in sorted(files)
    }
    return sha256_json({"files": hashes, "precision": precision})


def load_rationale_generator(
    checkpoint: str | Path,
    *,
    precision: str = "4bit",
    allow_download: bool = False,
    device: str = "cuda:0",
) -> LoadedRationaleGenerator:
    checkpoint_dir = resolve_rationale_checkpoint(checkpoint)
    metadata = json.loads((checkpoint_dir / "checkpoint.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("artifact_type") != "rationale_adapter_checkpoint":
        raise ValueError("invalid rationale checkpoint metadata")
    model_id = metadata.get("model_id")
    revision = metadata.get("model_revision")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("rationale checkpoint has no model_id")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("rationale checkpoint has no pinned model revision")
    if metadata.get("rationale_prompt_sha256") != sha256_text(RATIONALE_PROMPT_CONTRACT):
        raise ValueError("rationale train/inference prompt contract mismatch")
    if precision not in {"4bit", "bf16"}:
        raise ValueError("rationale precision must be '4bit' or 'bf16'")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-14B rationale inference")
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PeftModel = _imports()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir / "tokenizer",
        trust_remote_code=False,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    bnb = None
    if precision == "4bit":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
        local_files_only=not allow_download,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
        device_map={"": device},
    )
    model = PeftModel.from_pretrained(
        base,
        checkpoint_dir / "adapter",
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()
    max_input_length = int(metadata.get("max_input_length", 0))
    max_new_tokens = int(metadata.get("max_new_tokens", 0))
    if max_input_length <= 0 or max_new_tokens <= 0:
        raise ValueError("rationale checkpoint has invalid generation length limits")
    return LoadedRationaleGenerator(
        model=model,
        tokenizer=tokenizer,
        checkpoint_dir=checkpoint_dir,
        model_id=model_id,
        model_revision=revision,
        precision=precision,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
        generator_signature=_checkpoint_signature(checkpoint_dir, precision=precision),
    )


@torch.inference_mode()
def generate_rationale_for_record(
    loaded: LoadedRationaleGenerator,
    value: EssayInput | EssayRecord | dict[str, Any],
    scores: Mapping[str, float],
    *,
    max_attempts: int = 2,
) -> RationaleGenerationResult:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    record = ensure_essay_input(value)
    ledger = build_evidence_ledger(record)
    base_messages = build_rationale_messages(record, scores, ledger)
    attempts = []
    for attempt in range(1, max_attempts + 1):
        messages = [dict(message) for message in base_messages]
        if attempt > 1:
            messages[-1]["content"] += (
                "\n\n[재시도 지시] 정확한 세 문자열 JSON만 출력하고 exact evidence를 인용한다."
            )
        try:
            prompt = loaded.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = loaded.tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=False,
                return_tensors="pt",
            )
            input_length = int(encoded["input_ids"].shape[1])
            if input_length > loaded.max_input_length:
                raise ValueError(
                    f"rationale input length {input_length} exceeds {loaded.max_input_length}"
                )
            device = next(loaded.model.parameters()).device
            input_ids = encoded["input_ids"].to(device)
            generated = loaded.model.generate(
                input_ids=input_ids,
                attention_mask=encoded["attention_mask"].to(device),
                do_sample=False,
                max_new_tokens=loaded.max_new_tokens,
                pad_token_id=loaded.tokenizer.pad_token_id,
                eos_token_id=loaded.tokenizer.eos_token_id,
                use_cache=True,
            )
            raw = loaded.tokenizer.decode(
                generated[0, input_ids.shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            rationales = parse_rationales(raw)
            grounding = assess_grounding(rationales, essay=record.essay, ledger=ledger)
            attempts.append(
                {
                    "attempt": attempt,
                    "raw_output": raw,
                    "grounding_reasons": list(grounding.reasons),
                }
            )
            if grounding.accepted:
                return RationaleGenerationResult(
                    rationales=rationales,
                    fallback_used=False,
                    attempts=tuple(attempts),
                    evidence=ledger.to_dict(),
                )
        except Exception as error:
            attempts.append(
                {"attempt": attempt, "error": f"{type(error).__name__}: {error}"}
            )
    fallback = generate_grounded_rationales(ledger, scores)
    fallback_grounding = assess_grounding(fallback, essay=record.essay, ledger=ledger)
    if not fallback_grounding.accepted:
        raise RuntimeError(
            f"deterministic rationale fallback failed grounding: "
            f"{fallback_grounding.reasons}"
        )
    return RationaleGenerationResult(
        rationales=fallback,
        fallback_used=True,
        attempts=tuple(attempts),
        evidence=ledger.to_dict(),
    )
