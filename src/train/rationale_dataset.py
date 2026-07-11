from __future__ import annotations

import hashlib
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from src.data.schema import TRAITS, EssayInput, EssayRecord, ensure_essay_input
from src.rationale.evidence import build_evidence_ledger
from src.rationale.parsing import assess_grounding, serialize_rationales, validate_rationales
from src.rationale.prompting import build_rationale_messages


@dataclass(frozen=True)
class TokenizedRationaleExample:
    record_id: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


class RationaleSFTDataset(Dataset[dict[str, Any]]):
    """Causal-LM examples whose loss is restricted to rationale JSON tokens."""

    def __init__(
        self,
        records_by_id: Mapping[str, EssayInput | EssayRecord],
        silver_rows: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        *,
        max_length: int,
        score_jitter: float = 0.0,
        score_jitter_copies: int = 0,
        jitter_seed: int = 42,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if score_jitter < 0.0 or score_jitter > 0.25:
            raise ValueError("score_jitter must be in [0, 0.25]")
        if isinstance(score_jitter_copies, bool) or score_jitter_copies < 0:
            raise ValueError("score_jitter_copies must be a nonnegative integer")
        if score_jitter == 0.0 and score_jitter_copies != 0:
            raise ValueError("score_jitter_copies requires a positive score_jitter")
        self.items: list[TokenizedRationaleExample] = []
        seen: set[str] = set()
        for row in silver_rows:
            record_id = str(row.get("id", ""))
            if not record_id or record_id in seen:
                raise ValueError(f"invalid or duplicate silver id: {record_id!r}")
            if record_id not in records_by_id:
                raise ValueError(f"silver id is absent from essay inputs: {record_id}")
            seen.add(record_id)
            record = ensure_essay_input(records_by_id[record_id])
            if str(row.get("prompt_num")) != record.prompt_num:
                raise ValueError(f"silver prompt mismatch for {record_id}")
            scores = row.get("conditioned_scores")
            rationales = validate_rationales(row.get("rationales"))
            if not isinstance(scores, Mapping) or set(scores) != set(TRAITS):
                raise ValueError(f"silver conditioned_scores schema is invalid for {record_id}")
            canonical_scores: dict[str, float] = {}
            for trait in TRAITS:
                value = scores[trait]
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError(f"silver {trait} score must be numeric for {record_id}")
                score = float(value)
                if not 1.0 <= score <= 5.0:
                    raise ValueError(f"silver {trait} score is out of range for {record_id}")
                canonical_scores[trait] = score
            ledger = build_evidence_ledger(record)
            if row.get("evidence") != ledger.to_dict():
                raise ValueError(f"silver evidence ledger mismatch for {record_id}")
            grounding = assess_grounding(rationales, essay=record.essay, ledger=ledger)
            if not grounding.accepted:
                raise ValueError(
                    f"silver rationale is not grounded for {record_id}: {grounding.reasons}"
                )
            answer = serialize_rationales(rationales)
            variants: list[tuple[str, dict[str, float]]] = [("", canonical_scores)]
            for copy_index in range(score_jitter_copies):
                jittered = {}
                for trait in TRAITS:
                    digest = hashlib.blake2b(
                        f"{jitter_seed}\0{record_id}\0{copy_index}\0{trait}".encode("utf-8"),
                        digest_size=8,
                    ).digest()
                    unit = int.from_bytes(digest, "big") / float(2**64 - 1)
                    offset = (2.0 * unit - 1.0) * score_jitter
                    jittered[trait] = min(5.0, max(1.0, canonical_scores[trait] + offset))
                variants.append((f"#score_jitter_{copy_index + 1}", jittered))
            for suffix, variant_scores in variants:
                prompt_messages = build_rationale_messages(record, variant_scores, ledger)
                full_messages = [
                    *prompt_messages,
                    {"role": "assistant", "content": answer},
                ]
                prompt_ids = list(
                    tokenizer.apply_chat_template(
                        prompt_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
                full_ids = list(
                    tokenizer.apply_chat_template(
                        full_messages,
                        tokenize=True,
                        add_generation_prompt=False,
                        enable_thinking=False,
                    )
                )
                if full_ids[: len(prompt_ids)] != prompt_ids:
                    raise RuntimeError(
                        "assistant training sequence does not preserve the generation-prompt prefix"
                    )
                if len(full_ids) > max_length:
                    raise ValueError(
                        f"{record_id}{suffix} rationale sequence length {len(full_ids)} exceeds "
                        f"max_length={max_length}; truncation is forbidden"
                    )
                labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
                if not any(label != -100 for label in labels):
                    raise RuntimeError(
                        f"silver example has no assistant target tokens: {record_id}{suffix}"
                    )
                self.items.append(
                    TokenizedRationaleExample(
                        record_id=record_id + suffix,
                        input_ids=full_ids,
                        attention_mask=[1] * len(full_ids),
                        labels=labels,
                    )
                )
        if not self.items:
            raise ValueError("rationale SFT dataset is empty")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        return {
            "id": item.record_id,
            "input_ids": item.input_ids,
            "attention_mask": item.attention_mask,
            "labels": item.labels,
        }


class RationaleSFTCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        padded = self.tokenizer.pad(
            [
                {
                    "input_ids": example["input_ids"],
                    "attention_mask": example["attention_mask"],
                }
                for example in examples
            ],
            padding=True,
            return_tensors="pt",
        )
        sequence_length = int(padded["input_ids"].shape[1])
        labels = []
        for example in examples:
            padding = sequence_length - len(example["labels"])
            if self.tokenizer.padding_side == "left":
                labels.append([-100] * padding + list(example["labels"]))
            else:
                labels.append(list(example["labels"]) + [-100] * padding)
        return {
            "ids": [str(example["id"]) for example in examples],
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }
