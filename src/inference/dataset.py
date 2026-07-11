from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset

from src.data.schema import EssayInput, EssayRecord, ensure_essay_input
from src.train.prompting import render_scoring_prompt


@dataclass(frozen=True)
class TokenizedScoringInput:
    record_id: str
    prompt_num: str
    input_ids: list[int]
    attention_mask: list[int]


class EssayInferenceDataset(Dataset[dict[str, Any]]):
    """Tokenize labeled or unlabeled essays without ever consulting score labels."""

    def __init__(
        self,
        records: Sequence[EssayInput | EssayRecord | Mapping[str, Any]],
        tokenizer: Any,
        *,
        max_length: int,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.items: list[TokenizedScoringInput] = []
        for value in records:
            record = ensure_essay_input(value)
            prompt = render_scoring_prompt(tokenizer, record)
            encoded = tokenizer(prompt, add_special_tokens=False, truncation=False)
            input_ids = list(encoded["input_ids"])
            attention_mask = list(encoded.get("attention_mask", [1] * len(input_ids)))
            if len(input_ids) > max_length:
                raise ValueError(
                    f"{record.id} token length {len(input_ids)} exceeds max_length={max_length}; "
                    "increase max_length instead of truncating the essay"
                )
            self.items.append(
                TokenizedScoringInput(
                    record_id=record.id,
                    prompt_num=record.prompt_num,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        return {
            "id": item.record_id,
            "prompt_num": item.prompt_num,
            "input_ids": item.input_ids,
            "attention_mask": item.attention_mask,
        }


class EssayInferenceCollator:
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
        return {
            "ids": [str(example["id"]) for example in examples],
            "prompt_nums": [str(example["prompt_num"]) for example in examples],
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
        }
