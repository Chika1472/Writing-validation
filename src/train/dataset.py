from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from src.data.schema import EssayRecord, ensure_record
from src.train.prompting import render_scoring_prompt


@dataclass(frozen=True)
class TokenizedEssay:
    record_id: str
    prompt_num: str
    input_ids: list[int]
    attention_mask: list[int]
    targets: tuple[float, float, float]


class EssayScoringDataset(Dataset[dict[str, Any]]):
    """Tokenize full essays and fail instead of silently truncating."""

    def __init__(
        self,
        records: Sequence[EssayRecord | Mapping[str, Any]],
        tokenizer: Any,
        *,
        max_length: int,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.items: list[TokenizedEssay] = []
        for value in records:
            record = ensure_record(value)
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
                TokenizedEssay(
                    record_id=record.id,
                    prompt_num=record.prompt_num,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    targets=record.score.trait_values,
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
            "targets": item.targets,
        }


class EssayBatchCollator:
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
            "targets": torch.tensor(
                [example["targets"] for example in examples], dtype=torch.float32
            ),
        }


class PromptPairBatchSampler(Sampler[list[int]]):
    """Yield deterministic two-item, same-prompt batches with a real score gap.

    Pairwise loss must not be enabled with ordinary random micro-batches: a
    batch can otherwise contain no same-prompt comparison at all. Each epoch
    uses a different deterministic selection of eligible pairs while keeping a
    stable number of optimizer micro-batches.
    """

    def __init__(
        self,
        prompt_nums: Sequence[str],
        targets: Sequence[Sequence[float]],
        *,
        seed: int,
        minimum_gap: float = 0.0,
    ) -> None:
        if len(prompt_nums) != len(targets):
            raise ValueError("prompt_nums and targets must have the same length")
        if minimum_gap < 0:
            raise ValueError("minimum_gap must be non-negative")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, prompt_num in enumerate(prompt_nums):
            grouped[str(prompt_num)].append(index)
        if not grouped:
            raise ValueError("pairwise batching requires at least one training record")

        target_rows = [tuple(float(value) for value in row) for row in targets]
        if any(len(row) != 3 for row in target_rows):
            raise ValueError("each target row must contain the three trait scores")
        eligible: dict[str, list[tuple[int, int]]] = {}
        for prompt_num, indices in grouped.items():
            pairs = [
                (left, right)
                for left_position, left in enumerate(indices)
                for right in indices[left_position + 1 :]
                if any(
                    abs(target_rows[left][trait] - target_rows[right][trait]) > minimum_gap
                    for trait in range(3)
                )
            ]
            if not pairs:
                raise ValueError(
                    "pairwise batching requires at least two differently scored essays "
                    f"for every prompt; no eligible pair for prompt {prompt_num!r}"
                )
            eligible[prompt_num] = pairs

        self._grouped = dict(grouped)
        self._eligible = eligible
        self._seed = int(seed)
        self._epoch = 0
        self._length = sum(math.ceil(len(indices) / 2) for indices in grouped.values())

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        rng = random.Random(self._seed + self._epoch)
        batches: list[list[int]] = []
        for prompt_num, indices in self._grouped.items():
            target_count = math.ceil(len(indices) / 2)
            candidates = list(self._eligible[prompt_num])
            rng.shuffle(candidates)
            uncovered = set(indices)
            selected: list[tuple[int, int]] = []
            while len(selected) < target_count:
                preferred = [
                    pair
                    for pair in candidates
                    if pair[0] in uncovered or pair[1] in uncovered
                ]
                pool = preferred or candidates or self._eligible[prompt_num]
                pair = rng.choice(pool)
                selected.append(pair)
                uncovered.discard(pair[0])
                uncovered.discard(pair[1])
                if pair in candidates:
                    candidates.remove(pair)
            batches.extend([[left, right] for left, right in selected])
        rng.shuffle(batches)
        yield from batches


class PromptGroupBatchSampler(Sampler[list[int]]):
    """Yield same-prompt micro-batches for a genuine multi-item soft-rank loss.

    Every example is visited once.  When a prompt has an odd one-item tail, one
    deterministic companion is repeated so the tail still supplies a rank
    comparison without increasing the configured micro-batch size.
    """

    def __init__(
        self,
        prompt_nums: Sequence[str],
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size < 2:
            raise ValueError("soft-rank batching requires batch_size >= 2")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, prompt_num in enumerate(prompt_nums):
            grouped[str(prompt_num)].append(index)
        if not grouped or not any(len(indices) >= 2 for indices in grouped.values()):
            raise ValueError("soft-rank batching requires a prompt with at least two essays")
        self._grouped = dict(grouped)
        self._batch_size = int(batch_size)
        self._seed = int(seed)
        self._epoch = 0
        self._length = sum(
            math.ceil(len(indices) / self._batch_size)
            for indices in self._grouped.values()
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        rng = random.Random(self._seed + self._epoch)
        batches: list[list[int]] = []
        for indices in self._grouped.values():
            ordered = list(indices)
            rng.shuffle(ordered)
            prompt_batches = [
                ordered[start : start + self._batch_size]
                for start in range(0, len(ordered), self._batch_size)
            ]
            if len(indices) >= 2 and len(prompt_batches[-1]) == 1:
                tail_id = prompt_batches[-1][0]
                companions = [index for index in ordered if index != tail_id]
                prompt_batches[-1].append(rng.choice(companions))
            batches.extend(prompt_batches)
        rng.shuffle(batches)
        yield from batches


def within_prompt_pair_indices(
    prompt_nums: Sequence[str],
    targets: Tensor,
    *,
    minimum_gap: float = 0.0,
) -> Tensor:
    """Build unique within-prompt pairs with any trait gap above the threshold."""

    if targets.ndim != 2 or targets.shape[0] != len(prompt_nums):
        raise ValueError("targets must have shape (batch, traits) and match prompt_nums")
    pairs: list[tuple[int, int]] = []
    detached = targets.detach().float().cpu()
    for left in range(len(prompt_nums)):
        for right in range(left + 1, len(prompt_nums)):
            if prompt_nums[left] != prompt_nums[right]:
                continue
            if torch.any((detached[left] - detached[right]).abs() > minimum_gap):
                pairs.append((left, right))
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long)
