from __future__ import annotations

import pytest

from src.train.dataset import PromptGroupBatchSampler, PromptPairBatchSampler


def test_prompt_pair_sampler_yields_only_real_same_prompt_pairs() -> None:
    prompts = ["Q1", "Q1", "Q1", "Q2", "Q2"]
    targets = [
        (1.0, 1.0, 1.0),
        (3.0, 3.0, 3.0),
        (5.0, 5.0, 5.0),
        (2.0, 2.0, 2.0),
        (4.0, 4.0, 4.0),
    ]
    sampler = PromptPairBatchSampler(prompts, targets, seed=42)

    for left, right in sampler:
        assert prompts[left] == prompts[right]
        assert targets[left] != targets[right]


def test_prompt_pair_sampler_rejects_prompt_without_rank_signal() -> None:
    with pytest.raises(ValueError, match="no eligible pair"):
        PromptPairBatchSampler(
            ["Q1", "Q1"],
            [(3.0, 3.0, 3.0), (3.0, 3.0, 3.0)],
            seed=42,
        )


def test_prompt_group_sampler_keeps_batches_within_prompt_and_covers_all_rows() -> None:
    prompts = ["Q1", "Q1", "Q1", "Q1", "Q1", "Q2", "Q2", "Q3"]
    sampler = PromptGroupBatchSampler(prompts, batch_size=3, seed=42)
    batches = list(sampler)
    covered = set()
    for batch in batches:
        assert 1 <= len(batch) <= 3
        assert len({prompts[index] for index in batch}) == 1
        covered.update(batch)
    assert covered == set(range(len(prompts)))
    assert any(len(batch) >= 2 for batch in batches)
