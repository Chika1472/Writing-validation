import importlib.util

import pytest
import torch

from src.models.losses import (
    EssayScoringLoss,
    gaussian_soft_targets,
    ordinal_distribution_loss,
    pairwise_relative_loss,
    score_mse_loss,
    within_group_soft_rank_loss,
)
from src.models.ordinal_heads import (
    BoundedRegressionHead,
    NativeGridOrdinalHead,
    expected_score,
    expected_score_from_logits,
    native_score_grid,
)
from src.models.qwen_scorer import Qwen3ForEssayScoring, last_non_padding_hidden


def test_native_grids_and_expected_scores() -> None:
    content = native_score_grid("content")
    organization = native_score_grid("organization")
    assert content.shape == (41,)
    assert organization.shape == (17,)
    assert content[[0, -1]].tolist() == [1.0, 5.0]
    assert organization[[0, -1]].tolist() == [1.0, 5.0]

    probabilities = torch.zeros(2, 41)
    probabilities[0, 0] = 1.0
    probabilities[1, -1] = 1.0
    assert expected_score(probabilities, content).tolist() == [1.0, 5.0]
    assert expected_score_from_logits(torch.zeros(3, 41), content) == pytest.approx(
        torch.full((3,), 3.0)
    )


def test_direct_and_ordinal_heads_are_bounded_and_differentiable() -> None:
    hidden = torch.randn(8, 6, requires_grad=True)
    direct = BoundedRegressionHead(6)
    ordinal = NativeGridOrdinalHead(6, "expression")
    direct_scores = direct(hidden)
    ordinal_scores = ordinal.score(ordinal(hidden))
    assert torch.all((direct_scores > 1.0) & (direct_scores < 5.0))
    assert torch.all((ordinal_scores >= 1.0) & (ordinal_scores <= 5.0))
    (direct_scores.mean() + ordinal_scores.mean()).backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_mse_and_ordinal_distribution_losses() -> None:
    predictions = torch.tensor([1.0, 3.0, 5.0], requires_grad=True)
    targets = torch.tensor([2.0, 3.0, 4.0])
    assert score_mse_loss(predictions, targets).item() == pytest.approx(2.0 / 3.0)

    grid = native_score_grid("organization")
    labels = torch.tensor([2.0, 4.0])
    soft_targets = gaussian_soft_targets(labels, grid, sigma=0.25)
    perfect_logits = soft_targets.log().detach().requires_grad_()
    loss = ordinal_distribution_loss(perfect_logits, labels, grid, sigma=0.25)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    loss.backward()
    assert perfect_logits.grad is not None


def test_pairwise_loss_rewards_correct_order_and_ignores_ties() -> None:
    targets_left = torch.tensor([5.0, 4.0])
    targets_right = torch.tensor([1.0, 2.0])
    good = pairwise_relative_loss(
        torch.tensor([5.0, 4.0]),
        torch.tensor([1.0, 2.0]),
        targets_left,
        targets_right,
    )
    bad = pairwise_relative_loss(
        torch.tensor([1.0, 2.0]),
        torch.tensor([5.0, 4.0]),
        targets_left,
        targets_right,
    )
    assert good < bad

    score = torch.tensor([2.0], requires_grad=True)
    tie_loss = pairwise_relative_loss(score, torch.tensor([4.0]), torch.tensor([3.0]), torch.tensor([3.0]))
    assert tie_loss.item() == 0.0
    tie_loss.backward()
    assert score.grad is not None


def test_soft_rank_loss_rewards_within_prompt_order_and_handles_singletons() -> None:
    targets = torch.tensor([1.0, 3.0, 5.0, 2.0])
    group_ids = ["p1", "p1", "p1", "p2"]
    good_scores = torch.tensor([1.0, 3.0, 5.0, 4.0], requires_grad=True)
    bad_scores = torch.tensor([5.0, 3.0, 1.0, 4.0])
    good = within_group_soft_rank_loss(good_scores, targets, group_ids)
    bad = within_group_soft_rank_loss(bad_scores, targets, group_ids)
    assert good < bad
    good.backward()
    assert good_scores.grad is not None

    singleton = torch.tensor([3.0], requires_grad=True)
    zero = within_group_soft_rank_loss(singleton, singleton.detach(), ["p1"])
    assert zero.item() == 0.0
    zero.backward()
    assert singleton.grad is not None


def test_combined_loss_uses_all_terms_and_backpropagates() -> None:
    scores = torch.tensor(
        [[2.0, 2.5, 3.0], [4.5, 4.0, 3.5], [3.0, 3.0, 3.0]],
        requires_grad=True,
    )
    targets = torch.tensor([[2.0, 2.25, 2.75], [4.7, 4.25, 3.75], [3.0, 3.0, 3.0]])
    ordinal_logits = {
        "content": torch.randn(3, 41, requires_grad=True),
        "organization": torch.randn(3, 17, requires_grad=True),
        "expression": torch.randn(3, 17, requires_grad=True),
    }
    result = EssayScoringLoss(soft_rank_weight=0.1)(
        scores,
        targets,
        ordinal_logits=ordinal_logits,
        pair_indices=torch.tensor([[0, 1], [1, 2]]),
        group_ids=["p1", "p1", "p1"],
    )
    assert set(result) == {"loss", "mse", "ordinal", "pairwise", "soft_rank"}
    assert all(torch.isfinite(value) for value in result.values())
    result["loss"].backward()
    assert scores.grad is not None
    assert all(logits.grad is not None for logits in ordinal_logits.values())


def test_last_non_padding_pooling_handles_left_and_right_padding() -> None:
    hidden = torch.arange(2 * 4 * 2).reshape(2, 4, 2).float()
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0]])
    pooled = last_non_padding_hidden(hidden, mask)
    assert torch.equal(pooled[0], hidden[0, 1])
    assert torch.equal(pooled[1], hidden[1, 2])


def test_qwen_wrapper_has_clear_optional_dependency_error() -> None:
    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("This assertion is specific to the torch-only test environment.")
    with pytest.raises(ImportError, match="optional 'qwen' dependencies"):
        Qwen3ForEssayScoring("Qwen/Qwen3-14B")
