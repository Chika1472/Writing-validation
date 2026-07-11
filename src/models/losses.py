"""Losses aligned with RMSE and within-prompt ranking objectives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .ordinal_heads import TRAITS, native_score_grid


def _reduce(values: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'.")


def score_mse_loss(predictions: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
    """MSE in FP32, matching the challenge's primary absolute-error metric."""

    if predictions.shape != targets.shape:
        raise ValueError(f"Prediction and target shapes differ: {predictions.shape} vs {targets.shape}.")
    return F.mse_loss(predictions.float(), targets.float(), reduction=reduction)


def gaussian_soft_targets(targets: Tensor, grid: Tensor, sigma: float) -> Tensor:
    """Turn scalar labels into normalized Gaussian distributions on a score grid."""

    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    if grid.ndim != 1 or grid.numel() < 2:
        raise ValueError("grid must be a one-dimensional tensor with at least two values.")
    distances = targets.float().unsqueeze(-1) - grid.to(targets.device, torch.float32)
    logits = -(distances.square()) / (2.0 * sigma * sigma)
    return torch.softmax(logits, dim=-1)


def ordinal_distribution_loss(
    logits: Tensor,
    targets: Tensor,
    grid: Tensor,
    *,
    sigma: float,
    reduction: str = "mean",
) -> Tensor:
    """KL divergence from Gaussian soft labels to predicted native-grid probabilities."""

    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"Ordinal logits must have target shape plus one grid dimension; got {logits.shape} "
            f"and {targets.shape}."
        )
    if logits.shape[-1] != grid.numel():
        raise ValueError(f"Logit width {logits.shape[-1]} does not match grid size {grid.numel()}.")
    soft_targets = gaussian_soft_targets(targets, grid, sigma)
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    per_item = F.kl_div(log_probabilities, soft_targets, reduction="none").sum(dim=-1)
    return _reduce(per_item, reduction)


def pairwise_relative_loss(
    left_scores: Tensor,
    right_scores: Tensor,
    left_targets: Tensor,
    right_targets: Tensor,
    *,
    tie_threshold: float = 0.0,
    temperature: float = 1.0,
    order_weight: float = 0.5,
    huber_beta: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """Combine RankNet-style ordering and score-difference Huber losses.

    Exact ties and pairs whose gold difference is at most ``tie_threshold`` are
    excluded. A batch with no valid pairs returns a differentiable zero.
    """

    shapes = {left_scores.shape, right_scores.shape, left_targets.shape, right_targets.shape}
    if len(shapes) != 1:
        raise ValueError("All pairwise score and target tensors must have identical shapes.")
    if tie_threshold < 0.0:
        raise ValueError("tie_threshold must be non-negative.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if not 0.0 <= order_weight <= 1.0:
        raise ValueError("order_weight must be between 0 and 1.")
    if huber_beta <= 0.0:
        raise ValueError("huber_beta must be positive.")

    predicted_delta = left_scores.float() - right_scores.float()
    target_delta = left_targets.float() - right_targets.float()
    valid = target_delta.abs() > tie_threshold
    if not torch.any(valid):
        return predicted_delta.sum() * 0.0

    predicted_delta = predicted_delta[valid]
    target_delta = target_delta[valid]
    order = F.softplus(-target_delta.sign() * predicted_delta / temperature)
    delta = F.smooth_l1_loss(
        predicted_delta,
        target_delta,
        beta=huber_beta,
        reduction="none",
    )
    return _reduce(order_weight * order + (1.0 - order_weight) * delta, reduction)


def within_group_soft_rank_loss(
    scores: Tensor,
    targets: Tensor,
    group_ids: Sequence[str],
    *,
    temperature: float = 0.25,
    tie_threshold: float = 0.0,
    reduction: str = "mean",
) -> Tensor:
    """Match differentiable percentile ranks inside independent prompt groups.

    For each item, the predicted percentile is its mean sigmoid win probability
    against the other items from the same prompt. Gold percentiles use exact
    wins/losses and assign ``0.5`` to ties. Groups with fewer than two items are
    ignored; if every group is a singleton, a differentiable zero is returned.
    """

    if scores.ndim != 1 or targets.ndim != 1 or scores.shape != targets.shape:
        raise ValueError("scores and targets must be same-shaped one-dimensional tensors")
    if len(group_ids) != scores.shape[0]:
        raise ValueError("group_ids length must match scores")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if tie_threshold < 0.0:
        raise ValueError("tie_threshold must be non-negative")

    grouped: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        grouped.setdefault(str(group_id), []).append(index)

    group_losses: list[Tensor] = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long, device=scores.device)
        group_scores = scores.float()[index_tensor]
        group_targets = targets.float()[index_tensor]
        predicted_delta = group_scores[:, None] - group_scores[None, :]
        target_delta = group_targets[:, None] - group_targets[None, :]
        off_diagonal = ~torch.eye(
            len(indices), dtype=torch.bool, device=scores.device
        )

        predicted_percentile = (
            torch.sigmoid(predicted_delta / temperature) * off_diagonal
        ).sum(dim=1) / (len(indices) - 1)
        target_wins = torch.where(
            target_delta > tie_threshold,
            torch.ones_like(target_delta),
            torch.where(
                target_delta < -tie_threshold,
                torch.zeros_like(target_delta),
                torch.full_like(target_delta, 0.5),
            ),
        )
        target_percentile = (target_wins * off_diagonal).sum(dim=1) / (
            len(indices) - 1
        )
        group_losses.append(
            F.smooth_l1_loss(
                predicted_percentile,
                target_percentile,
                beta=0.1,
                reduction="mean",
            )
        )

    if not group_losses:
        return scores.float().sum() * 0.0
    return _reduce(torch.stack(group_losses), reduction)


class EssayScoringLoss(nn.Module):
    """Weighted regression, ordinal, pairwise, and soft-rank objective."""

    def __init__(
        self,
        *,
        mse_weight: float = 1.0,
        ordinal_weight: float = 0.3,
        pairwise_weight: float = 0.2,
        soft_rank_weight: float = 0.0,
        ordinal_sigmas: Mapping[str, float] | None = None,
        tie_threshold: float = 0.0,
        pairwise_temperature: float = 1.0,
        pairwise_order_weight: float = 0.5,
        soft_rank_temperature: float = 0.25,
    ) -> None:
        super().__init__()
        weights = (mse_weight, ordinal_weight, pairwise_weight, soft_rank_weight)
        if any(weight < 0.0 for weight in weights):
            raise ValueError("Loss weights must be non-negative.")
        self.mse_weight = float(mse_weight)
        self.ordinal_weight = float(ordinal_weight)
        self.pairwise_weight = float(pairwise_weight)
        self.soft_rank_weight = float(soft_rank_weight)
        self.ordinal_sigmas = dict(
            ordinal_sigmas
            or {"content": 0.15, "organization": 0.25, "expression": 0.25}
        )
        if set(self.ordinal_sigmas) != set(TRAITS):
            raise ValueError(f"ordinal_sigmas must contain exactly {TRAITS}.")
        self.tie_threshold = float(tie_threshold)
        self.pairwise_temperature = float(pairwise_temperature)
        self.pairwise_order_weight = float(pairwise_order_weight)
        if soft_rank_temperature <= 0.0:
            raise ValueError("soft_rank_temperature must be positive.")
        self.soft_rank_temperature = float(soft_rank_temperature)

    def forward(
        self,
        scores: Tensor,
        targets: Tensor,
        *,
        ordinal_logits: Mapping[str, Tensor] | None = None,
        pair_indices: Tensor | None = None,
        group_ids: Sequence[str] | None = None,
    ) -> dict[str, Tensor]:
        if scores.shape != targets.shape or scores.ndim != 2 or scores.shape[1] != len(TRAITS):
            raise ValueError(f"scores and targets must both have shape (batch, {len(TRAITS)}).")

        mse = score_mse_loss(scores, targets)
        zero = scores.float().sum() * 0.0
        ordinal = zero
        if ordinal_logits is not None:
            if set(ordinal_logits) != set(TRAITS):
                raise ValueError(f"ordinal_logits must contain exactly {TRAITS}.")
            ordinal_terms = []
            for trait_index, trait in enumerate(TRAITS):
                grid = native_score_grid(trait, device=scores.device)
                ordinal_terms.append(
                    ordinal_distribution_loss(
                        ordinal_logits[trait],
                        targets[:, trait_index],
                        grid,
                        sigma=self.ordinal_sigmas[trait],
                    )
                )
            ordinal = torch.stack(ordinal_terms).mean()

        pairwise = zero
        if pair_indices is not None:
            if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
                raise ValueError("pair_indices must have shape (number_of_pairs, 2).")
            pair_indices = pair_indices.to(device=scores.device, dtype=torch.long)
            if pair_indices.numel() and (
                pair_indices.min() < 0 or pair_indices.max() >= scores.shape[0]
            ):
                raise ValueError("pair_indices contains an out-of-range batch index.")
            pair_terms = []
            for trait_index in range(len(TRAITS)):
                left, right = pair_indices[:, 0], pair_indices[:, 1]
                pair_terms.append(
                    pairwise_relative_loss(
                        scores[left, trait_index],
                        scores[right, trait_index],
                        targets[left, trait_index],
                        targets[right, trait_index],
                        tie_threshold=self.tie_threshold,
                        temperature=self.pairwise_temperature,
                        order_weight=self.pairwise_order_weight,
                    )
                )
            pairwise = torch.stack(pair_terms).mean()

        soft_rank = zero
        if self.soft_rank_weight > 0.0:
            if group_ids is None:
                raise ValueError("group_ids are required when soft_rank_weight is positive.")
            rank_terms = [
                within_group_soft_rank_loss(
                    scores[:, trait_index],
                    targets[:, trait_index],
                    group_ids,
                    temperature=self.soft_rank_temperature,
                    tie_threshold=self.tie_threshold,
                )
                for trait_index in range(len(TRAITS))
            ]
            soft_rank = torch.stack(rank_terms).mean()

        total = (
            self.mse_weight * mse
            + self.ordinal_weight * ordinal
            + self.pairwise_weight * pairwise
            + self.soft_rank_weight * soft_rank
        )
        return {
            "loss": total,
            "mse": mse,
            "ordinal": ordinal,
            "pairwise": pairwise,
            "soft_rank": soft_rank,
        }
