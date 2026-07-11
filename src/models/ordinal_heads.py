"""Small, task-specific scoring heads for the three essay traits."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


TRAITS = ("content", "organization", "expression")
NATIVE_SCORE_GRIDS: dict[str, tuple[float, ...]] = {
    "content": tuple(round(1.0 + 0.1 * index, 1) for index in range(41)),
    "organization": tuple(1.0 + 0.25 * index for index in range(17)),
    "expression": tuple(1.0 + 0.25 * index for index in range(17)),
}


def native_score_grid(
    trait: str,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return a fresh tensor containing the observed label grid for ``trait``."""

    try:
        values = NATIVE_SCORE_GRIDS[trait]
    except KeyError as error:
        raise ValueError(f"Unknown trait {trait!r}; expected one of {TRAITS}.") from error
    return torch.tensor(values, device=device, dtype=dtype)


def expected_score(probabilities: Tensor, grid: Tensor | Sequence[float]) -> Tensor:
    """Compute a distribution's score expectation along its final dimension."""

    grid_tensor = torch.as_tensor(grid, device=probabilities.device, dtype=torch.float32)
    if probabilities.ndim == 0 or probabilities.shape[-1] != grid_tensor.numel():
        raise ValueError(
            "The probability dimension and score grid must have the same length; "
            f"got {probabilities.shape} and {grid_tensor.numel()}."
        )
    return (probabilities.float() * grid_tensor).sum(dim=-1)


def expected_score_from_logits(logits: Tensor, grid: Tensor | Sequence[float]) -> Tensor:
    """Compute a continuous expected score from native-grid class logits."""

    return expected_score(torch.softmax(logits.float(), dim=-1), grid)


class BoundedRegressionHead(nn.Module):
    """Map a hidden representation to a continuous score in the open interval (1, 5)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden: Tensor) -> Tensor:
        raw_score = self.linear(hidden).squeeze(-1)
        return 1.0 + 4.0 * torch.sigmoid(raw_score.float())


class NativeGridOrdinalHead(nn.Module):
    """Predict a categorical distribution over one trait's native score grid."""

    def __init__(self, hidden_size: int, trait: str) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        grid = native_score_grid(trait)
        self.trait = trait
        self.linear = nn.Linear(hidden_size, grid.numel())
        self.register_buffer("grid", grid, persistent=True)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.linear(hidden)

    def score(self, logits: Tensor) -> Tensor:
        return expected_score_from_logits(logits, self.grid)


class TraitScoreHead(nn.Module):
    """Blend direct regression and native-grid ordinal expectations for one trait."""

    def __init__(self, hidden_size: int, trait: str, blend_weight: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= blend_weight <= 1.0:
            raise ValueError("blend_weight must be between 0 and 1.")
        self.direct_head = BoundedRegressionHead(hidden_size)
        self.ordinal_head = NativeGridOrdinalHead(hidden_size, trait)
        self.blend_weight = float(blend_weight)

    def forward(self, hidden: Tensor) -> dict[str, Tensor]:
        direct_score = self.direct_head(hidden)
        ordinal_logits = self.ordinal_head(hidden)
        ordinal_score = self.ordinal_head.score(ordinal_logits)
        score = self.blend_weight * direct_score + (1.0 - self.blend_weight) * ordinal_score
        return {
            "score": score,
            "direct_score": direct_score,
            "ordinal_score": ordinal_score,
            "ordinal_logits": ordinal_logits,
        }
