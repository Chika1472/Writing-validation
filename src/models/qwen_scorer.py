"""Qwen backbone wrapper for deterministic, non-generative essay scoring."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .ordinal_heads import TRAITS, TraitScoreHead
from .contracts import SCORER_ARCHITECTURE_VERSION

try:
    from transformers import AutoModel
except ImportError as error:  # Keep torch-only utilities importable without the Qwen extra.
    AutoModel = None  # type: ignore[assignment]
    _TRANSFORMERS_IMPORT_ERROR: ImportError | None = error
else:
    _TRANSFORMERS_IMPORT_ERROR = None


def last_non_padding_hidden(last_hidden_state: Tensor, attention_mask: Tensor | None) -> Tensor:
    """Select the final unmasked token for both left- and right-padded batches."""

    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must have shape (batch, sequence, hidden).")
    batch_size, sequence_length, _ = last_hidden_state.shape
    if attention_mask is None:
        return last_hidden_state[:, -1, :]
    if attention_mask.shape != (batch_size, sequence_length):
        raise ValueError("attention_mask shape must match the first two hidden-state dimensions.")
    mask = attention_mask.to(device=last_hidden_state.device, dtype=torch.bool)
    if not torch.all(mask.any(dim=1)):
        raise ValueError("Every sample must contain at least one non-padding token.")
    positions = torch.arange(sequence_length, device=last_hidden_state.device).expand(batch_size, -1)
    last_positions = positions.masked_fill(~mask, -1).max(dim=1).values
    batch_positions = torch.arange(batch_size, device=last_hidden_state.device)
    return last_hidden_state[batch_positions, last_positions]


def _hidden_size(config: Any) -> int:
    value = getattr(config, "hidden_size", None)
    if value is None and getattr(config, "text_config", None) is not None:
        value = getattr(config.text_config, "hidden_size", None)
    if not isinstance(value, int) or value <= 0:
        raise ValueError("Could not determine a positive hidden_size from the Qwen config.")
    return value


class Qwen3ForEssayScoring(nn.Module):
    """Last-token Qwen representation, shared projection, and three independent heads."""

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        backbone: nn.Module | None = None,
        projection_size: int = 512,
        dropout: float = 0.0,
        blend_weight: float | Mapping[str, float] = 0.5,
        **backbone_kwargs: Any,
    ) -> None:
        super().__init__()
        if backbone is None and AutoModel is None:
            raise ImportError(
                "Qwen3ForEssayScoring requires the optional 'qwen' dependencies. "
                "Install them with `pip install -e .[qwen]`."
            ) from _TRANSFORMERS_IMPORT_ERROR
        if projection_size <= 0:
            raise ValueError("projection_size must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if backbone is None:
            if not model_name_or_path:
                raise ValueError("model_name_or_path is required when backbone is not provided.")
            backbone = AutoModel.from_pretrained(model_name_or_path, **backbone_kwargs)
        elif backbone_kwargs:
            raise ValueError("backbone_kwargs cannot be used with an already constructed backbone.")

        self.backbone = backbone
        self.config = backbone.config
        backbone_hidden_size = _hidden_size(self.config)
        self.shared_projection = nn.Sequential(
            nn.Linear(backbone_hidden_size, projection_size),
            nn.GELU(),
            nn.LayerNorm(projection_size),
            nn.Dropout(dropout),
        )
        if isinstance(blend_weight, Mapping):
            if set(blend_weight) != set(TRAITS):
                raise ValueError(f"blend_weight mapping keys must be exactly {TRAITS}")
            blend_weights = {trait: float(blend_weight[trait]) for trait in TRAITS}
        else:
            blend_weights = {trait: float(blend_weight) for trait in TRAITS}
        self.trait_heads = nn.ModuleDict(
            {
                trait: TraitScoreHead(
                    projection_size,
                    trait,
                    blend_weight=blend_weights[trait],
                )
                for trait in TRAITS
            }
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        **backbone_kwargs: Any,
    ) -> dict[str, Any]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **backbone_kwargs,
        )
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            raise TypeError("The backbone output must expose last_hidden_state.")
        pooled_hidden = last_non_padding_hidden(last_hidden_state, attention_mask)
        shared_hidden = self.shared_projection(pooled_hidden)

        trait_outputs = {trait: self.trait_heads[trait](shared_hidden) for trait in TRAITS}
        return {
            "scores": torch.stack([trait_outputs[trait]["score"] for trait in TRAITS], dim=-1),
            "direct_scores": torch.stack(
                [trait_outputs[trait]["direct_score"] for trait in TRAITS], dim=-1
            ),
            "ordinal_scores": torch.stack(
                [trait_outputs[trait]["ordinal_score"] for trait in TRAITS], dim=-1
            ),
            "ordinal_logits": {
                trait: trait_outputs[trait]["ordinal_logits"] for trait in TRAITS
            },
            "pooled_hidden": pooled_hidden,
            "shared_hidden": shared_hidden,
        }
