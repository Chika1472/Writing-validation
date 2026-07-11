"""Restore a trained Qwen scorer artifact and produce score-only predictions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.schema import EssayInput, EssayRecord
from src.inference.dataset import EssayInferenceCollator, EssayInferenceDataset
from src.models.contracts import SCORER_ARCHITECTURE_VERSION
from src.models.qwen_scorer import Qwen3ForEssayScoring
from src.train.prompting import SCORING_PROMPT_CONTRACT
from src.utils.hashing import sha256_file, sha256_text


@dataclass(frozen=True)
class LoadedScorer:
    model: Qwen3ForEssayScoring
    tokenizer: Any
    checkpoint_dir: Path
    model_id: str
    model_revision: str
    max_length: int
    compute_dtype: torch.dtype
    precision: str


def resolve_checkpoint(path: str | Path) -> Path:
    """Resolve either an epoch directory or a run's best-pointer JSON."""

    supplied = Path(path).resolve()
    if supplied.is_file():
        payload = json.loads(supplied.read_text(encoding="utf-8"))
        relative = payload.get("checkpoint")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"checkpoint pointer has no checkpoint field: {supplied}")
        candidate = (supplied.parent / relative).resolve()
        try:
            candidate.relative_to(supplied.parent.resolve())
        except ValueError as error:
            raise ValueError("checkpoint pointer must remain inside its run directory") from error
        supplied = candidate
    required_files = (
        supplied / "adapter" / "adapter_config.json",
        supplied / "tokenizer" / "tokenizer_config.json",
        supplied / "scoring_heads.pt",
        supplied / "scoring_head_config.json",
        supplied / "checkpoint_provenance.json",
        supplied / "oof.jsonl",
    )
    missing = [str(value) for value in required_files if not value.is_file()]
    adapter_weights = (
        supplied / "adapter" / "adapter_model.safetensors",
        supplied / "adapter" / "adapter_model.bin",
    )
    if not any(value.is_file() for value in adapter_weights):
        missing.append(f"one of {[str(value) for value in adapter_weights]}")
    if missing:
        raise FileNotFoundError(
            f"incomplete scorer checkpoint {supplied}; missing={missing}"
        )
    provenance_path = supplied / "checkpoint_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        not isinstance(provenance, dict)
        or provenance.get("artifact_type") != "qwen_scorer_fold_checkpoint"
        or provenance.get("oof_file") != "oof.jsonl"
        or provenance.get("oof_sha256") != sha256_file(supplied / "oof.jsonl")
    ):
        raise ValueError(f"invalid checkpoint/OOF provenance contract: {provenance_path}")
    return supplied


def checkpoint_artifact_files(checkpoint_dir: str | Path) -> tuple[Path, ...]:
    """List every persisted file that determines restored scorer predictions."""

    resolved = resolve_checkpoint(checkpoint_dir)
    files = [
        resolved / "scoring_heads.pt",
        resolved / "scoring_head_config.json",
        resolved / "checkpoint_provenance.json",
        resolved / "oof.jsonl",
    ]
    files.extend(path for path in (resolved / "adapter").rglob("*") if path.is_file())
    files.extend(path for path in (resolved / "tokenizer").rglob("*") if path.is_file())
    return tuple(sorted(files))


def _run_manifest(checkpoint_dir: Path) -> dict[str, Any]:
    candidates = (checkpoint_dir / "manifest.json", checkpoint_dir.parent / "manifest.json")
    for candidate in candidates:
        if candidate.is_file():
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"manifest must contain a JSON object: {candidate}")
            return value
    return {}


def _optional_inference_imports() -> tuple[Any, ...]:
    try:
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise ImportError(
            "Qwen scorer inference requires `pip install -e .[qwen]`."
        ) from error
    return AutoModel, AutoTokenizer, BitsAndBytesConfig, PeftModel


def load_scorer_checkpoint(
    checkpoint: str | Path,
    *,
    model_id: str | None = None,
    model_revision: str | None = None,
    precision: str | None = None,
    device: str = "cuda:0",
    allow_download: bool = False,
) -> LoadedScorer:
    """Restore base revision, PEFT adapter, tokenizer, and independent score heads."""

    checkpoint_dir = resolve_checkpoint(checkpoint)
    manifest = _run_manifest(checkpoint_dir)
    head_config = json.loads(
        (checkpoint_dir / "scoring_head_config.json").read_text(encoding="utf-8")
    )
    if head_config.get("scorer_architecture_version") != SCORER_ARCHITECTURE_VERSION:
        raise ValueError(
            "checkpoint scorer architecture contract differs from the inference code"
        )
    manifest_model_id = manifest.get("model_id") or head_config.get("model_id")
    manifest_revision = manifest.get("model_revision") or head_config.get(
        "model_revision"
    )
    resolved_model_id = model_id or manifest_model_id
    resolved_revision = model_revision or manifest_revision
    if not isinstance(resolved_model_id, str) or not resolved_model_id:
        raise ValueError("model_id is absent from both arguments and run manifest")
    if not isinstance(resolved_revision, str) or not re.fullmatch(
        r"[0-9a-fA-F]{40}", resolved_revision
    ):
        raise ValueError("a pinned 40-character model commit SHA is required for inference")
    if model_id and manifest_model_id and model_id != manifest_model_id:
        raise ValueError("model_id override does not match the checkpoint manifest")
    if model_revision and manifest_revision and model_revision != manifest_revision:
        raise ValueError("model_revision override does not match the checkpoint manifest")

    expected_prompt_hash = manifest.get("prompt_template_sha256") or head_config.get(
        "prompt_template_sha256"
    )
    current_prompt_hash = sha256_text(SCORING_PROMPT_CONTRACT)
    if expected_prompt_hash != current_prompt_hash:
        raise ValueError(
            "training/inference prompt template hash mismatch; do not score with a changed template"
        )

    scorer_config = manifest.get("config", {}).get("scorer", {})
    model_config = scorer_config.get("model", {})
    quantization_config = scorer_config.get(
        "quantization", head_config.get("quantization", {})
    )
    max_length = int(model_config.get("max_length", head_config.get("max_length", 0)))
    if max_length <= 0:
        raise ValueError("manifest does not contain a positive model.max_length")
    resolved_precision = precision or (
        "4bit" if bool(quantization_config.get("load_in_4bit", True)) else "bf16"
    )
    if resolved_precision not in {"4bit", "bf16"}:
        raise ValueError("precision must be '4bit' or 'bf16'")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")

    AutoModel, AutoTokenizer, BitsAndBytesConfig, PeftModel = _optional_inference_imports()
    compute_dtype = torch.bfloat16
    bnb_config = None
    if resolved_precision == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(
                quantization_config.get("bnb_4bit_quant_type", "nf4")
            ),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(
                quantization_config.get("bnb_4bit_use_double_quant", True)
            ),
        )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir / "tokenizer",
        trust_remote_code=False,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    backbone = AutoModel.from_pretrained(
        resolved_model_id,
        revision=resolved_revision,
        trust_remote_code=False,
        local_files_only=not allow_download,
        torch_dtype=compute_dtype,
        quantization_config=bnb_config,
        device_map={"": device},
    )
    backbone = PeftModel.from_pretrained(
        backbone,
        checkpoint_dir / "adapter",
        is_trainable=False,
        local_files_only=True,
    )

    blend_weights = head_config.get("blend_weights", head_config.get("blend_weight", 0.5))
    model = Qwen3ForEssayScoring(
        backbone=backbone,
        projection_size=int(head_config["projection_size"]),
        dropout=float(head_config["dropout"]),
        blend_weight=blend_weights,
    )
    head_state = torch.load(
        checkpoint_dir / "scoring_heads.pt",
        map_location="cpu",
        weights_only=True,
    )
    if set(head_state) != {"shared_projection", "trait_heads"}:
        raise ValueError("scoring_heads.pt has an unexpected top-level state schema")
    model.shared_projection.load_state_dict(head_state["shared_projection"], strict=True)
    model.trait_heads.load_state_dict(head_state["trait_heads"], strict=True)
    model.shared_projection.to(device)
    model.trait_heads.to(device)
    model.eval()
    return LoadedScorer(
        model=model,
        tokenizer=tokenizer,
        checkpoint_dir=checkpoint_dir,
        model_id=resolved_model_id,
        model_revision=resolved_revision,
        max_length=max_length,
        compute_dtype=compute_dtype,
        precision=resolved_precision,
    )


@torch.inference_mode()
def predict_scores_and_embeddings(
    loaded: LoadedScorer,
    records: list[EssayInput | EssayRecord],
    *,
    batch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict scores and post-projection hidden vectors in one forward pass."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dataset = EssayInferenceDataset(
        records,
        loaded.tokenizer,
        max_length=loaded.max_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=EssayInferenceCollator(loaded.tokenizer),
    )
    device = next(loaded.model.shared_projection.parameters()).device
    score_batches: list[np.ndarray] = []
    embedding_batches: list[np.ndarray] = []
    ordered_ids: list[str] = []
    autocast_enabled = device.type == "cuda"
    for batch in loader:
        with torch.autocast(
            device_type=device.type,
            dtype=loaded.compute_dtype,
            enabled=autocast_enabled,
        ):
            outputs = loaded.model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
        score_batches.append(outputs["scores"].float().cpu().numpy())
        embedding_batches.append(outputs["shared_hidden"].float().cpu().numpy())
        ordered_ids.extend(batch["ids"])
    expected_ids = [record.id for record in records]
    if ordered_ids != expected_ids:
        raise RuntimeError("inference loader changed source-record order")
    if not score_batches:
        return np.empty((0, 3), dtype=float), np.empty((0, 0), dtype=np.float32)
    scores = np.concatenate(score_batches, axis=0)
    embeddings = np.concatenate(embedding_batches, axis=0).astype(
        np.float32, copy=False
    )
    if scores.shape != (len(records), 3) or not np.isfinite(scores).all():
        raise RuntimeError(f"scorer returned an invalid prediction matrix: {scores.shape}")
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(records)
        or not np.isfinite(embeddings).all()
        or np.any(np.linalg.norm(embeddings, axis=1) <= 0.0)
    ):
        raise RuntimeError(
            f"scorer returned invalid shared embeddings: {embeddings.shape}"
        )
    return np.clip(scores, 1.0, 5.0), embeddings


def predict_scores(
    loaded: LoadedScorer,
    records: list[EssayInput | EssayRecord],
    *,
    batch_size: int = 1,
) -> np.ndarray:
    """Predict a continuous `(n, 3)` matrix in source-record order."""

    scores, _ = predict_scores_and_embeddings(
        loaded, records, batch_size=batch_size
    )
    return scores


@torch.inference_mode()
def extract_shared_embeddings(
    loaded: LoadedScorer,
    records: list[EssayInput | EssayRecord],
    *,
    batch_size: int = 1,
) -> np.ndarray:
    """Return the scorer's post-projection representation in source-record order.

    These embeddings are checkpoint-specific.  Consumers must never mix vectors
    produced by different fold adapters in one cosine space.
    """

    _, embeddings = predict_scores_and_embeddings(
        loaded, records, batch_size=batch_size
    )
    return embeddings
