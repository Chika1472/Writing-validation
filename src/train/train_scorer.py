from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.folds import load_folds
from src.data.load import load_jsonl
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.predictions import prediction_records, write_predictions
from src.models.losses import EssayScoringLoss
from src.models.contracts import SCORER_ARCHITECTURE_VERSION
from src.models.qwen_scorer import Qwen3ForEssayScoring
from src.train.dataset import (
    EssayBatchCollator,
    EssayScoringDataset,
    PromptPairBatchSampler,
    PromptGroupBatchSampler,
    within_prompt_pair_indices,
)
from src.train.prompting import SCORING_PROMPT_CONTRACT
from src.utils.config import load_yaml, resolve_project_path
from src.utils.hashing import sha256_file, sha256_text
from src.utils.manifest import build_manifest, write_manifest
from src.utils.reproducibility import seed_everything


def _optional_training_imports() -> tuple[Any, ...]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModel,
            AutoTokenizer,
            BitsAndBytesConfig,
            get_cosine_schedule_with_warmup,
        )
    except ImportError as error:
        raise ImportError(
            "Qwen scorer training requires the optional dependencies. "
            "Install the pinned training environment with `pip install -e .[qwen]`."
        ) from error
    return (
        AutoModel,
        AutoTokenizer,
        BitsAndBytesConfig,
        get_cosine_schedule_with_warmup,
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )


def _dtype(name: str) -> torch.dtype:
    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return values[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported compute dtype: {name}") from error


def _head_state(model: Qwen3ForEssayScoring) -> dict[str, Any]:
    return {
        "shared_projection": model.shared_projection.state_dict(),
        "trait_heads": model.trait_heads.state_dict(),
    }


def _save_checkpoint(
    model: Qwen3ForEssayScoring,
    tokenizer: Any,
    directory: Path,
    metrics: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    artifact_metadata: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(directory / "adapter", safe_serialization=True)
    torch.save(_head_state(model), directory / "scoring_heads.pt")
    tokenizer.save_pretrained(directory / "tokenizer")
    head_config = {
        "artifact_version": 1,
        "projection_size": model.shared_projection[0].out_features,
        "dropout": model.shared_projection[3].p,
        "blend_weights": {
            trait: model.trait_heads[trait].blend_weight for trait in model.trait_heads
        },
        **artifact_metadata,
    }
    (directory / "scoring_head_config.json").write_text(
        json.dumps(head_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_predictions(directory / "oof.jsonl", prediction_rows)
    checkpoint_provenance = {
        "artifact_type": "qwen_scorer_fold_checkpoint",
        "fold": int(artifact_metadata["fold"]),
        "seed": int(artifact_metadata["seed"]),
        "epoch": int(metrics["epoch"]),
        "oof_file": "oof.jsonl",
        "oof_sha256": sha256_file(directory / "oof.jsonl"),
        "rows": len(prediction_rows),
        "precision": artifact_metadata["precision"],
        "train_sha256": artifact_metadata["train_sha256"],
        "folds_sha256": artifact_metadata["folds_sha256"],
        "prompt_template_sha256": artifact_metadata["prompt_template_sha256"],
        "scorer_architecture_version": artifact_metadata[
            "scorer_architecture_version"
        ],
    }
    (directory / "checkpoint_provenance.json").write_text(
        json.dumps(checkpoint_provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_best_pointer(
    output_dir: Path,
    *,
    name: str,
    epoch: int,
    metric: str,
    value: float,
) -> None:
    payload = {
        "epoch": epoch,
        "metric": metric,
        "value": value,
        "checkpoint": f"epoch_{epoch}",
        "warning": (
            "Outer-fold best-epoch selection is diagnostic only. Use a fixed epoch "
            "chosen by inner CV when constructing unbiased OOF predictions."
        ),
    }
    (output_dir / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@torch.no_grad()
def _evaluate(
    model: Qwen3ForEssayScoring,
    loader: DataLoader,
    records: list[Any],
    device: torch.device,
    compute_dtype: torch.dtype,
    *,
    model_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    score_batches: list[np.ndarray] = []
    ids: list[str] = []
    autocast_enabled = device.type == "cuda" and compute_dtype != torch.float32
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=compute_dtype,
            enabled=autocast_enabled,
        ):
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        score_batches.append(output["scores"].float().cpu().numpy())
        ids.extend(batch["ids"])

    matrix = np.concatenate(score_batches, axis=0)
    record_by_id = {record.id: record for record in records}
    ordered_records = [record_by_id[record_id] for record_id in ids]
    rows = prediction_records(ordered_records, matrix, model=model_name)
    return evaluate_predictions(ordered_records, rows), rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Qwen3 essay-scoring fold.")
    parser.add_argument("--config", default="configs/scorer_qlora.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--folds", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override training.seed; persisted in the run manifest.",
    )
    parser.add_argument("--allow-unpinned-revision", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scorer_config = load_yaml(args.config)
    data_config = load_yaml(args.data_config)
    model_config = scorer_config["model"]
    training_config = scorer_config["training"]
    loss_config = scorer_config["loss"]
    if args.seed is not None:
        if not 0 <= args.seed <= 2**32 - 1:
            raise ValueError("--seed must be between 0 and 2**32-1")
        training_config["seed"] = args.seed
    if bool(model_config.get("enable_thinking", False)):
        raise ValueError("the scorer input contract requires model.enable_thinking=false")
    if str(model_config.get("pooling", "last_token")) != "last_token":
        raise ValueError("only model.pooling=last_token is implemented")
    revision = args.model_revision or model_config.get("revision")
    pinned_revision = isinstance(revision, str) and re.fullmatch(
        r"[0-9a-fA-F]{40}", revision
    )
    if not pinned_revision and not args.allow_unpinned_revision:
        raise ValueError(
            "A pinned 40-character Qwen commit SHA is required. Set model.revision or pass "
            "--model-revision; use --allow-unpinned-revision only for smoke tests."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-14B training requires a CUDA GPU")

    (
        AutoModel,
        AutoTokenizer,
        BitsAndBytesConfig,
        get_cosine_schedule_with_warmup,
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    ) = _optional_training_imports()

    seed = int(training_config["seed"])
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("training.seed must be between 0 and 2**32-1")
    seed_everything(seed)
    compute_dtype = _dtype(str(scorer_config["quantization"]["compute_dtype"]))
    if compute_dtype is not torch.bfloat16:
        raise ValueError(
            "the current QLoRA loop is intentionally BF16-only; FP16 requires GradScaler"
        )
    device = torch.device("cuda:0")
    train_path = resolve_project_path(data_config, data_config["paths"]["train"])
    records = load_jsonl(train_path)
    fold_path = Path(args.folds).resolve()
    assignments = load_folds(fold_path)
    missing = [record.id for record in records if record.id not in assignments]
    record_ids = {record.id for record in records}
    extra = sorted(set(assignments).difference(record_ids))
    if missing or extra:
        raise ValueError(
            "fold file must match train ids exactly; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    train_records = [record for record in records if assignments[record.id] != args.fold]
    validation_records = [record for record in records if assignments[record.id] == args.fold]
    if not train_records or not validation_records:
        raise ValueError(f"fold {args.fold} does not create nonempty train and validation sets")

    local_files_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_id"],
        revision=revision,
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = None
    if bool(scorer_config["quantization"]["load_in_4bit"]):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=scorer_config["quantization"]["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(
                scorer_config["quantization"]["bnb_4bit_use_double_quant"]
            ),
        )

    backbone = AutoModel.from_pretrained(
        model_config["model_id"],
        revision=revision,
        trust_remote_code=False,
        local_files_only=local_files_only,
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
        device_map={"": 0},
    )
    if quantization_config is not None:
        backbone = prepare_model_for_kbit_training(
            backbone,
            use_gradient_checkpointing=bool(training_config["gradient_checkpointing"]),
        )
    elif bool(training_config["gradient_checkpointing"]):
        backbone.gradient_checkpointing_enable()

    lora_target = scorer_config["lora"]["target_modules"]
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=int(scorer_config["lora"]["rank"]),
        lora_alpha=int(scorer_config["lora"]["alpha"]),
        lora_dropout=float(scorer_config["lora"]["dropout"]),
        target_modules=lora_target,
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_config, revision=revision)
    model = Qwen3ForEssayScoring(
        backbone=backbone,
        projection_size=int(model_config["projection_size"]),
        dropout=float(model_config["head_dropout"]),
        blend_weight=float(model_config["blend_weight"]),
    )
    model.shared_projection.to(device)
    model.trait_heads.to(device)

    max_length = int(model_config["max_length"])
    train_dataset = EssayScoringDataset(train_records, tokenizer, max_length=max_length)
    validation_dataset = EssayScoringDataset(
        validation_records, tokenizer, max_length=max_length
    )
    collator = EssayBatchCollator(tokenizer)
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(training_config["micro_batch_size"])
    if batch_size <= 0:
        raise ValueError("training.micro_batch_size must be positive")
    pairwise_weight = float(loss_config["pairwise_weight"])
    soft_rank_weight = float(loss_config.get("soft_rank_weight", 0.0))
    ranking_weight = pairwise_weight + soft_rank_weight
    ranking_minimum_gap = float(
        loss_config.get(
            "ranking_minimum_gap",
            loss_config.get("pairwise_minimum_gap", 0.0),
        )
    )
    ranking_batch_sampler = None
    if soft_rank_weight > 0:
        ranking_batch_sampler = PromptGroupBatchSampler(
            [record.prompt_num for record in train_records],
            batch_size=batch_size,
            seed=seed,
        )
    elif pairwise_weight > 0:
        if batch_size != 2:
            raise ValueError(
                "pairwise loss without soft-rank requires micro_batch_size=2 with "
                "the prompt-pair sampler"
            )
        ranking_batch_sampler = PromptPairBatchSampler(
            [record.prompt_num for record in train_records],
            [record.score.trait_values for record in train_records],
            seed=seed,
            minimum_gap=ranking_minimum_gap,
        )
    if ranking_batch_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=ranking_batch_sampler,
            collate_fn=collator,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            generator=generator,
        )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    head_parameters = list(model.shared_projection.parameters()) + list(
        model.trait_heads.parameters()
    )
    head_ids = {id(parameter) for parameter in head_parameters}
    adapter_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    if not adapter_parameters:
        raise RuntimeError("PEFT exposed no trainable adapter parameters")
    optimizer = AdamW(
        [
            {
                "params": adapter_parameters,
                "lr": float(training_config["learning_rate"]),
            },
            {
                "params": head_parameters,
                "lr": float(training_config["head_learning_rate"]),
            },
        ],
        weight_decay=float(training_config["weight_decay"]),
    )
    gradient_accumulation = int(training_config["gradient_accumulation_steps"])
    epochs = int(training_config["epochs"])
    if gradient_accumulation <= 0 or epochs <= 0:
        raise ValueError("gradient_accumulation_steps and epochs must be positive")
    update_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation)
    total_steps = epochs * update_steps_per_epoch
    warmup_steps = int(total_steps * float(training_config["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = EssayScoringLoss(
        mse_weight=float(loss_config["regression_weight"]),
        ordinal_weight=float(loss_config["ordinal_weight"]),
        pairwise_weight=pairwise_weight,
        soft_rank_weight=soft_rank_weight,
        tie_threshold=ranking_minimum_gap,
        pairwise_temperature=float(loss_config.get("pairwise_temperature", 1.0)),
        pairwise_order_weight=float(loss_config.get("pairwise_order_weight", 0.5)),
        soft_rank_temperature=float(loss_config.get("soft_rank_temperature", 0.25)),
    )

    artifacts = resolve_project_path(data_config, data_config["paths"]["artifacts"])
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else artifacts / "models" / args.run_id
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    history: list[dict[str, Any]] = []
    best_rmse = float("inf")
    best_spearman = float("-inf")
    autocast_enabled = compute_dtype != torch.float32
    checkpoint_metadata = {
        "model_id": model_config["model_id"],
        "model_revision": revision,
        "max_length": max_length,
        "quantization": scorer_config["quantization"],
        "prompt_template_sha256": sha256_text(SCORING_PROMPT_CONTRACT),
        "fold": args.fold,
        "seed": seed,
        "precision": (
            "4bit"
            if bool(scorer_config["quantization"]["load_in_4bit"])
            else "bf16"
        ),
        "train_sha256": sha256_file(train_path),
        "folds_sha256": sha256_file(fold_path),
        "scorer_architecture_version": SCORER_ARCHITECTURE_VERSION,
    }

    for epoch in range(1, epochs + 1):
        if ranking_batch_sampler is not None:
            ranking_batch_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = {
            "loss": 0.0,
            "mse": 0.0,
            "ordinal": 0.0,
            "pairwise": 0.0,
            "soft_rank": 0.0,
        }
        pair_count = 0
        soft_rank_group_count = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            accumulation_group_start = (
                ((batch_index - 1) // gradient_accumulation) * gradient_accumulation + 1
            )
            accumulation_group_size = min(
                gradient_accumulation,
                len(train_loader) - accumulation_group_start + 1,
            )
            targets = batch["targets"].to(device)
            pair_indices = None
            if ranking_weight > 0:
                eligible_pairs = within_prompt_pair_indices(
                    batch["prompt_nums"], targets, minimum_gap=criterion.tie_threshold
                ).to(device)
                pair_count += int(eligible_pairs.shape[0])
                if criterion.soft_rank_weight > 0:
                    prompt_sizes = {
                        prompt: batch["prompt_nums"].count(prompt)
                        for prompt in set(batch["prompt_nums"])
                    }
                    soft_rank_group_count += sum(
                        int(size >= 2) for size in prompt_sizes.values()
                    )
                if criterion.pairwise_weight > 0:
                    pair_indices = eligible_pairs
            with torch.autocast(
                device_type="cuda",
                dtype=compute_dtype,
                enabled=autocast_enabled,
            ):
                output = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                losses = criterion(
                    output["scores"],
                    targets,
                    ordinal_logits=output["ordinal_logits"],
                    pair_indices=pair_indices,
                    group_ids=(
                        batch["prompt_nums"]
                        if criterion.soft_rank_weight > 0
                        else None
                    ),
                )
                scaled_loss = losses["loss"] / accumulation_group_size
            scaled_loss.backward()
            for name in running:
                running[name] += float(losses[name].detach().float().cpu())

            should_step = (
                batch_index % gradient_accumulation == 0 or batch_index == len(train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if pairwise_weight > 0 and pair_count == 0:
            raise RuntimeError(
                "pairwise loss was enabled but the epoch produced no eligible comparison"
            )
        if soft_rank_weight > 0 and soft_rank_group_count == 0:
            raise RuntimeError(
                "soft-rank loss was enabled but the epoch produced no multi-item prompt group"
            )

        validation_metrics, prediction_rows = _evaluate(
            model,
            validation_loader,
            validation_records,
            device,
            compute_dtype,
            model_name=f"{args.run_id}_epoch{epoch}",
        )
        epoch_report = {
            "epoch": epoch,
            "train": {
                **{name: value / len(train_loader) for name, value in running.items()},
                "pair_count": pair_count,
                "soft_rank_group_count": soft_rank_group_count,
            },
            "validation": validation_metrics,
        }
        history.append(epoch_report)
        epoch_prediction_path = output_dir / f"oof_epoch{epoch}.jsonl"
        write_predictions(epoch_prediction_path, prediction_rows)
        _save_checkpoint(
            model,
            tokenizer,
            output_dir / f"epoch_{epoch}",
            epoch_report,
            prediction_rows,
            checkpoint_metadata,
        )

        macro_rmse = float(validation_metrics["macro"]["rmse"])
        macro_spearman = float(validation_metrics["macro"]["spearman"])
        if macro_rmse < best_rmse:
            best_rmse = macro_rmse
            _write_best_pointer(
                output_dir,
                name="best_rmse",
                epoch=epoch,
                metric="macro_rmse",
                value=macro_rmse,
            )
        if macro_spearman > best_spearman:
            best_spearman = macro_spearman
            _write_best_pointer(
                output_dir,
                name="best_spearman",
                epoch=epoch,
                metric="macro_spearman",
                value=macro_spearman,
            )

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    public_config = {
        "scorer": {key: value for key, value in scorer_config.items() if not key.startswith("_")},
        "data": {key: value for key, value in data_config.items() if not key.startswith("_")},
    }
    manifest = build_manifest(
        run_id=args.run_id,
        project_root=data_config["_project_root"],
        config=public_config,
        input_files=(train_path, fold_path),
        extra={
            "model_id": model_config["model_id"],
            "model_revision": revision,
            "prompt_template_sha256": sha256_text(SCORING_PROMPT_CONTRACT),
            "fold": args.fold,
            "seed": seed,
            "history": str(history_path),
            "history_sha256": sha256_file(history_path),
        },
    )
    write_manifest(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "best_rmse": best_rmse, "best_spearman": best_spearman}))


if __name__ == "__main__":
    main()
