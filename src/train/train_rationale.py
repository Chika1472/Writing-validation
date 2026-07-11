from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.load import load_inference_jsonl
from src.rationale.prompting import RATIONALE_PROMPT_CONTRACT
from src.train.rationale_dataset import RationaleSFTCollator, RationaleSFTDataset
from src.utils.config import load_yaml
from src.utils.hashing import sha256_file, sha256_text
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths
from src.utils.reproducibility import seed_everything


def _imports() -> tuple[Any, ...]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            get_cosine_schedule_with_warmup,
        )
    except ImportError as error:
        raise ImportError("Install the pinned qwen dependencies with `pip install -e .[qwen]`.") from error
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        get_cosine_schedule_with_warmup,
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank silver JSONL line: {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"silver row {line_number} must be an object")
            rows.append(value)
    if not rows:
        raise ValueError("silver rationale file is empty")
    return rows


def _split(rows: list[dict[str, Any]], fraction: float) -> tuple[list[dict], list[dict]]:
    if not 0.0 <= fraction < 0.5:
        raise ValueError("validation_fraction must be in [0, 0.5)")
    if fraction == 0.0:
        return rows, []
    validation_count = max(1, int(round(len(rows) * fraction)))
    if validation_count >= len(rows):
        raise ValueError("silver dataset is too small for the requested validation fraction")
    ordered = sorted(
        rows,
        key=lambda row: sha256_text(str(row.get("id", ""))),
    )
    return ordered[validation_count:], ordered[:validation_count]


def _revision(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise ValueError("rationale SFT requires a pinned 40-character model commit SHA")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the evidence-conditioned rationale QLoRA.")
    parser.add_argument("--config", default="configs/rationale_sft.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-unpromoted-silver", action="store_true")
    parser.add_argument(
        "--allow-non-oof-silver",
        action="store_true",
        help="Smoke-test override; production rationale SFT requires OOF-conditioned scores.",
    )
    return parser.parse_args()


@torch.no_grad()
def _validation_loss(model: Any, loader: DataLoader, device: torch.device) -> float | None:
    if len(loader) == 0:
        return None
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                use_cache=False,
            )
        total += float(output.loss.detach().float().cpu())
        count += 1
    return total / count


def _save_epoch(
    model: Any,
    tokenizer: Any,
    directory: Path,
    *,
    epoch: int,
    train_loss: float,
    validation_loss: float | None,
    metadata: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(directory / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(directory / "tokenizer")
    payload = {
        "artifact_type": "rationale_adapter_checkpoint",
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        **metadata,
    }
    (directory / "checkpoint.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    input_path = Path(args.input).resolve()
    silver_path = Path(args.silver).resolve()
    silver_manifest_path = silver_path.with_suffix(".manifest.json")
    config = load_yaml(config_path)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(config["_project_root"]) / "artifacts" / "models" / args.run_id
    )
    require_distinct_paths(
        config=config_path,
        input=input_path,
        silver=silver_path,
        silver_manifest=silver_manifest_path,
        output=output_dir,
    )
    require_new_paths(output=output_dir)
    if not silver_manifest_path.is_file():
        raise FileNotFoundError(f"silver manifest is required: {silver_manifest_path}")
    silver_manifest = json.loads(silver_manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(silver_manifest, dict)
        or silver_manifest.get("artifact_type") != "grounded_silver_rationales"
        or silver_manifest.get("accepted_file") != silver_path.name
        or silver_manifest.get("accepted_sha256") != sha256_file(silver_path)
    ):
        raise ValueError("silver file does not match its adjacent manifest")
    if not silver_manifest.get("promotion_eligible") and not args.allow_unpromoted_silver:
        raise ValueError(
            "silver artifact did not meet its acceptance gate; use the override only for smoke tests"
        )
    if (
        silver_manifest.get("score_provenance_type") != "out_of_fold_predictions"
        and not args.allow_non_oof_silver
    ):
        raise ValueError("production rationale SFT requires OOF-conditioned silver scores")
    if silver_manifest.get("input_sha256") != sha256_file(input_path):
        raise ValueError("silver rationales were not generated from the supplied essay input")
    if silver_manifest.get("rationale_prompt_sha256") != sha256_text(
        RATIONALE_PROMPT_CONTRACT
    ):
        raise ValueError("silver rationale prompt contract differs from the training contract")
    source_root = Path(__file__).resolve().parents[2]
    if silver_manifest.get("evidence_code_sha256") != sha256_file(
        source_root / "src" / "rationale" / "evidence.py"
    ):
        raise ValueError("silver evidence extraction code differs from the training contract")
    if silver_manifest.get("grounding_code_sha256") != sha256_file(
        source_root / "src" / "rationale" / "parsing.py"
    ):
        raise ValueError("silver grounding validator differs from the training contract")
    if silver_manifest.get("require_grounding") is not True:
        raise ValueError("rationale SFT requires silver generated with grounding enforced")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-14B rationale QLoRA requires CUDA")

    model_config = config["model"]
    quantization = config["quantization"]
    training = config["training"]
    if bool(model_config.get("enable_thinking", False)):
        raise ValueError("rationale SFT requires model.enable_thinking=false")
    for field in ("max_input_length", "max_new_tokens", "max_sequence_length"):
        if int(model_config.get(field, 0)) <= 0:
            raise ValueError(f"model.{field} must be positive")
    if int(model_config["max_sequence_length"]) < (
        int(model_config["max_input_length"]) + int(model_config["max_new_tokens"])
    ):
        raise ValueError(
            "model.max_sequence_length must cover max_input_length + max_new_tokens"
        )
    revision = _revision(args.model_revision or model_config.get("revision"))
    if silver_manifest.get("model_id") != model_config.get("model_id"):
        raise ValueError("rationale SFT model_id must match the silver generator model")
    if silver_manifest.get("model_revision") != revision:
        raise ValueError("rationale SFT base revision must match the silver generator revision")
    if str(quantization.get("compute_dtype", "bfloat16")) != "bfloat16":
        raise ValueError("rationale SFT currently supports BF16 compute only")
    seed = int(training["seed"])
    seed_everything(seed)
    (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        get_cosine_schedule_with_warmup,
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    ) = _imports()
    local_files_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_id"],
        revision=revision,
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    records = load_inference_jsonl(input_path)
    records_by_id = {record.id: record for record in records}
    silver_rows = _jsonl(silver_path)
    if (
        len(silver_rows) < int(config["silver"]["minimum_examples"])
        and not args.allow_unpromoted_silver
    ):
        raise ValueError("accepted silver row count is below silver.minimum_examples")
    train_rows, validation_rows = _split(
        silver_rows, float(training.get("validation_fraction", 0.1))
    )
    max_length = int(model_config["max_sequence_length"])
    train_dataset = RationaleSFTDataset(
        records_by_id,
        train_rows,
        tokenizer,
        max_length=max_length,
        score_jitter=float(training.get("score_jitter", 0.0)),
        score_jitter_copies=int(training.get("score_jitter_copies", 0)),
        jitter_seed=seed,
    )
    validation_dataset = (
        RationaleSFTDataset(records_by_id, validation_rows, tokenizer, max_length=max_length)
        if validation_rows
        else None
    )
    collator = RationaleSFTCollator(tokenizer)
    batch_size = int(training["micro_batch_size"])
    if batch_size <= 0:
        raise ValueError("training.micro_batch_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset or [],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    bnb = None
    if bool(quantization.get("load_in_4bit", True)):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quantization.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=bool(
                quantization.get("bnb_4bit_use_double_quant", True)
            ),
        )
    base = AutoModelForCausalLM.from_pretrained(
        model_config["model_id"],
        revision=revision,
        trust_remote_code=False,
        local_files_only=local_files_only,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
        device_map={"": 0},
    )
    base.config.use_cache = False
    if bnb is not None:
        base = prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=bool(training["gradient_checkpointing"]),
        )
    elif bool(training["gradient_checkpointing"]):
        base.gradient_checkpointing_enable()
    lora = config["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=lora["target_modules"],
        bias="none",
    )
    model = get_peft_model(base, peft_config, revision=revision)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("PEFT exposed no trainable rationale adapter parameters")
    optimizer = AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    accumulation = int(training["gradient_accumulation_steps"])
    epochs = int(training["epochs"])
    if accumulation <= 0 or epochs <= 0:
        raise ValueError("gradient_accumulation_steps and epochs must be positive")
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = epochs * updates_per_epoch
    warmup = int(total_updates * float(training["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_updates)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    history = []
    best_loss = float("inf")
    best_epoch = None
    metadata = {
        "model_id": model_config["model_id"],
        "model_revision": revision,
        "silver_sha256": sha256_file(silver_path),
        "rationale_prompt_sha256": sha256_text(RATIONALE_PROMPT_CONTRACT),
        "max_sequence_length": max_length,
        "max_input_length": int(model_config["max_input_length"]),
        "max_new_tokens": int(model_config["max_new_tokens"]),
    }
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for batch_index, batch in enumerate(train_loader, start=1):
            group_start = ((batch_index - 1) // accumulation) * accumulation + 1
            group_size = min(accumulation, len(train_loader) - group_start + 1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                    use_cache=False,
                )
                loss = output.loss
            (loss / group_size).backward()
            total_loss += float(loss.detach().float().cpu())
            if batch_index % accumulation == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        train_loss = total_loss / len(train_loader)
        validation_loss = _validation_loss(model, validation_loader, device)
        epoch_dir = output_dir / f"epoch_{epoch}"
        _save_epoch(
            model,
            tokenizer,
            epoch_dir,
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            metadata=metadata,
        )
        report = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "checkpoint": epoch_dir.name,
        }
        history.append(report)
        selection_loss = validation_loss if validation_loss is not None else train_loss
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_epoch = epoch
            (output_dir / "best.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    history_path = output_dir / "history.json"
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = build_manifest(
        run_id=args.run_id,
        project_root=config["_project_root"],
        config={key: value for key, value in config.items() if not key.startswith("_")},
        input_files=(config_path, input_path, silver_path, silver_manifest_path),
        extra={
            "artifact_type": "rationale_sft_run",
            **metadata,
            "train_rows": len(train_rows),
            "train_examples_after_score_jitter": len(train_dataset),
            "validation_rows": len(validation_rows),
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "history_sha256": sha256_file(history_path),
        },
    )
    write_manifest(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {"output_dir": str(output_dir), "best_epoch": best_epoch, "best_loss": best_loss},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
