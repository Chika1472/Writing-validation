from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load import load_inference_jsonl
from src.evaluation.prediction_provenance import (
    prediction_manifest_path,
    validate_prediction_provenance,
)
from src.evaluation.oof_provenance import validate_oof_provenance
from src.evaluation.predictions import read_canonical_predictions
from src.rationale.evidence import build_evidence_ledger
from src.rationale.parsing import assess_grounding, parse_rationales
from src.rationale.prompting import (
    RATIONALE_PROMPT_CONTRACT,
    build_rationale_messages,
)
from src.utils.config import load_yaml
from src.utils.hashing import sha256_file, sha256_text
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths
from src.utils.reproducibility import seed_everything


def _imports() -> tuple[Any, ...]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise ImportError("Install the pinned qwen dependencies with `pip install -e .[qwen]`.") from error
    return AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _pinned_revision(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise ValueError("rationale generation requires a pinned 40-character model commit SHA")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and grounding-filter local Qwen silver rationales."
    )
    parser.add_argument("--config", default="configs/rationale_sft.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument(
        "--folds",
        default=None,
        help="Required when --scores is a genuine OOF artifact; forbidden otherwise.",
    )
    parser.add_argument("--output", required=True, help="Accepted silver JSONL.")
    parser.add_argument("--rejected-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


@torch.inference_mode()
def _generate(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_input_length: int,
    max_new_tokens: int,
) -> tuple[str, int]:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=False,
        return_tensors="pt",
    )
    input_length = int(encoded["input_ids"].shape[1])
    if input_length > max_input_length:
        raise ValueError(
            f"rationale input length {input_length} exceeds max_input_length={max_input_length}"
        )
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    text = tokenizer.decode(
        generated[0, input_ids.shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return text.strip(), input_length


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    input_path = Path(args.input).resolve()
    score_path = Path(args.scores).resolve()
    score_manifest_path = prediction_manifest_path(score_path)
    fold_path = Path(args.folds).resolve() if args.folds else None
    output_path = Path(args.output).resolve()
    rejected_path = Path(args.rejected_output).resolve()
    report_path = Path(args.report).resolve()
    manifest_path = output_path.with_suffix(".manifest.json")
    require_distinct_paths(
        config=config_path,
        input=input_path,
        scores=score_path,
        score_manifest=score_manifest_path,
        output=output_path,
        rejected=rejected_path,
        report=report_path,
        manifest=manifest_path,
        folds=fold_path,
    )
    require_new_paths(
        output=output_path,
        rejected=rejected_path,
        report=report_path,
        manifest=manifest_path,
    )
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-14B silver generation requires CUDA")

    config = load_yaml(config_path)
    model_config = config["model"]
    quantization = config["quantization"]
    silver_config = config["silver"]
    if bool(model_config.get("enable_thinking", False)):
        raise ValueError("silver generation requires enable_thinking=false")
    if str(quantization.get("compute_dtype", "bfloat16")) != "bfloat16":
        raise ValueError("silver generation currently supports BF16 compute only")
    revision = _pinned_revision(args.model_revision or model_config.get("revision"))
    seed_everything(int(config["training"]["seed"]))
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = _imports()
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
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_id"],
        revision=revision,
        trust_remote_code=False,
        local_files_only=local_files_only,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
        device_map={"": 0},
    )
    model.eval()

    records = load_inference_jsonl(input_path)
    raw_score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_score_manifest, dict):
        raise ValueError("score manifest must contain a JSON object")
    if raw_score_manifest.get("artifact_type") == "out_of_fold_predictions":
        if fold_path is None:
            raise ValueError("--folds is required for OOF-conditioned silver generation")
        score_manifest = validate_oof_provenance(
            prediction_path=score_path,
            gold_path=input_path,
            fold_path=fold_path,
        )
        score_provenance_type = "out_of_fold_predictions"
    elif raw_score_manifest.get("artifact_type") == "scorer_predictions":
        if fold_path is not None:
            raise ValueError("--folds is only valid for an OOF score artifact")
        score_manifest = validate_prediction_provenance(score_path)
        if score_manifest.get("input_sha256") != sha256_file(input_path):
            raise ValueError("score predictions do not belong to the supplied input")
        score_provenance_type = "scorer_predictions"
    else:
        raise ValueError("scores must have a scorer_predictions or OOF manifest")
    score_rows = read_canonical_predictions(score_path)
    if int(score_manifest.get("rows", -1)) != len(score_rows):
        raise ValueError("score row count disagrees with its provenance manifest")
    if {row["model"] for row in score_rows} != {score_manifest["scorer_name"]}:
        raise ValueError("score rows do not match their scorer provenance")
    score_by_id = {row["id"]: row for row in score_rows}
    selected = records[: args.limit] if args.limit else records
    missing = [record.id for record in selected if record.id not in score_by_id]
    if missing:
        raise ValueError(f"scores are missing selected ids: {missing[:5]}")
    if args.limit is None:
        input_ids = {record.id for record in records}
        extra_scores = sorted(set(score_by_id).difference(input_ids))
        if extra_scores:
            raise ValueError(f"scores contain ids absent from input: {extra_scores[:5]}")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    max_attempts = int(silver_config.get("max_attempts", 2))
    if max_attempts <= 0:
        raise ValueError("silver.max_attempts must be positive")
    for record in selected:
        score_row = score_by_id[record.id]
        if score_row["prompt_num"] != record.prompt_num:
            raise ValueError(f"prompt mismatch for {record.id}")
        scores = {
            trait: float(score_row["prediction"][trait])
            for trait in ("content", "organization", "expression")
        }
        ledger = build_evidence_ledger(record)
        base_messages = build_rationale_messages(record, scores, ledger)
        attempt_reports = []
        accepted_row = None
        for attempt in range(1, max_attempts + 1):
            messages = [dict(message) for message in base_messages]
            if attempt > 1:
                messages[-1]["content"] += (
                    "\n\n[재시도 지시] 이전 출력이 형식 또는 grounding 검사를 통과하지 못했다. "
                    "정확한 JSON과 제공된 exact evidence 인용만 사용한다."
                )
            try:
                raw, input_tokens = _generate(
                    model,
                    tokenizer,
                    messages,
                    max_input_length=int(model_config["max_input_length"]),
                    max_new_tokens=int(model_config["max_new_tokens"]),
                )
                rationales = parse_rationales(raw)
                grounding = assess_grounding(
                    rationales, essay=record.essay, ledger=ledger
                )
                attempt_reports.append(
                    {
                        "attempt": attempt,
                        "raw_output": raw,
                        "input_tokens": input_tokens,
                        "grounding": {
                            "accepted": grounding.accepted,
                            "reasons": list(grounding.reasons),
                            "exact_evidence_hits": grounding.exact_evidence_hits,
                        },
                    }
                )
                if grounding.accepted or not bool(silver_config.get("require_grounding", True)):
                    accepted_row = {
                        "id": record.id,
                        "prompt_num": record.prompt_num,
                        "conditioned_scores": scores,
                        "rationales": rationales,
                        "evidence": ledger.to_dict(),
                        "accepted_attempt": attempt,
                        "attempts": attempt_reports,
                    }
                    break
            except Exception as error:
                attempt_reports.append(
                    {"attempt": attempt, "error": f"{type(error).__name__}: {error}"}
                )
        if accepted_row is not None:
            accepted.append(accepted_row)
        else:
            rejected.append(
                {
                    "id": record.id,
                    "prompt_num": record.prompt_num,
                    "conditioned_scores": scores,
                    "attempts": attempt_reports,
                }
            )

    total = len(selected)
    acceptance_rate = len(accepted) / total if total else 0.0
    promotion_eligible = (
        args.limit is None
        and bool(silver_config.get("require_grounding", True))
        and len(accepted) >= int(silver_config["minimum_examples"])
        and acceptance_rate >= float(silver_config["minimum_acceptance_rate"])
    )
    report = {
        "rows": total,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": acceptance_rate,
        "minimum_examples": int(silver_config["minimum_examples"]),
        "minimum_acceptance_rate": float(silver_config["minimum_acceptance_rate"]),
        "promotion_eligible": promotion_eligible,
        "limited_run": args.limit is not None,
    }
    for path, rows in ((output_path, accepted), (rejected_path, rejected)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                    + "\n"
                )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = build_manifest(
        run_id=output_path.stem,
        project_root=config["_project_root"],
        config={key: value for key, value in config.items() if not key.startswith("_")},
        input_files=(
            config_path,
            input_path,
            score_path,
            score_manifest_path,
            *((fold_path,) if fold_path else ()),
        ),
        extra={
            "artifact_type": "grounded_silver_rationales",
            "accepted_file": output_path.name,
            "accepted_sha256": sha256_file(output_path),
            "rejected_file": rejected_path.name,
            "rejected_sha256": sha256_file(rejected_path),
            "report_file": report_path.name,
            "report_sha256": sha256_file(report_path),
            "model_id": model_config["model_id"],
            "model_revision": revision,
            "input_sha256": sha256_file(input_path),
            "score_file_sha256": sha256_file(score_path),
            "score_scorer_signature": score_manifest["scorer_signature"],
            "score_provenance_type": score_provenance_type,
            "folds_sha256": sha256_file(fold_path) if fold_path else None,
            "rationale_prompt_sha256": sha256_text(RATIONALE_PROMPT_CONTRACT),
            "evidence_code_sha256": sha256_file(
                Path(__file__).resolve().parents[1] / "src" / "rationale" / "evidence.py"
            ),
            "grounding_code_sha256": sha256_file(
                Path(__file__).resolve().parents[1] / "src" / "rationale" / "parsing.py"
            ),
            "require_grounding": bool(silver_config.get("require_grounding", True)),
            "promotion_eligible": promotion_eligible,
            "rows": total,
            "accepted_rows": len(accepted),
            "rejected_rows": len(rejected),
        },
    )
    write_manifest(manifest_path, manifest)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
