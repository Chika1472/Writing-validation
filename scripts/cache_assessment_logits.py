from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assessment.cache import (
    ASSESSMENT_CACHE_ARTIFACT,
    assessment_cache_manifest_path,
    load_assessment_cache,
    prompt_hashes,
    write_assessment_npz,
)
from src.assessment.codebook import DEFAULT_ANSWER_CODES
from src.assessment.extraction import (
    extract_assessment_probabilities,
    load_assessment_extractor,
)
from src.assessment.prompting import ASSESSMENT_QUERY_SHA256
from src.assessment.questions import (
    QUESTION_IDS,
    QUESTION_VERSION,
    QUESTIONS_SHA256,
    question_contract,
)
from src.data.load import load_inference_jsonl
from src.utils.config import load_yaml
from src.utils.hashing import sha256_file, sha256_json
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache restricted ordinal answer probabilities from Qwen3-14B."
    )
    parser.add_argument("--config", default="configs/assessment_questions.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--precision", choices=("4bit", "bf16"), default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    branch_config = config["branch"]
    model_config = config["model"]
    quantization_config = config["quantization"]
    if model_config.get("model_id") != "Qwen/Qwen3-14B":
        raise ValueError("assessment question v1 is fixed to Qwen/Qwen3-14B")
    if quantization_config.get("compute_dtype") != "bfloat16":
        raise ValueError("assessment extraction implements BF16 compute only")
    if branch_config.get("question_version") != QUESTION_VERSION:
        raise ValueError("assessment config question_version does not match code")
    if branch_config.get("feature_type") != "restricted_answer_probabilities":
        raise ValueError("assessment v1 implements restricted answer probabilities only")
    if branch_config.get("candidate_only") is not True or branch_config.get(
        "auto_promote"
    ) is not False:
        raise ValueError("assessment branch must remain candidate-only and non-promoting")
    configured_codes = tuple(str(value) for value in model_config["answer_codes"])
    if configured_codes != DEFAULT_ANSWER_CODES:
        raise ValueError(
            "assessment query v1 is bound to the exact answer codes A, B, C, D, E"
        )

    revision = args.model_revision or model_config.get("revision")
    tokenizer_revision = (
        args.tokenizer_revision or model_config.get("tokenizer_revision") or revision
    )
    for name, value in (
        ("model revision", revision),
        ("tokenizer revision", tokenizer_revision),
    ):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
            raise ValueError(f"a pinned 40-character {name} commit SHA is required")
    precision = args.precision or str(model_config["precision"])
    if precision not in {"4bit", "bf16"}:
        raise ValueError("assessment precision must be '4bit' or 'bf16'")
    batch_size = (
        int(model_config["batch_size"])
        if args.batch_size is None
        else args.batch_size
    )
    max_length = int(model_config["max_length"])
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("assessment batch_size and max_length must be positive")
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("assessment cache output must end in .npz")
    manifest_path = assessment_cache_manifest_path(output_path)
    temporary_cache = output_path.with_name(output_path.stem + ".tmp-artifact.npz")
    temporary_manifest = manifest_path.with_name(
        manifest_path.name + ".tmp-artifact"
    )
    require_distinct_paths(
        config=config_path,
        input=input_path,
        output=output_path,
        manifest=manifest_path,
        temporary_cache=temporary_cache,
        temporary_manifest=temporary_manifest,
    )
    require_new_paths(
        output=output_path,
        manifest=manifest_path,
        temporary_cache=temporary_cache,
        temporary_manifest=temporary_manifest,
    )

    records = load_inference_jsonl(input_path)
    loaded = load_assessment_extractor(
        model_id=str(model_config["model_id"]),
        model_revision=revision,
        tokenizer_revision=tokenizer_revision,
        precision=precision,
        batch_size=batch_size,
        max_length=max_length,
        seed=args.seed,
        quantization={
            "bnb_4bit_quant_type": str(
                quantization_config["bnb_4bit_quant_type"]
            ),
            "bnb_4bit_use_double_quant": bool(
                quantization_config["bnb_4bit_use_double_quant"]
            ),
            "compute_dtype": str(quantization_config["compute_dtype"]),
        },
        answer_codes=configured_codes,
        device="cuda:0",
        allow_download=args.allow_download,
    )
    logits, probabilities = extract_assessment_probabilities(loaded, records)
    codebook = loaded.codebook
    feature_signature_payload = loaded.feature_payload
    feature_signature = loaded.feature_signature
    public_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    try:
        write_assessment_npz(
            temporary_cache,
            ids=[record.id for record in records],
            prompt_nums=[record.prompt_num for record in records],
            logits=logits,
            probabilities=probabilities,
        )
        manifest = build_manifest(
            run_id=output_path.stem,
            project_root=config["_project_root"],
            config=public_config,
            input_files=(config_path, input_path),
            extra={
                "artifact_type": ASSESSMENT_CACHE_ARTIFACT,
                "artifact_version": 1,
                "candidate_branch": True,
                "auto_promoted": False,
                "label_free_feature_extraction": True,
                "cache_file": output_path.name,
                "cache_creation_path": str(output_path),
                "cache_sha256": sha256_file(temporary_cache),
                "source_data": str(input_path),
                "source_data_sha256": sha256_file(input_path),
                "rows": len(records),
                "probability_shape": list(probabilities.shape),
                "logit_shape": list(logits.shape),
                "ordered_ids_sha256": sha256_json([record.id for record in records]),
                "prompt_hashes": prompt_hashes(records),
                "question_version": QUESTION_VERSION,
                "questions_sha256": QUESTIONS_SHA256,
                "question_contract": question_contract(),
                "question_ids": list(QUESTION_IDS),
                "assessment_query_sha256": ASSESSMENT_QUERY_SHA256,
                "model_id": model_config["model_id"],
                "model_revision": revision,
                "tokenizer_revision": tokenizer_revision,
                "precision": precision,
                "max_length": max_length,
                "batch_size": batch_size,
                "seed": args.seed,
                "deterministic_torch": True,
                "extractor_code_sha256": feature_signature_payload[
                    "extractor_code_sha256"
                ],
                "reproducibility_code_sha256": feature_signature_payload[
                    "reproducibility_code_sha256"
                ],
                "quantization": feature_signature_payload["quantization"],
                "feature_type": "restricted_answer_probabilities",
                "answer_codes": codebook["answer_codes"],
                "answer_values": codebook["answer_values"],
                "answer_token_ids": codebook["answer_token_ids"],
                "codebook_sha256": codebook["codebook_sha256"],
                "codebook_context_hashes": codebook["context_hashes"],
                "feature_signature_payload": feature_signature_payload,
                "feature_signature": feature_signature,
            },
        )
        write_manifest(temporary_manifest, manifest)
        temporary_cache.replace(output_path)
        temporary_manifest.replace(manifest_path)
        load_assessment_cache(output_path)
    finally:
        temporary_cache.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "cache": str(output_path),
                "manifest": str(manifest_path),
                "rows": len(records),
                "feature_signature": feature_signature,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
