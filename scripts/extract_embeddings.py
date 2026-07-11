from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.anchors.embeddings import (
    EMBEDDING_ARTIFACT_TYPE,
    embedding_extraction_contract_sha256,
    embedding_manifest_path,
    save_embedding_matrix,
)
from src.data.load import load_inference_jsonl
from src.evaluation.oof_provenance import checkpoint_ensemble_signature
from src.inference.scorer import (
    checkpoint_artifact_files,
    extract_shared_embeddings,
    load_scorer_checkpoint,
    resolve_checkpoint,
)
from src.utils.hashing import sha256_file, sha256_text
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths, require_outside_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract checkpoint-specific shared scorer embeddings for anchor KNN."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New .npz embedding artifact.")
    parser.add_argument("--precision", choices=("4bit", "bf16"), default="4bit")
    parser.add_argument(
        "--role",
        choices=("reference", "query"),
        required=True,
        help=(
            "reference is only for the exact scorer training set used to build the "
            "anchor bank; query is for validation/test inputs."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    checkpoint = resolve_checkpoint(args.checkpoint)
    output_path = Path(args.output).resolve()
    manifest_path = embedding_manifest_path(output_path)
    if output_path.suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    require_distinct_paths(
        input=input_path,
        checkpoint=checkpoint,
        output=output_path,
        manifest=manifest_path,
    )
    require_outside_roots(
        {"checkpoint": checkpoint}, output=output_path, manifest=manifest_path
    )
    require_new_paths(output=output_path, manifest=manifest_path)

    records = load_inference_jsonl(input_path)
    checkpoint_provenance = json.loads(
        (checkpoint / "checkpoint_provenance.json").read_text(encoding="utf-8")
    )
    checkpoint_fold = checkpoint_provenance.get("fold")
    if isinstance(checkpoint_fold, bool) or not isinstance(checkpoint_fold, int):
        raise ValueError("checkpoint provenance has no integer fold")
    training_gold_sha256 = checkpoint_provenance.get("train_sha256")
    training_folds_sha256 = checkpoint_provenance.get("folds_sha256")
    if not isinstance(training_gold_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", training_gold_sha256
    ) is None:
        raise ValueError("checkpoint provenance has no valid training data hash")
    if args.role == "reference" and training_gold_sha256 != sha256_file(input_path):
        raise ValueError(
            "anchor reference embeddings must be extracted from the exact scorer training data"
        )
    if not isinstance(training_folds_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", training_folds_sha256
    ) is None:
        raise ValueError("checkpoint provenance has no training fold hash")
    run_manifest_path = (
        checkpoint / "manifest.json"
        if (checkpoint / "manifest.json").is_file()
        else checkpoint.parent / "manifest.json"
    )
    if not run_manifest_path.is_file():
        raise FileNotFoundError("anchor extraction requires the scorer run manifest")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_inputs = run_manifest.get("inputs") if isinstance(run_manifest, dict) else None
    if (
        not isinstance(run_inputs, dict)
        or run_manifest.get("fold") != checkpoint_fold
        or training_gold_sha256 not in run_inputs.values()
        or training_folds_sha256 not in run_inputs.values()
    ):
        raise ValueError("scorer run manifest does not bind this fold, gold, and fold file")
    scorer_signature = checkpoint_ensemble_signature(
        [checkpoint], precision=args.precision
    )
    loaded = load_scorer_checkpoint(
        checkpoint,
        precision=args.precision,
        allow_download=args.allow_download,
    )
    embeddings = extract_shared_embeddings(
        loaded, records, batch_size=args.batch_size
    )
    actual_precision = loaded.precision
    model_id = loaded.model_id
    model_revision = loaded.model_revision
    del loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    save_embedding_matrix(
        output_path,
        ids=[record.id for record in records],
        prompt_hashes=[sha256_text(record.prompt) for record in records],
        embeddings=embeddings,
    )
    extraction_contract_sha256 = embedding_extraction_contract_sha256()
    artifact_inputs = (
        input_path,
        run_manifest_path,
        *checkpoint_artifact_files(checkpoint),
    )
    manifest = build_manifest(
        run_id=output_path.stem,
        project_root=Path(__file__).resolve().parents[1],
        config={
            "precision": actual_precision,
            "batch_size": args.batch_size,
            "representation": "shared_hidden_post_projection",
            "embedding_role": args.role,
            "extraction_contract_sha256": extraction_contract_sha256,
        },
        input_files=artifact_inputs,
        extra={
            "artifact_type": EMBEDDING_ARTIFACT_TYPE,
            "embedding_file": output_path.name,
            "embedding_sha256": sha256_file(output_path),
            "input_sha256": sha256_file(input_path),
            "rows": len(records),
            "hidden_size": int(embeddings.shape[1]),
            "scorer_signature": scorer_signature,
            "checkpoint": str(checkpoint),
            "checkpoint_fold": checkpoint_fold,
            "training_gold_sha256": training_gold_sha256,
            "training_folds_sha256": training_folds_sha256,
            "scorer_run_manifest_sha256": sha256_file(run_manifest_path),
            "model_id": model_id,
            "model_revision": model_revision,
            "precision": actual_precision,
            "prompt_identity": "sha256_exact_prompt_text",
            "embedding_role": args.role,
            "extraction_contract_sha256": extraction_contract_sha256,
        },
    )
    write_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "embeddings": str(output_path),
                "rows": len(records),
                "hidden_size": int(embeddings.shape[1]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
