from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.anchors.artifact import anchor_manifest_path, load_anchor_bank
from src.anchors.embeddings import embedding_manifest_path, load_embedding_matrix
from src.anchors.knn import prompt_aware_knn_predict
from src.data.load import load_inference_jsonl
from src.evaluation.prediction_provenance import (
    prediction_manifest_path,
    prediction_provenance_fields,
)
from src.evaluation.predictions import prediction_records, write_predictions
from src.utils.hashing import sha256_file, sha256_text
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths


def _fold_path(value: str) -> tuple[int, Path]:
    fold_text, separator, path_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("--embedding must have the form FOLD=PATH")
    try:
        fold = int(fold_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("embedding fold must be an integer") from error
    if fold < 0 or not path_text.strip():
        raise argparse.ArgumentTypeError("embedding fold/path is invalid")
    return fold, Path(path_text).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict an unseen split with a frozen multi-checkpoint anchor bank."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--anchor-bank", required=True)
    parser.add_argument("--embedding", action="append", type=_fold_path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--model-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding_paths = dict(args.embedding)
    if len(embedding_paths) != len(args.embedding):
        raise ValueError("every target embedding fold key must be unique")
    input_path = Path(args.input).resolve()
    bank_path = Path(args.anchor_bank).resolve()
    bank_manifest = anchor_manifest_path(bank_path)
    output_path = Path(args.output).resolve()
    output_manifest = prediction_manifest_path(output_path)
    diagnostics_path = Path(args.diagnostics).resolve()
    named_paths = {
        "input": input_path,
        "bank": bank_path,
        "bank_manifest": bank_manifest,
        "output": output_path,
        "output_manifest": output_manifest,
        "diagnostics": diagnostics_path,
    }
    for fold, path in embedding_paths.items():
        named_paths[f"embedding_{fold}"] = path
        named_paths[f"embedding_manifest_{fold}"] = embedding_manifest_path(path)
    require_distinct_paths(**named_paths)
    require_new_paths(
        output=output_path,
        output_manifest=output_manifest,
        diagnostics=diagnostics_path,
    )

    records = load_inference_jsonl(input_path)
    bank = load_anchor_bank(bank_path)
    if args.model_name != bank.manifest["scorer_name"]:
        raise ValueError("--model-name must match the frozen anchor bank scorer_name")
    expected_folds = sorted(bank.embeddings_by_fold)
    if sorted(embedding_paths) != expected_folds:
        raise ValueError(
            f"target embeddings must cover bank folds {expected_folds} exactly"
        )
    config = bank.manifest.get("config", {})
    k = int(config.get("k", 0))
    temperature = float(config.get("temperature", 0.0))
    unknown_policy = str(config.get("unknown_prompt_policy", ""))
    if k <= 0 or temperature <= 0.0 or unknown_policy not in {"global", "error"}:
        raise ValueError("anchor bank manifest contains invalid KNN configuration")
    target_ids = [record.id for record in records]
    target_prompts = [sha256_text(record.prompt) for record in records]
    input_hash = sha256_file(input_path)
    per_fold_predictions: list[np.ndarray] = []
    per_id_diagnostics: dict[str, list[dict]] = {record_id: [] for record_id in target_ids}
    input_files: list[Path] = [input_path, bank_path, bank_manifest]

    contracts = bank.manifest.get("embedding_contracts")
    if not isinstance(contracts, dict):
        raise ValueError("anchor bank has no embedding contracts")
    for fold in expected_folds:
        target_path = embedding_paths[fold]
        expected_contract = contracts.get(str(fold))
        if not isinstance(expected_contract, dict):
            raise ValueError(f"anchor bank has no contract for fold {fold}")
        target = load_embedding_matrix(
            target_path,
            expected_input_sha256=input_hash,
            expected_scorer_signature=expected_contract.get("scorer_signature"),
        )
        if target.manifest.get("embedding_role") != "query":
            raise ValueError(f"fold {fold} target embedding must declare role=query")
        if target.manifest.get("checkpoint_fold") != fold:
            raise ValueError(f"target embedding checkpoint fold mismatch for fold {fold}")
        if target.manifest.get("extraction_contract_sha256") != expected_contract.get(
            "extraction_contract_sha256"
        ):
            raise ValueError(f"target/reference embedding extraction contract differs for fold {fold}")
        if set(target.ids.tolist()) != set(target_ids):
            raise ValueError(f"target embedding ids do not match inference input for fold {fold}")
        positions = [int(np.flatnonzero(target.ids == record_id)[0]) for record_id in target_ids]
        target_matrix = target.embeddings[positions]
        if target.prompt_hashes[positions].tolist() != target_prompts:
            raise ValueError(f"target prompt hashes disagree with input for fold {fold}")
        result = prompt_aware_knn_predict(
            query_ids=target_ids,
            query_prompt_hashes=target_prompts,
            query_embeddings=target_matrix,
            bank_ids=bank.ids.tolist(),
            bank_prompt_hashes=bank.prompt_hashes.tolist(),
            bank_embeddings=bank.embeddings_by_fold[fold],
            bank_scores=bank.scores,
            k=k,
            temperature=temperature,
            unknown_prompt_policy=unknown_policy,
        )
        per_fold_predictions.append(result.predictions)
        for index, record_id in enumerate(target_ids):
            per_id_diagnostics[record_id].append(
                {
                    "fold": fold,
                    "mean_cosine_distance": float(result.mean_cosine_distance[index]),
                    "neighbor_score_variance": result.neighbor_score_variance[
                        index
                    ].astype(float).tolist(),
                    "used_global_fallback": bool(result.used_global_fallback[index]),
                    "neighbor_ids": list(result.neighbor_ids[index]),
                }
            )
        input_files.extend((target_path, embedding_manifest_path(target_path)))

    prediction_matrix = np.mean(np.stack(per_fold_predictions, axis=0), axis=0)
    anchor_signature = str(bank.manifest["anchor_signature"])
    rows = prediction_records(records, prediction_matrix, model=args.model_name)
    write_predictions(output_path, rows)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with diagnostics_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record_id in target_ids:
            handle.write(
                json.dumps(
                    {"id": record_id, "fold_results": per_id_diagnostics[record_id]},
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    manifest = build_manifest(
        run_id=output_path.stem,
        project_root=Path(__file__).resolve().parents[1],
        config={
            "k": k,
            "temperature": temperature,
            "unknown_prompt_policy": unknown_policy,
            "folds": expected_folds,
        },
        input_files=input_files,
        extra={
            **prediction_provenance_fields(
                prediction_path=output_path,
                input_path=input_path,
                rows=len(rows),
                scorer_name=args.model_name,
                scorer_signature=anchor_signature,
            ),
            "prediction_subtype": "anchor_knn_predictions",
            "anchor_bank_sha256": sha256_file(bank_path),
            "anchor_bank_signature": bank.manifest["anchor_signature"],
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": sha256_file(diagnostics_path),
        },
    )
    write_manifest(output_manifest, manifest)
    print(json.dumps({"predictions": str(output_path), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
