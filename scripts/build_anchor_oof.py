from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.anchors.artifact import (
    ANCHOR_BANK_ARTIFACT_TYPE,
    anchor_scorer_signature,
    anchor_manifest_path,
    save_anchor_bank,
)
from src.anchors.embeddings import embedding_manifest_path, load_embedding_matrix
from src.anchors.knn import prompt_aware_knn_predict
from src.data.folds import load_folds
from src.data.load import load_jsonl
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.oof_provenance import (
    oof_manifest_path,
    oof_provenance_fields,
)
from src.evaluation.predictions import prediction_records, write_predictions
from src.utils.config import load_yaml
from src.utils.hashing import sha256_file, sha256_text
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths


def _fold_path(value: str) -> tuple[int, Path]:
    fold_text, separator, path_text = value.partition("=")
    if not separator or not fold_text.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError("--embedding must have the form FOLD=PATH")
    try:
        fold = int(fold_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("embedding fold must be an integer") from error
    if fold < 0:
        raise argparse.ArgumentTypeError("embedding fold must be non-negative")
    return fold, Path(path_text).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe anchor OOF: each held fold uses only other-fold labels "
            "in the embedding space of its own fixed-epoch scorer checkpoint."
        )
    )
    parser.add_argument("--config", default="configs/anchor_knn.yaml")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--embedding", action="append", type=_fold_path, required=True)
    parser.add_argument("--oof-output", required=True)
    parser.add_argument("--anchor-bank", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--model-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding_paths = dict(args.embedding)
    if len(embedding_paths) != len(args.embedding):
        raise ValueError("every --embedding fold key must be unique")
    config_path = Path(args.config).resolve()
    gold_path = Path(args.gold).resolve()
    fold_path = Path(args.folds).resolve()
    oof_path = Path(args.oof_output).resolve()
    oof_manifest = oof_manifest_path(oof_path)
    bank_path = Path(args.anchor_bank).resolve()
    bank_manifest = anchor_manifest_path(bank_path)
    diagnostics_path = Path(args.diagnostics).resolve()
    report_path = Path(args.report).resolve()
    named_paths = {
        "config": config_path,
        "gold": gold_path,
        "folds": fold_path,
        "oof": oof_path,
        "oof_manifest": oof_manifest,
        "bank": bank_path,
        "bank_manifest": bank_manifest,
        "diagnostics": diagnostics_path,
        "report": report_path,
    }
    for fold, path in embedding_paths.items():
        named_paths[f"embedding_{fold}"] = path
        named_paths[f"embedding_manifest_{fold}"] = embedding_manifest_path(path)
    require_distinct_paths(**named_paths)
    require_new_paths(
        oof=oof_path,
        oof_manifest=oof_manifest,
        bank=bank_path,
        bank_manifest=bank_manifest,
        diagnostics=diagnostics_path,
        report=report_path,
    )
    if bank_path.suffix.lower() != ".npz":
        raise ValueError("--anchor-bank must end in .npz")

    config = load_yaml(config_path)
    if config.get("method") != "prompt_aware_cosine_knn":
        raise ValueError("anchor config method must be prompt_aware_cosine_knn")
    if config.get("selection_policy") != "prespecified_without_outer_oof_labels":
        raise ValueError("anchor k/temperature must be prespecified before outer OOF")
    k = int(config.get("k", 0))
    temperature = float(config.get("temperature", 0.0))
    unknown_policy = str(config.get("unknown_prompt_policy", ""))
    if k <= 0 or temperature <= 0.0 or unknown_policy not in {"global", "error"}:
        raise ValueError("invalid k, temperature, or unknown_prompt_policy")

    gold = load_jsonl(gold_path)
    assignments = load_folds(fold_path)
    gold_ids = [record.id for record in gold]
    if set(assignments) != set(gold_ids):
        raise ValueError("fold assignments must match gold ids exactly")
    fold_values = sorted(set(assignments.values()))
    if set(embedding_paths) != set(fold_values):
        raise ValueError(
            f"one embedding artifact is required per fold: expected={fold_values}, "
            f"received={sorted(embedding_paths)}"
        )
    prompt_hashes = [sha256_text(record.prompt) for record in gold]
    score_matrix = np.asarray([record.score.trait_values for record in gold], dtype=np.float32)
    fold_vector = np.asarray([assignments[record.id] for record in gold], dtype=int)
    predictions = np.full((len(gold), 3), np.nan, dtype=np.float32)
    diagnostics: list[dict | None] = [None] * len(gold)
    bank_embeddings: dict[int, np.ndarray] = {}
    embedding_contracts: dict[str, dict] = {}
    input_hash = sha256_file(gold_path)

    for fold in fold_values:
        embedding_path = embedding_paths[fold]
        artifact = load_embedding_matrix(
            embedding_path, expected_input_sha256=input_hash
        )
        if artifact.manifest.get("checkpoint_fold") != fold:
            raise ValueError(
                f"embedding fold key {fold} disagrees with checkpoint provenance"
            )
        if artifact.manifest.get("embedding_role") != "reference":
            raise ValueError(f"fold {fold} OOF requires a reference-role embedding artifact")
        if artifact.manifest.get("training_gold_sha256") != input_hash:
            raise ValueError(f"fold {fold} checkpoint was trained from different gold data")
        if artifact.manifest.get("training_folds_sha256") != sha256_file(fold_path):
            raise ValueError(f"fold {fold} checkpoint was trained with different folds")
        if set(artifact.ids.tolist()) != set(gold_ids):
            raise ValueError(f"fold {fold} embedding ids do not match gold")
        positions = [
            int(np.flatnonzero(artifact.ids == record_id)[0]) for record_id in gold_ids
        ]
        ordered_embeddings = artifact.embeddings[positions]
        ordered_prompts = artifact.prompt_hashes[positions].tolist()
        if ordered_prompts != prompt_hashes:
            raise ValueError(f"fold {fold} prompt hashes do not match exact gold prompts")
        bank_embeddings[fold] = ordered_embeddings
        held_indices = np.flatnonzero(fold_vector == fold)
        train_indices = np.flatnonzero(fold_vector != fold)
        if len(held_indices) == 0 or len(train_indices) == 0:
            raise ValueError(f"fold {fold} does not define a non-empty train/held split")
        result = prompt_aware_knn_predict(
            query_ids=[gold_ids[index] for index in held_indices],
            query_prompt_hashes=[prompt_hashes[index] for index in held_indices],
            query_embeddings=ordered_embeddings[held_indices],
            bank_ids=[gold_ids[index] for index in train_indices],
            bank_prompt_hashes=[prompt_hashes[index] for index in train_indices],
            bank_embeddings=ordered_embeddings[train_indices],
            bank_scores=score_matrix[train_indices],
            k=k,
            temperature=temperature,
            unknown_prompt_policy=unknown_policy,
        )
        predictions[held_indices] = result.predictions
        for local_index, source_index in enumerate(held_indices.tolist()):
            diagnostics[source_index] = {
                "id": gold_ids[source_index],
                "fold": fold,
                "mean_cosine_distance": float(result.mean_cosine_distance[local_index]),
                "neighbor_score_variance": result.neighbor_score_variance[
                    local_index
                ].astype(float).tolist(),
                "used_global_fallback": bool(result.used_global_fallback[local_index]),
                "neighbor_ids": list(result.neighbor_ids[local_index]),
            }
        embedding_contracts[str(fold)] = {
            "embedding_sha256": sha256_file(embedding_path),
            "embedding_manifest_sha256": sha256_file(
                embedding_manifest_path(embedding_path)
            ),
            "scorer_signature": artifact.manifest["scorer_signature"],
            "extraction_contract_sha256": artifact.manifest[
                "extraction_contract_sha256"
            ],
            "checkpoint_fold": fold,
            "training_gold_sha256": artifact.manifest["training_gold_sha256"],
            "training_folds_sha256": artifact.manifest["training_folds_sha256"],
            "scorer_run_manifest_sha256": artifact.manifest[
                "scorer_run_manifest_sha256"
            ],
        }

    if not np.isfinite(predictions).all() or any(item is None for item in diagnostics):
        raise RuntimeError("anchor OOF did not fill every gold row exactly once")
    oof_rows = prediction_records(gold, predictions, model=args.model_name)
    write_predictions(oof_path, oof_rows)
    save_anchor_bank(
        bank_path,
        ids=gold_ids,
        prompt_hashes=prompt_hashes,
        scores=score_matrix,
        embeddings_by_fold=bank_embeddings,
    )
    anchor_signature, anchor_signature_payload = anchor_scorer_signature(
        anchor_bank_sha256=sha256_file(bank_path),
        k=k,
        temperature=temperature,
        unknown_prompt_policy=unknown_policy,
        folds=fold_values,
        embedding_contracts=embedding_contracts,
    )
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with diagnostics_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in diagnostics:
            handle.write(
                json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + "\n"
            )
    report = {
        "metrics": evaluate_predictions(gold, oof_rows),
        "rows": len(gold),
        "folds": fold_values,
        "global_fallback_count": sum(
            int(bool(item["used_global_fallback"])) for item in diagnostics if item
        ),
        "warning": (
            "Candidate branch only. Promote through cross-fitted comparison/stacking; "
            "do not select it from validation labels."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    input_files = (
        config_path,
        gold_path,
        fold_path,
        *(path for path in embedding_paths.values()),
        *(embedding_manifest_path(path) for path in embedding_paths.values()),
    )
    common_config = {key: value for key, value in config.items() if not key.startswith("_")}
    bank_meta = build_manifest(
        run_id=bank_path.stem,
        project_root=Path(__file__).resolve().parents[1],
        config=common_config,
        input_files=input_files,
        extra={
            "artifact_type": ANCHOR_BANK_ARTIFACT_TYPE,
            "anchor_bank_sha256": sha256_file(bank_path),
            "anchor_signature": anchor_signature,
            "scorer_name": args.model_name,
            "anchor_signature_payload": anchor_signature_payload,
            "rows": len(gold),
            "folds": fold_values,
            "hidden_size": int(next(iter(bank_embeddings.values())).shape[1]),
            "prompt_identity": "sha256_exact_prompt_text",
            "embedding_contracts": embedding_contracts,
            "gold_sha256": input_hash,
            "folds_sha256": sha256_file(fold_path),
        },
    )
    write_manifest(bank_manifest, bank_meta)
    oof_meta = build_manifest(
        run_id=oof_path.stem,
        project_root=Path(__file__).resolve().parents[1],
        config=common_config,
        input_files=(*input_files, bank_path, bank_manifest),
        extra={
            **oof_provenance_fields(
                prediction_path=oof_path,
                gold_path=gold_path,
                fold_path=fold_path,
                rows=len(oof_rows),
                scorer_name=args.model_name,
                scorer_signature=anchor_signature,
                oof_level="base_model_oof",
            ),
            "anchor_bank": str(bank_path),
            "anchor_bank_sha256": sha256_file(bank_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
            "leakage_guard": "all held-fold labels excluded from neighbor bank",
        },
    )
    write_manifest(oof_manifest, oof_meta)
    print(json.dumps({"oof": str(oof_path), "anchor_bank": str(bank_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
