from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.folds import load_folds
from src.data.load import load_jsonl
from src.evaluation.oof_provenance import (
    checkpoint_ensemble_signature,
    checkpoint_fingerprint,
    checkpoint_set_signature,
    oof_manifest_path,
    oof_provenance_fields,
)
from src.evaluation.predictions import prediction_records, write_predictions
from src.inference.scorer import (
    checkpoint_artifact_files,
    load_scorer_checkpoint,
    predict_scores,
    resolve_checkpoint,
)
from src.models.contracts import SCORER_ARCHITECTURE_VERSION
from src.utils.hashing import sha256_file
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import (
    require_distinct_paths,
    require_new_paths,
    require_outside_roots,
)
from src.utils.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run fixed fold checkpoints at one inference precision and create "
            "checkpoint-bound held-out OOF predictions."
        )
    )
    parser.add_argument("--gold", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Repeat once per concrete epoch_<N> fold checkpoint directory.",
    )
    parser.add_argument("--precision", choices=("4bit", "bf16"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True, help="Stable experiment label.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def _checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    provenance_path = checkpoint / "checkpoint_provenance.json"
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint provenance must be an object: {provenance_path}")
    fold = payload.get("fold")
    epoch = payload.get("epoch")
    training_precision = payload.get("precision")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
        raise ValueError(f"checkpoint has invalid fold: {checkpoint}")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError(f"checkpoint has invalid epoch: {checkpoint}")
    if training_precision not in {"4bit", "bf16"}:
        raise ValueError(f"checkpoint has invalid training precision: {checkpoint}")
    if payload.get("scorer_architecture_version") != SCORER_ARCHITECTURE_VERSION:
        raise ValueError(f"checkpoint has incompatible scorer architecture: {checkpoint}")
    return payload


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    supplied_checkpoints = [Path(value).resolve() for value in args.checkpoint]
    if any(path.is_file() for path in supplied_checkpoints):
        raise ValueError(
            "precision OOF requires concrete epoch directories; outer-fold best-pointer "
            "files are diagnostic and would bias epoch selection"
        )
    checkpoints = [resolve_checkpoint(path) for path in supplied_checkpoints]
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("checkpoint directories must be unique")

    gold_path = Path(args.gold).resolve()
    fold_path = Path(args.folds).resolve()
    output_path = Path(args.output).resolve()
    manifest_path = oof_manifest_path(output_path)
    named_paths: dict[str, Path] = {
        "gold": gold_path,
        "folds": fold_path,
        "output": output_path,
        "manifest": manifest_path,
    }
    named_paths.update(
        {f"checkpoint_{index}": path for index, path in enumerate(checkpoints)}
    )
    require_distinct_paths(**named_paths)
    require_outside_roots(
        {f"checkpoint_{index}": path for index, path in enumerate(checkpoints)},
        output=output_path,
        manifest=manifest_path,
    )
    require_new_paths(output=output_path, manifest=manifest_path)

    gold = load_jsonl(gold_path)
    assignments = load_folds(fold_path)
    gold_ids = [record.id for record in gold]
    gold_id_set = set(gold_ids)
    missing_assignments = [record_id for record_id in gold_ids if record_id not in assignments]
    extra_assignments = sorted(set(assignments).difference(gold_id_set))
    if missing_assignments or extra_assignments:
        raise ValueError(
            "fold assignments must match gold ids exactly; "
            f"missing={missing_assignments[:5]}, extra={extra_assignments[:5]}"
        )

    metadata = [_checkpoint_metadata(path) for path in checkpoints]
    if any(
        item.get("train_sha256") != sha256_file(gold_path)
        or item.get("folds_sha256") != sha256_file(fold_path)
        for item in metadata
    ):
        raise ValueError("precision checkpoints were trained from different gold/folds")
    checkpoint_folds = [int(value["fold"]) for value in metadata]
    expected_folds = sorted(set(assignments.values()))
    if sorted(checkpoint_folds) != expected_folds:
        raise ValueError(
            "checkpoint folds must cover the fold file exactly once; "
            f"expected={expected_folds}, got={sorted(checkpoint_folds)}"
        )
    epochs = {int(value["epoch"]) for value in metadata}
    if len(epochs) != 1:
        raise ValueError(
            "all outer folds must use one fixed epoch chosen without outer-fold labels; "
            f"got epochs={sorted(epochs)}"
        )
    training_precisions = {str(value["precision"]) for value in metadata}
    if len(training_precisions) != 1:
        raise ValueError(
            "all fold checkpoints must share one training precision; "
            f"got {sorted(training_precisions)}"
        )

    seed_everything(args.seed)
    merged: dict[str, dict[str, Any]] = {}
    common_contract: tuple[str, str, int] | None = None
    fold_artifacts: list[dict[str, Any]] = []
    for checkpoint, provenance in sorted(
        zip(checkpoints, metadata, strict=True), key=lambda item: int(item[1]["fold"])
    ):
        fold = int(provenance["fold"])
        held_out = [record for record in gold if assignments[record.id] == fold]
        if not held_out:
            raise ValueError(f"fold {fold} has no held-out records")
        if int(provenance.get("rows", -1)) != len(held_out):
            raise ValueError(
                f"checkpoint fold {fold} row count does not match fold assignments"
            )

        loaded = load_scorer_checkpoint(
            checkpoint,
            precision=args.precision,
            allow_download=args.allow_download,
        )
        contract = (loaded.model_id, loaded.model_revision, loaded.max_length)
        if common_contract is None:
            common_contract = contract
        elif contract != common_contract:
            raise ValueError(
                "all fold checkpoints must share model id, revision, and max length"
            )
        matrix = predict_scores(loaded, held_out, batch_size=args.batch_size)
        rows = prediction_records(
            held_out,
            matrix,
            model=f"{args.model}:{args.precision}",
        )
        for row in rows:
            if row["id"] in merged:
                raise ValueError(f"OOF id predicted by multiple checkpoints: {row['id']}")
            merged[row["id"]] = row
        fold_artifacts.append(
            {
                "fold": fold,
                "epoch": int(provenance["epoch"]),
                "checkpoint": str(checkpoint),
                "checkpoint_fingerprint": checkpoint_fingerprint(checkpoint),
                "training_precision": provenance["precision"],
                "inference_precision": args.precision,
                "rows": len(rows),
            }
        )
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    missing_predictions = [record_id for record_id in gold_ids if record_id not in merged]
    extra_predictions = sorted(set(merged).difference(gold_id_set))
    if missing_predictions or extra_predictions:
        raise RuntimeError(
            "precision OOF must predict every gold row exactly once; "
            f"missing={missing_predictions[:5]}, extra={extra_predictions[:5]}"
        )
    rows = [merged[record_id] for record_id in gold_ids]
    write_predictions(output_path, rows)

    set_signature = checkpoint_set_signature(checkpoints)
    scorer_signature = checkpoint_ensemble_signature(
        checkpoints, precision=args.precision
    )
    artifact_inputs: list[Path] = [gold_path, fold_path]
    for checkpoint in checkpoints:
        artifact_inputs.extend(checkpoint_artifact_files(checkpoint))
    manifest = build_manifest(
        run_id=output_path.stem,
        project_root=Path.cwd(),
        config={
            "model": args.model,
            "precision": args.precision,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        input_files=artifact_inputs,
        extra={
            **oof_provenance_fields(
                prediction_path=output_path,
                gold_path=gold_path,
                fold_path=fold_path,
                rows=len(rows),
                scorer_name=f"{args.model}:{args.precision}",
                scorer_signature=scorer_signature,
            ),
            "precision": args.precision,
            "checkpoint_set_signature": set_signature,
            "fixed_epoch": next(iter(epochs)),
            "training_precision": next(iter(training_precisions)),
            "fold_artifacts": fold_artifacts,
        },
    )
    write_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "oof": str(output_path),
                "manifest": str(manifest_path),
                "precision": args.precision,
                "rows": len(rows),
                "checkpoint_set_signature": set_signature,
                "oof_sha256": sha256_file(output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
