from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.folds import load_folds
from src.data.load import load_jsonl
from src.evaluation.metrics import prediction_matrix
from src.evaluation.oof_provenance import (
    checkpoint_ensemble_signature,
    checkpoint_fingerprint,
    checkpoint_set_signature,
    oof_manifest_path,
    oof_provenance_fields,
    validate_oof_provenance,
)
from src.evaluation.predictions import (
    prediction_records,
    read_canonical_predictions,
    write_predictions,
)
from src.models.contracts import SCORER_ARCHITECTURE_VERSION
from src.utils.hashing import sha256_file
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths, require_outside_roots


def _seed_source(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator or not path_text.strip():
        raise argparse.ArgumentTypeError("--source must have the form SEED=OOF_PATH")
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("source seed must be an integer") from error
    return seed, Path(path_text).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average complete seed-level fixed-epoch OOF artifacts and bind the exact "
            "union of deployment checkpoints."
        )
    )
    parser.add_argument("--gold", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--source", action="append", type=_seed_source, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = dict(args.source)
    if len(args.source) < 2 or len(sources) != len(args.source):
        raise ValueError("seed ensemble requires at least two uniquely keyed OOF sources")
    gold_path = Path(args.gold).resolve()
    fold_path = Path(args.folds).resolve()
    output_path = Path(args.output).resolve()
    manifest_path = oof_manifest_path(output_path)
    paths = {
        "gold": gold_path,
        "folds": fold_path,
        "output": output_path,
        "manifest": manifest_path,
    }
    for seed, path in sources.items():
        paths[f"source_{seed}"] = path
        paths[f"source_manifest_{seed}"] = oof_manifest_path(path)
    require_distinct_paths(**paths)
    require_new_paths(output=output_path, manifest=manifest_path)

    gold = load_jsonl(gold_path)
    assignments = load_folds(fold_path)
    if set(assignments) != {record.id for record in gold}:
        raise ValueError("fold assignments must match gold ids exactly")
    expected_folds = sorted(set(assignments.values()))
    gold_hash = sha256_file(gold_path)
    folds_hash = sha256_file(fold_path)
    matrices: list[np.ndarray] = []
    all_checkpoints: list[Path] = []
    source_contracts: dict[str, dict] = {}
    precisions: set[str] = set()
    epochs: set[int] = set()
    model_contracts: set[tuple[str, str, int]] = set()

    for declared_seed, source_path in sorted(sources.items()):
        provenance = validate_oof_provenance(
            prediction_path=source_path,
            gold_path=gold_path,
            fold_path=fold_path,
        )
        if provenance.get("oof_level") != "base_model_oof":
            raise ValueError("seed ensemble accepts only complete base_model_oof sources")
        rows = read_canonical_predictions(source_path)
        if int(provenance["rows"]) != len(rows) or len(rows) != len(gold):
            raise ValueError(f"seed {declared_seed} OOF row count mismatch")
        if {row["model"] for row in rows} != {provenance["scorer_name"]}:
            raise ValueError(f"seed {declared_seed} row model/provenance mismatch")
        fold_entries = provenance.get("fold_files")
        entry_mode = "checkpoint_oof_files"
        if not isinstance(fold_entries, list):
            fold_entries = provenance.get("fold_artifacts")
            entry_mode = "precision_rerun"
        if not isinstance(fold_entries, list) or len(fold_entries) != len(expected_folds):
            raise ValueError(f"seed {declared_seed} OOF has no complete fold checkpoint list")
        precision = provenance.get("precision")
        if precision not in {"4bit", "bf16"}:
            raise ValueError(f"seed {declared_seed} has invalid precision")
        observed_folds: set[int] = set()
        source_checkpoints: list[Path] = []
        source_epochs: set[int] = set()
        for item in fold_entries:
            if not isinstance(item, dict):
                raise ValueError("invalid seed OOF fold file entry")
            if entry_mode == "checkpoint_oof_files":
                if not isinstance(item.get("path"), str):
                    raise ValueError("seed OOF fold entry has no prediction path")
                prediction_file = Path(item["path"]).resolve()
                checkpoint = prediction_file.parent
            else:
                if not isinstance(item.get("checkpoint"), str):
                    raise ValueError("precision OOF fold entry has no checkpoint path")
                checkpoint = Path(item["checkpoint"]).resolve()
                prediction_file = checkpoint / "oof.jsonl"
            checkpoint_meta_path = checkpoint / "checkpoint_provenance.json"
            checkpoint_meta = json.loads(checkpoint_meta_path.read_text(encoding="utf-8"))
            if not isinstance(checkpoint_meta, dict):
                raise ValueError(f"checkpoint provenance must be an object: {checkpoint}")
            fold = checkpoint_meta.get("fold")
            epoch = checkpoint_meta.get("epoch")
            if (
                checkpoint_meta.get("artifact_type") != "qwen_scorer_fold_checkpoint"
                or checkpoint_meta.get("seed") != declared_seed
                or isinstance(fold, bool)
                or not isinstance(fold, int)
                or isinstance(epoch, bool)
                or not isinstance(epoch, int)
                or epoch <= 0
                or checkpoint_meta.get("train_sha256") != gold_hash
                or checkpoint_meta.get("folds_sha256") != folds_hash
                or checkpoint_meta.get("scorer_architecture_version")
                != SCORER_ARCHITECTURE_VERSION
                or checkpoint_meta.get("oof_file") != prediction_file.name
                or checkpoint_meta.get("oof_sha256") != sha256_file(prediction_file)
                or item.get("fold") != fold
            ):
                raise ValueError(
                    f"seed {declared_seed} checkpoint provenance mismatch: {checkpoint}"
                )
            if entry_mode == "checkpoint_oof_files":
                if (
                    checkpoint_meta.get("precision") != precision
                    or item.get("sha256") != sha256_file(prediction_file)
                    or item.get("precision") != precision
                ):
                    raise ValueError(f"seed {declared_seed} fold precision/hash mismatch")
            elif (
                item.get("training_precision") != checkpoint_meta.get("precision")
                or item.get("inference_precision") != precision
                or item.get("epoch") != epoch
                or item.get("rows") != checkpoint_meta.get("rows")
                or item.get("checkpoint_fingerprint")
                != checkpoint_fingerprint(checkpoint)
            ):
                raise ValueError(f"seed {declared_seed} precision rerun contract mismatch")
            head_config = json.loads(
                (checkpoint / "scoring_head_config.json").read_text(encoding="utf-8")
            )
            if not isinstance(head_config, dict):
                raise ValueError(f"scoring head config must be an object: {checkpoint}")
            model_id = head_config.get("model_id")
            revision = head_config.get("model_revision")
            max_length = head_config.get("max_length")
            if (
                not isinstance(model_id, str)
                or not model_id
                or not isinstance(revision, str)
                or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None
                or isinstance(max_length, bool)
                or not isinstance(max_length, int)
                or max_length <= 0
                or head_config.get("scorer_architecture_version")
                != SCORER_ARCHITECTURE_VERSION
            ):
                raise ValueError(f"invalid scorer model contract: {checkpoint}")
            model_contracts.add((model_id, revision.lower(), max_length))
            observed_folds.add(fold)
            source_epochs.add(epoch)
            source_checkpoints.append(checkpoint)
        if sorted(observed_folds) != expected_folds or len(source_epochs) != 1:
            raise ValueError(
                f"seed {declared_seed} must cover every fold at one fixed epoch"
            )
        if provenance.get("scorer_signature") != checkpoint_ensemble_signature(
            source_checkpoints, precision=str(precision)
        ) or provenance.get("checkpoint_set_signature") != checkpoint_set_signature(
            source_checkpoints
        ):
            raise ValueError(f"seed {declared_seed} OOF checkpoint signature mismatch")
        precisions.add(str(precision))
        epochs.update(source_epochs)
        all_checkpoints.extend(source_checkpoints)
        matrices.append(prediction_matrix(rows, gold))
        source_contracts[str(declared_seed)] = {
            "scorer_name": provenance["scorer_name"],
            "scorer_signature": provenance["scorer_signature"],
            "oof_sha256": provenance["oof_sha256"],
            "oof_manifest_sha256": sha256_file(oof_manifest_path(source_path)),
            "checkpoint_set_signature": provenance.get("checkpoint_set_signature"),
        }

    if len(precisions) != 1 or len(epochs) != 1 or len(model_contracts) != 1:
        raise ValueError(
            f"all seed OOF sources must share model contract, precision, and fixed epoch; "
            f"models={model_contracts}, precisions={precisions}, epochs={epochs}"
        )
    if len(set(all_checkpoints)) != len(all_checkpoints):
        raise ValueError("a checkpoint appears in more than one seed source")
    require_outside_roots(
        {f"checkpoint_{index}": path for index, path in enumerate(all_checkpoints)},
        output=output_path,
        manifest=manifest_path,
    )
    precision = next(iter(precisions))
    averaged = np.mean(np.stack(matrices, axis=0), axis=0)
    rows = prediction_records(gold, averaged, model=args.model)
    write_predictions(output_path, rows)
    scorer_signature = checkpoint_ensemble_signature(
        all_checkpoints, precision=precision
    )
    manifest = build_manifest(
        run_id=output_path.stem,
        project_root=Path(__file__).resolve().parents[1],
        config={
            "model": args.model,
            "precision": precision,
            "aggregation": (
                "oof_equal_seed_held_checkpoint_mean;"
                "deployment_equal_all_seed_fold_checkpoints"
            ),
            "seeds": sorted(sources),
            "fixed_epoch": next(iter(epochs)),
        },
        input_files=(
            gold_path,
            fold_path,
            *sources.values(),
            *(oof_manifest_path(path) for path in sources.values()),
        ),
        extra={
            **oof_provenance_fields(
                prediction_path=output_path,
                gold_path=gold_path,
                fold_path=fold_path,
                rows=len(rows),
                scorer_name=args.model,
                scorer_signature=scorer_signature,
            ),
            "precision": precision,
            "fixed_epoch": next(iter(epochs)),
            "seed_source_contracts": source_contracts,
            "ensemble_size": len(all_checkpoints),
            "checkpoint_set_signature": checkpoint_set_signature(all_checkpoints),
            "checkpoint_directories": [str(path) for path in all_checkpoints],
        },
    )
    write_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "oof": str(output_path),
                "rows": len(rows),
                "seeds": sorted(sources),
                "ensemble_size": len(all_checkpoints),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
