from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.folds import load_folds
from src.data.load import load_jsonl
from src.evaluation.oof_provenance import (
    checkpoint_ensemble_signature,
    checkpoint_set_signature,
    oof_manifest_path,
    oof_provenance_fields,
)
from src.evaluation.predictions import canonical_prediction, read_predictions, write_predictions
from src.models.contracts import SCORER_ARCHITECTURE_VERSION
from src.utils.hashing import sha256_file
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import (
    require_distinct_paths,
    require_new_paths,
    require_outside_roots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge checkpoint-bound, disjoint fold predictions into one OOF JSONL."
    )
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", action="append", required=True, help="Repeat once per fold file.")
    parser.add_argument("--folds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_path = Path(args.gold).resolve()
    prediction_paths = [Path(value).resolve() for value in args.pred]
    fold_path = Path(args.folds).resolve()
    output_path = Path(args.output).resolve()
    manifest_path = oof_manifest_path(output_path)
    path_contract = {
        "gold": gold_path,
        "folds": fold_path,
        "output": output_path,
        "manifest": manifest_path,
    }
    path_contract.update(
        {f"prediction_{index}": path for index, path in enumerate(prediction_paths)}
    )
    require_distinct_paths(**path_contract)
    require_outside_roots(
        {
            f"checkpoint_{index}": path.parent
            for index, path in enumerate(prediction_paths)
        },
        output=output_path,
        manifest=manifest_path,
    )
    require_new_paths(output=output_path, manifest=manifest_path)
    gold = load_jsonl(gold_path)
    assignments = load_folds(fold_path)
    gold_hash = sha256_file(gold_path)
    folds_hash = sha256_file(fold_path)
    gold_ids = [record.id for record in gold]
    gold_id_set = set(gold_ids)
    missing_fold_ids = [record_id for record_id in gold_ids if record_id not in assignments]
    extra_fold_ids = sorted(set(assignments).difference(gold_id_set))
    if missing_fold_ids or extra_fold_ids:
        raise ValueError(
            "fold assignments must match gold ids exactly; "
            f"missing={missing_fold_ids[:5]}, extra={extra_fold_ids[:5]}"
        )
    prompt_by_id = {record.id: record.prompt_num for record in gold}

    merged: dict[str, dict] = {}
    seen_folds: set[int] = set()
    observed_precisions: set[str] = set()
    observed_epochs: set[int] = set()
    observed_seeds: set[int] = set()
    fold_files: list[dict[str, object]] = []
    model_contracts: set[tuple[str, str, int]] = set()
    for path in prediction_paths:
        fold_rows = read_predictions(path)
        row_fold_ids = {
            assignments[row["id"]]
            for row in fold_rows
            if row["id"] in assignments
        }
        unknown_ids = [row["id"] for row in fold_rows if row["id"] not in assignments]
        if unknown_ids:
            raise ValueError(f"fold prediction contains unknown ids: {unknown_ids[:5]}")
        if len(row_fold_ids) != 1:
            raise ValueError(
                f"each prediction file must contain exactly one held-out fold; "
                f"{path} has folds={sorted(row_fold_ids)}"
            )
        fold_id = int(next(iter(row_fold_ids)))
        if fold_id in seen_folds:
            raise ValueError(f"more than one prediction file claims fold {fold_id}")
        checkpoint_dir = path.parent
        checkpoint_provenance_path = checkpoint_dir / "checkpoint_provenance.json"
        if not checkpoint_provenance_path.is_file():
            raise FileNotFoundError(
                f"fold prediction lacks checkpoint provenance: {checkpoint_provenance_path}"
            )
        checkpoint_provenance = json.loads(
            checkpoint_provenance_path.read_text(encoding="utf-8")
        )
        if not isinstance(checkpoint_provenance, dict):
            raise ValueError(f"invalid checkpoint provenance: {checkpoint_provenance_path}")
        epoch = checkpoint_provenance.get("epoch")
        seed = checkpoint_provenance.get("seed")
        if (
            checkpoint_provenance.get("artifact_type")
            != "qwen_scorer_fold_checkpoint"
            or isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch <= 0
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or checkpoint_provenance.get("train_sha256") != gold_hash
            or checkpoint_provenance.get("folds_sha256") != folds_hash
        ):
            raise ValueError(f"invalid checkpoint provenance: {checkpoint_provenance_path}")
        recorded_oof = checkpoint_provenance.get("oof_file")
        if (
            not isinstance(recorded_oof, str)
            or (checkpoint_dir / recorded_oof).resolve() != path
            or checkpoint_provenance.get("oof_sha256") != sha256_file(path)
            or checkpoint_provenance.get("rows") != len(fold_rows)
            or checkpoint_provenance.get("scorer_architecture_version")
            != SCORER_ARCHITECTURE_VERSION
        ):
            raise ValueError(
                f"fold prediction does not match its checkpoint provenance: {path}"
            )
        if int(checkpoint_provenance.get("fold", -1)) != fold_id:
            raise ValueError(
                f"row fold {fold_id} does not match checkpoint fold for {path}"
            )
        precision = checkpoint_provenance.get("precision")
        if precision not in {"4bit", "bf16"}:
            raise ValueError(f"checkpoint has invalid OOF precision: {precision!r}")
        observed_precisions.add(str(precision))
        observed_epochs.add(epoch)
        observed_seeds.add(seed)
        head_config = json.loads(
            (checkpoint_dir / "scoring_head_config.json").read_text(encoding="utf-8")
        )
        if (
            not isinstance(head_config, dict)
            or head_config.get("scorer_architecture_version")
            != SCORER_ARCHITECTURE_VERSION
            or not isinstance(head_config.get("model_id"), str)
            or not head_config["model_id"]
            or not isinstance(head_config.get("model_revision"), str)
            or (
                re.fullmatch(r"[0-9a-fA-F]{40}", head_config["model_revision"])
                is None
            )
            or isinstance(head_config.get("max_length"), bool)
            or not isinstance(head_config.get("max_length"), int)
            or head_config["max_length"] <= 0
            or head_config.get("fold") != fold_id
            or head_config.get("seed") != seed
            or head_config.get("precision") != precision
            or head_config.get("train_sha256") != gold_hash
            or head_config.get("folds_sha256") != folds_hash
        ):
            raise ValueError(f"invalid scorer head contract: {checkpoint_dir}")
        model_contracts.add(
            (
                head_config["model_id"],
                head_config["model_revision"].lower(),
                head_config["max_length"],
            )
        )
        seen_folds.add(fold_id)
        fold_files.append(
            {
                "fold": fold_id,
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(fold_rows),
                "checkpoint_provenance_sha256": sha256_file(
                    checkpoint_provenance_path
                ),
                "precision": precision,
            }
        )
        for row in fold_rows:
            record_id = row["id"]
            if record_id in merged:
                raise ValueError(f"OOF id appears in more than one fold file: {record_id}")
            if record_id not in prompt_by_id:
                raise ValueError(f"prediction id is not present in gold data: {record_id}")
            if row["prompt_num"] != prompt_by_id[record_id]:
                raise ValueError(f"prompt mismatch for {record_id}")
            merged[record_id] = row

    missing = [record_id for record_id in gold_ids if record_id not in merged]
    if missing:
        raise ValueError(f"OOF files are missing {len(missing)} gold ids; first={missing[:5]}")
    if len(observed_precisions) != 1:
        raise ValueError(
            f"all fold OOF predictions must share one actual precision: {observed_precisions}"
        )
    if len(observed_epochs) != 1:
        raise ValueError(
            "all outer folds must use one preselected fixed epoch; "
            f"got epochs={sorted(observed_epochs)}"
        )
    if len(observed_seeds) != 1:
        raise ValueError(
            "build one complete OOF artifact per seed before seed ensembling; "
            f"got seeds={sorted(observed_seeds)}"
        )
    expected_folds = set(assignments.values())
    if seen_folds != expected_folds:
        raise ValueError(
            "prediction files must cover every configured fold exactly once; "
            f"expected={sorted(expected_folds)}, got={sorted(seen_folds)}"
        )
    if len(model_contracts) != 1:
        raise ValueError(f"fold checkpoints do not share one model contract: {model_contracts}")
    resolved_precision = next(iter(observed_precisions))
    rows = [
        canonical_prediction(
            record_id,
            prompt_by_id[record_id],
            merged[record_id]["prediction"],
            args.model,
        )
        for record_id in gold_ids
    ]
    scorer_signature = checkpoint_ensemble_signature(
        [path.parent for path in prediction_paths],
        precision=resolved_precision,
    )
    write_predictions(output_path, rows)

    project_root = Path.cwd().resolve()
    manifest = build_manifest(
        run_id=output_path.stem,
        project_root=project_root,
        config={
            "model": args.model,
            "precision": resolved_precision,
            "fixed_epoch": next(iter(observed_epochs)),
            "seed": next(iter(observed_seeds)),
        },
        input_files=(gold_path, fold_path, *prediction_paths),
        extra={
            **oof_provenance_fields(
                prediction_path=output_path,
                gold_path=gold_path,
                fold_path=fold_path,
                rows=len(rows),
                scorer_name=args.model,
                scorer_signature=scorer_signature,
            ),
            "fold_files": fold_files,
            "precision": resolved_precision,
            "fixed_epoch": next(iter(observed_epochs)),
            "seed": next(iter(observed_seeds)),
            "checkpoint_set_signature": checkpoint_set_signature(
                [path.parent for path in prediction_paths]
            ),
        },
    )
    write_manifest(manifest_path, manifest)
    print(json.dumps({"oof": str(output_path), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
