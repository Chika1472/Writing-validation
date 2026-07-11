from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.folds import load_folds
from src.data.load import load_jsonl
from src.ensemble.crossfit import cross_fit_simplex_stacker
from src.ensemble.contracts import stacker_inference_code_contract
from src.evaluation.metrics import evaluate_predictions, gold_matrix, prediction_matrix
from src.evaluation.oof_provenance import (
    oof_manifest_path,
    oof_provenance_fields,
    validate_oof_provenance,
)
from src.evaluation.predictions import (
    prediction_records,
    read_canonical_predictions,
    write_predictions,
)
from src.utils.config import load_yaml
from src.utils.hashing import sha256_file, sha256_json
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import (
    require_distinct_paths,
    require_new_paths,
    require_outside_roots,
)


def _source(value: str) -> tuple[str, Path]:
    alias, separator, path = value.partition("=")
    if not separator or not alias.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--source must have the form ALIAS=PATH")
    return alias.strip(), Path(path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a multi-source trait simplex stacker from genuine base OOF files."
    )
    parser.add_argument("--config", default="configs/stacker.yaml")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--source", action="append", type=_source, required=True)
    parser.add_argument("--output", required=True, help="Deployment stacker JSON.")
    parser.add_argument("--crossfit-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--model-name", required=True)
    return parser.parse_args()


def _weight_stability(
    fold_reports: tuple[dict, ...], source_order: tuple[str, ...]
) -> dict[str, dict[str, dict[str, float | int]]]:
    output = {}
    for trait in ("content", "organization", "expression"):
        output[trait] = {}
        for source in source_order:
            values = np.asarray(
                [report["stacker"]["weights"][trait][source] for report in fold_reports],
                dtype=float,
            )
            output[trait][source] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "boundary_count": int(((values <= 1e-8) | (values >= 1.0 - 1e-8)).sum()),
            }
    return output


def main() -> None:
    args = parse_args()
    if len(args.source) < 2:
        raise ValueError("the simplex stacker requires at least two --source values")
    source_paths = dict(args.source)
    if len(source_paths) != len(args.source):
        raise ValueError("source aliases must be unique")
    source_order = tuple(alias for alias, _ in args.source)
    config_path = Path(args.config).resolve()
    gold_path = Path(args.gold).resolve()
    fold_path = Path(args.folds).resolve()
    output_path = Path(args.output).resolve()
    output_manifest_path = output_path.with_suffix(".manifest.json")
    crossfit_path = Path(args.crossfit_output).resolve()
    crossfit_manifest_path = oof_manifest_path(crossfit_path)
    report_path = Path(args.report).resolve()
    paths = {
        "config": config_path,
        "gold": gold_path,
        "folds": fold_path,
        "stacker": output_path,
        "stacker_manifest": output_manifest_path,
        "crossfit": crossfit_path,
        "crossfit_manifest": crossfit_manifest_path,
        "report": report_path,
    }
    paths.update({f"source_{alias}": path for alias, path in source_paths.items()})
    paths.update(
        {
            f"source_manifest_{alias}": oof_manifest_path(path)
            for alias, path in source_paths.items()
        }
    )
    require_distinct_paths(**paths)
    require_new_paths(
        stacker=output_path,
        stacker_manifest=output_manifest_path,
        crossfit=crossfit_path,
        crossfit_manifest=crossfit_manifest_path,
        report=report_path,
    )

    config = load_yaml(config_path)
    if config.get("method") not in {"trait_mse_simplex", "trait_two_source_mse_simplex"}:
        raise ValueError("stacker method must be trait_mse_simplex")
    calibration_config = config.get("calibration", {})
    if (
        calibration_config.get("method") != "affine_prompt_shrinkage"
        or calibration_config.get("positive_slope") is not True
    ):
        raise ValueError("stacker requires positive-slope affine_prompt_shrinkage")

    gold = load_jsonl(gold_path)
    assignments = load_folds(fold_path)
    gold_ids = {record.id for record in gold}
    missing = [record.id for record in gold if record.id not in assignments]
    extra = sorted(set(assignments).difference(gold_ids))
    if missing or extra:
        raise ValueError(
            "fold assignments must match gold exactly; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    source_matrices = {}
    source_contracts = {}
    source_rows = {}
    checkpoint_roots: dict[str, Path] = {}
    for alias in source_order:
        path = source_paths[alias]
        provenance = validate_oof_provenance(
            prediction_path=path,
            gold_path=gold_path,
            fold_path=fold_path,
        )
        if provenance.get("oof_level") != "base_model_oof":
            raise ValueError(
                f"source {alias} is not base-model OOF; recursive meta stacking is forbidden"
            )
        rows = read_canonical_predictions(path)
        if len(rows) != len(gold) or int(provenance["rows"]) != len(rows):
            raise ValueError(f"OOF row count mismatch for source {alias}")
        row_models = {row["model"] for row in rows}
        if row_models != {provenance["scorer_name"]}:
            raise ValueError(f"OOF model/provenance mismatch for source {alias}")
        source_rows[alias] = rows
        source_matrices[alias] = prediction_matrix(rows, gold)
        source_contracts[alias] = {
            "scorer_name": provenance["scorer_name"],
            "scorer_signature": provenance["scorer_signature"],
            "oof_sha256": provenance["oof_sha256"],
            "oof_manifest_sha256": sha256_file(oof_manifest_path(path)),
        }
        fold_files = provenance.get("fold_files", [])
        if isinstance(fold_files, list):
            for index, item in enumerate(fold_files):
                if isinstance(item, dict) and isinstance(item.get("path"), str):
                    checkpoint_roots[f"{alias}_checkpoint_{index}"] = (
                        Path(item["path"]).resolve().parent
                    )

    require_outside_roots(
        checkpoint_roots,
        stacker=output_path,
        stacker_manifest=output_manifest_path,
        crossfit=crossfit_path,
        crossfit_manifest=crossfit_manifest_path,
        report=report_path,
    )

    result = cross_fit_simplex_stacker(
        gold_matrix(gold),
        source_matrices,
        [assignments[record.id] for record in gold],
        [record.prompt_num for record in gold],
        source_order=source_order,
        epsilon=float(config.get("epsilon", 1e-12)),
        prompt_shrinkage=float(calibration_config.get("prompt_shrinkage", 20.0)),
        clip_min=float(calibration_config.get("clip_min", 1.0)),
        clip_max=float(calibration_config.get("clip_max", 5.0)),
    )
    signature_payload = {
        "artifact_version": 1,
        "method": "trait_mse_simplex",
        "fit_source": "base_oof",
        "scorer_name": args.model_name,
        "source_order": list(source_order),
        "source_contracts": source_contracts,
        "stacker": result.final_stacker.to_dict(),
        "calibrator": result.final_calibrator.to_dict(),
        "gold_sha256": sha256_file(gold_path),
        "folds_sha256": sha256_file(fold_path),
        "inference_code_contract": stacker_inference_code_contract(),
        "config": {
            key: value for key, value in config.items() if not key.startswith("_")
        },
    }
    stacker_signature = sha256_json(signature_payload)
    artifact = {
        "artifact_type": "trait_simplex_stacker",
        **signature_payload,
        "stacker_signature": stacker_signature,
        "evaluation_scope": "level1_meta_crossfit_not_fully_nested",
    }

    stacked_rows = prediction_records(
        gold,
        result.cross_fitted_predictions,
        model=args.model_name,
    )
    report = {
        "evaluation_scope": "level1_meta_crossfit_not_fully_nested",
        "warning": (
            "Meta weights/calibration are cross-fitted, but base scorers were not "
            "retrained in a fully nested outer loop."
        ),
        "sources": {
            alias: evaluate_predictions(gold, rows)
            for alias, rows in source_rows.items()
        },
        "stacked_crossfit": evaluate_predictions(gold, stacked_rows),
        "fold_reports": list(result.fold_reports),
        "weight_stability": _weight_stability(result.fold_reports, source_order),
        "final_stacker": result.final_stacker.to_dict(),
        "final_calibrator": result.final_calibrator.to_dict(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_predictions(crossfit_path, stacked_rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    input_files = (
        config_path,
        gold_path,
        fold_path,
        *source_paths.values(),
        *(oof_manifest_path(path) for path in source_paths.values()),
    )
    stacker_manifest = build_manifest(
        run_id=output_path.stem,
        project_root=Path.cwd(),
        config=signature_payload["config"],
        input_files=input_files,
        extra={
            "artifact_type": "trait_simplex_stacker_manifest",
            "stacker_file": output_path.name,
            "stacker_sha256": sha256_file(output_path),
            "stacker_signature": stacker_signature,
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
        },
    )
    write_manifest(output_manifest_path, stacker_manifest)
    crossfit_manifest = build_manifest(
        run_id=crossfit_path.stem,
        project_root=Path.cwd(),
        config=signature_payload["config"],
        input_files=(*input_files, output_path),
        extra={
            **oof_provenance_fields(
                prediction_path=crossfit_path,
                gold_path=gold_path,
                fold_path=fold_path,
                rows=len(stacked_rows),
                scorer_name=args.model_name,
                scorer_signature=stacker_signature,
                oof_level="level1_meta_crossfit_not_fully_nested",
            ),
            "stacker_file": output_path.name,
            "stacker_sha256": sha256_file(output_path),
        },
    )
    write_manifest(crossfit_manifest_path, crossfit_manifest)
    print(
        json.dumps(
            {
                "stacker": str(output_path),
                "crossfit": str(crossfit_path),
                "report": str(report_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
