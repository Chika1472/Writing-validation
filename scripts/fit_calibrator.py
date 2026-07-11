from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration.affine import AffinePromptCalibrator, DOMAINS
from src.calibration.contracts import calibration_inference_code_contract
from src.data.folds import load_folds
from src.data.load import load_jsonl
from src.evaluation.metrics import gold_matrix, prediction_matrix
from src.evaluation.oof_provenance import (
    oof_manifest_path,
    validate_oof_provenance,
)
from src.evaluation.predictions import prediction_records, read_predictions, write_predictions
from src.utils.config import load_yaml, resolve_project_path
from src.utils.hashing import sha256_file
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import (
    require_distinct_paths,
    require_new_paths,
    require_outside_roots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit an affine + prompt residual OOF calibrator.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--gold", required=True, help="Gold JSONL used to create the OOF rows.")
    parser.add_argument("--pred", required=True, help="Canonical OOF prediction JSONL.")
    parser.add_argument("--output", required=True, help="Calibrator JSON path.")
    parser.add_argument("--calibrated-output", default=None)
    parser.add_argument(
        "--folds",
        required=True,
        help="Fold assignment JSONL bound to the required OOF provenance sidecar.",
    )
    parser.add_argument("--prompt-shrinkage", type=float, default=None)
    parser.add_argument("--fit-source", default="oof")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fit_source.lower() != "oof":
        raise ValueError("calibration must be fitted from out-of-fold predictions")
    config = load_yaml(args.config)
    calibration_config_path = resolve_project_path(config, "configs/calibration.yaml")
    calibration_config = load_yaml(calibration_config_path)
    if calibration_config.get("method") != "affine_prompt_shrinkage":
        raise ValueError("only calibration.method=affine_prompt_shrinkage is implemented")
    if calibration_config.get("positive_slope") is not True:
        raise ValueError("the implemented calibrator requires positive_slope=true")
    prompt_shrinkage = (
        args.prompt_shrinkage
        if args.prompt_shrinkage is not None
        else float(calibration_config["prompt_shrinkage"])
    )
    gold_path = Path(args.gold).resolve()
    prediction_path = Path(args.pred).resolve()
    output_path = Path(args.output).resolve()
    fold_path = Path(args.folds).resolve()
    output_manifest_path = output_path.with_suffix(".manifest.json")
    calibrated_output_path = (
        Path(args.calibrated_output).resolve() if args.calibrated_output else None
    )
    require_distinct_paths(
        data_config=Path(args.config).resolve(),
        calibration_config=calibration_config_path,
        gold=gold_path,
        prediction=prediction_path,
        prediction_manifest=oof_manifest_path(prediction_path),
        folds=fold_path,
        calibrator=output_path,
        calibrator_manifest=output_manifest_path,
        calibrated_output=calibrated_output_path,
    )
    require_new_paths(
        calibrator=output_path,
        calibrator_manifest=output_manifest_path,
        calibrated_output=calibrated_output_path,
    )
    gold = load_jsonl(gold_path)
    predictions = read_predictions(prediction_path)
    provenance = validate_oof_provenance(
        prediction_path=prediction_path,
        gold_path=gold_path,
        fold_path=fold_path,
    )
    if provenance.get("oof_level") != "base_model_oof":
        raise ValueError("standalone calibrator accepts only base_model_oof predictions")
    checkpoint_roots: dict[str, Path] = {}
    fold_files = provenance.get("fold_files", [])
    if isinstance(fold_files, list):
        for index, item in enumerate(fold_files):
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                checkpoint_roots[f"checkpoint_{index}"] = Path(item["path"]).resolve().parent
    require_outside_roots(
        checkpoint_roots,
        calibrator=output_path,
        calibrator_manifest=output_manifest_path,
        calibrated_output=calibrated_output_path,
    )
    if int(provenance["rows"]) != len(predictions) or len(predictions) != len(gold):
        raise ValueError(
            "OOF provenance, prediction, and gold row counts must be identical"
        )
    source_models = {str(row["model"]) for row in predictions}
    if len(source_models) != 1:
        raise ValueError(f"OOF predictions must contain one model name, got {source_models}")
    source_model = next(iter(source_models))
    if source_model != provenance["scorer_name"]:
        raise ValueError("OOF model name does not match its provenance sidecar")

    truth = gold_matrix(gold)
    raw = prediction_matrix(predictions, gold)
    gold_by_domain = {domain: truth[:, index] for index, domain in enumerate(DOMAINS)}
    pred_by_domain = {domain: raw[:, index] for index, domain in enumerate(DOMAINS)}
    prompts = [record.prompt_num for record in gold]
    calibrator = AffinePromptCalibrator.fit(
        gold_by_domain,
        pred_by_domain,
        prompts,
        prompt_shrinkage=prompt_shrinkage,
        clip_min=float(calibration_config.get("clip_min", 1.0)),
        clip_max=float(calibration_config.get("clip_max", 5.0)),
        fit_source="oof",
    )

    calibrator_payload = {
        **calibrator.to_dict(),
        "inference_code_contract": calibration_inference_code_contract(),
        "source": {
            "scorer_name": source_model,
            "scorer_signature": provenance["scorer_signature"],
            "oof_prediction_sha256": provenance["oof_sha256"],
            "oof_manifest_sha256": sha256_file(oof_manifest_path(prediction_path)),
            "folds_sha256": provenance["folds_sha256"],
        },
    }
    calibrated_output = None
    calibrated_rows = None
    if args.calibrated_output:
        assignments = load_folds(fold_path)
        gold_ids = {record.id for record in gold}
        missing_ids = [record.id for record in gold if record.id not in assignments]
        extra_ids = sorted(set(assignments).difference(gold_ids))
        if missing_ids or extra_ids:
            raise ValueError(
                "fold assignments must match gold ids exactly; "
                f"missing={missing_ids[:5]}, extra={extra_ids[:5]}"
            )
        fold_ids = np.asarray([assignments[record.id] for record in gold])
        matrix = np.full_like(raw, np.nan, dtype=float)
        prompt_array = np.asarray(prompts, dtype=str)
        for fold_id in np.unique(fold_ids):
            held_out = fold_ids == fold_id
            fit_rows = ~held_out
            if int(held_out.sum()) == 0 or int(fit_rows.sum()) < 2:
                raise ValueError(f"fold {fold_id!r} does not support calibration cross-fitting")
            fold_calibrator = AffinePromptCalibrator.fit(
                {
                    domain: gold_by_domain[domain][fit_rows]
                    for domain in DOMAINS
                },
                {
                    domain: pred_by_domain[domain][fit_rows]
                    for domain in DOMAINS
                },
                prompt_array[fit_rows].tolist(),
                prompt_shrinkage=prompt_shrinkage,
                clip_min=float(calibration_config.get("clip_min", 1.0)),
                clip_max=float(calibration_config.get("clip_max", 5.0)),
                fit_source="cross_fit_train_oof",
            )
            transformed = fold_calibrator.transform(
                {
                    domain: pred_by_domain[domain][held_out]
                    for domain in DOMAINS
                },
                prompt_array[held_out].tolist(),
            )
            for domain_index, domain in enumerate(DOMAINS):
                matrix[held_out, domain_index] = transformed[domain]
        if not np.isfinite(matrix).all():
            raise RuntimeError("cross-fitted calibration left one or more OOF rows unset")
        calibrated_rows = prediction_records(
            gold, matrix, model="cross_fitted_calibrated_oof"
        )
        assert calibrated_output_path is not None
        calibrated_output = calibrated_output_path

    public_config = {
        "data": {key: value for key, value in config.items() if not key.startswith("_")},
        "calibration": {
            key: value for key, value in calibration_config.items() if not key.startswith("_")
        },
    }
    temporary_calibrator = output_path.with_name(output_path.name + ".tmp-artifact")
    temporary_manifest = output_manifest_path.with_name(
        output_manifest_path.name + ".tmp-artifact"
    )
    temporary_calibrated = (
        calibrated_output_path.with_name(calibrated_output_path.name + ".tmp-artifact")
        if calibrated_output_path
        else None
    )
    require_distinct_paths(
        gold=gold_path,
        prediction=prediction_path,
        prediction_manifest=oof_manifest_path(prediction_path),
        folds=fold_path,
        calibrator=output_path,
        calibrator_manifest=output_manifest_path,
        calibrated_output=calibrated_output_path,
        temporary_calibrator=temporary_calibrator,
        temporary_manifest=temporary_manifest,
        temporary_calibrated=temporary_calibrated,
    )
    require_outside_roots(
        checkpoint_roots,
        temporary_calibrator=temporary_calibrator,
        temporary_manifest=temporary_manifest,
        temporary_calibrated=temporary_calibrated,
    )
    require_new_paths(
        temporary_calibrator=temporary_calibrator,
        temporary_manifest=temporary_manifest,
        temporary_calibrated=temporary_calibrated,
    )
    try:
        temporary_calibrator.parent.mkdir(parents=True, exist_ok=True)
        temporary_calibrator.write_text(
            json.dumps(calibrator_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        calibrated_sha256 = None
        if temporary_calibrated is not None:
            assert calibrated_rows is not None
            write_predictions(temporary_calibrated, calibrated_rows)
            calibrated_sha256 = sha256_file(temporary_calibrated)
        manifest = build_manifest(
            run_id=output_path.stem,
            project_root=config["_project_root"],
            config=public_config,
            input_files=(
                gold_path,
                prediction_path,
                oof_manifest_path(prediction_path),
                fold_path,
            ),
            extra={
                "artifact_type": "oof_affine_prompt_calibrator",
                "calibrator_file": output_path.name,
                "calibrator_creation_path": str(output_path),
                "calibrator_sha256": sha256_file(temporary_calibrator),
                "source_scorer_name": source_model,
                "source_scorer_signature": provenance["scorer_signature"],
                "calibrated_output": (
                    str(calibrated_output) if calibrated_output else None
                ),
                "calibrated_output_sha256": calibrated_sha256,
                "calibrated_output_mode": (
                    "cross_fitted" if calibrated_output else None
                ),
            },
        )
        write_manifest(temporary_manifest, manifest)
        temporary_calibrator.replace(output_path)
        if temporary_calibrated is not None:
            assert calibrated_output_path is not None
            temporary_calibrated.replace(calibrated_output_path)
        temporary_manifest.replace(output_manifest_path)
    finally:
        for temporary in (
            temporary_calibrator,
            temporary_calibrated,
            temporary_manifest,
        ):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(json.dumps({"calibrator": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
