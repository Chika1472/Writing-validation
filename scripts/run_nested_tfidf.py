from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.contracts import baseline_inference_code_contract
from src.baselines.nested_tuning import (
    nested_tfidf_oof,
    nested_tuning_code_contract,
)
from src.data.folds import load_folds
from src.data.load import load_jsonl
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.oof_provenance import oof_manifest_path, oof_provenance_fields
from src.evaluation.predictions import prediction_records, write_predictions
from src.utils.config import load_yaml, resolve_project_path
from src.utils.hashing import sha256_file, sha256_json
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune TF-IDF Ridge inside every outer-training partition and create "
            "strict outer OOF plus hash-bound target fold models."
        )
    )
    parser.add_argument("--config", default="configs/tfidf_nested.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--folds", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="nested_tfidf_ridge")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not isinstance(args.model_name, str) or not args.model_name.strip():
        raise ValueError("--model-name must be nonempty")
    config_path = Path(args.config).resolve()
    data_config_path = Path(args.data_config).resolve()
    fold_path = Path(args.folds).resolve()
    output_dir = Path(args.output_dir).resolve()
    oof_path = output_dir / "nested_tfidf_oof.jsonl"
    oof_sidecar_path = oof_manifest_path(oof_path)
    selection_path = output_dir / "selection_report.json"
    model_artifact_path = output_dir / "nested_tfidf_ensemble.json"
    run_manifest_path = output_dir / "manifest.json"
    require_distinct_paths(
        config=config_path,
        data_config=data_config_path,
        folds=fold_path,
        output_dir=output_dir,
        oof=oof_path,
        oof_manifest=oof_sidecar_path,
        selection=selection_path,
        model_artifact=model_artifact_path,
        run_manifest=run_manifest_path,
    )
    require_new_paths(output_dir=output_dir)

    config = load_yaml(config_path)
    data_config = load_yaml(data_config_path)
    train_path = resolve_project_path(data_config, data_config["paths"]["train"])
    if train_path in {config_path, data_config_path, fold_path, output_dir}:
        raise ValueError("training data must be distinct from configs, folds, and output")
    immutable_inputs = {
        "config_sha256": sha256_file(config_path),
        "data_config_sha256": sha256_file(data_config_path),
        "train_sha256": sha256_file(train_path),
        "folds_sha256": sha256_file(fold_path),
    }
    tuning_contract = nested_tuning_code_contract()
    inference_contract = baseline_inference_code_contract()
    train = load_jsonl(train_path)
    assignments = load_folds(fold_path)
    train_ids = [record.id for record in train]
    missing = sorted(set(train_ids).difference(assignments))
    extra = sorted(set(assignments).difference(train_ids))
    if missing or extra:
        raise ValueError(
            "outer folds must match training ids exactly; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    result = nested_tfidf_oof(train, assignments, config)
    current_inputs = {
        "config_sha256": sha256_file(config_path),
        "data_config_sha256": sha256_file(data_config_path),
        "train_sha256": sha256_file(train_path),
        "folds_sha256": sha256_file(fold_path),
    }
    if current_inputs != immutable_inputs:
        raise RuntimeError("nested TF-IDF inputs changed while tuning was in progress")
    if (
        nested_tuning_code_contract() != tuning_contract
        or baseline_inference_code_contract() != inference_contract
    ):
        raise RuntimeError("nested TF-IDF source code changed while tuning was in progress")
    scorer_name = args.model_name.strip()
    oof_rows = prediction_records(
        train, result.oof_predictions, model=scorer_name
    )
    selection_report = {
        "artifact_type": "nested_tfidf_selection_report",
        "artifact_version": 1,
        "candidate_branch": True,
        "auto_promoted": False,
        "fit_source": "outer_train_refit_after_nested_inner_cv",
        "selection_scope": "outer_train_only",
        "outer_held_labels_used_for_selection": False,
        "selection_metric": "trait_mean_rmse",
        "selection_config": result.normalized_config,
        "selection_config_sha256": sha256_json(result.normalized_config),
        "training_source": {
            **immutable_inputs,
            "rows": len(train),
        },
        "tuning_code_contract": tuning_contract,
        "outer_folds": result.outer_reports,
        "oof_metrics": evaluate_predictions(train, oof_rows),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    write_predictions(oof_path, oof_rows)
    selection_path.write_text(
        json.dumps(
            selection_report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fold_model_paths: dict[str, Path] = {}
    for fold_id, model in sorted(result.fold_models.items(), key=lambda item: int(item[0])):
        model_path = output_dir / f"nested_tfidf_fold{fold_id}.joblib"
        joblib.dump(model, model_path, compress=3)
        fold_model_paths[fold_id] = model_path
    fold_model_hashes = {
        fold_id: sha256_file(path)
        for fold_id, path in sorted(fold_model_paths.items(), key=lambda item: int(item[0]))
    }
    signature_payload = {
        "scorer": scorer_name,
        "method": "nested_inner_cv_tfidf_ridge",
        "fit_source": "outer_train_refit_after_nested_inner_cv",
        "train_sha256": immutable_inputs["train_sha256"],
        "folds_sha256": immutable_inputs["folds_sha256"],
        "raw_config_sha256": immutable_inputs["config_sha256"],
        "selection_config": result.normalized_config,
        "selection_config_sha256": sha256_json(result.normalized_config),
        "selection_report_sha256": sha256_file(selection_path),
        "fold_model_hashes": fold_model_hashes,
        "tuning_code_contract": tuning_contract,
        "inference_code_contract": inference_contract,
    }
    scorer_signature = sha256_json(signature_payload)
    model_artifact = {
        "artifact_type": "baseline_fold_ensemble",
        "artifact_version": 2,
        "model_family": "tfidf_ridge_nested_cv",
        "candidate_branch": True,
        "auto_promoted": False,
        "fit_source": "outer_train_refit_after_nested_inner_cv",
        "aggregation": "equal_weight_mean",
        "scorer_name": scorer_name,
        "scorer_signature": scorer_signature,
        "signature_payload": signature_payload,
        "selection_report": selection_path.name,
        "selection_report_sha256": sha256_file(selection_path),
        "fold_models": {
            fold_id: {
                "file": path.name,
                "sha256": fold_model_hashes[fold_id],
            }
            for fold_id, path in sorted(
                fold_model_paths.items(), key=lambda item: int(item[0])
            )
        },
    }
    model_artifact_path.write_text(
        json.dumps(model_artifact, ensure_ascii=False, allow_nan=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    public_config = {
        "nested_tfidf": result.normalized_config,
        "data": {
            key: value for key, value in data_config.items() if not key.startswith("_")
        },
    }
    artifact_inputs = (
        config_path,
        data_config_path,
        train_path,
        fold_path,
        selection_path,
        model_artifact_path,
        *fold_model_paths.values(),
    )
    oof_manifest = build_manifest(
        run_id=oof_path.stem,
        project_root=config["_project_root"],
        config=public_config,
        input_files=artifact_inputs,
        extra={
            **oof_provenance_fields(
                prediction_path=oof_path,
                gold_path=train_path,
                fold_path=fold_path,
                rows=len(oof_rows),
                scorer_name=scorer_name,
                scorer_signature=scorer_signature,
                oof_level="base_model_oof",
            ),
            "candidate_branch": True,
            "auto_promoted": False,
            "nested_inner_cv": True,
            "selection_scope": "outer_train_only",
            "outer_held_labels_used_for_selection": False,
            "selection_report": selection_path.name,
            "selection_report_sha256": sha256_file(selection_path),
            "model_artifact": model_artifact_path.name,
            "model_artifact_sha256": sha256_file(model_artifact_path),
        },
    )
    write_manifest(oof_sidecar_path, oof_manifest)
    run_manifest = build_manifest(
        run_id=output_dir.name,
        project_root=config["_project_root"],
        config=public_config,
        input_files=(config_path, data_config_path, train_path, fold_path),
        extra={
            "artifact_type": "nested_tfidf_run",
            "candidate_branch": True,
            "auto_promoted": False,
            "oof_file": oof_path.name,
            "oof_sha256": sha256_file(oof_path),
            "oof_manifest": oof_sidecar_path.name,
            "oof_manifest_sha256": sha256_file(oof_sidecar_path),
            "selection_report": selection_path.name,
            "selection_report_sha256": sha256_file(selection_path),
            "model_artifact": model_artifact_path.name,
            "model_artifact_sha256": sha256_file(model_artifact_path),
            "fold_model_hashes": fold_model_hashes,
            "scorer_name": scorer_name,
            "scorer_signature": scorer_signature,
        },
    )
    write_manifest(run_manifest_path, run_manifest)
    print(
        json.dumps(
            {
                "oof": str(oof_path),
                "model_artifact": str(model_artifact_path),
                "selection_report": str(selection_path),
                "rows": len(oof_rows),
                "candidate_only": True,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
