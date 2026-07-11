from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assessment.artifact import build_deployment_artifact, load_deployment_artifact
from src.assessment.cache import (
    assessment_cache_manifest_path,
    load_assessment_cache,
    validate_cache_source,
)
from src.assessment.questions import QUESTION_VERSION, QUESTIONS_SHA256
from src.assessment.ridge import nested_oof_ridge
from src.data.folds import load_folds
from src.data.load import load_jsonl
from src.evaluation.metrics import evaluate_predictions, gold_matrix
from src.evaluation.oof_provenance import oof_manifest_path, oof_provenance_fields
from src.evaluation.predictions import prediction_records, write_predictions
from src.utils.config import load_yaml
from src.utils.hashing import sha256_file
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a nested-CV assessment-question Ridge candidate and its exact "
            "fold-ensemble deployment artifact."
        )
    )
    parser.add_argument("--config", default="configs/assessment_questions.yaml")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    branch_config = config["branch"]
    ridge_config = config["ridge"]
    if branch_config.get("question_version") != QUESTION_VERSION:
        raise ValueError("assessment config question_version does not match code")
    if branch_config.get("candidate_only") is not True or branch_config.get(
        "auto_promote"
    ) is not False:
        raise ValueError("assessment branch must remain candidate-only and non-promoting")
    if ridge_config.get("selection_metric") != "rmse":
        raise ValueError("assessment nested selection implements only RMSE")
    if ridge_config.get("normalization") != "inner_train_standardization":
        raise ValueError("assessment normalization must be fit inside each CV fit split")

    gold_path = Path(args.gold).resolve()
    cache_path = Path(args.cache).resolve()
    cache_manifest_path = assessment_cache_manifest_path(cache_path)
    fold_path = Path(args.folds).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_path = output_dir / "assessment_ridge.json"
    model_manifest_path = output_dir / "assessment_ridge.manifest.json"
    oof_path = output_dir / "assessment_oof.jsonl"
    oof_sidecar_path = oof_manifest_path(oof_path)
    report_path = output_dir / "selection_report.json"
    require_distinct_paths(
        config=config_path,
        gold=gold_path,
        cache=cache_path,
        cache_manifest=cache_manifest_path,
        folds=fold_path,
        model=model_path,
        model_manifest=model_manifest_path,
        oof=oof_path,
        oof_manifest=oof_sidecar_path,
        report=report_path,
    )
    require_new_paths(output_dir=output_dir)

    gold = load_jsonl(gold_path)
    cache = load_assessment_cache(cache_path)
    validate_cache_source(cache, gold, gold_path)
    assignments = load_folds(fold_path)
    gold_ids = [record.id for record in gold]
    missing_folds = [record_id for record_id in gold_ids if record_id not in assignments]
    extra_folds = sorted(set(assignments).difference(gold_ids))
    if missing_folds or extra_folds:
        raise ValueError(
            "fold assignments must match assessment gold ids exactly; "
            f"missing={missing_folds[:5]}, extra={extra_folds[:5]}"
        )

    result = nested_oof_ridge(
        cache.probabilities,
        gold_matrix(gold),
        [assignments[record_id] for record_id in gold_ids],
        gold_ids,
        question_counts=ridge_config["question_count_candidates"],
        alphas=ridge_config["alpha_grid"],
        clip_min=float(ridge_config["clip_min"]),
        clip_max=float(ridge_config["clip_max"]),
    )
    scorer_name = args.model_name or str(branch_config["name"])
    selection_config = {
        "question_count_candidates": list(ridge_config["question_count_candidates"]),
        "alpha_grid": list(ridge_config["alpha_grid"]),
        "selection_metric": "rmse",
        "normalization": "inner_train_standardization",
        "held_fold_exclusion": "strict",
    }
    training_source = {
        "gold_sha256": sha256_file(gold_path),
        "cache_sha256": sha256_file(cache_path),
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "folds_sha256": sha256_file(fold_path),
        "config_sha256": sha256_file(config_path),
        "rows": len(gold),
    }
    artifact = build_deployment_artifact(
        scorer_name=scorer_name,
        feature_signature=cache.feature_signature,
        feature_contract=cache.manifest["feature_signature_payload"],
        fold_models=result.fold_models,
        outer_selection=result.outer_reports,
        clip_min=float(ridge_config["clip_min"]),
        clip_max=float(ridge_config["clip_max"]),
        training_source=training_source,
        selection_config=selection_config,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    model_path.write_text(
        json.dumps(artifact, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    load_deployment_artifact(model_path)
    oof_rows = prediction_records(
        gold,
        result.oof_predictions,
        model=scorer_name,
    )
    write_predictions(oof_path, oof_rows)
    report = {
        "artifact_type": "assessment_nested_cv_report",
        "candidate_branch": True,
        "auto_promoted": False,
        "question_version": QUESTION_VERSION,
        "questions_sha256": QUESTIONS_SHA256,
        "feature_signature": cache.feature_signature,
        "selection_config": selection_config,
        "outer_folds": result.outer_reports,
        "oof_metrics": evaluate_predictions(gold, oof_rows),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )

    public_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    input_files = (
        config_path,
        gold_path,
        cache_path,
        cache_manifest_path,
        fold_path,
    )
    model_manifest = build_manifest(
        run_id=model_path.stem,
        project_root=config["_project_root"],
        config=public_config,
        input_files=input_files,
        extra={
            "artifact_type": "assessment_ridge_deployment_manifest",
            "candidate_branch": True,
            "auto_promoted": False,
            "model_file": model_path.name,
            "model_sha256": sha256_file(model_path),
            "scorer_name": scorer_name,
            "scorer_signature": artifact["artifact_signature"],
            "feature_signature": cache.feature_signature,
            "selection_report": report_path.name,
            "selection_report_sha256": sha256_file(report_path),
        },
    )
    write_manifest(model_manifest_path, model_manifest)
    oof_manifest = build_manifest(
        run_id=oof_path.stem,
        project_root=config["_project_root"],
        config=public_config,
        input_files=(*input_files, model_path, report_path),
        extra={
            **oof_provenance_fields(
                prediction_path=oof_path,
                gold_path=gold_path,
                fold_path=fold_path,
                rows=len(oof_rows),
                scorer_name=scorer_name,
                scorer_signature=artifact["artifact_signature"],
            ),
            "candidate_branch": True,
            "auto_promoted": False,
            "nested_inner_cv": True,
            "feature_signature": cache.feature_signature,
            "deployment_model": model_path.name,
            "deployment_model_sha256": sha256_file(model_path),
            "selection_report": report_path.name,
            "selection_report_sha256": sha256_file(report_path),
        },
    )
    write_manifest(oof_sidecar_path, oof_manifest)
    print(
        json.dumps(
            {
                "model": str(model_path),
                "oof": str(oof_path),
                "report": str(report_path),
                "candidate_only": True,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
