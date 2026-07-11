from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assessment.artifact import load_deployment_artifact
from src.assessment.cache import (
    assessment_cache_manifest_path,
    load_assessment_cache,
    validate_cache_source,
)
from src.assessment.ridge import predict_assessment_fold_ensemble
from src.data.load import load_inference_jsonl
from src.evaluation.prediction_provenance import (
    prediction_manifest_path,
    prediction_provenance_fields,
)
from src.evaluation.predictions import prediction_records, write_predictions
from src.utils.hashing import sha256_file
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the isolated assessment-question fold-ensemble candidate."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    cache_path = Path(args.cache).resolve()
    cache_manifest_path = assessment_cache_manifest_path(cache_path)
    model_path = Path(args.model).resolve()
    model_manifest_path = model_path.with_suffix(".manifest.json")
    output_path = Path(args.output).resolve()
    output_manifest_path = prediction_manifest_path(output_path)
    require_distinct_paths(
        input=input_path,
        cache=cache_path,
        cache_manifest=cache_manifest_path,
        model=model_path,
        model_manifest=model_manifest_path,
        output=output_path,
        output_manifest=output_manifest_path,
    )
    require_new_paths(output=output_path, output_manifest=output_manifest_path)

    artifact = load_deployment_artifact(model_path)
    if not model_manifest_path.is_file():
        raise FileNotFoundError(
            f"assessment deployment manifest is required: {model_manifest_path}"
        )
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(model_manifest, dict)
        or model_manifest.get("artifact_type")
        != "assessment_ridge_deployment_manifest"
        or model_manifest.get("candidate_branch") is not True
        or model_manifest.get("auto_promoted") is not False
        or model_manifest.get("model_file") != model_path.name
        or model_manifest.get("model_sha256") != sha256_file(model_path)
        or model_manifest.get("scorer_name") != artifact["scorer_name"]
        or model_manifest.get("scorer_signature") != artifact["artifact_signature"]
        or model_manifest.get("feature_signature") != artifact["feature_signature"]
    ):
        raise ValueError("assessment deployment manifest does not match its model")

    records = load_inference_jsonl(input_path)
    cache = load_assessment_cache(cache_path)
    validate_cache_source(cache, records, input_path)
    if cache.feature_signature != artifact["feature_signature"]:
        raise ValueError(
            "assessment cache was produced by a different question/model/tokenizer contract"
        )
    matrix = predict_assessment_fold_ensemble(
        cache.probabilities,
        artifact["fold_models"],
        clip_min=float(artifact["clip_min"]),
        clip_max=float(artifact["clip_max"]),
    )
    scorer_name = str(artifact["scorer_name"])
    rows = prediction_records(records, matrix, model=scorer_name)
    write_predictions(output_path, rows)
    manifest = build_manifest(
        run_id=output_path.stem,
        project_root=Path.cwd(),
        config={
            "branch": "assessment_question_ridge_candidate",
            "candidate_only": True,
            "auto_promote": False,
            "model": str(model_path),
            "feature_signature": cache.feature_signature,
        },
        input_files=(
            input_path,
            cache_path,
            cache_manifest_path,
            model_path,
            model_manifest_path,
        ),
        extra={
            **prediction_provenance_fields(
                prediction_path=output_path,
                input_path=input_path,
                rows=len(rows),
                scorer_name=scorer_name,
                scorer_signature=artifact["artifact_signature"],
                model_artifact=model_path,
            ),
            "candidate_branch": True,
            "auto_promoted": False,
            "feature_signature": cache.feature_signature,
            "cache_sha256": sha256_file(cache_path),
            "cache_manifest_sha256": sha256_file(cache_manifest_path),
        },
    )
    write_manifest(output_manifest_path, manifest)
    print(
        json.dumps(
            {
                "predictions": str(output_path),
                "rows": len(rows),
                "candidate_only": True,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
