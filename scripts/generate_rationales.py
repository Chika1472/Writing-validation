from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load import load_inference_jsonl
from src.evaluation.prediction_provenance import (
    prediction_manifest_path,
    validate_prediction_provenance,
)
from src.evaluation.predictions import read_canonical_predictions
from src.inference.rationale_generator import (
    generate_rationale_for_record,
    load_rationale_generator,
    rationale_checkpoint_files,
    resolve_rationale_checkpoint,
)
from src.rationale.parsing import serialize_rationales
from src.utils.hashing import sha256_file
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths, require_outside_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grounded rationales with adapter and deterministic fallback."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--precision", choices=("4bit", "bf16"), default="4bit")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    score_path = Path(args.scores).resolve()
    score_manifest_path = prediction_manifest_path(score_path)
    checkpoint = resolve_rationale_checkpoint(args.checkpoint)
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    manifest_path = output_path.with_suffix(".manifest.json")
    require_distinct_paths(
        input=input_path,
        scores=score_path,
        score_manifest=score_manifest_path,
        checkpoint=checkpoint,
        output=output_path,
        report=report_path,
        manifest=manifest_path,
    )
    require_outside_roots(
        {"rationale_checkpoint": checkpoint},
        output=output_path,
        report=report_path,
        manifest=manifest_path,
    )
    require_new_paths(output=output_path, report=report_path, manifest=manifest_path)
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")

    records = load_inference_jsonl(input_path)
    score_manifest = validate_prediction_provenance(score_path)
    if score_manifest.get("input_sha256") != sha256_file(input_path):
        raise ValueError("scores do not belong to the supplied essay input")
    score_rows = read_canonical_predictions(score_path)
    if {row["model"] for row in score_rows} != {score_manifest["scorer_name"]}:
        raise ValueError("score model/provenance mismatch")
    score_by_id = {row["id"]: row for row in score_rows}
    ids = {record.id for record in records}
    missing = [record.id for record in records if record.id not in score_by_id]
    extra = sorted(set(score_by_id).difference(ids))
    if missing or extra:
        raise ValueError(f"score/input id mismatch: missing={missing[:5]}, extra={extra[:5]}")

    loaded = load_rationale_generator(
        checkpoint,
        precision=args.precision,
        allow_download=args.allow_download,
    )
    rows = []
    fallback_count = 0
    for record in records:
        score_row = score_by_id[record.id]
        if score_row["prompt_num"] != record.prompt_num:
            raise ValueError(f"prompt mismatch for {record.id}")
        scores = {
            trait: float(score_row["prediction"][trait])
            for trait in ("content", "organization", "expression")
        }
        result = generate_rationale_for_record(
            loaded,
            record,
            scores,
            max_attempts=args.max_attempts,
        )
        serialize_rationales(result.rationales)
        fallback_count += int(result.fallback_used)
        rows.append(
            {
                "id": record.id,
                "prompt_num": record.prompt_num,
                "conditioned_scores": scores,
                "rationales": result.rationales,
                "fallback_used": result.fallback_used,
                "attempts": list(result.attempts),
                "evidence": result.evidence,
            }
        )
    generator_signature = loaded.generator_signature
    generator_model_id = loaded.model_id
    generator_model_revision = loaded.model_revision
    del loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + "\n"
            )
    fallback_rate = fallback_count / len(rows) if rows else 0.0
    report = {
        "rows": len(rows),
        "fallback_count": fallback_count,
        "fallback_rate": fallback_rate,
        "generator_signature": generator_signature,
        "precision": args.precision,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = build_manifest(
        run_id=output_path.stem,
        project_root=Path.cwd(),
        config={
            "precision": args.precision,
            "max_attempts": args.max_attempts,
        },
        input_files=(
            input_path,
            score_path,
            score_manifest_path,
            *rationale_checkpoint_files(checkpoint),
        ),
        extra={
            "artifact_type": "generated_grounded_rationales",
            "rationale_file": output_path.name,
            "rationale_sha256": sha256_file(output_path),
            "report_file": report_path.name,
            "report_sha256": sha256_file(report_path),
            "input_sha256": sha256_file(input_path),
            "score_sha256": sha256_file(score_path),
            "score_scorer_signature": score_manifest["scorer_signature"],
            "generator_signature": generator_signature,
            "model_id": generator_model_id,
            "model_revision": generator_model_revision,
            "checkpoint": str(checkpoint),
            "rows": len(rows),
            "fallback_count": fallback_count,
            "fallback_rate": fallback_rate,
        },
    )
    write_manifest(manifest_path, manifest)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
