from __future__ import annotations

import argparse
import json
import sys
from numbers import Real
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load import load_inference_jsonl
from src.evaluation.prediction_provenance import (
    prediction_manifest_path,
    prediction_provenance_fields,
    validate_prediction_provenance,
)
from src.evaluation.predictions import read_canonical_predictions
from src.inference.finalize import final_prediction_row
from src.inference.serializer import serialize_prediction, strict_parse_prediction
from src.rationale.deterministic import (
    RATIONALE_TEMPLATE_VERSION,
    generate_grounded_rationales,
)
from src.rationale.evidence import build_evidence_ledger
from src.rationale.parsing import assess_grounding, validate_rationales
from src.utils.hashing import sha256_file, sha256_json
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach validated generated or fallback rationales without changing scores."
    )
    parser.add_argument("--input", required=True, help="Labeled or unlabeled essay JSONL.")
    parser.add_argument("--scores", required=True, help="Canonical scorer prediction JSONL.")
    parser.add_argument("--output", required=True, help="ID-bearing final prediction JSONL.")
    parser.add_argument("--ledger", required=True, help="Internal evidence-ledger JSONL.")
    parser.add_argument("--bare-output", default=None, help="Optional schema-only JSONL in input order.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--rationales",
        default=None,
        help=(
            "Optional grounded rationale JSONL from generate_rationales.py. "
            "If omitted, the deterministic exact-evidence fallback is used."
        ),
    )
    return parser.parse_args()


def _load_generated_rationales(
    path: Path,
    *,
    input_path: Path,
    score_path: Path,
    score_manifest: dict,
) -> tuple[dict[str, dict], dict, Path]:
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("generated rationale file and adjacent manifest are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("artifact_type") != "generated_grounded_rationales"
        or manifest.get("rationale_file") != path.name
        or manifest.get("rationale_sha256") != sha256_file(path)
    ):
        raise ValueError("generated rationale artifact does not match its manifest")
    expected = {
        "input_sha256": sha256_file(input_path),
        "score_sha256": sha256_file(score_path),
        "score_scorer_signature": score_manifest["scorer_signature"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"generated rationale {key} does not match finalization inputs")
    generator_signature = manifest.get("generator_signature")
    if not isinstance(generator_signature, str) or not generator_signature:
        raise ValueError("generated rationale manifest has no generator signature")
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank rationale JSONL row")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid rationale JSON: {error.msg}"
                ) from error
            required = {
                "id",
                "prompt_num",
                "conditioned_scores",
                "rationales",
                "fallback_used",
                "attempts",
                "evidence",
            }
            if not isinstance(row, dict) or set(row) != required:
                raise ValueError(f"{path}:{line_number}: unexpected rationale row schema")
            record_id = row.get("id")
            if not isinstance(record_id, str) or not record_id or record_id in rows:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate rationale id")
            if not isinstance(row.get("fallback_used"), bool):
                raise ValueError(f"{path}:{line_number}: fallback_used must be boolean")
            if not isinstance(row.get("attempts"), list) or not isinstance(row.get("evidence"), dict):
                raise ValueError(f"{path}:{line_number}: attempts/evidence schema is invalid")
            validate_rationales(row.get("rationales"))
            rows[record_id] = row
    if int(manifest.get("rows", -1)) != len(rows):
        raise ValueError("generated rationale row count disagrees with its manifest")
    return rows, manifest, manifest_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    score_path = Path(args.scores).resolve()
    score_manifest_path = prediction_manifest_path(score_path)
    output_path = Path(args.output).resolve()
    output_manifest_path = output_path.with_suffix(".manifest.json")
    ledger_path = Path(args.ledger).resolve()
    bare_path = Path(args.bare_output).resolve() if args.bare_output else None
    rationale_path = Path(args.rationales).resolve() if args.rationales else None
    rationale_manifest_path = (
        rationale_path.with_suffix(".manifest.json") if rationale_path else None
    )
    require_distinct_paths(
        input=input_path,
        scores=score_path,
        score_manifest=score_manifest_path,
        output=output_path,
        output_manifest=output_manifest_path,
        ledger=ledger_path,
        bare_output=bare_path,
        rationales=rationale_path,
        rationale_manifest=rationale_manifest_path,
    )
    require_new_paths(
        output=output_path,
        output_manifest=output_manifest_path,
        ledger=ledger_path,
        bare_output=bare_path,
    )

    records = load_inference_jsonl(input_path)
    score_manifest = validate_prediction_provenance(score_path)
    if score_manifest.get("input_sha256") != sha256_file(input_path):
        raise ValueError("score predictions were not produced from the supplied essay input")
    score_rows = read_canonical_predictions(score_path)
    if int(score_manifest["rows"]) != len(score_rows):
        raise ValueError("score prediction row count disagrees with its manifest")
    if {row["model"] for row in score_rows} != {score_manifest["scorer_name"]}:
        raise ValueError("score prediction model disagrees with its manifest")
    score_by_id = {row["id"]: row for row in score_rows}
    input_ids = [record.id for record in records]
    missing = [record_id for record_id in input_ids if record_id not in score_by_id]
    extra = sorted(set(score_by_id).difference(input_ids))
    if missing or extra:
        raise ValueError(f"score/input id mismatch: missing={missing[:5]}, extra={extra[:5]}")

    generated_by_id: dict[str, dict] | None = None
    rationale_manifest: dict | None = None
    if rationale_path is not None:
        generated_by_id, rationale_manifest, rationale_manifest_path = _load_generated_rationales(
            rationale_path,
            input_path=input_path,
            score_path=score_path,
            score_manifest=score_manifest,
        )
        missing_rationales = [record_id for record_id in input_ids if record_id not in generated_by_id]
        extra_rationales = sorted(set(generated_by_id).difference(input_ids))
        if missing_rationales or extra_rationales:
            raise ValueError(
                "rationale/input id mismatch: "
                f"missing={missing_rationales[:5]}, extra={extra_rationales[:5]}"
            )

    final_rows = []
    ledger_rows = []
    for record in records:
        score_row = score_by_id[record.id]
        if score_row["prompt_num"] != record.prompt_num:
            raise ValueError(f"prompt_num mismatch for {record.id}")
        scores = {
            trait: float(score_row["prediction"][trait])
            for trait in ("content", "organization", "expression")
        }
        ledger = build_evidence_ledger(record)
        generated_row = generated_by_id[record.id] if generated_by_id is not None else None
        if generated_row is None:
            rationales = generate_grounded_rationales(ledger, scores)
            fallback_grounding = assess_grounding(
                rationales, essay=record.essay, ledger=ledger
            )
            if not fallback_grounding.accepted:
                raise RuntimeError(
                    f"deterministic fallback is not grounded for {record.id}: "
                    f"{fallback_grounding.reasons}"
                )
            fallback_used = True
            attempts = []
            rationale_source = "deterministic_exact_evidence_fallback"
        else:
            if generated_row["prompt_num"] != record.prompt_num:
                raise ValueError(f"generated rationale prompt mismatch for {record.id}")
            conditioned_scores = generated_row["conditioned_scores"]
            if not isinstance(conditioned_scores, dict) or set(conditioned_scores) != set(scores):
                raise ValueError(f"invalid conditioned score schema for {record.id}")
            if any(
                isinstance(conditioned_scores[trait], bool)
                or not isinstance(conditioned_scores[trait], Real)
                for trait in scores
            ):
                raise ValueError(f"conditioned scores must be JSON numbers for {record.id}")
            if any(float(conditioned_scores[trait]) != scores[trait] for trait in scores):
                raise ValueError(f"generated rationale was conditioned on different scores for {record.id}")
            if generated_row["evidence"] != ledger.to_dict():
                raise ValueError(f"generated rationale evidence ledger mismatch for {record.id}")
            rationales = validate_rationales(generated_row["rationales"])
            grounding = assess_grounding(rationales, essay=record.essay, ledger=ledger)
            if not grounding.accepted:
                raise ValueError(
                    f"generated rationale is not grounded for {record.id}: {grounding.reasons}"
                )
            fallback_used = bool(generated_row["fallback_used"])
            attempts = generated_row["attempts"]
            rationale_source = "generated_grounded_rationales"
        row = final_prediction_row(
            record_id=record.id,
            prompt_num=record.prompt_num,
            model=args.model_name,
            scores=scores,
            rationales=rationales,
        )
        strict_parse_prediction(serialize_prediction(row["prediction"]))
        if any(float(row["prediction"][trait]["score"]) != scores[trait] for trait in scores):
            raise RuntimeError(f"rationale finalization changed a score for {record.id}")
        final_rows.append(row)
        ledger_rows.append(
            {
                "id": record.id,
                "prompt_num": record.prompt_num,
                "template_version": RATIONALE_TEMPLATE_VERSION,
                "scores": scores,
                "rationales": rationales,
                "evidence": ledger.to_dict(),
                "rationale_source": rationale_source,
                "fallback_used": fallback_used,
                "attempts": attempts,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in final_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + "\n"
            )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ledger_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + "\n"
            )
    if bare_path is not None:
        bare_path.parent.mkdir(parents=True, exist_ok=True)
        with bare_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in final_rows:
                serialized = serialize_prediction(row["prediction"])
                strict_parse_prediction(serialized)
                handle.write(serialized + "\n")

    source_root = Path(__file__).resolve().parents[1]
    rationale_signature_payload = {
        "template_version": RATIONALE_TEMPLATE_VERSION,
        "evidence_code_sha256": sha256_file(source_root / "src" / "rationale" / "evidence.py"),
        "template_code_sha256": sha256_file(
            source_root / "src" / "rationale" / "deterministic.py"
        ),
        "finalizer_code_sha256": sha256_file(source_root / "src" / "inference" / "finalize.py"),
        "serializer_code_sha256": sha256_file(source_root / "src" / "inference" / "serializer.py"),
        "rationale_parser_code_sha256": sha256_file(
            source_root / "src" / "rationale" / "parsing.py"
        ),
        "rationale_prompt_code_sha256": sha256_file(
            source_root / "src" / "rationale" / "prompting.py"
        ),
        "generated_rationale_sha256": sha256_file(rationale_path) if rationale_path else None,
        "generated_rationale_manifest_sha256": (
            sha256_file(rationale_manifest_path) if rationale_manifest_path else None
        ),
        "generator_signature": (
            rationale_manifest["generator_signature"] if rationale_manifest else None
        ),
    }
    rationale_signature = sha256_json(rationale_signature_payload)
    final_signature = sha256_json(
        {
            "score_scorer_signature": score_manifest["scorer_signature"],
            "rationale_signature": rationale_signature,
            "schema": "content_organization_expression_score_rationale_v1",
            "model_name": args.model_name,
        }
    )
    manifest_inputs = [input_path, score_path, score_manifest_path]
    if rationale_path is not None and rationale_manifest_path is not None:
        manifest_inputs.extend((rationale_path, rationale_manifest_path))
    manifest = build_manifest(
        run_id=output_path.stem,
        project_root=source_root,
        config={
            "model_name": args.model_name,
            "rationale_template_version": RATIONALE_TEMPLATE_VERSION,
        },
        input_files=manifest_inputs,
        extra={
            **prediction_provenance_fields(
                prediction_path=output_path,
                input_path=input_path,
                rows=len(final_rows),
                scorer_name=args.model_name,
                scorer_signature=final_signature,
            ),
            "artifact_type": "final_challenge_predictions",
            "score_source_file": score_path.name,
            "score_source_sha256": sha256_file(score_path),
            "score_source_scorer_signature": score_manifest["scorer_signature"],
            "rationale_signature": rationale_signature,
            "ledger_file": ledger_path.name,
            "ledger_sha256": sha256_file(ledger_path),
            "bare_output_file": bare_path.name if bare_path else None,
            "bare_output_sha256": sha256_file(bare_path) if bare_path else None,
            "score_preservation_verified": True,
            "rationale_source": (
                "generated_grounded_rationales"
                if rationale_path
                else "deterministic_exact_evidence_fallback"
            ),
            "generated_rationale_file": rationale_path.name if rationale_path else None,
            "generated_rationale_sha256": sha256_file(rationale_path) if rationale_path else None,
            "generated_rationale_manifest_sha256": (
                sha256_file(rationale_manifest_path) if rationale_manifest_path else None
            ),
        },
    )
    write_manifest(output_manifest_path, manifest)
    print(
        json.dumps(
            {
                "final_predictions": str(output_path),
                "ledger": str(ledger_path),
                "bare_output": str(bare_path) if bare_path else None,
                "rows": len(final_rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
