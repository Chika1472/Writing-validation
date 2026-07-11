from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load import load_inference_jsonl
from src.evaluation.prediction_provenance import prediction_provenance_fields
from src.inference.deployment import load_deployment_config
from src.inference.serializer import serialize_prediction, strict_parse_prediction
from src.inference.submission import SubmissionEngine
from src.utils.hashing import sha256_file
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths, require_outside_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SDK-neutral offline submission engine. Adapt only this CLI boundary "
            "after the competition publishes its official model interface."
        )
    )
    parser.add_argument("--config", default="configs/inference_l40s.yaml")
    parser.add_argument("--input", required=True, help="Strict labeled or unlabeled essay JSONL.")
    parser.add_argument("--output", required=True, help="ID-bearing final prediction JSONL.")
    parser.add_argument("--ledger", required=True, help="New internal evidence/audit JSONL.")
    parser.add_argument(
        "--bare-output",
        default=None,
        help="Optional schema-only JSONL in source input order.",
    )
    return parser.parse_args()


def _write_jsonl(path: Path, rows: list[dict] | tuple[dict, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    manifest_path = output_path.with_suffix(".manifest.json")
    ledger_path = Path(args.ledger).resolve()
    bare_path = Path(args.bare_output).resolve() if args.bare_output else None
    require_distinct_paths(
        config=config_path,
        input=input_path,
        output=output_path,
        output_manifest=manifest_path,
        ledger=ledger_path,
        bare_output=bare_path,
    )
    require_new_paths(
        output=output_path,
        output_manifest=manifest_path,
        ledger=ledger_path,
        bare_output=bare_path,
    )
    config = load_deployment_config(config_path)
    immutable_roots = {
        "input": input_path,
        **{
            f"checkpoint_{index}": checkpoint
            for index, checkpoint in enumerate(config.qwen.checkpoints)
        },
    }
    if config.runtime.package_manifest is not None:
        immutable_roots["signed_package"] = config.project_root
    require_outside_roots(
        immutable_roots,
        output=output_path,
        output_manifest=manifest_path,
        ledger=ledger_path,
        bare_output=bare_path,
    )
    records = load_inference_jsonl(input_path)
    engine = SubmissionEngine.from_config(config_path)
    temporary_output = output_path.with_name(output_path.name + ".tmp-artifact")
    temporary_ledger = ledger_path.with_name(ledger_path.name + ".tmp-artifact")
    temporary_bare = (
        bare_path.with_name(bare_path.name + ".tmp-artifact") if bare_path else None
    )
    require_distinct_paths(
        input=input_path,
        output=output_path,
        manifest=manifest_path,
        ledger=ledger_path,
        bare=bare_path,
        temporary_output=temporary_output,
        temporary_ledger=temporary_ledger,
        temporary_bare=temporary_bare,
    )
    require_new_paths(
        temporary_output=temporary_output,
        temporary_ledger=temporary_ledger,
        temporary_bare=temporary_bare,
    )
    validated_immutable_roots = {
        "input": input_path,
        **{
            f"artifact_root_{index}": path
            for index, path in enumerate(engine.artifacts.immutable_roots)
        },
    }
    if config.runtime.hf_home is not None:
        validated_immutable_roots["hf_home"] = config.runtime.hf_home
    if config.runtime.package_manifest is not None:
        validated_immutable_roots["signed_package"] = config.project_root
    require_outside_roots(
        validated_immutable_roots,
        output=output_path,
        output_manifest=manifest_path,
        ledger=ledger_path,
        bare_output=bare_path,
        temporary_output=temporary_output,
        temporary_ledger=temporary_ledger,
        temporary_bare=temporary_bare,
    )

    result = engine.predict(records)
    if len(result.rows) != len(records):
        raise RuntimeError("submission engine changed the input row count")
    if [row["id"] for row in result.rows] != [record.id for record in records]:
        raise RuntimeError("submission engine changed input row order")
    committed: list[Path] = []
    try:
        _write_jsonl(temporary_output, result.rows)
        _write_jsonl(temporary_ledger, result.ledger_rows)
        if temporary_bare is not None:
            temporary_bare.parent.mkdir(parents=True, exist_ok=True)
            with temporary_bare.open("w", encoding="utf-8", newline="\n") as handle:
                for row in result.rows:
                    serialized = serialize_prediction(row["prediction"])
                    strict_parse_prediction(serialized)
                    handle.write(serialized + "\n")
        for row in result.rows:
            strict_parse_prediction(serialize_prediction(row["prediction"]))
        temporary_output.replace(output_path)
        committed.append(output_path)
        temporary_ledger.replace(ledger_path)
        committed.append(ledger_path)
        if temporary_bare is not None and bare_path is not None:
            temporary_bare.replace(bare_path)
            committed.append(bare_path)
    except Exception:
        for temporary in (temporary_output, temporary_ledger, temporary_bare):
            if temporary is not None and temporary.exists():
                temporary.unlink()
        for path in committed:
            if path.exists():
                path.unlink()
        raise

    fallback_rate = result.fallback_count / len(result.rows)
    try:
        manifest = build_manifest(
            run_id=output_path.stem,
            project_root=config.project_root,
            config={
                "deployment_config": str(config_path),
                "deployment_config_sha256": sha256_file(config_path),
                "scoring_mode": config.scoring_mode,
                "rationale_mode": config.rationale.mode,
                "scorer_precision": config.runtime.scorer_precision,
                "rationale_precision": config.runtime.rationale_precision,
                "offline_required": config.runtime.offline_required,
            },
            input_files=(
                input_path,
                config_path,
                *engine.artifacts.all_files,
                *(
                    (config.runtime.package_manifest,)
                    if config.runtime.package_manifest is not None
                    else ()
                ),
                *(
                    (config.runtime.requirements_lock,)
                    if config.runtime.requirements_lock is not None
                    else ()
                ),
            ),
            extra={
                **prediction_provenance_fields(
                    prediction_path=output_path,
                    input_path=input_path,
                    rows=len(result.rows),
                    scorer_name=config.output_model_name,
                    scorer_signature=result.final_signature,
                ),
                "artifact_type": "final_challenge_predictions",
                "score_scorer_name": result.score_scorer_name,
                "score_scorer_signature": result.score_scorer_signature,
                "rationale_signature": result.rationale_signature,
                "ledger_file": ledger_path.name,
                "ledger_sha256": sha256_file(ledger_path),
                "bare_output_file": bare_path.name if bare_path else None,
                "bare_output_sha256": sha256_file(bare_path) if bare_path else None,
                "fallback_count": result.fallback_count,
                "fallback_rate": fallback_rate,
                "strict_parse_verified": True,
                "score_preservation_verified": True,
                "input_order_preserved": True,
            },
        )
        write_manifest(manifest_path, manifest)
    except Exception:
        for path in (*committed, manifest_path):
            if path.exists():
                path.unlink()
        raise
    print(
        json.dumps(
            {
                "output": str(output_path),
                "ledger": str(ledger_path),
                "bare_output": str(bare_path) if bare_path else None,
                "manifest": str(manifest_path),
                "rows": len(result.rows),
                "fallback_count": result.fallback_count,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
