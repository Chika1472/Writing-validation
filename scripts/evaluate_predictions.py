from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load import load_jsonl
from src.evaluation.predictions import (
    read_canonical_predictions,
    read_final_predictions,
    read_predictions,
)
from src.evaluation.slices import evaluation_report
from src.utils.config import load_yaml, resolve_project_path
from src.utils.hashing import sha256_file
from src.utils.manifest import build_manifest, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a canonical prediction JSONL.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--gold", default=None)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--final-schema",
        action="store_true",
        help="Require exact ID-bearing score+rationale final rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    root = config["_project_root"]
    gold_path = (
        Path(args.gold).resolve()
        if args.gold
        else resolve_project_path(config, config["paths"]["validation"])
    )
    prediction_path = Path(args.pred).resolve()
    gold = load_jsonl(gold_path)
    if args.strict and args.final_schema:
        raise ValueError("--strict and --final-schema are mutually exclusive")
    if args.final_schema:
        if args.model is not None:
            raise ValueError("--model coercion is incompatible with --final-schema")
        predictions = read_final_predictions(prediction_path)
    elif args.strict:
        if args.model is not None:
            raise ValueError("--model coercion is incompatible with --strict")
        predictions = read_canonical_predictions(prediction_path)
    else:
        predictions = read_predictions(
            prediction_path,
            model=args.model,
            validate_range=True,
            require_unique_ids=True,
        )
    report = evaluation_report(gold, predictions)
    report["schema_mode"] = (
        "final" if args.final_schema else "canonical" if args.strict else "relaxed"
    )
    report["prediction_rows"] = len(predictions)

    artifacts = resolve_project_path(config, config["paths"]["artifacts"])
    output_path = (
        Path(args.output).resolve()
        if args.output
        else artifacts / "reports" / f"{prediction_path.stem}_metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    manifest = build_manifest(
        run_id=f"evaluate_{prediction_path.stem}",
        project_root=root,
        config=public_config,
        input_files=(gold_path, prediction_path),
        extra={"report": str(output_path), "report_sha256": sha256_file(output_path)},
    )
    write_manifest(output_path.with_suffix(".manifest.json"), manifest)
    print(json.dumps({"report": str(output_path), "n": len(predictions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
