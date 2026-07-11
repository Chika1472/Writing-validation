from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load import load_jsonl
from src.inference.parsing import parse_repaired_json, parse_strict_json
from src.utils.config import load_yaml, resolve_project_path
from src.utils.hashing import sha256_file
from src.utils.manifest import build_manifest, write_manifest


DOMAINS = ("content", "organization", "expression")


def _gold_score(record: Any, domain: str) -> float:
    scores = record.score
    if isinstance(scores, dict):
        return float(scores[domain])
    return float(getattr(scores, domain))


def _metric(gold: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(np.square(predicted - gold)))),
        "mae": float(np.mean(np.abs(predicted - gold))),
        "spearman": float(spearmanr(gold, predicted).statistic),
        "bias": float(np.mean(predicted - gold)),
        "gold_mean": float(gold.mean()),
        "gold_std": float(gold.std(ddof=1)),
        "pred_mean": float(predicted.mean()),
        "pred_std": float(predicted.std(ddof=1)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-audit the persisted Qwen3-14B zero-shot run.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--repaired-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    root = config["_project_root"]
    gold_path = resolve_project_path(config, config["paths"]["validation"])
    prediction_path = resolve_project_path(config, config["paths"]["zero_shot_predictions"])
    gold_records = load_jsonl(gold_path)
    gold_by_id = {record.id: record for record in gold_records}

    persisted: list[dict[str, Any]] = []
    with prediction_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                value["_line_number"] = line_number
                persisted.append(value)

    strict_count = 0
    repaired_count = 0
    persisted_parse_true = sum(record.get("parse_ok") is True for record in persisted)
    repaired_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for record in persisted:
        raw = str(record.get("raw_output", ""))
        strict_ok = False
        try:
            parsed = parse_strict_json(raw)
            strict_count += 1
            strict_ok = True
        except Exception as strict_error:
            try:
                parsed = parse_repaired_json(raw)
            except Exception as repaired_error:
                errors.append(
                    {
                        "id": record.get("id"),
                        "strict_error": str(strict_error),
                        "repaired_error": str(repaired_error),
                    }
                )
                continue
        repaired_count += 1
        repaired_rows.append(
            {
                "id": record["id"],
                "prompt_num": record.get("prompt_num"),
                "model": record.get("model_id"),
                "parse_mode": "strict" if strict_ok else "repaired",
                "prediction": {domain: float(parsed[domain]["score"]) for domain in DOMAINS},
                "rationales": {domain: str(parsed[domain]["rationale"]) for domain in DOMAINS},
            }
        )

    predicted_by_id = {row["id"]: row for row in repaired_rows}
    metrics: dict[str, dict[str, float]] = {}
    for domain in DOMAINS:
        shared_ids = [record.id for record in gold_records if record.id in predicted_by_id]
        gold = np.asarray([_gold_score(gold_by_id[id_], domain) for id_ in shared_ids])
        predicted = np.asarray([predicted_by_id[id_]["prediction"][domain] for id_ in shared_ids])
        metrics[domain] = _metric(gold, predicted)

    report = {
        "input_rows": len(persisted),
        "unique_ids": len({record.get("id") for record in persisted}),
        "persisted_parse_ok_true": persisted_parse_true,
        "strict_parsed_rows": strict_count,
        "repaired_parsed_rows": repaired_count,
        "strict_parse_rate": strict_count / len(persisted) if persisted else 0.0,
        "repaired_parse_rate": repaired_count / len(persisted) if persisted else 0.0,
        "errors": errors,
        "metrics_on_repaired_rows": metrics,
    }

    artifacts = resolve_project_path(config, config["paths"]["artifacts"])
    output_path = Path(args.output).resolve() if args.output else artifacts / "reports" / "zero_shot_reaudit.json"
    repaired_path = (
        Path(args.repaired_output).resolve()
        if args.repaired_output
        else artifacts / "predictions" / "qwen3_14b_zero_shot_repaired.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with repaired_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in repaired_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    manifest = build_manifest(
        run_id="zero_shot_reaudit",
        project_root=root,
        config=public_config,
        input_files=(gold_path, prediction_path),
        extra={
            "report": str(output_path),
            "report_sha256": sha256_file(output_path),
            "repaired_predictions": str(repaired_path),
            "repaired_predictions_sha256": sha256_file(repaired_path),
        },
    )
    write_manifest(output_path.with_name("zero_shot_reaudit_manifest.json"), manifest)
    print(json.dumps({"report": str(output_path), "strict": strict_count, "repaired": repaired_count}))


if __name__ == "__main__":
    main()
