from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load import load_jsonl
from src.utils.config import load_yaml, resolve_project_path
from src.utils.hashing import sha256_file
from src.utils.manifest import build_manifest, write_manifest


DOMAINS = ("content", "organization", "expression")


def _score(record: Any, domain: str) -> float:
    scores = record.score
    if isinstance(scores, dict):
        return float(scores[domain])
    return float(getattr(scores, domain))


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "unique": int(len(np.unique(array))),
    }


def audit_split(records: list[Any]) -> dict[str, Any]:
    essays = [record.essay for record in records]
    return {
        "rows": len(records),
        "unique_ids": len({record.id for record in records}),
        "unique_document_ids": len({record.document_id for record in records}),
        "unique_essays": len(set(essays)),
        "prompt_counts": dict(sorted(Counter(record.prompt_num for record in records).items())),
        "unique_prompt_texts": len({record.prompt for record in records}),
        "scores": {
            domain: _summary([_score(record, domain) for record in records])
            for domain in DOMAINS
        },
        "essay_characters": _summary([float(len(text)) for text in essays]),
        "format": {
            "leading_whitespace": sum(bool(text[:1].isspace()) for text in essays),
            "trailing_whitespace": sum(bool(text[-1:].isspace()) for text in essays),
            "contains_lf": sum("\n" in text for text in essays),
            "contains_tab": sum("\t" in text for text in essays),
            "contains_carriage_return": sum("\r" in text for text in essays),
            "contains_double_space": sum("  " in text for text in essays),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the official train/validation JSONL files.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    root = config["_project_root"]
    train_path = resolve_project_path(config, config["paths"]["train"])
    validation_path = resolve_project_path(config, config["paths"]["validation"])
    train = load_jsonl(train_path)
    validation = load_jsonl(validation_path)

    report = {
        "files": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "validation": {"path": str(validation_path), "sha256": sha256_file(validation_path)},
        },
        "train": audit_split(train),
        "validation": audit_split(validation),
        "cross_split": {
            "id_overlap": len({r.id for r in train} & {r.id for r in validation}),
            "document_id_overlap": len(
                {r.document_id for r in train} & {r.document_id for r in validation}
            ),
            "essay_overlap": len({r.essay for r in train} & {r.essay for r in validation}),
            "prompt_text_overlap": len({r.prompt for r in train} & {r.prompt for r in validation}),
        },
    }

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = resolve_project_path(config, config["paths"]["artifacts"]) / "reports" / "data_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    manifest = build_manifest(
        run_id="data_audit",
        project_root=root,
        config=public_config,
        input_files=(train_path, validation_path),
        extra={"report": str(output_path), "report_sha256": sha256_file(output_path)},
    )
    write_manifest(output_path.with_name("data_audit_manifest.json"), manifest)
    print(json.dumps({"report": str(output_path), "rows": [len(train), len(validation)]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
