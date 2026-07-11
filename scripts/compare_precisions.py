from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load import load_jsonl
from src.evaluation.bootstrap import paired_stratified_bootstrap
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.oof_provenance import oof_manifest_path
from src.evaluation.precision import (
    precision_promotion_gate,
    validate_precision_pair,
)
from src.evaluation.predictions import read_canonical_predictions
from src.utils.config import load_yaml
from src.utils.hashing import sha256_file
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare precision-controlled OOF predictions and emit a conservative "
            "promotion decision."
        )
    )
    parser.add_argument("--config", default="configs/precision_comparison.yaml")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _validate_rows(
    label: str,
    rows: list[dict],
    gold: list,
    provenance: dict,
) -> None:
    if provenance["rows"] != len(rows):
        raise ValueError(f"{label} OOF row count does not match provenance")
    expected = {record.id: record.prompt_num for record in gold}
    observed = {row["id"]: row["prompt_num"] for row in rows}
    missing = [record_id for record_id in expected if record_id not in observed]
    extra = sorted(set(observed).difference(expected))
    prompt_mismatches = [
        record_id
        for record_id in expected.keys() & observed.keys()
        if expected[record_id] != observed[record_id]
    ]
    if missing or extra or prompt_mismatches:
        raise ValueError(
            f"{label} OOF row contract mismatch; missing={missing[:5]}, "
            f"extra={extra[:5]}, prompt_mismatch={prompt_mismatches[:5]}"
        )
    models = {row["model"] for row in rows}
    if models != {provenance["scorer_name"]}:
        raise ValueError(
            f"{label} OOF model field does not match provenance scorer_name"
        )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    candidate_precision = str(config["candidate_precision"])
    baseline_precision = str(config["baseline_precision"])
    bootstrap_config = config["bootstrap"]
    gate_config = config["promotion_gate"]

    gold_path = Path(args.gold).resolve()
    fold_path = Path(args.folds).resolve()
    candidate_path = Path(args.candidate).resolve()
    baseline_path = Path(args.baseline).resolve()
    output_path = Path(args.output).resolve()
    report_manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    require_distinct_paths(
        config=config_path,
        gold=gold_path,
        folds=fold_path,
        candidate=candidate_path,
        baseline=baseline_path,
        output=output_path,
        report_manifest=report_manifest_path,
        candidate_manifest=oof_manifest_path(candidate_path),
        baseline_manifest=oof_manifest_path(baseline_path),
    )
    require_new_paths(output=output_path, report_manifest=report_manifest_path)

    candidate_provenance, baseline_provenance = validate_precision_pair(
        candidate_path=candidate_path,
        baseline_path=baseline_path,
        gold_path=gold_path,
        fold_path=fold_path,
        candidate_precision=candidate_precision,
        baseline_precision=baseline_precision,
    )
    gold = load_jsonl(gold_path)
    candidate = read_canonical_predictions(candidate_path)
    baseline = read_canonical_predictions(baseline_path)
    if len(candidate) != len(gold) or len(baseline) != len(gold):
        raise ValueError("precision OOF prediction rows must match gold rows")
    _validate_rows("candidate", candidate, gold, candidate_provenance)
    _validate_rows("baseline", baseline, gold, baseline_provenance)

    bootstrap = paired_stratified_bootstrap(
        gold,
        candidate,
        baseline,
        n_resamples=int(bootstrap_config["n_resamples"]),
        confidence=float(bootstrap_config["confidence"]),
        seed=int(bootstrap_config["seed"]),
    )
    gate = precision_promotion_gate(
        bootstrap,
        max_rmse_increase=float(gate_config["max_rmse_increase"]),
        max_spearman_drop=float(gate_config["max_spearman_drop"]),
        min_rmse_improvement=float(gate_config["min_rmse_improvement"]),
        min_spearman_improvement=float(gate_config["min_spearman_improvement"]),
        min_probability=float(gate_config["min_probability"]),
    )
    report = {
        "artifact_type": "precision_oof_comparison",
        "candidate_precision": candidate_precision,
        "baseline_precision": baseline_precision,
        "checkpoint_set_signature": candidate_provenance["checkpoint_set_signature"],
        "candidate_oof_sha256": candidate_provenance["oof_sha256"],
        "baseline_oof_sha256": baseline_provenance["oof_sha256"],
        "candidate_metrics": evaluate_predictions(gold, candidate),
        "baseline_metrics": evaluate_predictions(gold, baseline),
        "paired_bootstrap": bootstrap,
        "promotion_gate": gate,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = build_manifest(
        run_id=output_path.stem,
        project_root=config["_project_root"],
        config={key: value for key, value in config.items() if not key.startswith("_")},
        input_files=(
            config_path,
            gold_path,
            fold_path,
            candidate_path,
            baseline_path,
            oof_manifest_path(candidate_path),
            oof_manifest_path(baseline_path),
        ),
        extra={
            "artifact_type": "precision_oof_comparison_manifest",
            "report": str(output_path),
            "report_sha256": sha256_file(output_path),
            "checkpoint_set_signature": candidate_provenance[
                "checkpoint_set_signature"
            ],
            "promote_candidate": gate["promote_candidate"],
        },
    )
    write_manifest(report_manifest_path, manifest)
    print(
        json.dumps(
            {
                "report": str(output_path),
                "promote_candidate": gate["promote_candidate"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
