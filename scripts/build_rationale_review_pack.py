from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load import load_inference_jsonl
from src.evaluation.predictions import read_final_predictions
from src.inference.serializer import TRAITS
from src.utils.hashing import sha256_file
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic prompt/score-stratified blind A/B rationale "
            "review pack. Candidate and baseline scores must be numerically identical."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key-output", required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _stable(seed: int, *parts: str) -> str:
    value = "\0".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _score_payload(row: dict) -> dict[str, float]:
    return {trait: float(row["prediction"][trait]["score"]) for trait in TRAITS}


def _rationale_payload(row: dict) -> dict[str, str]:
    return {trait: str(row["prediction"][trait]["rationale"]) for trait in TRAITS}


def _balanced_ids(
    rows: list[dict], *, sample_size: int, seed: int
) -> list[str]:
    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for row in rows:
        scores = _score_payload(row)
        mean_score = sum(scores.values()) / len(scores)
        score_band = min(4, max(1, int(mean_score)))
        groups[(row["prompt_num"], score_band)].append(row["id"])
    queues = {
        key: deque(sorted(ids, key=lambda record_id: _stable(seed, record_id)))
        for key, ids in groups.items()
    }
    keys = sorted(queues, key=lambda key: _stable(seed, key[0], str(key[1])))
    selected: list[str] = []
    while len(selected) < sample_size and any(queues[key] for key in keys):
        for key in keys:
            if queues[key] and len(selected) < sample_size:
                selected.append(queues[key].popleft())
    return selected


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    input_path = Path(args.input).resolve()
    candidate_path = Path(args.candidate).resolve()
    baseline_path = Path(args.baseline).resolve()
    output_path = Path(args.output).resolve()
    key_path = Path(args.key_output).resolve()
    manifest_path = output_path.with_suffix(".manifest.json")
    require_distinct_paths(
        input=input_path,
        candidate=candidate_path,
        baseline=baseline_path,
        output=output_path,
        key=key_path,
        manifest=manifest_path,
    )
    require_new_paths(output=output_path, key=key_path, manifest=manifest_path)

    records = load_inference_jsonl(input_path)
    candidate = read_final_predictions(candidate_path)
    baseline = read_final_predictions(baseline_path)
    records_by_id = {record.id: record for record in records}
    candidate_by_id = {row["id"]: row for row in candidate}
    baseline_by_id = {row["id"]: row for row in baseline}
    expected_ids = set(records_by_id)
    if set(candidate_by_id) != expected_ids or set(baseline_by_id) != expected_ids:
        raise ValueError("input, candidate, and baseline id sets must match exactly")
    for record in records:
        left = candidate_by_id[record.id]
        right = baseline_by_id[record.id]
        if left["prompt_num"] != record.prompt_num or right["prompt_num"] != record.prompt_num:
            raise ValueError(f"prompt mismatch for {record.id}")
        if _score_payload(left) != _score_payload(right):
            raise ValueError(
                f"blind rationale review requires identical fixed scores: {record.id}"
            )
    if args.sample_size > len(records):
        raise ValueError("--sample-size exceeds available aligned rows")
    selected_ids = _balanced_ids(candidate, sample_size=args.sample_size, seed=args.seed)

    review_rows = []
    key_rows = []
    for index, record_id in enumerate(selected_ids, start=1):
        record = records_by_id[record_id]
        candidate_row = candidate_by_id[record_id]
        baseline_row = baseline_by_id[record_id]
        candidate_is_a = int(_stable(args.seed, record_id, "assignment")[-1], 16) % 2 == 0
        option_a = candidate_row if candidate_is_a else baseline_row
        option_b = baseline_row if candidate_is_a else candidate_row
        review_id = f"R{index:04d}"
        review_rows.append(
            {
                "review_id": review_id,
                "prompt_num": record.prompt_num,
                "prompt": record.prompt,
                "essay": record.essay,
                "fixed_scores": _score_payload(candidate_row),
                "option_a": _rationale_payload(option_a),
                "option_b": _rationale_payload(option_b),
                "review": {
                    "grounded_in_essay": None,
                    "specific_and_helpful": None,
                    "trait_separation": None,
                    "consistent_with_fixed_scores": None,
                    "overall_preference": None,
                    "notes": "",
                },
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "id": record_id,
                "option_a": "candidate" if candidate_is_a else "baseline",
                "option_b": "baseline" if candidate_is_a else "candidate",
            }
        )

    for path, rows in ((output_path, review_rows), (key_path, key_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                    + "\n"
                )
    manifest = build_manifest(
        run_id=output_path.stem,
        project_root=Path(__file__).resolve().parents[1],
        config={"sample_size": args.sample_size, "seed": args.seed},
        input_files=(input_path, candidate_path, baseline_path),
        extra={
            "artifact_type": "blind_rationale_review_pack",
            "review_file": output_path.name,
            "review_sha256": sha256_file(output_path),
            "key_file": key_path.name,
            "key_sha256": sha256_file(key_path),
            "rows": len(review_rows),
            "score_equality_verified": True,
            "assignment_hidden_from_review_file": True,
        },
    )
    write_manifest(manifest_path, manifest)
    print(json.dumps({"review_pack": str(output_path), "rows": len(review_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
