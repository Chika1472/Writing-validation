from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.folds import make_folds, save_folds
from src.data.load import load_jsonl
from src.utils.config import load_yaml, resolve_project_path
from src.utils.hashing import sha256_file
from src.utils.manifest import build_manifest, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create immutable deterministic train folds.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--score-bins", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _require_safe_create_only_paths(
    *,
    destinations: tuple[Path, ...],
    protected_inputs: tuple[Path, ...],
) -> None:
    if len(set(destinations)) != len(destinations):
        raise ValueError("fold, summary, and manifest destinations must be distinct")
    protected = set(protected_inputs)
    for destination in destinations:
        if destination in protected:
            raise ValueError(
                f"fold artifact destination aliases a protected input: {destination}"
            )
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite immutable fold artifact: {destination}"
            )


def _write_json_create_only(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        handle = path.open("x", encoding="utf-8", newline="\n")
        created = True
        with handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    except Exception:
        if created:
            path.unlink(missing_ok=True)
        raise


def _require_unchanged_files(expected_sha256: dict[Path, str]) -> None:
    for path, expected in expected_sha256.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"immutable fold input changed during the run: {path}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_sha256 = sha256_file(config_path)
    config = load_yaml(config_path)
    _require_unchanged_files({config_path: config_sha256})
    root = config["_project_root"]
    fold_config = config["folds"]
    n_splits = (
        args.n_folds if args.n_folds is not None else int(fold_config["n_splits"])
    )
    seed = args.seed if args.seed is not None else int(fold_config["seed"])
    score_bins = (
        args.score_bins
        if args.score_bins is not None
        else int(fold_config["score_bins"])
    )
    train_path = resolve_project_path(config, config["paths"]["train"])
    immutable_inputs = {
        config_path: config_sha256,
        train_path: sha256_file(train_path),
    }
    artifacts = resolve_project_path(config, config["paths"]["artifacts"])
    output_path = (
        Path(args.output).resolve()
        if args.output
        else artifacts / "folds" / f"folds_{n_splits}fold_seed{seed}.jsonl"
    )
    summary_path = output_path.with_suffix(".summary.json")
    manifest_path = output_path.with_suffix(".manifest.json")
    _require_safe_create_only_paths(
        destinations=(output_path, summary_path, manifest_path),
        protected_inputs=(
            train_path,
            config_path,
        ),
    )

    records = load_jsonl(train_path)
    _require_unchanged_files(immutable_inputs)
    assignments = make_folds(
        records,
        n_splits=n_splits,
        seed=seed,
        score_bins=score_bins,
    )

    by_prompt: dict[str, Counter[int]] = defaultdict(Counter)
    prompt_by_id = {record.id: record.prompt_num for record in records}
    for record_id, fold in assignments.items():
        by_prompt[prompt_by_id[record_id]][fold] += 1
    summary = {
        "n_records": len(records),
        "n_splits": n_splits,
        "seed": seed,
        "score_bins": score_bins,
        "fold_sizes": dict(sorted(Counter(assignments.values()).items())),
        "prompt_fold_sizes": {
            prompt: dict(sorted(counts.items())) for prompt, counts in sorted(by_prompt.items())
        },
    }
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    created: list[Path] = []
    try:
        save_folds(assignments, output_path, create_only=True)
        created.append(output_path)
        _write_json_create_only(summary_path, summary)
        created.append(summary_path)
        manifest = build_manifest(
            run_id=f"folds_{n_splits}fold_seed{seed}",
            project_root=root,
            config=public_config,
            input_files=tuple(immutable_inputs),
            extra={
                "folds": str(output_path),
                "folds_sha256": sha256_file(output_path),
                "summary": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
            },
        )
        write_manifest(manifest_path, manifest, create_only=True)
        created.append(manifest_path)
        _require_unchanged_files(immutable_inputs)
    except Exception:
        for created_path in reversed(created):
            created_path.unlink(missing_ok=True)
        raise
    print(json.dumps({"folds": str(output_path), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
