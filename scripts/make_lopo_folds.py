from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.folds import save_folds
from src.data.load import load_inference_jsonl
from src.data.lopo import build_lopo_contract, make_lopo_folds
from src.utils.config import load_yaml, resolve_project_path
from src.utils.hashing import sha256_file, sha256_json
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create immutable leave-one-prompt-out diagnostic folds."
    )
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _require_unchanged_files(expected_sha256: dict[Path, str]) -> None:
    for path, expected in expected_sha256.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"immutable LOPO input changed during the run: {path}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_sha256 = sha256_file(config_path)
    config = load_yaml(config_path)
    _require_unchanged_files({config_path: config_sha256})
    root = config["_project_root"]
    train_path = resolve_project_path(config, config["paths"]["train"])
    immutable_inputs = {
        config_path: config_sha256,
        train_path: sha256_file(train_path),
    }
    records = load_inference_jsonl(train_path)
    _require_unchanged_files(immutable_inputs)

    assignments = make_lopo_folds(records)
    prompt_fold_contract = build_lopo_contract(records, assignments)

    artifacts = resolve_project_path(config, config["paths"]["artifacts"])
    output_path = (
        Path(args.output).resolve()
        if args.output
        else artifacts / "folds" / "lopo_by_prompt.jsonl"
    )
    manifest_path = output_path.with_suffix(".manifest.json")
    require_distinct_paths(
        config=config_path,
        train=train_path,
        folds=output_path,
        manifest=manifest_path,
    )
    require_new_paths(folds=output_path, manifest=manifest_path)

    created: list[Path] = []
    try:
        save_folds(assignments, output_path, create_only=True)
        created.append(output_path)
        input_sha256 = immutable_inputs[train_path]
        fold_sizes = dict(sorted(Counter(assignments.values()).items()))
        public_config = {
            key: value for key, value in config.items() if not key.startswith("_")
        }
        manifest = build_manifest(
            run_id="lopo_by_prompt",
            project_root=root,
            config=public_config,
            input_files=tuple(immutable_inputs),
            extra={
                "split_kind": "leave_one_prompt_out",
                "input_path": str(train_path),
                "input_sha256": input_sha256,
                "n_records": len(records),
                "n_splits": prompt_fold_contract["n_splits"],
                "fold_sizes": fold_sizes,
                "prompt_fold_contract": prompt_fold_contract,
                "prompt_fold_contract_sha256": sha256_json(prompt_fold_contract),
                "folds": str(output_path),
                "folds_sha256": sha256_file(output_path),
            },
        )
        write_manifest(manifest_path, manifest, create_only=True)
        created.append(manifest_path)
        _require_unchanged_files(immutable_inputs)
    except Exception:
        for created_path in reversed(created):
            created_path.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "folds": str(output_path),
                "manifest": str(manifest_path),
                "n_records": len(records),
                "n_splits": prompt_fold_contract["n_splits"],
                "fold_sizes": fold_sizes,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
