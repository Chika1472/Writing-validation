from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.rationale_judge import (
    DECISION_FIELDS,
    JUDGE_CODE_FILES,
    JUDGE_SCHEMA_VERSION,
    LocalGGUFJudge,
    code_contract,
    judge_generation_contract,
    judge_one_order,
    judge_prompt_contract,
    load_verified_review_pack,
    reconcile_order_judgments,
    validate_judge_config,
    validate_result_rows,
)
from src.utils.config import load_yaml
from src.utils.hashing import sha256_file, sha256_json
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Blindly judge a rationale review pack with a local GGUF model in both "
            "A/B orders. This command deliberately has no assignment-key argument."
        )
    )
    parser.add_argument("--config", default="configs/rationale_judge.yaml")
    parser.add_argument("--review-pack", required=True)
    parser.add_argument(
        "--review-manifest",
        help="Defaults to REVIEW_PACK with its suffix replaced by .manifest.json.",
    )
    parser.add_argument("--model", required=True, help="Path to a local GGUF file.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        return None


def _write_jsonl_atomic(path: Path, temporary: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
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
    temporary.replace(path)


def _require_unchanged_files(expected_sha256: dict[Path, str]) -> None:
    for path, expected in expected_sha256.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"immutable judge input changed during the run: {path}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_file_sha256 = sha256_file(config_path)
    config = load_yaml(config_path)
    _require_unchanged_files({config_path: config_file_sha256})
    judge_config = validate_judge_config(config)
    project_root = Path(__file__).resolve().parents[1]
    configured_project_root = Path(config["_project_root"]).resolve()
    if configured_project_root != project_root:
        raise ValueError(
            "config project_root must resolve to the project containing this judge "
            f"script: expected {project_root}, got {configured_project_root}"
        )

    review_path = Path(args.review_pack).resolve()
    review_manifest_path = (
        Path(args.review_manifest).resolve()
        if args.review_manifest
        else review_path.with_suffix(".manifest.json")
    )
    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()
    output_manifest_path = output_path.with_suffix(
        output_path.suffix + ".manifest.json"
    )
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    manifest_temporary_path = output_manifest_path.with_suffix(
        output_manifest_path.suffix + ".tmp"
    )
    require_distinct_paths(
        config=config_path,
        review_pack=review_path,
        review_manifest=review_manifest_path,
        model=model_path,
        output=output_path,
        output_manifest=output_manifest_path,
        temporary=temporary_path,
        manifest_temporary=manifest_temporary_path,
    )
    require_new_paths(
        output=output_path,
        output_manifest=output_manifest_path,
        temporary=temporary_path,
        manifest_temporary=manifest_temporary_path,
    )
    if not model_path.is_file() or model_path.suffix.lower() != ".gguf":
        raise FileNotFoundError(f"--model must be an existing .gguf file: {model_path}")

    review_input_sha256 = {
        review_path: sha256_file(review_path),
        review_manifest_path: sha256_file(review_manifest_path),
    }
    review_rows, review_manifest = load_verified_review_pack(
        review_path,
        review_manifest_path,
    )
    _require_unchanged_files(review_input_sha256)
    # Hash before model loading so the manifest identifies the bytes actually selected.
    gguf_sha256 = sha256_file(model_path)
    immutable_input_sha256 = {
        config_path: config_file_sha256,
        **review_input_sha256,
        model_path: gguf_sha256,
    }
    prompt_contract = judge_prompt_contract(judge_config["reason_max_chars"])
    source_contract = code_contract(project_root, files=JUDGE_CODE_FILES)
    llama_cpp_version = _package_version()
    if llama_cpp_version is None:
        raise RuntimeError("llama-cpp-python package metadata is unavailable")
    generation_contract = judge_generation_contract(
        judge_config,
        llama_cpp_python_version=llama_cpp_version,
    )

    judge = LocalGGUFJudge(
        model_path,
        runtime=judge_config["runtime"],
        seed=judge_config["seed"],
    )
    judged_rows: list[dict] = []
    unstable_by_field = {field: 0 for field in DECISION_FIELDS}
    unstable_rows = 0
    for row in review_rows:
        forward = judge_one_order(
            row,
            reverse=False,
            base_seed=judge_config["seed"],
            reason_max_chars=judge_config["reason_max_chars"],
            generation=judge_config["generation"],
            complete=judge.complete,
        )
        reverse = judge_one_order(
            row,
            reverse=True,
            base_seed=judge_config["seed"],
            reason_max_chars=judge_config["reason_max_chars"],
            generation=judge_config["generation"],
            complete=judge.complete,
        )
        consensus, unstable = reconcile_order_judgments(
            forward["normalized_judgment"],
            reverse["normalized_judgment"],
            reason_max_chars=judge_config["reason_max_chars"],
        )
        if unstable:
            unstable_rows += 1
        for field in DECISION_FIELDS:
            unstable_by_field[field] += int(consensus[field]["unstable"])
        judged_rows.append(
            {
                "review_id": row["review_id"],
                "review_row_sha256": sha256_json(row),
                "orders": {"ab": forward, "ba": reverse},
                "consensus": consensus,
                "unstable": unstable,
            }
        )

    judged_rows = validate_result_rows(
        judged_rows,
        reason_max_chars=judge_config["reason_max_chars"],
        base_seed=judge_config["seed"],
        max_attempts=judge_config["generation"]["max_attempts"],
    )
    _require_unchanged_files(immutable_input_sha256)
    _write_jsonl_atomic(output_path, temporary_path, judged_rows)
    manifest = build_manifest(
        run_id=output_path.stem,
        project_root=project_root,
        config={key: value for key, value in config.items() if not key.startswith("_")},
        input_files=(),
        extra={
            "artifact_type": "local_blind_rationale_judgments",
            "schema_version": JUDGE_SCHEMA_VERSION,
            "result_file": output_path.name,
            "result_sha256": sha256_file(output_path),
            "rows": len(judged_rows),
            "unstable_rows": unstable_rows,
            "unstable_by_field": unstable_by_field,
            "review_pack_sha256": review_input_sha256[review_path],
            "review_pack_manifest_sha256": review_input_sha256[
                review_manifest_path
            ],
            "review_pack_manifest_verified": True,
            "expected_review_key_file": review_manifest["key_file"],
            "expected_review_key_sha256": review_manifest["key_sha256"],
            "assignment_key_read_by_judge": False,
            "assignment_key_cli_supported": False,
            "local_gguf_only": True,
            "llama_cpp_python_version": llama_cpp_version,
            "prompt_contract": prompt_contract,
            "prompt_contract_sha256": sha256_json(prompt_contract),
            "code_contract": source_contract,
            "code_contract_sha256": sha256_json(source_contract),
            "generation_contract": generation_contract,
            "generation_contract_sha256": sha256_json(generation_contract),
        },
    )
    _require_unchanged_files(immutable_input_sha256)
    # Reuse the stable start-of-run identities, including the large GGUF, so the
    # manifest cannot silently describe bytes other than those selected for judging.
    manifest["inputs"] = {
        str(path): digest for path, digest in immutable_input_sha256.items()
    }
    manifest["gguf_file"] = model_path.name
    manifest["gguf_sha256"] = gguf_sha256
    write_manifest(output_manifest_path, manifest)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "rows": len(judged_rows),
                "unstable_rows": unstable_rows,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
