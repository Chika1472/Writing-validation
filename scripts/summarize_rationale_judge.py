from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.rationale_judge import (
    JUDGE_CODE_FILES,
    JUDGE_SCHEMA_VERSION,
    SUMMARY_CODE_FILES,
    RationaleJudgeValidationError,
    code_contract,
    judge_generation_contract,
    judge_prompt_contract,
    read_key_rows,
    read_result_rows,
    strict_json_object,
    summarize_keyed_results,
)
from src.utils.hashing import sha256_file, sha256_json
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths


_SHA256 = re.compile(r"[0-9a-f]{64}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join already-completed blind local judgments with the hidden assignment "
            "key and report candidate/baseline wins. The key is read only in this stage."
        )
    )
    parser.add_argument("--judgments", required=True)
    parser.add_argument(
        "--judgment-manifest",
        help="Defaults to JUDGMENTS plus .manifest.json.",
    )
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load_manifest(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise RationaleJudgeValidationError(
            f"judgment manifest is not valid UTF-8: {path}"
        ) from error
    return strict_json_object(text, source=str(path))


def _validate_judgment_manifest(
    manifest: dict,
    *,
    judgments_path: Path,
) -> dict[str, object]:
    rows = manifest.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise RationaleJudgeValidationError(
            "judgment manifest rows must be a positive integer"
        )
    expected = {
        "artifact_type": "local_blind_rationale_judgments",
        "schema_version": JUDGE_SCHEMA_VERSION,
        "result_file": judgments_path.name,
        "result_sha256": sha256_file(judgments_path),
        "assignment_key_read_by_judge": False,
        "assignment_key_cli_supported": False,
        "review_pack_manifest_verified": True,
        "local_gguf_only": True,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise RationaleJudgeValidationError(
                f"judgment manifest mismatch for {field}: expected {value!r}"
            )
    for field in (
        "gguf_sha256",
        "prompt_contract_sha256",
        "code_contract_sha256",
        "generation_contract_sha256",
    ):
        value = manifest.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise RationaleJudgeValidationError(
                f"judgment manifest lacks valid {field}"
            )
    for contract_field, hash_field in (
        ("prompt_contract", "prompt_contract_sha256"),
        ("code_contract", "code_contract_sha256"),
        ("generation_contract", "generation_contract_sha256"),
    ):
        if sha256_json(manifest.get(contract_field)) != manifest[hash_field]:
            raise RationaleJudgeValidationError(
                f"judgment manifest {contract_field} does not match {hash_field}"
            )
    prompt_contract = manifest.get("prompt_contract")
    if not isinstance(prompt_contract, dict):
        raise RationaleJudgeValidationError(
            "judgment manifest prompt_contract must be an object"
        )
    reason_max_chars = prompt_contract.get("reason_max_chars")
    try:
        expected_prompt_contract = judge_prompt_contract(reason_max_chars)
    except (TypeError, ValueError) as error:
        raise RationaleJudgeValidationError(
            "judgment manifest prompt_contract is invalid"
        ) from error
    if prompt_contract != expected_prompt_contract:
        raise RationaleJudgeValidationError(
            "judgment manifest prompt_contract does not match the judge schema"
        )

    source_code_contract = manifest.get("code_contract")
    if (
        not isinstance(source_code_contract, dict)
        or set(source_code_contract) != set(JUDGE_CODE_FILES)
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in source_code_contract.values()
        )
    ):
        raise RationaleJudgeValidationError(
            "judgment manifest code_contract has invalid files or hashes"
        )

    llama_cpp_version = manifest.get("llama_cpp_python_version")
    if not isinstance(llama_cpp_version, str) or not llama_cpp_version.strip():
        raise RationaleJudgeValidationError(
            "judgment manifest lacks llama_cpp_python_version"
        )
    generation_contract = manifest.get("generation_contract")
    if not isinstance(generation_contract, dict):
        raise RationaleJudgeValidationError(
            "judgment manifest generation_contract must be an object"
        )
    try:
        expected_generation_contract = judge_generation_contract(
            {
                "seed": generation_contract.get("seed"),
                "reason_max_chars": generation_contract.get("reason_max_chars"),
                "runtime": generation_contract.get("runtime"),
                "generation": generation_contract.get("generation"),
            },
            llama_cpp_python_version=llama_cpp_version,
        )
    except (TypeError, ValueError) as error:
        raise RationaleJudgeValidationError(
            "judgment manifest generation_contract is invalid"
        ) from error
    if generation_contract != expected_generation_contract:
        raise RationaleJudgeValidationError(
            "judgment manifest generation_contract does not match the judge schema"
        )
    if generation_contract["reason_max_chars"] != reason_max_chars:
        raise RationaleJudgeValidationError(
            "prompt and generation reason_max_chars contracts differ"
        )
    key_sha256 = manifest.get("expected_review_key_sha256")
    if not isinstance(key_sha256, str) or _SHA256.fullmatch(key_sha256) is None:
        raise RationaleJudgeValidationError(
            "judgment manifest lacks expected_review_key_sha256"
        )
    return {
        "rows": rows,
        "key_sha256": key_sha256,
        "reason_max_chars": reason_max_chars,
        "base_seed": generation_contract["seed"],
        "max_attempts": generation_contract["generation"]["max_attempts"],
    }


def _write_json_atomic(path: Path, temporary: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_unchanged_files(expected_sha256: dict[Path, str]) -> None:
    for path, expected in expected_sha256.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"immutable summary input changed during the run: {path}")


def main() -> None:
    args = parse_args()
    judgments_path = Path(args.judgments).resolve()
    judgment_manifest_path = (
        Path(args.judgment_manifest).resolve()
        if args.judgment_manifest
        else judgments_path.with_suffix(judgments_path.suffix + ".manifest.json")
    )
    key_path = Path(args.key).resolve()
    output_path = Path(args.output).resolve()
    output_manifest_path = output_path.with_suffix(
        output_path.suffix + ".manifest.json"
    )
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    manifest_temporary_path = output_manifest_path.with_suffix(
        output_manifest_path.suffix + ".tmp"
    )
    require_distinct_paths(
        judgments=judgments_path,
        judgment_manifest=judgment_manifest_path,
        key=key_path,
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

    immutable_input_sha256 = {
        judgments_path: sha256_file(judgments_path),
        judgment_manifest_path: sha256_file(judgment_manifest_path),
        key_path: sha256_file(key_path),
    }
    judgment_manifest = _load_manifest(judgment_manifest_path)
    manifest_contract = _validate_judgment_manifest(
        judgment_manifest,
        judgments_path=judgments_path,
    )
    result_rows = read_result_rows(
        judgments_path,
        reason_max_chars=int(manifest_contract["reason_max_chars"]),
        base_seed=int(manifest_contract["base_seed"]),
        max_attempts=int(manifest_contract["max_attempts"]),
    )
    if len(result_rows) != manifest_contract["rows"]:
        raise RationaleJudgeValidationError(
            "judgment row count does not match the judgment manifest"
        )
    expected_key_sha256 = str(manifest_contract["key_sha256"])
    actual_key_sha256 = immutable_input_sha256[key_path]
    if actual_key_sha256 != expected_key_sha256:
        raise RationaleJudgeValidationError(
            "assignment key hash does not match the verified review-pack manifest"
        )
    key_by_id = read_key_rows(key_path)
    summary = summarize_keyed_results(result_rows, key_by_id)
    _require_unchanged_files(immutable_input_sha256)
    summary.update(
        {
            "judge_result_sha256": immutable_input_sha256[judgments_path],
            "judge_manifest_sha256": immutable_input_sha256[
                judgment_manifest_path
            ],
            "assignment_key_sha256": actual_key_sha256,
        }
    )
    _write_json_atomic(output_path, temporary_path, summary)

    project_root = Path(__file__).resolve().parents[1]
    source_contract = code_contract(project_root, files=SUMMARY_CODE_FILES)
    manifest = build_manifest(
        run_id=output_path.stem,
        project_root=project_root,
        config={
            "stage": "keyed_summary_only",
            "judge_schema_version": JUDGE_SCHEMA_VERSION,
        },
        input_files=(),
        extra={
            "artifact_type": "rationale_judge_keyed_summary_manifest",
            "summary_file": output_path.name,
            "summary_sha256": sha256_file(output_path),
            "rows": summary["rows"],
            "key_read_stage": "summary_only",
            "source_judge_code_contract_sha256": judgment_manifest[
                "code_contract_sha256"
            ],
            "source_prompt_contract_sha256": judgment_manifest[
                "prompt_contract_sha256"
            ],
            "source_generation_contract_sha256": judgment_manifest[
                "generation_contract_sha256"
            ],
            "source_gguf_sha256": judgment_manifest["gguf_sha256"],
            "code_contract": source_contract,
            "code_contract_sha256": sha256_json(source_contract),
        },
    )
    _require_unchanged_files(immutable_input_sha256)
    manifest["inputs"] = {
        str(path): digest for path, digest in immutable_input_sha256.items()
    }
    write_manifest(output_manifest_path, manifest)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "rows": summary["rows"],
                "overall": summary["criteria"]["overall"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
