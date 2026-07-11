from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load import load_inference_jsonl
from src.inference.deployment import (
    configure_offline_environment,
    is_runtime_cache_file,
    load_deployment_config,
    validate_artifact_contracts,
    validate_dependency_lock,
    validate_package_manifest,
)
from src.inference.serializer import TRAITS, serialize_prediction, strict_parse_prediction
from src.utils.hashing import sha256_file
from src.utils.paths import require_distinct_paths, require_new_paths, require_outside_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline L40S end-to-end smoke test. It validates strict output, input "
            "immutability, resource use, and optional two-load score determinism."
        )
    )
    parser.add_argument("--config", default="configs/inference_l40s.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Defaults to runtime.expected_rows (400 in the L40S template).",
    )
    parser.add_argument("--offline", action="store_true", help="Required safety acknowledgement.")
    parser.add_argument("--strict", action="store_true", help="Required strict-schema acknowledgement.")
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Run a second fresh model load and require bitwise-identical score matrices.",
    )
    parser.add_argument(
        "--require-package-manifest",
        action="store_true",
        help="Fail unless package.manifest.json is configured and valid.",
    )
    return parser.parse_args()


@contextlib.contextmanager
def _deny_network() -> Iterator[None]:
    """Raise on DNS lookup or outbound socket connection during the smoke run."""

    original_create = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def denied(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("network access attempted during mandatory offline smoke test")

    socket.create_connection = denied
    socket.getaddrinfo = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    try:
        yield
    finally:
        socket.create_connection = original_create
        socket.getaddrinfo = original_getaddrinfo
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex


def _score_digest(matrix: np.ndarray) -> str:
    canonical = np.ascontiguousarray(matrix, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _validate_result(result: Any, records: list[Any]) -> dict[str, Any]:
    if len(result.rows) != len(records):
        raise RuntimeError("submission result row count changed")
    expected_ids = [record.id for record in records]
    actual_ids = [row.get("id") for row in result.rows]
    if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise RuntimeError("submission result IDs/order do not match the input")
    expected_prompts = [record.prompt_num for record in records]
    if [row.get("prompt_num") for row in result.rows] != expected_prompts:
        raise RuntimeError("submission result prompt_num/order do not match the input")
    strict_count = 0
    rationale_lengths: list[int] = []
    for row in result.rows:
        if set(row) != {"id", "prompt_num", "prediction", "model"}:
            raise ValueError("final row has an unexpected outer schema")
        parsed = strict_parse_prediction(serialize_prediction(row["prediction"]))
        if parsed != row["prediction"]:
            raise RuntimeError("strict serialization changed a prediction")
        strict_count += 1
        rationale_lengths.extend(len(str(parsed[trait]["rationale"])) for trait in TRAITS)
    matrix = np.asarray(result.score_matrix, dtype=float)
    if matrix.shape != (len(records), len(TRAITS)) or not np.isfinite(matrix).all():
        raise RuntimeError("submission result contains an invalid score matrix")
    for index, row in enumerate(result.rows):
        for trait_index, trait in enumerate(TRAITS):
            if float(row["prediction"][trait]["score"]) != float(matrix[index, trait_index]):
                raise RuntimeError(f"final score differs from score pass: {row['id']}/{trait}")
    return {
        "strict_rows": strict_count,
        "score_sha256": _score_digest(matrix),
        "rationale_char_min": min(rationale_lengths),
        "rationale_char_max": max(rationale_lengths),
        "fallback_count": int(result.fallback_count),
    }


def _run_pass(config: Any, artifacts: Any, records: list[Any]) -> tuple[Any, dict[str, Any]]:
    import torch
    from src.inference.submission import SubmissionEngine

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    engine = SubmissionEngine(config, artifacts)
    result = engine.predict(records)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    resources = {
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_mb": (
            torch.cuda.max_memory_allocated(0) / (1024**2)
            if torch.cuda.is_available()
            else 0.0
        ),
        "peak_cuda_reserved_mb": (
            torch.cuda.max_memory_reserved(0) / (1024**2)
            if torch.cuda.is_available()
            else 0.0
        ),
    }
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, resources


def main() -> None:
    args = parse_args()
    if not args.offline:
        raise ValueError("--offline is mandatory; smoke testing with network access is forbidden")
    if not args.strict:
        raise ValueError("--strict is mandatory; permissive parsing is not a smoke test")
    if args.expected_count is not None and args.expected_count <= 0:
        raise ValueError("--expected-count must be positive")
    config_path = Path(args.config).resolve()
    input_path = Path(args.input).resolve()
    report_path = Path(args.report).resolve()
    require_distinct_paths(config=config_path, input=input_path, report=report_path)
    require_new_paths(report=report_path)

    config = load_deployment_config(config_path)
    if args.require_package_manifest and config.runtime.package_manifest is None:
        raise ValueError("--require-package-manifest was set, but config has none")
    expected_count = args.expected_count or config.runtime.expected_rows
    if expected_count != config.runtime.expected_rows:
        raise ValueError(
            "--expected-count must equal runtime.expected_rows so engine and smoke contracts agree"
        )
    configure_offline_environment(config)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    package_manifest = validate_package_manifest(config)
    installed_versions = validate_dependency_lock(config)
    artifacts = validate_artifact_contracts(config)
    signed_package_files = (
        tuple(
            (config.project_root / relative).resolve()
            for relative in package_manifest["files"]
        )
        if package_manifest is not None
        else ()
    )
    package_file_set_before = (
        {
            path.relative_to(config.project_root).as_posix()
            for path in config.project_root.rglob("*")
            if path.is_file()
            and not is_runtime_cache_file(path.relative_to(config.project_root))
        }
        if package_manifest is not None
        else None
    )
    immutable_files = tuple(
        dict.fromkeys(
            (
                config_path,
                input_path,
                *artifacts.all_files,
                *signed_package_files,
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
            )
        )
    )
    immutable_hashes = {str(path): sha256_file(path) for path in immutable_files}
    immutable_roots = {
        "input": input_path,
        **{
            f"artifact_root_{index}": path
            for index, path in enumerate(artifacts.immutable_roots)
        },
    }
    if config.runtime.hf_home is not None:
        immutable_roots["hf_home"] = config.runtime.hf_home
    if config.runtime.package_manifest is not None:
        immutable_roots["signed_package"] = config.project_root
    require_outside_roots(
        immutable_roots,
        report=report_path,
    )
    records = load_inference_jsonl(input_path)
    if len(records) != expected_count:
        raise ValueError(f"expected {expected_count} rows, got {len(records)}")

    import torch

    device_report = {
        "cuda_available": torch.cuda.is_available(),
        "visible_cuda_devices": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
    }
    with _deny_network():
        first, first_resources = _run_pass(config, artifacts, records)
        first_validation = _validate_result(first, records)
        second = None
        second_resources = None
        second_validation = None
        if args.verify_determinism:
            second, second_resources = _run_pass(config, artifacts, records)
            second_validation = _validate_result(second, records)
            if first.score_matrix.shape != second.score_matrix.shape or not np.array_equal(
                first.score_matrix, second.score_matrix
            ):
                raise RuntimeError("score matrices are not bitwise identical across fresh loads")
            if first_validation["score_sha256"] != second_validation["score_sha256"]:
                raise RuntimeError("score byte digests differ across fresh loads")

    changed = {}
    for path, digest in immutable_hashes.items():
        candidate = Path(path)
        after = sha256_file(candidate) if candidate.is_file() else None
        if after != digest:
            changed[path] = {"before": digest, "after": after}
    if package_file_set_before is not None:
        package_file_set_after = {
            path.relative_to(config.project_root).as_posix()
            for path in config.project_root.rglob("*")
            if path.is_file()
            and not is_runtime_cache_file(path.relative_to(config.project_root))
        }
        if package_file_set_after != package_file_set_before:
            changed["<package-file-set>"] = {
                "added": sorted(package_file_set_after - package_file_set_before),
                "removed": sorted(package_file_set_before - package_file_set_after),
            }
    if changed:
        raise RuntimeError(f"smoke test modified an immutable input/artifact: {changed}")
    report = {
        "artifact_type": "submission_offline_smoke_report",
        "passed": True,
        "offline_enforced": True,
        "network_guard_active_during_inference": True,
        "strict_schema_verified": True,
        "immutable_inputs_verified": True,
        "expected_rows": expected_count,
        "actual_rows": len(records),
        "package_manifest_required": bool(args.require_package_manifest),
        "package_signature": (
            package_manifest.get("package_signature") if package_manifest else None
        ),
        "deployment_config_sha256": sha256_file(config_path),
        "locked_runtime_versions": installed_versions,
        "input_sha256": sha256_file(input_path),
        "device": device_report,
        "first_pass": {**first_resources, **first_validation},
        "determinism_checked": bool(args.verify_determinism),
        "second_pass": (
            {**second_resources, **second_validation}
            if second_resources is not None and second_validation is not None
            else None
        ),
        "score_deterministic": bool(args.verify_determinism),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    require_new_paths(temporary_report=temporary)
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
