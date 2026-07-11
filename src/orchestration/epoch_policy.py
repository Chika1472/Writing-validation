"""Fixed-epoch selection that cannot consume outer-fold validation metrics.

The scorer trainer intentionally persists per-epoch outer-fold diagnostics.  Those
diagnostics are useful after an experiment, but choosing each fold's best epoch
from them would make OOF estimates optimistic.  This module therefore accepts
only one of two explicit policies:

* an epoch fixed before outer-fold training; or
* a single global epoch selected from separately declared inner-dev evidence.

The evidence contract is tamper-evident, not a proof of experimental conduct.
Callers remain responsible for constructing genuinely disjoint inner-dev splits.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.hashing import sha256_file, sha256_json


FIXED_EPOCH_POLICY_TYPE = "fixed_epoch_policy"
INNER_DEV_EVIDENCE_TYPE = "inner_dev_epoch_metrics"
SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_epoch(value: Any, *, field: str = "epoch") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _average_ranks(values: Mapping[int, float], *, higher_is_better: bool) -> dict[int, float]:
    """Return deterministic 1-based average ranks, preserving metric ties."""

    ordered = sorted(
        values.items(),
        key=lambda item: ((-item[1] if higher_is_better else item[1]), item[0]),
    )
    ranks: dict[int, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average = ((index + 1) + end) / 2.0
        for epoch, _ in ordered[index:end]:
            ranks[epoch] = average
        index = end
    return ranks


def _validated_inner_evidence(path: str | Path) -> dict[str, Any]:
    evidence_path = Path(path).resolve()
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"inner-dev evidence must be a JSON object: {evidence_path}")
    if payload.get("artifact_type") != INNER_DEV_EVIDENCE_TYPE:
        raise ValueError(
            f"inner-dev evidence has the wrong artifact_type: {evidence_path}"
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported inner-dev evidence schema: {evidence_path}")
    if payload.get("split_role") != "inner_dev":
        raise ValueError("epoch evidence split_role must be exactly 'inner_dev'")
    if payload.get("outer_holdout_labels_used") is not False:
        raise ValueError(
            "epoch evidence must explicitly declare outer_holdout_labels_used=false"
        )
    split_signature = payload.get("split_signature")
    if not isinstance(split_signature, str) or not split_signature.strip():
        raise ValueError("inner-dev evidence requires a nonempty split_signature")
    source_run_id = payload.get("source_run_id")
    if not isinstance(source_run_id, str) or not source_run_id.strip():
        raise ValueError("inner-dev evidence requires a nonempty source_run_id")

    metric_rows = payload.get("metrics")
    if not isinstance(metric_rows, list) or not metric_rows:
        raise ValueError("inner-dev evidence metrics must be a nonempty list")
    validated_metrics: list[dict[str, float | int]] = []
    seen_epochs: set[int] = set()
    for index, row in enumerate(metric_rows):
        if not isinstance(row, dict):
            raise ValueError(f"inner-dev metrics[{index}] must be an object")
        epoch = _positive_epoch(row.get("epoch"), field=f"metrics[{index}].epoch")
        if epoch in seen_epochs:
            raise ValueError(f"inner-dev evidence repeats epoch {epoch}")
        rmse = _finite_number(
            row.get("macro_rmse"), field=f"metrics[{index}].macro_rmse"
        )
        spearman = _finite_number(
            row.get("macro_spearman"), field=f"metrics[{index}].macro_spearman"
        )
        if rmse < 0:
            raise ValueError("macro_rmse must be non-negative")
        if not -1.0 <= spearman <= 1.0:
            raise ValueError("macro_spearman must be between -1 and 1")
        seen_epochs.add(epoch)
        validated_metrics.append(
            {"epoch": epoch, "macro_rmse": rmse, "macro_spearman": spearman}
        )
    return {
        **payload,
        "_path": evidence_path,
        "metrics": validated_metrics,
    }


def _signed_policy(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("policy_signature", None)
    return {**unsigned, "policy_signature": sha256_json(unsigned)}


def create_prespecified_policy(epoch: int, *, reason: str) -> dict[str, Any]:
    selected = _positive_epoch(epoch, field="fixed_epoch")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("a nonempty reason is required for a prespecified epoch")
    return _signed_policy(
        {
            "artifact_type": FIXED_EPOCH_POLICY_TYPE,
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "fixed_epoch": selected,
            "selection_source": "prespecified_before_outer_training",
            "selection_method": "manual_prespecification",
            "reason": reason.strip(),
            "outer_fold_metrics_used": False,
            "evidence": [],
        }
    )


def create_inner_dev_policy(
    evidence_paths: Sequence[str | Path],
    *,
    rmse_weight: float = 0.5,
    spearman_weight: float = 0.5,
) -> dict[str, Any]:
    if not evidence_paths:
        raise ValueError("at least one inner-dev evidence file is required")
    rmse_weight = _finite_number(rmse_weight, field="rmse_weight")
    spearman_weight = _finite_number(spearman_weight, field="spearman_weight")
    if rmse_weight < 0 or spearman_weight < 0 or rmse_weight + spearman_weight <= 0:
        raise ValueError("metric weights must be non-negative with a positive sum")
    total_weight = rmse_weight + spearman_weight
    rmse_weight /= total_weight
    spearman_weight /= total_weight

    evidence = [_validated_inner_evidence(path) for path in evidence_paths]
    paths = [item["_path"] for item in evidence]
    if len(set(paths)) != len(paths):
        raise ValueError("inner-dev evidence paths must be unique")
    source_ids = [str(item["source_run_id"]) for item in evidence]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("inner-dev evidence source_run_id values must be unique")

    candidate_sets = [
        {int(row["epoch"]) for row in item["metrics"]} for item in evidence
    ]
    candidates = candidate_sets[0]
    if any(candidate_set != candidates for candidate_set in candidate_sets[1:]):
        raise ValueError("all inner-dev evidence must evaluate the same epoch candidates")

    rank_totals = {epoch: 0.0 for epoch in candidates}
    per_evidence: list[dict[str, Any]] = []
    for item in evidence:
        rmse = {int(row["epoch"]): float(row["macro_rmse"]) for row in item["metrics"]}
        spearman = {
            int(row["epoch"]): float(row["macro_spearman"])
            for row in item["metrics"]
        }
        rmse_ranks = _average_ranks(rmse, higher_is_better=False)
        spearman_ranks = _average_ranks(spearman, higher_is_better=True)
        combined = {
            epoch: rmse_weight * rmse_ranks[epoch]
            + spearman_weight * spearman_ranks[epoch]
            for epoch in candidates
        }
        for epoch, value in combined.items():
            rank_totals[epoch] += value
        evidence_path = Path(item["_path"])
        per_evidence.append(
            {
                "path": str(evidence_path),
                "sha256": sha256_file(evidence_path),
                "source_run_id": item["source_run_id"],
                "split_signature": item["split_signature"],
                "combined_rank": {str(epoch): combined[epoch] for epoch in sorted(combined)},
            }
        )

    mean_ranks = {
        epoch: rank_totals[epoch] / len(evidence) for epoch in sorted(candidates)
    }
    fixed_epoch = min(mean_ranks, key=lambda epoch: (mean_ranks[epoch], epoch))
    return _signed_policy(
        {
            "artifact_type": FIXED_EPOCH_POLICY_TYPE,
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "fixed_epoch": fixed_epoch,
            "selection_source": "inner_dev_only",
            "selection_method": "mean_weighted_within_evidence_metric_rank",
            "metric_weights": {
                "macro_rmse": rmse_weight,
                "macro_spearman": spearman_weight,
            },
            "candidate_mean_rank": {
                str(epoch): mean_ranks[epoch] for epoch in sorted(mean_ranks)
            },
            "outer_fold_metrics_used": False,
            "evidence": per_evidence,
        }
    )


def validate_epoch_policy(
    policy: Mapping[str, Any], *, max_epoch: int | None = None
) -> dict[str, Any]:
    payload = dict(policy)
    if payload.get("artifact_type") != FIXED_EPOCH_POLICY_TYPE:
        raise ValueError("not a fixed-epoch policy artifact")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported fixed-epoch policy schema")
    fixed_epoch = _positive_epoch(payload.get("fixed_epoch"), field="fixed_epoch")
    if max_epoch is not None and fixed_epoch > _positive_epoch(
        max_epoch, field="max_epoch"
    ):
        raise ValueError(
            f"fixed_epoch={fixed_epoch} exceeds configured training epochs={max_epoch}"
        )
    if payload.get("outer_fold_metrics_used") is not False:
        raise ValueError("fixed-epoch policy must declare outer_fold_metrics_used=false")
    source = payload.get("selection_source")
    if source not in {"prespecified_before_outer_training", "inner_dev_only"}:
        raise ValueError("fixed-epoch policy has an unsafe selection_source")
    signature = payload.get("policy_signature")
    unsigned = dict(payload)
    unsigned.pop("policy_signature", None)
    if not isinstance(signature, str) or signature != sha256_json(unsigned):
        raise ValueError("fixed-epoch policy signature mismatch")

    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("fixed-epoch policy evidence must be a list")
    if source == "prespecified_before_outer_training" and evidence:
        raise ValueError("a prespecified policy must not contain metric evidence")
    if source == "inner_dev_only" and not evidence:
        raise ValueError("an inner-dev policy must contain evidence")
    if source == "prespecified_before_outer_training":
        if payload.get("selection_method") != "manual_prespecification":
            raise ValueError("prespecified epoch policy has an invalid selection_method")
        if not isinstance(payload.get("reason"), str) or not payload["reason"].strip():
            raise ValueError("prespecified epoch policy requires a nonempty reason")
    if source == "inner_dev_only":
        if (
            payload.get("selection_method")
            != "mean_weighted_within_evidence_metric_rank"
        ):
            raise ValueError("inner-dev epoch policy has an invalid selection_method")
        candidate_ranks = payload.get("candidate_mean_rank")
        if not isinstance(candidate_ranks, dict) or not candidate_ranks:
            raise ValueError("inner-dev policy requires candidate_mean_rank")
        validated_ranks: dict[int, float] = {}
        for raw_epoch, raw_rank in candidate_ranks.items():
            try:
                epoch = int(raw_epoch)
            except (TypeError, ValueError) as error:
                raise ValueError("candidate_mean_rank keys must be epochs") from error
            if str(epoch) != str(raw_epoch) or epoch <= 0 or epoch in validated_ranks:
                raise ValueError("candidate_mean_rank contains an invalid epoch")
            rank = _finite_number(raw_rank, field=f"candidate_mean_rank.{raw_epoch}")
            if rank <= 0:
                raise ValueError("candidate mean ranks must be positive")
            validated_ranks[epoch] = rank
        selected_from_ranks = min(
            validated_ranks,
            key=lambda epoch: (validated_ranks[epoch], epoch),
        )
        if fixed_epoch != selected_from_ranks:
            raise ValueError("fixed_epoch is not the deterministic best inner-dev rank")
    for index, reference in enumerate(evidence):
        if not isinstance(reference, dict):
            raise ValueError(f"policy evidence[{index}] must be an object")
        path = reference.get("path")
        digest = reference.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError(f"policy evidence[{index}] lacks path/sha256")
        evidence_path = Path(path).resolve()
        if not evidence_path.is_file() or sha256_file(evidence_path) != digest:
            raise ValueError(f"policy evidence file drifted or is missing: {evidence_path}")
        restored_evidence = _validated_inner_evidence(evidence_path)
        if (
            reference.get("source_run_id") != restored_evidence["source_run_id"]
            or reference.get("split_signature") != restored_evidence["split_signature"]
        ):
            raise ValueError(
                f"policy evidence identity does not match its file: {evidence_path}"
            )
    return payload


def load_epoch_policy(path: str | Path, *, max_epoch: int | None = None) -> dict[str, Any]:
    policy_path = Path(path).resolve()
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"fixed-epoch policy must be a JSON object: {policy_path}")
    return validate_epoch_policy(payload, max_epoch=max_epoch)


def write_epoch_policy(path: str | Path, policy: Mapping[str, Any]) -> Path:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"fixed-epoch policy already exists: {target}")
    validated = validate_epoch_policy(policy)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(validated, ensure_ascii=False, indent=2) + "\n")
    return target


__all__ = [
    "FIXED_EPOCH_POLICY_TYPE",
    "INNER_DEV_EVIDENCE_TYPE",
    "create_inner_dev_policy",
    "create_prespecified_policy",
    "load_epoch_policy",
    "validate_epoch_policy",
    "write_epoch_policy",
]
