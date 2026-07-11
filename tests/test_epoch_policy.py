from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.orchestration.epoch_policy import (
    INNER_DEV_EVIDENCE_TYPE,
    create_inner_dev_policy,
    create_prespecified_policy,
    load_epoch_policy,
    write_epoch_policy,
)


def _write_evidence(
    path: Path,
    *,
    source_run_id: str,
    outer_labels_used: bool = False,
) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_type": INNER_DEV_EVIDENCE_TYPE,
                "schema_version": 1,
                "split_role": "inner_dev",
                "outer_holdout_labels_used": outer_labels_used,
                "source_run_id": source_run_id,
                "split_signature": f"split-{source_run_id}",
                "metrics": [
                    {"epoch": 1, "macro_rmse": 0.90, "macro_spearman": 0.50},
                    {"epoch": 2, "macro_rmse": 0.70, "macro_spearman": 0.65},
                    {"epoch": 3, "macro_rmse": 0.75, "macro_spearman": 0.60},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_prespecified_policy_roundtrip_and_signature_guard(tmp_path: Path) -> None:
    path = tmp_path / "epoch_policy.json"
    policy = create_prespecified_policy(2, reason="chosen from an earlier experiment")
    write_epoch_policy(path, policy)

    assert load_epoch_policy(path, max_epoch=4)["fixed_epoch"] == 2

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["fixed_epoch"] = 3
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="signature mismatch"):
        load_epoch_policy(path, max_epoch=4)


def test_inner_dev_policy_uses_one_global_epoch_and_binds_evidence(tmp_path: Path) -> None:
    first = tmp_path / "inner_a.json"
    second = tmp_path / "inner_b.json"
    _write_evidence(first, source_run_id="inner-a")
    _write_evidence(second, source_run_id="inner-b")

    policy = create_inner_dev_policy([first, second])
    output = tmp_path / "selected.json"
    write_epoch_policy(output, policy)
    assert load_epoch_policy(output)["fixed_epoch"] == 2

    first.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="drifted or is missing"):
        load_epoch_policy(output)


def test_inner_dev_policy_rejects_outer_holdout_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "unsafe.json"
    _write_evidence(
        evidence,
        source_run_id="unsafe",
        outer_labels_used=True,
    )
    with pytest.raises(ValueError, match="outer_holdout_labels_used=false"):
        create_inner_dev_policy([evidence])
