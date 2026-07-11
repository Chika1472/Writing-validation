from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.epoch_policy import (
    create_inner_dev_policy,
    create_prespecified_policy,
    write_epoch_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a signed global fixed-epoch policy without consulting outer-fold "
            "validation metrics."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--preselected-epoch",
        type=int,
        help="Epoch chosen before outer-fold training.",
    )
    source.add_argument(
        "--inner-evidence",
        action="append",
        help=(
            "Repeat for independently generated inner-dev metric artifacts. Each must "
            "declare artifact_type=inner_dev_epoch_metrics, split_role=inner_dev, and "
            "outer_holdout_labels_used=false."
        ),
    )
    parser.add_argument("--reason", default=None)
    parser.add_argument("--rmse-weight", type=float, default=0.5)
    parser.add_argument("--spearman-weight", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preselected_epoch is not None:
        if args.reason is None or not args.reason.strip():
            raise ValueError("--reason is required with --preselected-epoch")
        policy = create_prespecified_policy(
            args.preselected_epoch,
            reason=args.reason,
        )
    else:
        if args.reason is not None:
            raise ValueError("--reason applies only to --preselected-epoch")
        policy = create_inner_dev_policy(
            args.inner_evidence,
            rmse_weight=args.rmse_weight,
            spearman_weight=args.spearman_weight,
        )
    output = write_epoch_policy(args.output, policy)
    print(
        json.dumps(
            {
                "epoch_policy": str(output),
                "fixed_epoch": policy["fixed_epoch"],
                "selection_source": policy["selection_source"],
                "policy_signature": policy["policy_signature"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
