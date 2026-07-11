from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration.affine import AffinePromptCalibrator, DOMAINS
from src.calibration.contracts import calibration_inference_code_contract
from src.data.load import load_inference_jsonl
from src.evaluation.oof_provenance import checkpoint_ensemble_signature
from src.evaluation.prediction_provenance import prediction_provenance_fields
from src.evaluation.predictions import prediction_records, write_predictions
from src.inference.scorer import (
    checkpoint_artifact_files,
    load_scorer_checkpoint,
    predict_scores,
    resolve_checkpoint,
)
from src.utils.hashing import sha256_file, sha256_json
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import (
    require_distinct_paths,
    require_new_paths,
    require_outside_roots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one or more Qwen scorer checkpoints sequentially on one GPU."
    )
    parser.add_argument("--input", required=True, help="Labeled or unlabeled essay JSONL.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Repeat an epoch directory or best-pointer JSON to form an equal-weight ensemble.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibrator", default=None)
    parser.add_argument("--precision", choices=("4bit", "bf16"), default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--model-name", default="qwen3_scorer_ensemble")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_manifest_path = output_path.with_suffix(".manifest.json")
    calibrator_path = Path(args.calibrator).resolve() if args.calibrator else None
    calibrator_manifest_path = (
        calibrator_path.with_suffix(".manifest.json") if calibrator_path else None
    )
    records = load_inference_jsonl(input_path)
    resolved_checkpoints = [resolve_checkpoint(value) for value in args.checkpoint]
    path_contract = {
        "input": input_path,
        "output": output_path,
        "output_manifest": output_manifest_path,
        "calibrator": calibrator_path,
        "calibrator_manifest": calibrator_manifest_path,
    }
    path_contract.update(
        {
            f"checkpoint_{index}": checkpoint
            for index, checkpoint in enumerate(resolved_checkpoints)
        }
    )
    require_distinct_paths(**path_contract)
    require_outside_roots(
        {
            f"checkpoint_{index}": checkpoint
            for index, checkpoint in enumerate(resolved_checkpoints)
        },
        output=output_path,
        output_manifest=output_manifest_path,
    )
    require_new_paths(output=output_path, output_manifest=output_manifest_path)
    score_matrices: list[np.ndarray] = []
    artifact_inputs: list[Path] = [input_path]
    ensemble_identity: tuple[str, str, int, str] | None = None
    actual_precision: str | None = None
    scorer_signature: str | None = None
    calibrator = None
    calibrator_source = None
    calibrator_manifest = None
    if calibrator_path:
        assert calibrator_manifest_path is not None
        if not calibrator_manifest_path.is_file():
            raise FileNotFoundError(
                f"calibrator manifest is required: {calibrator_manifest_path}"
            )
        calibrator_manifest = json.loads(
            calibrator_manifest_path.read_text(encoding="utf-8")
        )
        if (
            not isinstance(calibrator_manifest, dict)
            or calibrator_manifest.get("artifact_type")
            != "oof_affine_prompt_calibrator"
            or calibrator_manifest.get("calibrator_file") != calibrator_path.name
            or calibrator_manifest.get("calibrator_sha256")
            != sha256_file(calibrator_path)
        ):
            raise ValueError("calibrator file does not match its adjacent manifest")
        calibrator_payload = json.loads(calibrator_path.read_text(encoding="utf-8"))
        if (
            calibrator_payload.get("inference_code_contract")
            != calibration_inference_code_contract()
        ):
            raise ValueError("calibrator inference source changed since OOF fitting")
        calibrator = AffinePromptCalibrator.from_dict(calibrator_payload)
        if calibrator.fit_source != "oof":
            raise ValueError("deployment calibrator must declare fit_source='oof'")
        calibrator_source = calibrator_payload.get("source")
        if not isinstance(calibrator_source, dict):
            raise ValueError("deployment calibrator is missing its OOF source contract")
        if calibrator_source.get("scorer_name") != args.model_name:
            raise ValueError(
                "calibrator scorer_name does not match --model-name; refusing a silent mismatch"
            )
        if (
            calibrator_manifest.get("source_scorer_name")
            != calibrator_source.get("scorer_name")
            or calibrator_manifest.get("source_scorer_signature")
            != calibrator_source.get("scorer_signature")
        ):
            raise ValueError("calibrator source contract disagrees with its manifest")

    for checkpoint_dir in resolved_checkpoints:
        loaded = load_scorer_checkpoint(
            checkpoint_dir,
            model_id=args.model_id,
            model_revision=args.model_revision,
            precision=args.precision,
            allow_download=args.allow_download,
        )
        identity = (
            loaded.model_id,
            loaded.model_revision,
            loaded.max_length,
            loaded.precision,
        )
        if ensemble_identity is None:
            ensemble_identity = identity
        elif identity != ensemble_identity:
            raise ValueError(
                "all ensemble checkpoints must share model id, revision, max length, and precision"
            )
        if scorer_signature is None:
            scorer_signature = checkpoint_ensemble_signature(
                resolved_checkpoints,
                precision=loaded.precision,
            )
            if (
                calibrator_source is not None
                and calibrator_source.get("scorer_signature") != scorer_signature
            ):
                raise ValueError(
                    "calibrator was not fitted from OOF predictions of this exact "
                    "checkpoint ensemble and precision"
                )
        score_matrices.append(
            predict_scores(loaded, records, batch_size=args.batch_size)
        )
        actual_precision = loaded.precision
        artifact_inputs.extend(checkpoint_artifact_files(checkpoint_dir))
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not score_matrices:
        raise RuntimeError("no checkpoint prediction was produced")
    score_matrix = np.mean(np.stack(score_matrices, axis=0), axis=0)
    assert actual_precision is not None
    assert scorer_signature is not None
    if calibrator_path:
        assert calibrator is not None
        calibrated = calibrator.transform(
            {
                domain: score_matrix[:, index]
                for index, domain in enumerate(DOMAINS)
            },
            [record.prompt_num for record in records],
        )
        score_matrix = np.column_stack([calibrated[domain] for domain in DOMAINS])
        artifact_inputs.extend((calibrator_path, calibrator_manifest_path))

    output_scorer_signature = scorer_signature
    if calibrator_path:
        output_scorer_signature = sha256_json(
            {
                "base_scorer_signature": scorer_signature,
                "calibrator_sha256": sha256_file(calibrator_path),
                "transform": "affine_prompt_shrinkage",
            }
        )

    rows = prediction_records(records, score_matrix, model=args.model_name)
    write_predictions(output_path, rows)
    manifest = build_manifest(
        run_id=output_path.stem,
        project_root=Path.cwd(),
        config={
            "checkpoints": [str(value) for value in resolved_checkpoints],
            "precision": actual_precision,
            "batch_size": args.batch_size,
            "model_name": args.model_name,
            "base_scorer_signature": scorer_signature,
            "scorer_signature": output_scorer_signature,
            "calibrator": str(calibrator_path) if calibrator_path else None,
        },
        input_files=artifact_inputs,
        extra={
            **prediction_provenance_fields(
                prediction_path=output_path,
                input_path=input_path,
                rows=len(rows),
                scorer_name=args.model_name,
                scorer_signature=output_scorer_signature,
            ),
            "ensemble_size": len(score_matrices),
        },
    )
    write_manifest(output_manifest_path, manifest)
    print(
        json.dumps(
            {
                "predictions": str(output_path),
                "rows": len(rows),
                "ensemble_size": len(score_matrices),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
