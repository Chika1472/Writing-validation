from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.contracts import baseline_inference_code_contract
from src.data.load import load_inference_jsonl
from src.evaluation.prediction_provenance import (
    prediction_manifest_path,
    prediction_provenance_fields,
)
from src.evaluation.predictions import prediction_records, write_predictions
from src.utils.hashing import sha256_file, sha256_json
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a hash-verified full-train CPU baseline artifact."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--model", required=True, help="Saved baseline_fold_ensemble JSON artifact."
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()
    output_manifest_path = prediction_manifest_path(output_path)
    require_distinct_paths(
        input=input_path,
        model=model_path,
        output=output_path,
        output_manifest=output_manifest_path,
    )
    require_new_paths(output=output_path, output_manifest=output_manifest_path)

    model_manifest = json.loads(model_path.read_text(encoding="utf-8"))
    if (
        not isinstance(model_manifest, dict)
        or model_manifest.get("artifact_type") != "baseline_fold_ensemble"
        or model_manifest.get("aggregation") != "equal_weight_mean"
    ):
        raise ValueError("invalid baseline fold-ensemble artifact")
    scorer_name = model_manifest.get("scorer_name")
    scorer_signature = model_manifest.get("scorer_signature")
    if not isinstance(scorer_name, str) or not scorer_name:
        raise ValueError("baseline model manifest has no scorer_name")
    if not isinstance(scorer_signature, str) or not scorer_signature:
        raise ValueError("baseline model manifest has no scorer_signature")
    signature_payload = model_manifest.get("signature_payload")
    if not isinstance(signature_payload, dict) or sha256_json(signature_payload) != scorer_signature:
        raise ValueError("baseline ensemble signature payload is invalid")
    if signature_payload.get("scorer") != scorer_name:
        raise ValueError("baseline ensemble scorer name/signature mismatch")
    if (
        signature_payload.get("inference_code_contract")
        != baseline_inference_code_contract()
    ):
        raise ValueError(
            "baseline inference source has changed since the artifact was created"
        )

    fold_models = model_manifest.get("fold_models")
    if not isinstance(fold_models, dict) or not fold_models:
        raise ValueError("baseline ensemble contains no fold models")
    model_files: list[Path] = []
    for fold_id, item in fold_models.items():
        if not isinstance(item, dict) or not isinstance(item.get("file"), str):
            raise ValueError(f"invalid fold model entry: {fold_id!r}")
        if Path(item["file"]).name != item["file"]:
            raise ValueError("baseline fold model filenames must not contain directories")
        candidate = (model_path.parent / item["file"]).resolve()
        try:
            candidate.relative_to(model_path.parent.resolve())
        except ValueError as error:
            raise ValueError("fold model path must remain beside the ensemble artifact") from error
        if item.get("sha256") != sha256_file(candidate):
            raise ValueError(f"fold model hash mismatch: {candidate}")
        model_files.append(candidate)
    if signature_payload.get("fold_model_hashes") != {
        str(fold_id): item["sha256"] for fold_id, item in fold_models.items()
    }:
        raise ValueError("baseline signature does not bind the listed fold models")
    auxiliary_files: list[Path] = []
    selection_name = model_manifest.get("selection_report")
    selection_hash = model_manifest.get("selection_report_sha256")
    if (selection_name is None) != (selection_hash is None):
        raise ValueError("baseline selection report name/hash must be present together")
    if selection_name is not None:
        if not isinstance(selection_name, str) or Path(selection_name).name != selection_name:
            raise ValueError("baseline selection report must be an adjacent filename")
        selection_path = (model_path.parent / selection_name).resolve()
        if selection_hash != sha256_file(selection_path):
            raise ValueError("baseline selection report hash mismatch")
        if signature_payload.get("selection_report_sha256") != selection_hash:
            raise ValueError("baseline signature does not bind its selection report")
        auxiliary_files.append(selection_path)
    require_distinct_paths(
        output=output_path,
        output_manifest=output_manifest_path,
        **{f"fold_model_{index}": path for index, path in enumerate(model_files)},
        **{f"auxiliary_{index}": path for index, path in enumerate(auxiliary_files)},
    )

    records = load_inference_jsonl(input_path)
    fold_predictions: list[np.ndarray] = []
    for path in model_files:
        model = joblib.load(path)
        if not hasattr(model, "predict"):
            raise TypeError(f"baseline fold artifact does not expose predict(records): {path}")
        fold_predictions.append(np.asarray(model.predict(records), dtype=float))
    matrix = np.mean(np.stack(fold_predictions, axis=0), axis=0)
    if matrix.shape != (len(records), 3) or not np.isfinite(matrix).all():
        raise RuntimeError(f"baseline returned an invalid prediction matrix: {matrix.shape}")
    rows = prediction_records(records, matrix, model=scorer_name)
    write_predictions(output_path, rows)
    manifest = build_manifest(
        run_id=output_path.stem,
        project_root=Path.cwd(),
        config={
            "model": str(model_path),
            "scorer_name": scorer_name,
            "scorer_signature": scorer_signature,
        },
        input_files=(input_path, model_path, *model_files, *auxiliary_files),
        extra=prediction_provenance_fields(
            prediction_path=output_path,
            input_path=input_path,
            rows=len(rows),
            scorer_name=scorer_name,
            scorer_signature=scorer_signature,
            model_artifact=model_path,
        ),
    )
    write_manifest(output_manifest_path, manifest)
    print(
        json.dumps(
            {"predictions": str(output_path), "rows": len(rows)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
