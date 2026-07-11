from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration.affine import AffinePromptCalibrator, DOMAINS
from src.ensemble.simplex import TraitSimplexStacker
from src.ensemble.contracts import stacker_inference_code_contract
from src.evaluation.prediction_provenance import (
    prediction_manifest_path,
    prediction_provenance_fields,
    validate_prediction_provenance,
)
from src.evaluation.predictions import (
    prediction_records,
    read_canonical_predictions,
    write_predictions,
)
from src.evaluation.metrics import prediction_matrix
from src.utils.hashing import sha256_file, sha256_json
from src.utils.manifest import build_manifest, write_manifest
from src.utils.paths import require_distinct_paths, require_new_paths


def _source(value: str) -> tuple[str, Path]:
    alias, separator, path = value.partition("=")
    if not separator or not alias.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--source must have the form ALIAS=PATH")
    return alias.strip(), Path(path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a hash-bound multi-source stacker and its OOF calibrator."
    )
    parser.add_argument("--stacker", required=True)
    parser.add_argument("--source", action="append", type=_source, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _signature_payload(artifact: dict) -> dict:
    fields = (
        "artifact_version",
        "method",
        "fit_source",
        "scorer_name",
        "source_order",
        "source_contracts",
        "stacker",
        "calibrator",
        "gold_sha256",
        "folds_sha256",
        "inference_code_contract",
        "config",
    )
    if any(field not in artifact for field in fields):
        raise ValueError("stacker artifact is missing a signed field")
    return {field: artifact[field] for field in fields}


def main() -> None:
    args = parse_args()
    stacker_path = Path(args.stacker).resolve()
    stacker_manifest_path = stacker_path.with_suffix(".manifest.json")
    output_path = Path(args.output).resolve()
    output_manifest_path = prediction_manifest_path(output_path)
    source_paths = dict(args.source)
    if len(args.source) < 2 or len(source_paths) != len(args.source):
        raise ValueError("the stacker requires at least two unique source aliases")
    paths = {
        "stacker": stacker_path,
        "stacker_manifest": stacker_manifest_path,
        "output": output_path,
        "output_manifest": output_manifest_path,
    }
    paths.update({f"source_{alias}": path for alias, path in source_paths.items()})
    paths.update(
        {
            f"source_manifest_{alias}": prediction_manifest_path(path)
            for alias, path in source_paths.items()
        }
    )
    require_distinct_paths(**paths)
    require_new_paths(output=output_path, output_manifest=output_manifest_path)

    artifact = json.loads(stacker_path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or artifact.get("artifact_type") != "trait_simplex_stacker":
        raise ValueError("invalid trait simplex stacker artifact")
    signature_payload = _signature_payload(artifact)
    stacker_signature = sha256_json(signature_payload)
    if artifact.get("stacker_signature") != stacker_signature:
        raise ValueError("stacker artifact signature mismatch")
    if (
        signature_payload.get("inference_code_contract")
        != stacker_inference_code_contract()
    ):
        raise ValueError("stacker inference source changed since OOF fitting")
    if not stacker_manifest_path.is_file():
        raise FileNotFoundError(f"stacker manifest is required: {stacker_manifest_path}")
    stacker_manifest = json.loads(stacker_manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(stacker_manifest, dict)
        or stacker_manifest.get("artifact_type") != "trait_simplex_stacker_manifest"
        or stacker_manifest.get("stacker_file") != stacker_path.name
        or stacker_manifest.get("stacker_sha256") != sha256_file(stacker_path)
        or stacker_manifest.get("stacker_signature") != stacker_signature
    ):
        raise ValueError("stacker does not match its adjacent manifest")

    source_order = artifact.get("source_order")
    source_contracts = artifact.get("source_contracts")
    if (
        not isinstance(source_order, list)
        or set(source_order) != set(source_paths)
        or not isinstance(source_contracts, dict)
        or set(source_contracts) != set(source_order)
    ):
        raise ValueError("stacker source aliases do not match --source")
    source_rows = {}
    source_matrices = {}
    source_manifests = {}
    reference_rows = None
    reference_ids = None
    reference_prompts = None
    input_hash = None
    for alias in source_order:
        contract = source_contracts.get(alias)
        if not isinstance(contract, dict):
            raise ValueError(f"stacker has no source contract for {alias}")
        manifest = validate_prediction_provenance(
            source_paths[alias],
            expected_scorer_name=contract.get("scorer_name"),
            expected_scorer_signature=contract.get("scorer_signature"),
        )
        rows = read_canonical_predictions(source_paths[alias])
        if int(manifest["rows"]) != len(rows):
            raise ValueError(f"prediction row count mismatch for source {alias}")
        if {row["model"] for row in rows} != {manifest["scorer_name"]}:
            raise ValueError(f"prediction model/provenance mismatch for source {alias}")
        ids = [row["id"] for row in rows]
        prompts = {row["id"]: row["prompt_num"] for row in rows}
        if reference_rows is None:
            reference_rows = rows
            reference_ids = ids
            reference_prompts = prompts
        else:
            assert reference_ids is not None and reference_prompts is not None
            if set(ids) != set(reference_ids):
                raise ValueError("stacker source ID sets do not match")
            if any(prompts[record_id] != reference_prompts[record_id] for record_id in reference_ids):
                raise ValueError("stacker source prompt_num values do not match")
        manifest_input_hash = manifest.get("input_sha256")
        if not isinstance(manifest_input_hash, str) or len(manifest_input_hash) != 64:
            raise ValueError(f"source {alias} has no valid input SHA256")
        if input_hash is None:
            input_hash = manifest_input_hash
        elif manifest_input_hash != input_hash:
            raise ValueError("stacker sources were not produced from the same input file")
        source_rows[alias] = rows
        source_manifests[alias] = manifest

    assert reference_rows is not None and reference_ids is not None
    for alias in source_order:
        source_matrices[alias] = prediction_matrix(source_rows[alias], reference_rows)
    stacker = TraitSimplexStacker.from_dict(artifact["stacker"])
    raw = stacker.transform(source_matrices)
    calibrator = AffinePromptCalibrator.from_dict(artifact["calibrator"])
    if calibrator.fit_source != "base_oof_stacked":
        raise ValueError("stacker calibrator was not fitted from base OOF predictions")
    calibrated = calibrator.transform(
        {domain: raw[:, index] for index, domain in enumerate(DOMAINS)},
        [row["prompt_num"] for row in reference_rows],
    )
    matrix = np.column_stack([calibrated[domain] for domain in DOMAINS])
    scorer_name = artifact["scorer_name"]
    rows = prediction_records(reference_rows, matrix, model=scorer_name)
    write_predictions(output_path, rows)
    manifest = build_manifest(
        run_id=output_path.stem,
        project_root=Path.cwd(),
        config={
            "stacker": str(stacker_path),
            "stacker_signature": stacker_signature,
            "source_order": source_order,
        },
        input_files=(
            stacker_path,
            stacker_manifest_path,
            *source_paths.values(),
            *(prediction_manifest_path(path) for path in source_paths.values()),
        ),
        extra={
            **prediction_provenance_fields(
                prediction_path=output_path,
                input_path=source_paths[source_order[0]],
                rows=len(rows),
                scorer_name=scorer_name,
                scorer_signature=stacker_signature,
                model_artifact=stacker_path,
            ),
            "input_creation_path": None,
            "input_sha256": input_hash,
            "input_provenance": "inherited_from_all_source_prediction_manifests",
            "source_predictions": {
                alias: {
                    "file": source_paths[alias].name,
                    "sha256": sha256_file(source_paths[alias]),
                    "scorer_signature": source_manifests[alias]["scorer_signature"],
                }
                for alias in source_order
            },
        },
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
