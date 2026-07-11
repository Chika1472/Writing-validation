from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.mean_baseline import (
    MeanBaseline,
    PromptMeanBaseline,
    oof_predict_with_models,
)
from src.baselines.contracts import baseline_inference_code_contract
from src.baselines.tfidf_ridge import SurfaceOLSBaseline, TfidfRidgeBaseline
from src.data.folds import load_folds
from src.data.load import load_train_validation
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.oof_provenance import oof_manifest_path, oof_provenance_fields
from src.evaluation.prediction_provenance import (
    prediction_manifest_path,
    prediction_provenance_fields,
)
from src.evaluation.predictions import prediction_records, write_predictions
from src.utils.config import load_yaml, resolve_project_path
from src.utils.hashing import sha256_file, sha256_json
from src.utils.manifest import build_manifest, write_manifest


def _tuple2(value: list[int] | tuple[int, int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError("ngram_range must contain exactly two integers")
    return int(value[0]), int(value[1])


def _models(config: dict[str, Any]) -> dict[str, Any]:
    tfidf = config["tfidf_ridge"]
    return {
        "global_mean": MeanBaseline(),
        "prompt_mean": PromptMeanBaseline(),
        "surface_ols": SurfaceOLSBaseline(
            include_prompt=bool(config["surface_ols"]["include_prompt"])
        ),
        "tfidf_ridge": TfidfRidgeBaseline(
            alpha=float(tfidf["alpha"]),
            char_ngram_range=_tuple2(tfidf["char_ngram_range"]),
            word_ngram_range=_tuple2(tfidf["word_ngram_range"]),
            char_min_df=tfidf["char_min_df"],
            word_min_df=tfidf["word_min_df"],
            prompt_min_df=tfidf["prompt_min_df"],
            max_char_features=tfidf.get("max_char_features"),
            max_word_features=tfidf.get("max_word_features"),
            include_prompt=bool(tfidf["include_prompt"]),
            include_prompt_text=bool(tfidf["include_prompt_text"]),
            include_surface=bool(tfidf["include_surface"]),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train OOF and validation CPU baselines.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--baseline-config", default="configs/baselines.yaml")
    parser.add_argument("--folds", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_config = load_yaml(args.config)
    baseline_config = load_yaml(args.baseline_config)
    train_path = resolve_project_path(data_config, data_config["paths"]["train"])
    validation_path = resolve_project_path(data_config, data_config["paths"]["validation"])
    train, validation = load_train_validation(train_path, validation_path)
    fold_path = Path(args.folds).resolve()
    assignments = load_folds(fold_path)
    missing = [record.id for record in train if record.id not in assignments]
    train_ids = {record.id for record in train}
    extra = sorted(set(assignments).difference(train_ids))
    if missing or extra:
        raise ValueError(
            "fold file must match train ids exactly; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    fold_ids = [assignments[record.id] for record in train]

    artifacts = resolve_project_path(data_config, data_config["paths"]["artifacts"])
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else artifacts / "predictions" / "cpu_baselines"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {}
    public_config = {
        "data": {key: value for key, value in data_config.items() if not key.startswith("_")},
        "baselines": {
            key: value for key, value in baseline_config.items() if not key.startswith("_")
        },
    }

    for name, estimator in _models(baseline_config).items():
        oof_matrix, fold_models = oof_predict_with_models(
            estimator, train, fold_ids
        )
        validation_matrix = np.mean(
            np.stack(
                [
                    np.asarray(model.predict(validation), dtype=float)
                    for model in fold_models.values()
                ],
                axis=0,
            ),
            axis=0,
        )
        fold_model_artifacts: dict[str, Path] = {}
        for fold_id, fitted_model in fold_models.items():
            model_path = output_dir / f"{name}_fold{fold_id}.joblib"
            joblib.dump(fitted_model, model_path, compress=3)
            fold_model_artifacts[fold_id] = model_path
        oof_rows = prediction_records(train, oof_matrix, model=name)
        validation_rows = prediction_records(
            validation, validation_matrix, model=name
        )
        oof_path = write_predictions(output_dir / f"{name}_oof.jsonl", oof_rows)
        validation_output = write_predictions(
            output_dir / f"{name}_validation.jsonl", validation_rows
        )
        scorer_name = name
        fold_model_hashes = {
            fold_id: sha256_file(path)
            for fold_id, path in sorted(fold_model_artifacts.items())
        }
        signature_payload = {
            "scorer": name,
            "baseline_config": public_config["baselines"],
            "train_sha256": sha256_file(train_path),
            "folds_sha256": sha256_file(fold_path),
            "fold_model_hashes": fold_model_hashes,
            "inference_code_contract": baseline_inference_code_contract(),
        }
        scorer_signature = sha256_json(signature_payload)
        model_artifact = output_dir / f"{name}_ensemble.json"
        model_artifact.write_text(
            json.dumps(
                {
                    "artifact_type": "baseline_fold_ensemble",
                    "artifact_version": 1,
                    "scorer_name": scorer_name,
                    "scorer_signature": scorer_signature,
                    "signature_payload": signature_payload,
                    "fold_models": {
                        fold_id: {
                            "file": path.name,
                            "sha256": fold_model_hashes[fold_id],
                        }
                        for fold_id, path in sorted(fold_model_artifacts.items())
                    },
                    "aggregation": "equal_weight_mean",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        oof_manifest = build_manifest(
            run_id=scorer_name,
            project_root=data_config["_project_root"],
            config=public_config,
            input_files=(
                train_path,
                fold_path,
                model_artifact,
                *fold_model_artifacts.values(),
            ),
            extra=oof_provenance_fields(
                prediction_path=oof_path,
                gold_path=train_path,
                fold_path=fold_path,
                rows=len(oof_rows),
                scorer_name=scorer_name,
                scorer_signature=scorer_signature,
            ),
        )
        write_manifest(oof_manifest_path(oof_path), oof_manifest)
        validation_manifest = build_manifest(
            run_id=f"{name}_validation",
            project_root=data_config["_project_root"],
            config=public_config,
            input_files=(
                train_path,
                validation_path,
                fold_path,
                model_artifact,
                *fold_model_artifacts.values(),
            ),
            extra=prediction_provenance_fields(
                prediction_path=validation_output,
                input_path=validation_path,
                rows=len(validation_rows),
                scorer_name=scorer_name,
                scorer_signature=scorer_signature,
                model_artifact=model_artifact,
            ),
        )
        write_manifest(
            prediction_manifest_path(validation_output), validation_manifest
        )
        report[name] = {
            "oof": evaluate_predictions(train, oof_rows),
            "validation": evaluate_predictions(validation, validation_rows),
            "oof_path": str(oof_path),
            "validation_path": str(validation_output),
            "model_artifact": str(model_artifact),
            "scorer_signature": scorer_signature,
        }

    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = build_manifest(
        run_id="cpu_baselines",
        project_root=data_config["_project_root"],
        config=public_config,
        input_files=(train_path, validation_path, fold_path),
        extra={"report": str(report_path), "report_sha256": sha256_file(report_path)},
    )
    write_manifest(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "models": list(report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
