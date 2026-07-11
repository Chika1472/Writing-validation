"""Portable, hash-bound assessment Ridge deployment artifact."""

from __future__ import annotations

import json
import re
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from src.assessment.questions import QUESTION_VERSION, QUESTIONS_SHA256, TRAITS
from src.assessment.codebook import DEFAULT_ANSWER_CODES
from src.assessment.contracts import assessment_extraction_code_sha256
from src.assessment.prompting import ASSESSMENT_QUERY_SHA256
from src.assessment.ridge import predict_assessment_fold_ensemble
from src.utils.hashing import sha256_file, sha256_json


ASSESSMENT_DEPLOYMENT_ARTIFACT = "assessment_ridge_deployment"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def build_deployment_artifact(
    *,
    scorer_name: str,
    feature_signature: str,
    feature_contract: dict[str, Any],
    fold_models: dict[str, dict[str, dict[str, Any]]],
    outer_selection: list[dict[str, Any]],
    clip_min: float,
    clip_max: float,
    training_source: dict[str, Any],
    selection_config: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(scorer_name, str) or not scorer_name.strip():
        raise ValueError("assessment scorer_name must be nonempty")
    if _SHA256.fullmatch(feature_signature) is None:
        raise ValueError("assessment feature_signature must be a SHA-256 digest")
    if (
        not isinstance(feature_contract, dict)
        or sha256_json(feature_contract) != feature_signature
    ):
        raise ValueError("assessment feature contract/signature mismatch")
    if not clip_min < clip_max:
        raise ValueError("assessment clip_min must be smaller than clip_max")
    payload = {
        "artifact_type": ASSESSMENT_DEPLOYMENT_ARTIFACT,
        "artifact_version": 1,
        "candidate_branch": True,
        "auto_promoted": False,
        "fit_source": "outer_train_models_after_nested_inner_cv",
        "aggregation": "equal_weight_fold_mean",
        "scorer_name": scorer_name,
        "feature_signature": feature_signature,
        "feature_contract": feature_contract,
        "question_version": QUESTION_VERSION,
        "questions_sha256": QUESTIONS_SHA256,
        "trait_order": list(TRAITS),
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
        "selection_config": selection_config,
        "outer_selection": outer_selection,
        "fold_models": fold_models,
        "training_source": training_source,
    }
    return {**payload, "artifact_signature": sha256_json(payload)}


def load_deployment_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path).resolve()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("assessment deployment artifact must contain a JSON object")
    signature = payload.get("artifact_signature")
    unsigned = {key: value for key, value in payload.items() if key != "artifact_signature"}
    if (
        payload.get("artifact_type") != ASSESSMENT_DEPLOYMENT_ARTIFACT
        or payload.get("artifact_version") != 1
        or payload.get("candidate_branch") is not True
        or payload.get("auto_promoted") is not False
        or payload.get("fit_source")
        != "outer_train_models_after_nested_inner_cv"
        or payload.get("aggregation") != "equal_weight_fold_mean"
        or not isinstance(signature, str)
        or sha256_json(unsigned) != signature
    ):
        raise ValueError("assessment deployment artifact signature/contract is invalid")
    if payload.get("question_version") != QUESTION_VERSION or payload.get(
        "questions_sha256"
    ) != QUESTIONS_SHA256:
        raise ValueError("assessment deployment question contract changed")
    if payload.get("trait_order") != list(TRAITS):
        raise ValueError("assessment deployment trait order mismatch")
    selection_config = payload.get("selection_config")
    if (
        not isinstance(selection_config, dict)
        or selection_config.get("held_fold_exclusion") != "strict"
        or selection_config.get("normalization") != "inner_train_standardization"
        or selection_config.get("selection_metric") != "rmse"
    ):
        raise ValueError("assessment deployment held-fold exclusion is not strict")
    question_counts = selection_config.get("question_count_candidates")
    alpha_grid = selection_config.get("alpha_grid")
    if (
        not isinstance(question_counts, list)
        or not question_counts
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 3 or value > 6
            for value in question_counts
        )
        or not isinstance(alpha_grid, list)
        or not alpha_grid
    ):
        raise ValueError("assessment deployment selection grid is invalid")
    allowed_counts = set(question_counts)
    allowed_alphas = {float(value) for value in alpha_grid}
    if any(not np.isfinite(value) or value <= 0.0 for value in allowed_alphas):
        raise ValueError("assessment deployment alpha grid is invalid")
    feature_signature = payload.get("feature_signature")
    if not isinstance(feature_signature, str) or _SHA256.fullmatch(feature_signature) is None:
        raise ValueError("assessment deployment feature signature is invalid")
    feature_contract = payload.get("feature_contract")
    if (
        not isinstance(feature_contract, dict)
        or sha256_json(feature_contract) != feature_signature
    ):
        raise ValueError("assessment deployment feature contract is invalid")
    source_root = Path(__file__).resolve().parents[2]
    expected_code_hashes = {
        "extractor_code_sha256": assessment_extraction_code_sha256(),
        "reproducibility_code_sha256": sha256_file(
            source_root / "src" / "utils" / "reproducibility.py"
        ),
    }
    if any(feature_contract.get(key) != value for key, value in expected_code_hashes.items()):
        raise ValueError("assessment deployment extraction source contract changed")
    required_feature_fields = {
        "artifact_version",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "precision",
        "max_length",
        "batch_size",
        "seed",
        "deterministic_torch",
        "extractor_code_sha256",
        "reproducibility_code_sha256",
        "quantization",
        "feature_type",
        "question_version",
        "questions_sha256",
        "assessment_query_sha256",
        "answer_codes",
        "answer_token_ids",
        "codebook_sha256",
    }
    token_ids = feature_contract.get("answer_token_ids")
    if (
        set(feature_contract) != required_feature_fields
        or feature_contract.get("artifact_version") != 1
        or feature_contract.get("model_id") != "Qwen/Qwen3-14B"
        or not isinstance(feature_contract.get("model_revision"), str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", feature_contract["model_revision"])
        is None
        or not isinstance(feature_contract.get("tokenizer_revision"), str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", feature_contract["tokenizer_revision"])
        is None
        or feature_contract.get("precision") not in {"4bit", "bf16"}
        or feature_contract.get("deterministic_torch") is not True
        or feature_contract.get("feature_type")
        != "restricted_answer_probabilities"
        or feature_contract.get("question_version") != QUESTION_VERSION
        or feature_contract.get("questions_sha256") != QUESTIONS_SHA256
        or feature_contract.get("assessment_query_sha256")
        != ASSESSMENT_QUERY_SHA256
        or feature_contract.get("answer_codes") != list(DEFAULT_ANSWER_CODES)
        or not isinstance(token_ids, list)
        or len(token_ids) != 5
        or len(set(token_ids)) != 5
        or any(isinstance(value, bool) or not isinstance(value, int) for value in token_ids)
        or not isinstance(feature_contract.get("codebook_sha256"), str)
        or _SHA256.fullmatch(feature_contract["codebook_sha256"]) is None
    ):
        raise ValueError("assessment deployment feature fields are invalid")
    for field in ("max_length", "batch_size"):
        value = feature_contract.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"assessment deployment {field} is invalid")
    seed = feature_contract.get("seed")
    quantization = feature_contract.get("quantization")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("assessment deployment seed is invalid")
    if (
        not isinstance(quantization, dict)
        or set(quantization)
        != {
            "bnb_4bit_quant_type",
            "bnb_4bit_use_double_quant",
            "compute_dtype",
        }
        or quantization.get("bnb_4bit_quant_type") != "nf4"
        or not isinstance(quantization.get("bnb_4bit_use_double_quant"), bool)
        or quantization.get("compute_dtype") != "bfloat16"
    ):
        raise ValueError("assessment deployment quantization is invalid")
    clip_min = float(payload.get("clip_min"))
    clip_max = float(payload.get("clip_max"))
    if not np.isfinite(clip_min) or not np.isfinite(clip_max) or not clip_min < clip_max:
        raise ValueError("assessment deployment clip bounds are invalid")
    fold_models = payload.get("fold_models")
    if not isinstance(fold_models, dict) or len(fold_models) < 3:
        raise ValueError("assessment deployment fold models are missing")
    outer_selection = payload.get("outer_selection")
    if not isinstance(outer_selection, list) or len(outer_selection) != len(fold_models):
        raise ValueError("assessment deployment outer selection audit is invalid")
    reports_by_fold: dict[str, dict[str, Any]] = {}
    for report in outer_selection:
        if (
            not isinstance(report, dict)
            or report.get("selection_scope") != "outer_train_only"
            or isinstance(report.get("outer_fold"), bool)
            or not isinstance(report.get("outer_fold"), int)
            or not isinstance(report.get("fit_ids_sha256"), str)
            or _SHA256.fullmatch(report["fit_ids_sha256"]) is None
            or not isinstance(report.get("held_ids_sha256"), str)
            or _SHA256.fullmatch(report["held_ids_sha256"]) is None
            or isinstance(report.get("fit_rows"), bool)
            or not isinstance(report.get("fit_rows"), int)
            or report["fit_rows"] <= 0
            or isinstance(report.get("held_rows"), bool)
            or not isinstance(report.get("held_rows"), int)
            or report["held_rows"] <= 0
        ):
            raise ValueError("assessment deployment contains an invalid outer-fold report")
        reports_by_fold[str(report["outer_fold"])] = report
    if set(reports_by_fold) != set(fold_models):
        raise ValueError("assessment fold models and outer selection reports differ")
    for fold_id, models in fold_models.items():
        report_traits = reports_by_fold[fold_id].get("traits")
        if not isinstance(models, dict) or not isinstance(report_traits, dict):
            raise ValueError("assessment fold model/selection trait mapping is invalid")
        for trait in TRAITS:
            selection = report_traits.get(trait)
            selected = selection.get("selected") if isinstance(selection, dict) else None
            model = models.get(trait)
            selected_alpha = selected.get("alpha") if isinstance(selected, dict) else None
            if (
                not isinstance(selected, dict)
                or not isinstance(model, dict)
                or selected.get("question_count") not in allowed_counts
                or isinstance(selected_alpha, bool)
                or not isinstance(selected_alpha, Real)
                or float(selected_alpha) not in allowed_alphas
                or model.get("question_count") != selected.get("question_count")
                or model.get("alpha") != selected.get("alpha")
            ):
                raise ValueError(
                    "assessment fold model does not match its inner-CV selection"
                )
    predict_assessment_fold_ensemble(
        np.full((1, 18, 5), 0.2, dtype=np.float64),
        fold_models,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    return payload


__all__ = [
    "ASSESSMENT_DEPLOYMENT_ARTIFACT",
    "build_deployment_artifact",
    "load_deployment_artifact",
]
