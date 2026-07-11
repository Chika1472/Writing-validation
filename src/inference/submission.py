"""SDK-neutral, offline submission inference engine.

The competition's official Python class/function contract is not public in this
workspace.  ``SubmissionEngine`` is therefore a small stable API that an official
adapter can call without changing scoring or rationale logic.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.data.schema import EssayInput, EssayRecord, ensure_essay_input
from src.inference.deployment import (
    DeploymentConfig,
    ValidatedArtifacts,
    configure_offline_environment,
    load_deployment_config,
    validate_artifact_contracts,
    validate_dependency_lock,
    validate_package_manifest,
)
from src.inference.finalize import final_prediction_row
from src.inference.serializer import TRAITS, serialize_prediction, strict_parse_prediction
from src.rationale.deterministic import (
    RATIONALE_TEMPLATE_VERSION,
    generate_grounded_rationales,
)
from src.rationale.evidence import build_evidence_ledger
from src.rationale.parsing import assess_grounding
from src.utils.hashing import sha256_file, sha256_json
from src.utils.reproducibility import seed_everything


@dataclass(frozen=True)
class SubmissionResult:
    rows: tuple[dict[str, Any], ...]
    ledger_rows: tuple[dict[str, Any], ...]
    score_matrix: np.ndarray
    score_scorer_name: str
    score_scorer_signature: str
    rationale_signature: str
    final_signature: str
    fallback_count: int


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must contain a JSON object: {path}")
    return payload


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _matrix(value: Any, *, rows: int, where: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (rows, len(TRAITS)):
        raise ValueError(f"{where} must have shape ({rows}, 3), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{where} contains a non-finite score")
    if (matrix < 1.0).any() or (matrix > 5.0).any():
        raise ValueError(f"{where} contains a score outside [1, 5]")
    return matrix


class SubmissionEngine:
    """Run the frozen score pipeline, then attach grounded rationales.

    One ``predict`` call is expected to receive the complete test split.  Models
    are loaded sequentially and released before the rationale adapter is loaded,
    keeping the single-GPU memory contract explicit.
    """

    def __init__(
        self,
        config: DeploymentConfig,
        artifacts: ValidatedArtifacts,
    ) -> None:
        self.config = config
        self.artifacts = artifacts
        configure_offline_environment(config)
        self._validate_runtime()

    @classmethod
    def from_config(cls, path: str | Path) -> "SubmissionEngine":
        config = load_deployment_config(path)
        configure_offline_environment(config)
        validate_package_manifest(config)
        validate_dependency_lock(config)
        artifacts = validate_artifact_contracts(config)
        return cls(config, artifacts)

    def _validate_runtime(self) -> None:
        import torch

        runtime = self.config.runtime
        if runtime.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("deployment config requires CUDA, but CUDA is unavailable")
        if runtime.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested {runtime.device}, but CUDA is unavailable")
        if runtime.require_single_visible_gpu and torch.cuda.device_count() != 1:
            raise RuntimeError(
                "exactly one visible GPU is required; set CUDA_VISIBLE_DEVICES before startup"
            )
        if runtime.required_gpu_name_contains is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("a required GPU name was configured without CUDA")
            actual = torch.cuda.get_device_name(0)
            if runtime.required_gpu_name_contains.lower() not in actual.lower():
                raise RuntimeError(
                    f"GPU mismatch: required substring {runtime.required_gpu_name_contains!r}, "
                    f"actual={actual!r}"
                )
        seed_everything(
            runtime.seed,
            deterministic_torch=runtime.deterministic_algorithms,
        )

    def _score_qwen(
        self, records: list[EssayInput]
    ) -> tuple[np.ndarray, str, dict[int, np.ndarray]]:
        from src.calibration.affine import AffinePromptCalibrator, DOMAINS
        from src.inference.scorer import (
            load_scorer_checkpoint,
            predict_scores_and_embeddings,
        )

        matrices: list[np.ndarray] = []
        embeddings_by_fold: dict[int, np.ndarray] = {}
        identity: tuple[str, str, int, str] | None = None
        for checkpoint in self.artifacts.scorer_checkpoints:
            loaded = load_scorer_checkpoint(
                checkpoint,
                model_id=self.config.qwen.model_id,
                model_revision=self.config.qwen.model_revision,
                precision=self.config.runtime.scorer_precision,
                device=self.config.runtime.device,
                allow_download=False,
            )
            current = (
                loaded.model_id,
                loaded.model_revision,
                loaded.max_length,
                loaded.precision,
            )
            if identity is None:
                identity = current
            elif current != identity:
                raise ValueError(
                    "all scorer checkpoints must share model, revision, max length, and precision"
                )
            checkpoint_scores, checkpoint_embeddings = predict_scores_and_embeddings(
                loaded,
                records,
                batch_size=self.config.runtime.batch_size,
            )
            matrices.append(
                _matrix(
                    checkpoint_scores,
                    rows=len(records),
                    where=f"Qwen checkpoint {checkpoint.name}",
                )
            )
            if self.config.anchor_artifact is not None:
                provenance = _json_object(checkpoint / "checkpoint_provenance.json")
                fold = provenance.get("fold")
                if (
                    isinstance(fold, bool)
                    or not isinstance(fold, int)
                    or fold in embeddings_by_fold
                ):
                    raise ValueError(
                        "anchor scoring requires one uniquely keyed checkpoint per fold"
                    )
                embeddings_by_fold[fold] = checkpoint_embeddings
            del loaded
            _release_cuda()
        if not matrices:
            raise RuntimeError("no Qwen score matrix was produced")
        matrix = np.mean(np.stack(matrices, axis=0), axis=0)
        signature = self.artifacts.scorer_signature
        calibrator_path = self.config.qwen.calibrator
        if calibrator_path is not None:
            payload = _json_object(calibrator_path)
            source = payload.get("source")
            if not isinstance(source, dict):
                raise ValueError("Qwen calibrator is missing its OOF source contract")
            if source.get("scorer_name") != self.config.qwen.scorer_name:
                raise ValueError("Qwen calibrator scorer_name does not match deployment config")
            if source.get("scorer_signature") != signature:
                raise ValueError("Qwen calibrator was fitted on a different checkpoint ensemble")
            calibrator = AffinePromptCalibrator.from_dict(payload)
            if calibrator.fit_source != "oof":
                raise ValueError("Qwen deployment calibrator must declare fit_source='oof'")
            transformed = calibrator.transform(
                {trait: matrix[:, index] for index, trait in enumerate(DOMAINS)},
                [record.prompt_num for record in records],
            )
            matrix = np.column_stack([transformed[trait] for trait in DOMAINS])
            signature = sha256_json(
                {
                    "base_scorer_signature": signature,
                    "calibrator_sha256": sha256_file(calibrator_path),
                    "transform": "affine_prompt_shrinkage",
                }
            )
        return (
            _matrix(matrix, rows=len(records), where="Qwen ensemble"),
            signature,
            embeddings_by_fold,
        )

    def _score_baseline(self, records: list[EssayInput]) -> np.ndarray:
        import joblib

        if self.config.baseline_artifact is None:
            raise RuntimeError("baseline scoring requested without an artifact")
        payload = _json_object(self.config.baseline_artifact)
        fold_models = payload.get("fold_models")
        if not isinstance(fold_models, dict) or not fold_models:
            raise ValueError("baseline artifact contains no fold models")
        matrices = []
        for fold_id, item in fold_models.items():
            if not isinstance(item, dict) or not isinstance(item.get("file"), str):
                raise ValueError(f"invalid baseline fold entry: {fold_id!r}")
            path = (self.config.baseline_artifact.parent / item["file"]).resolve()
            model = joblib.load(path)
            if not hasattr(model, "predict"):
                raise TypeError(f"baseline fold model has no predict method: {path}")
            matrices.append(
                _matrix(
                    model.predict(records),
                    rows=len(records),
                    where=f"baseline fold {fold_id}",
                )
            )
            del model
            gc.collect()
        return _matrix(
            np.mean(np.stack(matrices, axis=0), axis=0),
            rows=len(records),
            where="baseline ensemble",
        )

    def _score_anchor(
        self,
        records: list[EssayInput],
        embeddings_by_fold: dict[int, np.ndarray],
    ) -> np.ndarray:
        from src.anchors.artifact import load_anchor_bank
        from src.anchors.knn import prompt_aware_knn_predict
        from src.utils.hashing import sha256_text

        if self.config.anchor_artifact is None:
            raise RuntimeError("anchor scoring requested without an artifact")
        bank = load_anchor_bank(self.config.anchor_artifact)
        if set(embeddings_by_fold) != set(bank.embeddings_by_fold):
            raise ValueError("query embeddings do not cover the anchor bank folds")
        config = bank.manifest.get("config")
        if not isinstance(config, dict):
            raise ValueError("anchor bank has no KNN configuration")
        k = int(config.get("k", 0))
        temperature = float(config.get("temperature", 0.0))
        unknown_policy = str(config.get("unknown_prompt_policy", ""))
        if k <= 0 or temperature <= 0.0 or unknown_policy not in {"global", "error"}:
            raise ValueError("anchor bank has an invalid KNN configuration")
        ids = [record.id for record in records]
        prompt_hashes = [sha256_text(record.prompt) for record in records]
        matrices: list[np.ndarray] = []
        for fold in sorted(bank.embeddings_by_fold):
            result = prompt_aware_knn_predict(
                query_ids=ids,
                query_prompt_hashes=prompt_hashes,
                query_embeddings=embeddings_by_fold[fold],
                bank_ids=bank.ids.tolist(),
                bank_prompt_hashes=bank.prompt_hashes.tolist(),
                bank_embeddings=bank.embeddings_by_fold[fold],
                bank_scores=bank.scores,
                k=k,
                temperature=temperature,
                unknown_prompt_policy=unknown_policy,
            )
            matrices.append(result.predictions)
        return _matrix(
            np.mean(np.stack(matrices, axis=0), axis=0),
            rows=len(records),
            where="anchor ensemble",
        )

    def _score_assessment(self, records: list[EssayInput]) -> np.ndarray:
        from src.assessment.artifact import load_deployment_artifact
        from src.assessment.extraction import (
            extract_assessment_probabilities,
            load_assessment_extractor,
        )
        from src.assessment.ridge import predict_assessment_fold_ensemble

        if self.config.assessment_artifact is None:
            raise RuntimeError("assessment scoring requested without an artifact")
        artifact = load_deployment_artifact(self.config.assessment_artifact)
        contract = artifact.get("feature_contract")
        if not isinstance(contract, dict):
            raise ValueError("assessment artifact has no feature contract")
        loaded = load_assessment_extractor(
            model_id=str(contract["model_id"]),
            model_revision=str(contract["model_revision"]),
            tokenizer_revision=str(contract["tokenizer_revision"]),
            precision=str(contract["precision"]),
            batch_size=int(contract["batch_size"]),
            max_length=int(contract["max_length"]),
            seed=int(contract["seed"]),
            quantization=dict(contract["quantization"]),
            answer_codes=tuple(str(value) for value in contract["answer_codes"]),
            device=self.config.runtime.device,
            allow_download=False,
        )
        try:
            if (
                loaded.feature_payload != contract
                or loaded.feature_signature != artifact.get("feature_signature")
            ):
                raise ValueError(
                    "runtime assessment extractor differs from its OOF contract"
                )
            _, probabilities = extract_assessment_probabilities(loaded, records)
        finally:
            del loaded
            _release_cuda()
        matrix = predict_assessment_fold_ensemble(
            probabilities,
            artifact["fold_models"],
            clip_min=float(artifact["clip_min"]),
            clip_max=float(artifact["clip_max"]),
        )
        return _matrix(matrix, rows=len(records), where="assessment ensemble")

    def predict_scores(
        self, records: Sequence[EssayInput | EssayRecord | dict[str, Any]]
    ) -> tuple[np.ndarray, str, str]:
        canonical = [ensure_essay_input(record) for record in records]
        if len(canonical) != self.config.runtime.expected_rows:
            raise ValueError(
                f"expected {self.config.runtime.expected_rows} input rows, got {len(canonical)}"
            )
        if len({record.id for record in canonical}) != len(canonical):
            raise ValueError("submission inputs contain duplicate IDs")
        qwen_matrix, qwen_signature, embeddings_by_fold = self._score_qwen(canonical)
        if self.config.scoring_mode == "qwen_ensemble":
            return qwen_matrix, self.config.qwen.scorer_name, qwen_signature

        from src.calibration.affine import AffinePromptCalibrator, DOMAINS
        from src.ensemble.simplex import TraitSimplexStacker

        assert self.config.stacker is not None
        aliases = dict(self.config.stacker.source_aliases)
        source_matrices: dict[str, np.ndarray] = {
            aliases["qwen"]: qwen_matrix,
        }
        source_identities: dict[str, tuple[str, str]] = {
            aliases["qwen"]: (self.config.qwen.scorer_name, qwen_signature),
        }
        if "baseline" in aliases:
            if (
                self.artifacts.baseline_scorer_name is None
                or self.artifacts.baseline_scorer_signature is None
            ):
                raise RuntimeError("validated baseline source identity is absent")
            source_matrices[aliases["baseline"]] = self._score_baseline(canonical)
            source_identities[aliases["baseline"]] = (
                self.artifacts.baseline_scorer_name,
                self.artifacts.baseline_scorer_signature,
            )
        if "anchor" in aliases:
            if (
                self.artifacts.anchor_scorer_name is None
                or self.artifacts.anchor_scorer_signature is None
            ):
                raise RuntimeError("validated anchor source identity is absent")
            source_matrices[aliases["anchor"]] = self._score_anchor(
                canonical, embeddings_by_fold
            )
            source_identities[aliases["anchor"]] = (
                self.artifacts.anchor_scorer_name,
                self.artifacts.anchor_scorer_signature,
            )
        if "assessment" in aliases:
            if (
                self.artifacts.assessment_scorer_name is None
                or self.artifacts.assessment_scorer_signature is None
            ):
                raise RuntimeError("validated assessment source identity is absent")
            source_matrices[aliases["assessment"]] = self._score_assessment(canonical)
            source_identities[aliases["assessment"]] = (
                self.artifacts.assessment_scorer_name,
                self.artifacts.assessment_scorer_signature,
            )
        stacker_payload = _json_object(self.config.stacker.artifact)
        source_order = stacker_payload.get("source_order")
        source_contracts = stacker_payload.get("source_contracts")
        expected_aliases = set(source_matrices)
        if (
            not isinstance(source_order, list)
            or len(source_order) != len(expected_aliases)
            or set(source_order) != expected_aliases
        ):
            raise ValueError("stacker source aliases do not match deployment config")
        if not isinstance(source_contracts, dict) or set(source_contracts) != expected_aliases:
            raise ValueError("stacker source contracts are absent")
        expected_contracts = {
            alias: {"scorer_name": name, "scorer_signature": signature}
            for alias, (name, signature) in source_identities.items()
        }
        for alias, expected in expected_contracts.items():
            contract = source_contracts.get(alias)
            if not isinstance(contract, dict):
                raise ValueError(f"stacker source contract is absent for {alias}")
            if any(contract.get(field) != value for field, value in expected.items()):
                raise ValueError(f"stacker source contract mismatch for {alias}")
        stacker = TraitSimplexStacker.from_dict(stacker_payload["stacker"])
        raw = stacker.transform(source_matrices)
        calibrator = AffinePromptCalibrator.from_dict(stacker_payload["calibrator"])
        if calibrator.fit_source != "base_oof_stacked":
            raise ValueError("stacker calibrator was not fitted from base OOF stacking")
        transformed = calibrator.transform(
            {trait: raw[:, index] for index, trait in enumerate(DOMAINS)},
            [record.prompt_num for record in canonical],
        )
        matrix = np.column_stack([transformed[trait] for trait in DOMAINS])
        signature = self.artifacts.stacker_signature
        if not isinstance(signature, str) or not signature:
            raise RuntimeError("validated stacker has no signature")
        scorer_name = stacker_payload.get("scorer_name")
        if not isinstance(scorer_name, str) or not scorer_name:
            raise ValueError("stacker artifact has no scorer_name")
        return _matrix(matrix, rows=len(canonical), where="stacked scores"), scorer_name, signature

    def _rationales(
        self,
        records: list[EssayInput],
        score_matrix: np.ndarray,
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]], int, str]:
        mode = self.config.rationale.mode
        rationales: list[dict[str, str]] = []
        ledger_rows: list[dict[str, Any]] = []
        fallback_count = 0
        if mode == "deterministic":
            for index, record in enumerate(records):
                scores = {
                    trait: float(score_matrix[index, trait_index])
                    for trait_index, trait in enumerate(TRAITS)
                }
                evidence = build_evidence_ledger(record)
                value = generate_grounded_rationales(evidence, scores)
                grounding = assess_grounding(value, essay=record.essay, ledger=evidence)
                if not grounding.accepted:
                    raise RuntimeError(
                        f"deterministic fallback is not grounded for {record.id}: "
                        f"{grounding.reasons}"
                    )
                rationales.append(value)
                ledger_rows.append(
                    {
                        "id": record.id,
                        "prompt_num": record.prompt_num,
                        "mode": mode,
                        "fallback_used": True,
                        "attempts": [],
                        "scores": scores,
                        "rationales": value,
                        "evidence": evidence.to_dict(),
                    }
                )
            fallback_count = len(records)
            root = Path(__file__).resolve().parents[1]
            signature = sha256_json(
                {
                    "template_version": RATIONALE_TEMPLATE_VERSION,
                    "evidence_code_sha256": sha256_file(root / "rationale" / "evidence.py"),
                    "template_code_sha256": sha256_file(
                        root / "rationale" / "deterministic.py"
                    ),
                    "grounding_code_sha256": sha256_file(
                        root / "rationale" / "parsing.py"
                    ),
                }
            )
            return rationales, ledger_rows, fallback_count, signature

        from src.inference.rationale_generator import (
            generate_rationale_for_record,
            load_rationale_generator,
        )

        if self.artifacts.rationale_checkpoint is None:
            raise RuntimeError("adapter rationale mode has no validated checkpoint")
        loaded = load_rationale_generator(
            self.artifacts.rationale_checkpoint,
            precision=self.config.runtime.rationale_precision,
            allow_download=False,
            device=self.config.runtime.device,
        )
        signature = loaded.generator_signature
        for index, record in enumerate(records):
            scores = {
                trait: float(score_matrix[index, trait_index])
                for trait_index, trait in enumerate(TRAITS)
            }
            result = generate_rationale_for_record(
                loaded,
                record,
                scores,
                max_attempts=self.config.rationale.max_attempts,
            )
            rationales.append(result.rationales)
            fallback_count += int(result.fallback_used)
            ledger_rows.append(
                {
                    "id": record.id,
                    "prompt_num": record.prompt_num,
                    "mode": mode,
                    "fallback_used": result.fallback_used,
                    "attempts": list(result.attempts),
                    "scores": scores,
                    "rationales": result.rationales,
                    "evidence": result.evidence,
                }
            )
        del loaded
        _release_cuda()
        return rationales, ledger_rows, fallback_count, signature

    def predict(
        self, records: Sequence[EssayInput | EssayRecord | dict[str, Any]]
    ) -> SubmissionResult:
        canonical = [ensure_essay_input(record) for record in records]
        score_matrix, scorer_name, scorer_signature = self.predict_scores(canonical)
        rationale_values, ledger_rows, fallback_count, rationale_signature = self._rationales(
            canonical, score_matrix
        )
        final_signature = sha256_json(
            {
                "score_scorer_signature": scorer_signature,
                "rationale_signature": rationale_signature,
                "schema": "content_organization_expression_score_rationale_v1",
                "model_name": self.config.output_model_name,
            }
        )
        rows = []
        for index, record in enumerate(canonical):
            scores = {
                trait: float(score_matrix[index, trait_index])
                for trait_index, trait in enumerate(TRAITS)
            }
            row = final_prediction_row(
                record_id=record.id,
                prompt_num=record.prompt_num,
                model=self.config.output_model_name,
                scores=scores,
                rationales=rationale_values[index],
            )
            strict_parse_prediction(serialize_prediction(row["prediction"]))
            for trait in TRAITS:
                if float(row["prediction"][trait]["score"]) != scores[trait]:
                    raise RuntimeError(f"rationale composition changed {record.id}/{trait}")
            rows.append(row)
        return SubmissionResult(
            rows=tuple(rows),
            ledger_rows=tuple(ledger_rows),
            score_matrix=np.array(score_matrix, dtype=float, copy=True),
            score_scorer_name=scorer_name,
            score_scorer_signature=scorer_signature,
            rationale_signature=rationale_signature,
            final_signature=final_signature,
            fallback_count=fallback_count,
        )


__all__ = ["SubmissionEngine", "SubmissionResult"]
