from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from src.evaluation.metrics import TRAITS


def _matrix(value: Any, *, rows: int | None = None, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    expected = (rows, len(TRAITS)) if rows is not None else None
    if matrix.ndim != 2 or matrix.shape[1] != len(TRAITS):
        raise ValueError(f"{name} must have shape (n, 3), got {matrix.shape}")
    if expected is not None and matrix.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains a non-finite value")
    return matrix


@dataclass(frozen=True)
class TraitSimplexStacker:
    """Trait-wise nonnegative MSE blend whose source weights sum to one."""

    source_names: tuple[str, ...]
    weights: dict[str, tuple[float, ...]]
    degenerate: dict[str, bool]
    epsilon: float = 1e-12

    @classmethod
    def fit(
        cls,
        gold: Any,
        source_predictions: Mapping[str, Any],
        *,
        source_order: Sequence[str] | None = None,
        epsilon: float = 1e-12,
    ) -> "TraitSimplexStacker":
        if epsilon <= 0 or not np.isfinite(epsilon):
            raise ValueError("epsilon must be positive and finite")
        names = tuple(source_order or source_predictions.keys())
        if (
            len(names) < 2
            or len(set(names)) != len(names)
            or set(names) != set(source_predictions)
            or not all(isinstance(name, str) and name for name in names)
        ):
            raise ValueError("simplex stacker requires at least two uniquely named sources")
        truth = _matrix(gold, name="gold")
        sources = {
            name: _matrix(source_predictions[name], rows=len(truth), name=name)
            for name in names
        }
        weights: dict[str, tuple[float, ...]] = {}
        degenerate: dict[str, bool] = {}
        for index, trait in enumerate(TRAITS):
            design = np.column_stack([sources[name][:, index] for name in names])
            if len(names) == 2:
                delta = design[:, 0] - design[:, 1]
                denominator = float(np.dot(delta, delta))
                if denominator <= epsilon:
                    fitted = np.asarray([0.5, 0.5], dtype=float)
                    degenerate[trait] = True
                else:
                    numerator = float(np.dot(delta, truth[:, index] - design[:, 1]))
                    left_weight = float(np.clip(numerator / denominator, 0.0, 1.0))
                    fitted = np.asarray([left_weight, 1.0 - left_weight], dtype=float)
                    degenerate[trait] = False
            else:
                target = truth[:, index]

                def objective(value: np.ndarray) -> float:
                    residual = design @ value - target
                    return float(np.dot(residual, residual))

                def gradient(value: np.ndarray) -> np.ndarray:
                    return 2.0 * design.T @ (design @ value - target)

                result = minimize(
                    objective,
                    np.full(len(names), 1.0 / len(names), dtype=float),
                    jac=gradient,
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * len(names),
                    constraints={
                        "type": "eq",
                        "fun": lambda value: float(value.sum() - 1.0),
                        "jac": lambda value: np.ones_like(value),
                    },
                    options={"ftol": epsilon, "maxiter": 2000},
                )
                if not result.success or not np.isfinite(result.x).all():
                    raise RuntimeError(
                        f"simplex optimization failed for {trait}: {result.message}"
                    )
                fitted = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
                if fitted.sum() <= epsilon:
                    raise RuntimeError(f"simplex optimizer returned zero weights for {trait}")
                fitted /= fitted.sum()
                contrasts = design[:, :-1] - design[:, [-1]]
                degenerate[trait] = bool(
                    np.linalg.matrix_rank(contrasts, tol=epsilon) < len(names) - 1
                )
            weights[trait] = tuple(float(value) for value in fitted)
        return cls(
            source_names=tuple(str(name) for name in names),
            weights=weights,
            degenerate=degenerate,
            epsilon=float(epsilon),
        )

    def transform(self, source_predictions: Mapping[str, Any]) -> np.ndarray:
        if set(source_predictions) != set(self.source_names):
            raise ValueError("source names do not match the fitted stacker")
        matrices = {
            name: _matrix(source_predictions[name], name=name)
            for name in self.source_names
        }
        row_counts = {len(matrix) for matrix in matrices.values()}
        if len(row_counts) != 1:
            raise ValueError("source prediction row counts do not match")
        output = np.empty((next(iter(row_counts)), len(TRAITS)), dtype=float)
        for index, trait in enumerate(TRAITS):
            output[:, index] = sum(
                self.weights[trait][source_index] * matrices[name][:, index]
                for source_index, name in enumerate(self.source_names)
            )
        return np.clip(output, 1.0, 5.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": (
                "trait_two_source_mse_simplex"
                if len(self.source_names) == 2
                else "trait_multi_source_mse_simplex"
            ),
            "source_order": list(self.source_names),
            "weights": {
                trait: {
                    name: self.weights[trait][index]
                    for index, name in enumerate(self.source_names)
                }
                for trait in TRAITS
            },
            "degenerate": self.degenerate,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TraitSimplexStacker":
        if payload.get("method") not in {
            "trait_two_source_mse_simplex",
            "trait_multi_source_mse_simplex",
        }:
            raise ValueError("unsupported stacker method")
        source_order = payload.get("source_order")
        if (
            not isinstance(source_order, list)
            or len(source_order) < 2
            or not all(isinstance(value, str) and value for value in source_order)
            or len(set(source_order)) != len(source_order)
        ):
            raise ValueError("invalid stacker source_order")
        if (
            payload.get("method") == "trait_two_source_mse_simplex"
            and len(source_order) != 2
        ) or (
            payload.get("method") == "trait_multi_source_mse_simplex"
            and len(source_order) <= 2
        ):
            raise ValueError("stacker method/source count mismatch")
        raw_weights = payload.get("weights")
        raw_degenerate = payload.get("degenerate")
        if (
            not isinstance(raw_weights, Mapping)
            or not isinstance(raw_degenerate, Mapping)
            or set(raw_weights) != set(TRAITS)
            or set(raw_degenerate) != set(TRAITS)
        ):
            raise ValueError("stacker weights and degenerate flags must be mappings")
        weights: dict[str, tuple[float, ...]] = {}
        degenerate: dict[str, bool] = {}
        for trait in TRAITS:
            item = raw_weights.get(trait)
            if not isinstance(item, Mapping) or set(item) != set(source_order):
                raise ValueError(f"invalid weights for {trait}")
            values = tuple(float(item[name]) for name in source_order)
            if not all(np.isfinite(values)) or min(values) < 0 or abs(sum(values) - 1.0) > 1e-8:
                raise ValueError(f"weights for {trait} must lie on the simplex")
            flag = raw_degenerate.get(trait)
            if not isinstance(flag, bool):
                raise ValueError(f"invalid degenerate flag for {trait}")
            weights[trait] = values
            degenerate[trait] = flag
        epsilon = float(payload.get("epsilon", 1e-12))
        if epsilon <= 0 or not np.isfinite(epsilon):
            raise ValueError("stacker epsilon must be positive and finite")
        return cls(
            source_names=tuple(source_order),
            weights=weights,
            degenerate=degenerate,
            epsilon=epsilon,
        )
