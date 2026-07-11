from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


DOMAINS = ("content", "organization", "expression")


@dataclass(frozen=True)
class DomainCalibration:
    intercept: float
    slope: float
    prompt_offsets: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "intercept": self.intercept,
            "slope": self.slope,
            "prompt_offsets": self.prompt_offsets,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "DomainCalibration":
        offsets = value.get("prompt_offsets", {})
        if not isinstance(offsets, Mapping):
            raise ValueError("prompt_offsets must be a mapping")
        return cls(
            intercept=float(value["intercept"]),
            slope=float(value["slope"]),
            prompt_offsets={str(k): float(v) for k, v in offsets.items()},
        )


@dataclass(frozen=True)
class AffinePromptCalibrator:
    domains: dict[str, DomainCalibration]
    clip_min: float = 1.0
    clip_max: float = 5.0
    prompt_shrinkage: float = 20.0
    fit_source: str = "oof"

    @classmethod
    def fit(
        cls,
        gold: Mapping[str, Sequence[float]],
        predicted: Mapping[str, Sequence[float]],
        prompts: Sequence[str],
        *,
        prompt_shrinkage: float = 20.0,
        clip_min: float = 1.0,
        clip_max: float = 5.0,
        min_slope: float = 1e-6,
        fit_source: str = "oof",
    ) -> "AffinePromptCalibrator":
        if prompt_shrinkage < 0:
            raise ValueError("prompt_shrinkage must be non-negative")
        prompt_array = np.asarray(prompts, dtype=str)
        calibrated: dict[str, DomainCalibration] = {}

        for domain in DOMAINS:
            y = np.asarray(gold[domain], dtype=float)
            x = np.asarray(predicted[domain], dtype=float)
            if y.shape != x.shape or y.ndim != 1 or len(y) != len(prompt_array):
                raise ValueError(f"shape mismatch for {domain}")
            if len(y) < 2 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
                raise ValueError(f"invalid calibration data for {domain}")

            centered_x = x - x.mean()
            variance = float(np.dot(centered_x, centered_x))
            if variance == 0.0:
                slope = min_slope
            else:
                slope = max(float(np.dot(centered_x, y - y.mean()) / variance), min_slope)
            intercept = float(y.mean() - slope * x.mean())

            residual = y - (intercept + slope * x)
            offsets: dict[str, float] = {}
            for prompt in np.unique(prompt_array):
                mask = prompt_array == prompt
                count = int(mask.sum())
                weight = count / (count + prompt_shrinkage) if prompt_shrinkage else 1.0
                offsets[str(prompt)] = float(weight * residual[mask].mean())

            calibrated[domain] = DomainCalibration(intercept, slope, offsets)

        return cls(
            domains=calibrated,
            clip_min=clip_min,
            clip_max=clip_max,
            prompt_shrinkage=prompt_shrinkage,
            fit_source=fit_source,
        )

    def transform(
        self,
        predicted: Mapping[str, Sequence[float]],
        prompts: Sequence[str],
    ) -> dict[str, np.ndarray]:
        prompt_array = np.asarray(prompts, dtype=str)
        output: dict[str, np.ndarray] = {}
        for domain in DOMAINS:
            values = np.asarray(predicted[domain], dtype=float)
            if values.ndim != 1 or len(values) != len(prompt_array):
                raise ValueError(f"shape mismatch for {domain}")
            params = self.domains[domain]
            offsets = np.fromiter(
                (params.prompt_offsets.get(str(prompt), 0.0) for prompt in prompt_array),
                dtype=float,
                count=len(prompt_array),
            )
            calibrated = params.intercept + params.slope * values + offsets
            output[domain] = np.clip(calibrated, self.clip_min, self.clip_max)
        return output

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "affine_prompt_shrinkage",
            "fit_source": self.fit_source,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "prompt_shrinkage": self.prompt_shrinkage,
            "domains": {key: value.to_dict() for key, value in self.domains.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AffinePromptCalibrator":
        domains = value.get("domains")
        if not isinstance(domains, Mapping):
            raise ValueError("domains must be a mapping")
        parsed = {
            domain: DomainCalibration.from_dict(domains[domain])  # type: ignore[arg-type]
            for domain in DOMAINS
        }
        return cls(
            domains=parsed,
            clip_min=float(value.get("clip_min", 1.0)),
            clip_max=float(value.get("clip_max", 5.0)),
            prompt_shrinkage=float(value.get("prompt_shrinkage", 20.0)),
            fit_source=str(value.get("fit_source", "oof")),
        )

