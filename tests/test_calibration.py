import numpy as np

from src.calibration.affine import AffinePromptCalibrator, DOMAINS


def test_affine_calibration_removes_known_bias_and_keeps_positive_slope():
    x = np.linspace(1.5, 4.5, 30)
    predicted = {domain: x for domain in DOMAINS}
    gold = {domain: 0.5 + 0.8 * x for domain in DOMAINS}
    prompts = ["Q1"] * 15 + ["Q2"] * 15

    model = AffinePromptCalibrator.fit(gold, predicted, prompts, prompt_shrinkage=20)
    transformed = model.transform(predicted, prompts)

    for domain in DOMAINS:
        assert model.domains[domain].slope > 0
        np.testing.assert_allclose(transformed[domain], gold[domain], atol=1e-10)


def test_unknown_prompt_uses_global_affine_only_and_roundtrips():
    predicted = {domain: [2.0, 3.0, 4.0] for domain in DOMAINS}
    gold = {domain: [2.5, 3.5, 4.5] for domain in DOMAINS}
    model = AffinePromptCalibrator.fit(gold, predicted, ["Q1", "Q1", "Q1"])
    restored = AffinePromptCalibrator.from_dict(model.to_dict())

    actual = restored.transform({domain: [3.0] for domain in DOMAINS}, ["UNKNOWN"])
    for domain in DOMAINS:
        assert actual[domain][0] == 3.5

