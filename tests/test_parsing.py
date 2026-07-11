import json

import pytest

from src.inference.parsing import parse_repaired_json, parse_strict_json


def _valid_payload() -> dict:
    return {
        domain: {"score": 3.25, "rationale": f"{domain} 근거"}
        for domain in ("content", "organization", "expression")
    }


def test_strict_parser_accepts_decimal_scores():
    assert parse_strict_json(json.dumps(_valid_payload(), ensure_ascii=False))["content"]["score"] == 3.25


def test_repaired_parser_recovers_observed_closing_parenthesis_only():
    text = json.dumps(_valid_payload(), ensure_ascii=False)
    broken = text.rsplit("}", 2)[0] + ")" + "}"
    with pytest.raises(ValueError):
        parse_strict_json(broken)
    assert parse_repaired_json(broken)["expression"]["score"] == 3.25


def test_parser_rejects_out_of_range_score():
    payload = _valid_payload()
    payload["content"]["score"] = 5.1
    with pytest.raises(ValueError, match="between 1 and 5"):
        parse_strict_json(json.dumps(payload, ensure_ascii=False))


def test_strict_parser_rejects_string_score_and_extra_key():
    payload = _valid_payload()
    payload["content"]["score"] = "3.25"
    with pytest.raises(ValueError, match="real number"):
        parse_strict_json(json.dumps(payload, ensure_ascii=False))

    payload = _valid_payload()
    payload["extra"] = {}
    with pytest.raises(ValueError, match="keys must be exactly"):
        parse_strict_json(json.dumps(payload, ensure_ascii=False))
