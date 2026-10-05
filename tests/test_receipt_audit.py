"""Review clues must not turn punctuation changes into proven correct amounts."""

import importlib.util
from pathlib import Path

from precisionlab.metrics import exact, receipt_currency_spacing


spec = importlib.util.spec_from_file_location("analyze_study", Path(__file__).resolve().parents[1] / "experiments/analyze_service_study.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_equal_digits_are_only_a_lexical_clue_and_other_field_matches_are_visible():
    samples = [{"id": "total", "group_id": "receipt", "task": "total", "answer": "12.50", "image": "receipt.png"},
               {"id": "tax", "group_id": "receipt", "task": "tax", "answer": "1.00", "image": "receipt.png"},
               {"id": "cash", "group_id": "receipt", "task": "cash", "answer": "20.00", "image": "receipt.png"}]
    predictions = [{"id": "total", "prediction": "1,250", "success": True},
                   {"id": "tax", "prediction": "12.50", "success": True},
                   {"id": "cash", "prediction": "", "success": False}]
    result = audit.error_audit(samples, predictions)
    assert result["counts"] == {"same_signed_digit_string": 1, "matches_another_labelled_field": 1, "request_failed": 1}
    assert not exact("12.50", "1,250")
    assert result["cases"][1]["other_matching_fields"] == ["total"]
    assert "do not establish numeric equivalence" in result["scope"]


def test_a_lost_minus_sign_is_not_a_same_digit_clue():
    result = audit.error_audit([{"id": "x", "group_id": "r", "task": "amount", "answer": "-12.50", "image": "r.png"}],
                              [{"id": "x", "prediction": "12.50", "success": True}])
    assert result["counts"] == {"other_mismatch": 1}


def test_currency_spacing_view_preserves_amount_values_signs_and_currency():
    assert receipt_currency_spacing("Rp 20,446") == receipt_currency_spacing("Rp20,446")
    for left, right in (("Rp12.50", "Rp1,250"), ("-Rp12.50", "Rp12.50"),
                        ("Rp20,446", "20,446"), ("RP 20,446", "Rp20,446"), ("Rp1,818", "Rp1.818")):
        assert receipt_currency_spacing(left) != receipt_currency_spacing(right)
    assert not exact("Rp 20,446", "Rp20,446")
