import pandas as pd
import pytest

from src.evaluation.decision_economics_v2 import (
    evaluate_policy,
)


def test_break_even_threshold():
    data = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "default_next_month": [0, 1, 0],
        "raw_probability": [0.05, 0.10, 0.20],
    })

    result = evaluate_policy(data)

    assert result["threshold"] == pytest.approx(1 / 9)
    assert result["n_approved"] == 2
    assert result["approval_rate"] == pytest.approx(2 / 3)

    # Expected: (100 - 900 * 0.05)
    #         + (100 - 900 * 0.10) = 65
    assert result["expected_net_value"] == pytest.approx(65)

    # One performing loan earns $100.
    # One defaulted loan loses $800.
    assert result["illustrative_outcome_based_value"] == -700


def test_no_approvals():
    data = pd.DataFrame({
        "customer_id": [1, 2],
        "default_next_month": [0, 1],
        "raw_probability": [0.5, 0.8],
    })

    result = evaluate_policy(data)

    assert result["n_approved"] == 0
    assert result["expected_net_value"] == 0
    assert result["observed_default_rate_approved"] is None


def test_invalid_probability():
    data = pd.DataFrame({
        "customer_id": [1],
        "default_next_month": [0],
        "raw_probability": [1.2],
    })

    with pytest.raises(ValueError):
        evaluate_policy(data)
