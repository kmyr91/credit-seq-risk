import numpy as np
import pandas as pd
import pytest

from src.evaluation.outcomes import evaluate_outcomes


def sample_data():
    predictions = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "probability": [0.1, 0.8, 0.2, 0.9],
    })

    outcomes = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "default_next_month": [0, 1, 0, 1],
    })

    return predictions, outcomes


def test_all_labels_available():
    predictions, outcomes = sample_data()
    result = evaluate_outcomes(predictions, outcomes)

    assert result["n_labeled"] == 4
    assert result["label_coverage"] == 1.0
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["pr_auc"] == pytest.approx(1.0)


def test_pending_outcomes_not_counted_as_zero():
    predictions, outcomes = sample_data()
    outcomes.loc[3, "default_next_month"] = np.nan

    result = evaluate_outcomes(predictions, outcomes)

    assert result["n_labeled"] == 3
    assert result["n_pending"] == 1
    assert result["label_coverage"] == 0.75


def test_duplicate_prediction_rejected():
    predictions, outcomes = sample_data()
    predictions.loc[1, "customer_id"] = 1

    with pytest.raises(ValueError):
        evaluate_outcomes(predictions, outcomes)


def test_invalid_probability_rejected():
    predictions, outcomes = sample_data()
    predictions.loc[0, "probability"] = 1.5

    with pytest.raises(ValueError):
        evaluate_outcomes(predictions, outcomes)


def test_single_class_has_undefined_auc():
    predictions, outcomes = sample_data()
    outcomes["default_next_month"] = 0

    result = evaluate_outcomes(predictions, outcomes)

    assert np.isnan(result["roc_auc"])
    assert np.isnan(result["pr_auc"])
    assert np.isfinite(result["brier_score"])
