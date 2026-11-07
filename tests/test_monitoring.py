import numpy as np
import pandas as pd
import pytest

from src.evaluation.monitoring import (
    population_stability_index,
    drift_report,
)


def test_identical_distributions_have_zero_psi():
    sample = np.arange(1000, dtype=float)
    assert population_stability_index(
        sample, sample
    ) == pytest.approx(0.0, abs=1e-12)


def test_shift_is_detected():
    rng = np.random.default_rng(42)
    reference = rng.normal(0, 1, size=5000)
    current = rng.normal(1.5, 1, size=5000)

    assert population_stability_index(
        reference, current
    ) > 0.25


def test_missingness_shift_is_detected():
    reference = np.arange(1000, dtype=float)
    current = reference.copy()
    current[:300] = np.nan

    assert population_stability_index(
        reference, current
    ) > 0.10


def test_report_contains_expected_fields():
    reference = pd.DataFrame({
        "credit_limit": [100, 200, 300, 400, 500]
    })
    current = pd.DataFrame({
        "credit_limit": [100, 200, 300, 400, 500]
    })

    report = drift_report(
        reference, current, ["credit_limit"]
    )

    assert len(report) == 1
    assert report.loc[0, "feature"] == "credit_limit"
    assert report.loc[0, "status"] == "stable"
    assert report.loc[0, "psi"] == pytest.approx(0.0)


def test_empty_sample_rejected():
    with pytest.raises(ValueError):
        population_stability_index([], [1, 2, 3])


def test_infinity_rejected():
    with pytest.raises(ValueError):
        population_stability_index(
            [1, np.inf], [1, 2]
        )
