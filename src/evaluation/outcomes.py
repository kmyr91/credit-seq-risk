"""Evaluate predictions after default outcomes become available."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


INPUT = Path(
    "outputs/uci/stacking/"
    "lightgbm_full_control_predictions.parquet"
)
OUTPUT = Path("outputs/uci/outcomes")


def evaluate_outcomes(predictions, outcomes):
    """Join predictions to observed labels and calculate performance.

    Only customers with an available outcome are evaluated.
    Missing labels are never treated as non-defaults.
    """
    required_predictions = {"customer_id", "probability"}
    required_outcomes = {"customer_id", "default_next_month"}

    if not required_predictions.issubset(predictions.columns):
        raise ValueError("Missing prediction columns.")

    if not required_outcomes.issubset(outcomes.columns):
        raise ValueError("Missing outcome columns.")

    if predictions["customer_id"].isna().any():
        raise ValueError("Null prediction customer ID.")

    if outcomes["customer_id"].isna().any():
        raise ValueError("Null outcome customer ID.")

    if predictions["customer_id"].duplicated().any():
        raise ValueError("Duplicate prediction customer ID.")

    if outcomes["customer_id"].duplicated().any():
        raise ValueError("Duplicate outcome customer ID.")

    scores = pd.to_numeric(
        predictions["probability"], errors="raise"
    )

    if (
        scores.isna().any()
        or not np.isfinite(scores.to_numpy()).all()
        or not scores.between(0, 1).all()
    ):
        raise ValueError("Invalid prediction probabilities.")

    labels = outcomes["default_next_month"].dropna()

    if not labels.isin([0, 1]).all():
        raise ValueError("Outcomes must be binary or missing.")

    joined = predictions[
        ["customer_id", "probability"]
    ].merge(
        outcomes[["customer_id", "default_next_month"]],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    observed = joined.dropna(
        subset=["default_next_month"]
    ).copy()

    result = {
        "n_predictions": len(joined),
        "n_labeled": len(observed),
        "n_pending": len(joined) - len(observed),
        "label_coverage": (
            len(observed) / len(joined)
            if len(joined) else 0.0
        ),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "brier_score": np.nan,
    }

    if len(observed):
        y = observed["default_next_month"].astype(int)
        p = observed["probability"]

        result["brier_score"] = brier_score_loss(y, p)

        if y.nunique() == 2:
            result["roc_auc"] = roc_auc_score(y, p)
            result["pr_auc"] = average_precision_score(y, p)

    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(INPUT)

    # Demonstration only:
    # The historical file already contains predictions and labels.
    # Separate them to simulate a delayed-outcome join.
    predictions = data[["customer_id", "probability"]].copy()
    outcomes = data[
        ["customer_id", "default_next_month"]
    ].copy()

    rng = np.random.default_rng(42)

    # Simulate that 25% of outcomes have not arrived.
    pending = rng.choice(
        len(outcomes),
        size=len(outcomes) // 4,
        replace=False,
    )

    outcomes.loc[
        outcomes.index[pending], "default_next_month"
    ] = np.nan

    partial = evaluate_outcomes(predictions, outcomes)

    # Evaluate all recorded historical outcomes.
    complete = evaluate_outcomes(
        predictions,
        data[["customer_id", "default_next_month"]],
    )

    results = pd.DataFrame([
        {"scenario": "simulated_pending_labels", **partial},
        {"scenario": "all_historical_labels", **complete},
    ])

    print("\nDELAYED-OUTCOME MONITORING")
    print(results.round(4).to_string(index=False))

    results.to_csv(
        OUTPUT / "outcome_monitoring.csv",
        index=False,
    )

    print("\nSaved to:", OUTPUT)


if __name__ == "__main__":
    main()
