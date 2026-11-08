"""Illustrative credit-approval economics on policy validation data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/uci/calibration/"
    "policy_validation_predictions.parquet"
)
OUT = Path("outputs/uci/decision_economics")

REVENUE_PER_LOAN = 100.0
EXPOSURE_AT_DEFAULT = 1000.0
LOSS_GIVEN_DEFAULT = 0.80


def analyze_policy(data):
    y = data["default_next_month"].to_numpy(dtype=int)
    p = data["raw_probability"].to_numpy(dtype=float)

    if (
        len(data) == 0
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or not ((p >= 0) & (p <= 1)).all()
    ):
        raise ValueError("Invalid policy-validation data.")

    loss_if_default = (
        EXPOSURE_AT_DEFAULT * LOSS_GIVEN_DEFAULT
    )

    expected_value = (
        REVENUE_PER_LOAN - p * loss_if_default
    )

    thresholds = np.round(
        np.arange(0.0, 1.001, 0.005), 3
    )

    rows = []

    for threshold in thresholds:
        approved = p < threshold
        n_approved = int(approved.sum())

        observed_defaults = int(y[approved].sum())

        rows.append({
            "threshold": threshold,
            "approved": n_approved,
            "approval_rate": n_approved / len(y),
            "mean_predicted_pd": (
                float(p[approved].mean())
                if n_approved else np.nan
            ),
            "observed_default_rate": (
                observed_defaults / n_approved
                if n_approved else np.nan
            ),
            "expected_net_value": float(
                expected_value[approved].sum()
            ),
            "illustrative_realized_net_value": float(
                n_approved * REVENUE_PER_LOAN
                - observed_defaults * loss_if_default
            ),
        })

    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(INPUT)

    if len(data) != 2250:
        raise ValueError(
            "Expected 2,250 policy-validation customers."
        )

    if data["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs.")

    report = analyze_policy(data)

    best = report.loc[
        report["expected_net_value"].idxmax()
    ]

    break_even_pd = (
        REVENUE_PER_LOAN
        / (EXPOSURE_AT_DEFAULT * LOSS_GIVEN_DEFAULT)
    )

    print("\nILLUSTRATIVE UNDERWRITING ECONOMICS")
    print("Customers:", len(data))
    print("Revenue per approved loan:", REVENUE_PER_LOAN)
    print(
        "Loss if default:",
        EXPOSURE_AT_DEFAULT * LOSS_GIVEN_DEFAULT,
    )
    print("Break-even PD:", round(break_even_pd, 4))

    print("\nBEST GRID THRESHOLD")
    for name, value in best.items():
        print(f"{name}: {value:.4f}")

    report.to_csv(
        OUT / "economic_thresholds.csv",
        index=False,
    )

    assumptions = {
        "revenue_per_loan": REVENUE_PER_LOAN,
        "exposure_at_default": EXPOSURE_AT_DEFAULT,
        "loss_given_default": LOSS_GIVEN_DEFAULT,
        "break_even_pd": break_even_pd,
        "selected_threshold": float(best["threshold"]),
        "model": "lightgbm_fair_12_features",
        "probabilities": "raw",
        "selection_dataset": "policy_validation",
        "note": (
            "Hypothetical economics, not an actual "
            "lending policy or profit estimate."
        ),
    }

    with open(
        OUT / "policy_assumptions.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(assumptions, f, indent=2)

    print("\nSaved to:", OUT)
    print("Held-out test set remains untouched.")


if __name__ == "__main__":
    main()
