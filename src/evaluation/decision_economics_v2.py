"""Revised illustrative lending economics.

Revenue is received only on non-defaulted loans.
This revision was made AFTER the original held-out test evaluation.
No test data is loaded or evaluated here.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/uci/calibration/"
    "policy_validation_predictions.parquet"
)
OUTPUT = Path("outputs/uci/decision_economics_v2")

REVENUE_IF_PERFORMING = 100.0
EXPOSURE_AT_DEFAULT = 1000.0
LOSS_GIVEN_DEFAULT = 0.80


def evaluate_policy(data):
    required = {
        "customer_id",
        "default_next_month",
        "raw_probability",
    }

    if not required.issubset(data.columns):
        raise ValueError("Missing policy data columns.")

    if data.empty or data["customer_id"].duplicated().any():
        raise ValueError("Invalid customer records.")

    y = data["default_next_month"].to_numpy(dtype=int)
    p = data["raw_probability"].to_numpy(dtype=float)

    if (
        not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or not np.logical_and(p >= 0, p <= 1).all()
    ):
        raise ValueError("Invalid labels or probabilities.")

    loss_if_default = (
        EXPOSURE_AT_DEFAULT * LOSS_GIVEN_DEFAULT
    )

    threshold = (
        REVENUE_IF_PERFORMING
        / (REVENUE_IF_PERFORMING + loss_if_default)
    )

    approved = p < threshold
    n_approved = int(approved.sum())
    defaults = int(y[approved].sum())

    expected_values = (
        (1 - p) * REVENUE_IF_PERFORMING
        - p * loss_if_default
    )

    result = {
        "threshold": float(threshold),
        "n_customers": len(y),
        "n_approved": n_approved,
        "approval_rate": float(approved.mean()),
        "mean_predicted_pd_approved": (
            float(p[approved].mean())
            if n_approved else None
        ),
        "observed_default_rate_approved": (
            defaults / n_approved if n_approved else None
        ),
        "expected_net_value": float(
            expected_values[approved].sum()
        ),
        "illustrative_outcome_based_value": float(
            (n_approved - defaults) * REVENUE_IF_PERFORMING
            - defaults * loss_if_default
        ),
    }

    return result


def main():
    if OUTPUT.exists():
        raise RuntimeError(
            "Revised policy output already exists. "
            "Preserve previous analysis results."
        )

    data = pd.read_parquet(INPUT)

    if len(data) != 2250:
        raise ValueError("Unexpected validation sample size.")

    result = evaluate_policy(data)

    record = {
        "policy": result,
        "assumptions": {
            "revenue_if_performing": REVENUE_IF_PERFORMING,
            "revenue_if_default": 0.0,
            "exposure_at_default": EXPOSURE_AT_DEFAULT,
            "loss_given_default": LOSS_GIVEN_DEFAULT,
        },
        "disclosure": (
            "This policy was revised after the original "
            "held-out test results were viewed. "
            "Only policy-validation data was used here. "
            "No revised-policy test performance is claimed."
        ),
    }

    OUTPUT.mkdir(parents=True, exist_ok=False)

    with (OUTPUT / "revised_policy.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(record, f, indent=2)

    print("\nREVISED POLICY — VALIDATION ONLY")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print("\nNo held-out test data accessed.")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()
