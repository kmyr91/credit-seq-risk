"""Platt calibration for the selected 12-feature LightGBM."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)


INPUT = Path(
    "outputs/uci/lightgbm_fair/"
    "validation_predictions.parquet"
)
OUT = Path("outputs/uci/calibration")
SEED = 42


def logit(probabilities):
    p = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expected_calibration_error(y, p, bins=10):
    """ECE with equally spaced probability bins."""
    y = np.asarray(y)
    p = np.asarray(p)

    edges = np.linspace(0, 1, bins + 1)
    indices = np.minimum(
        np.searchsorted(edges, p, side="right") - 1,
        bins - 1,
    )

    error = 0.0

    for i in range(bins):
        mask = indices == i

        if mask.any():
            error += mask.mean() * abs(
                y[mask].mean() - p[mask].mean()
            )

    return float(error)


def reliability_points(y, p, bins=10):
    """Observed default frequency versus mean predicted PD."""
    y = np.asarray(y)
    p = np.asarray(p)

    # Quantile bins give approximately balanced bin sizes.
    groups = pd.qcut(
        p,
        q=bins,
        labels=False,
        duplicates="drop",
    )

    frame = pd.DataFrame({
        "label": y,
        "probability": p,
        "bin": groups,
    })

    return frame.groupby("bin").agg(
        predicted_pd=("probability", "mean"),
        observed_default_rate=("label", "mean"),
        count=("label", "size"),
    ).reset_index()


def metrics(y, p):
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "ece_10": expected_calibration_error(y, p),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(INPUT)

    if len(data) != 4500:
        raise ValueError("Unexpected validation size.")

    if data["customer_id"].duplicated().any():
        raise ValueError("Duplicate customers.")

    if (
        data["probability"].isna().any()
        or not data["probability"].between(0, 1).all()
        or not np.isfinite(data["probability"]).all()
    ):
        raise ValueError("Invalid prediction probabilities.")

    calibration, policy = train_test_split(
        data,
        test_size=0.50,
        random_state=SEED,
        stratify=data["default_next_month"],
    )

    calibration = calibration.copy()
    policy = policy.copy()

    # Fit the calibrator on the first half only.
    calibrator = LogisticRegression(
        random_state=SEED,
        max_iter=1000,
    )

    calibrator.fit(
        logit(calibration["probability"]).reshape(-1, 1),
        calibration["default_next_month"],
    )

    # Evaluate on customers not used to fit calibration.
    raw = policy["probability"].to_numpy()

    calibrated = calibrator.predict_proba(
        logit(raw).reshape(-1, 1)
    )[:, 1]

    y = policy["default_next_month"].to_numpy()

    results = pd.DataFrame({
        "raw": metrics(y, raw),
        "platt": metrics(y, calibrated),
    })

    print("\nCALIBRATION: POLICY-VALIDATION CUSTOMERS")
    print("Calibration customers:", len(calibration))
    print("Evaluation customers:", len(policy))
    print(results.round(6))

    policy["raw_probability"] = raw
    policy["calibrated_probability"] = calibrated

    policy[
        [
            "customer_id",
            "default_next_month",
            "raw_probability",
            "calibrated_probability",
        ]
    ].to_parquet(
        OUT / "policy_validation_predictions.parquet",
        index=False,
    )

    joblib.dump(
        calibrator,
        OUT / "platt_calibrator.joblib",
    )

    results.to_csv(OUT / "calibration_metrics.csv")

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        [0, 1], [0, 1],
        "--", color="gray",
        label="Perfect calibration",
    )

    for name, probabilities in [
        ("Raw LightGBM", raw),
        ("Platt calibrated", calibrated),
    ]:
        points = reliability_points(y, probabilities)

        ax.plot(
            points["predicted_pd"],
            points["observed_default_rate"],
            marker="o",
            label=name,
        )

    ax.set_xlabel("Mean predicted probability of default")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Credit Default Probability Calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUT / "reliability_diagram.png",
        dpi=180,
    )
    plt.close(fig)

    print("\nSaved to:", OUT)
    print("Final test set remains untouched.")


if __name__ == "__main__":
    main()
