"""One-time held-out evaluation of the frozen underwriting model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)

from src.models.train_lightgbm_raw import load_dataset
from src.evaluation.calibration import expected_calibration_error


MODEL_PATH = Path("outputs/uci/lightgbm_fair/model.joblib")
POLICY_PATH = Path(
    "outputs/uci/decision_economics/policy_assumptions.json"
)
OUT = Path("outputs/uci/final_test")

EXPECTED_THRESHOLD = 0.125
EXPECTED_FEATURE_COUNT = 12
SEED = 42


def calculate_metrics(y, p):
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece_10": float(
            expected_calibration_error(y, p, bins=10)
        ),
    }


def bootstrap_intervals(y, p, n_bootstrap=1000):
    """Stratified bootstrap 95% confidence intervals."""
    rng = np.random.default_rng(SEED)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    negatives = np.flatnonzero(y == 0)
    positives = np.flatnonzero(y == 1)

    if not len(negatives) or not len(positives):
        raise ValueError("Both outcome classes are required.")

    samples = []

    for _ in range(n_bootstrap):
        indices = np.concatenate([
            rng.choice(
                negatives,
                size=len(negatives),
                replace=True,
            ),
            rng.choice(
                positives,
                size=len(positives),
                replace=True,
            ),
        ])

        samples.append(calculate_metrics(y[indices], p[indices]))

    frame = pd.DataFrame(samples)

    return {
        metric: {
            "lower_95": float(frame[metric].quantile(0.025)),
            "upper_95": float(frame[metric].quantile(0.975)),
        }
        for metric in frame.columns
    }


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(
            "Final test outputs already exist. "
            "Do not repeatedly run or tune on the test set."
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    if not POLICY_PATH.is_file():
        raise FileNotFoundError(POLICY_PATH)

    with POLICY_PATH.open(encoding="utf-8") as f:
        policy = json.load(f)

    threshold = float(policy["selected_threshold"])
    revenue = float(policy["revenue_per_loan"])
    exposure = float(policy["exposure_at_default"])
    lgd = float(policy["loss_given_default"])

    if not np.isclose(threshold, EXPECTED_THRESHOLD):
        raise ValueError("Unexpected policy threshold.")

    if policy["model"] != "lightgbm_fair_12_features":
        raise ValueError("Unexpected policy model.")

    if policy["probabilities"] != "raw":
        raise ValueError("Unexpected probability calibration.")

    if min(revenue, exposure, lgd) < 0 or lgd > 1:
        raise ValueError("Invalid economic assumptions.")

    df, all_features = load_dataset()
    features = all_features[:EXPECTED_FEATURE_COUNT]

    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]
    test = df[df["split"] == "test"].copy()

    if not (
        len(train) == 21000
        and len(validation) == 4500
        and len(test) == 4500
    ):
        raise ValueError("Unexpected split sizes.")

    if set(test["customer_id"]) & set(train["customer_id"]):
        raise ValueError("Train/test customer overlap.")

    if set(test["customer_id"]) & set(
        validation["customer_id"]
    ):
        raise ValueError("Validation/test customer overlap.")

    model = joblib.load(MODEL_PATH)

    if list(model.feature_name_) != features:
        raise ValueError("Model feature schema mismatch.")

    X_test = test[features]
    y = test["default_next_month"].to_numpy(dtype=int)

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid test labels.")

    p = model.predict_proba(
        X_test, validate_features=True
    )[:, 1]

    if not (
        np.isfinite(p).all()
        and ((p >= 0) & (p <= 1)).all()
    ):
        raise ValueError("Invalid model probabilities.")

    metrics = calculate_metrics(y, p)
    confidence_intervals = bootstrap_intervals(y, p)

    approved = p < threshold
    n_approved = int(approved.sum())
    n_defaults = int(y[approved].sum())

    loss_per_default = exposure * lgd
    expected_value = revenue - p * loss_per_default

    decision_results = {
        "threshold": threshold,
        "n_customers": int(len(y)),
        "n_approved": n_approved,
        "approval_rate": float(approved.mean()),
        "mean_predicted_pd_approved": (
            float(p[approved].mean())
            if n_approved else None
        ),
        "observed_default_rate_approved": (
            float(y[approved].mean())
            if n_approved else None
        ),
        "observed_defaults_approved": n_defaults,
        "expected_net_value": float(
            expected_value[approved].sum()
        ),
        "illustrative_realized_net_value": float(
            n_approved * revenue
            - n_defaults * loss_per_default
        ),
    }

    print("\nFINAL HELD-OUT TEST — 12-FEATURE LIGHTGBM")
    print("Test customers:", len(y))

    for metric, value in metrics.items():
        interval = confidence_intervals[metric]
        print(
            f"{metric}: {value:.6f} "
            f"[{interval['lower_95']:.6f}, "
            f"{interval['upper_95']:.6f}]"
        )

    print("\nFROZEN UNDERWRITING POLICY")
    for name, value in decision_results.items():
        if isinstance(value, float):
            print(f"{name}: {value:.6f}")
        else:
            print(f"{name}: {value}")

    results = {
        "dataset": "UCI Default of Credit Card Clients",
        "model": "lightgbm_fair_12_features",
        "test_evaluation": "one_time_holdout",
        "n_features": len(features),
        "metrics": metrics,
        "bootstrap_95_intervals": confidence_intervals,
        "policy": decision_results,
        "assumptions": policy,
        "limitations": [
            "Retrospective evaluation on 2005 credit-card data.",
            "Business economics are hypothetical.",
            "Validation data informed model and policy selection.",
            "No production reject-inference correction.",
            "No temporal out-of-time validation.",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=False)

    with (OUT / "final_metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    predictions = test[
        ["customer_id", "default_next_month"]
    ].copy()

    predictions["probability"] = p
    predictions["approved"] = approved

    predictions.to_parquet(
        OUT / "test_predictions.parquet",
        index=False,
    )

    print("\nSaved to:", OUT)
    print("Final model and policy were not retrained.")


if __name__ == "__main__":
    main()
