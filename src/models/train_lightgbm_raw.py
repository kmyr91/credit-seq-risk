"""LightGBM with engineered and raw monthly credit features.

Uses an internal training split for early stopping.
Does not access the held-out test labels or evaluate the test set.
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


SEED = 42
DATA = Path("data/processed/uci")
OUT = Path("outputs/uci/lightgbm_raw")
OUT.mkdir(parents=True, exist_ok=True)

MONTHS = [
    "2005-04",
    "2005-05",
    "2005-06",
    "2005-07",
    "2005-08",
    "2005-09",
]

MONTHLY_COLUMNS = [
    "repayment_status",
    "bill_amount",
    "payment_amount",
]


def load_dataset():
    features = pd.read_parquet(
        DATA / "customer_features.parquet"
    )
    customers = pd.read_parquet(
        DATA / "customers.parquet"
    )
    splits = pd.read_parquet(
        DATA / "splits.parquet"
    )
    history = pd.read_parquet(
        DATA / "history.parquet"
    )

    # Keep original engineered-feature definitions.
    df = (
        features
        .merge(customers, on="customer_id", validate="one_to_one")
        .merge(splits, on="customer_id", validate="one_to_one")
    )

    df["credit_utilization"] = (
        df["latest_bill"].clip(lower=0)
        / df["credit_limit"].replace(0, np.nan)
    )

    df["payment_ratio"] = (
        df["total_payment"]
        / df["total_positive_bill"].replace(0, np.nan)
    )

    excluded = {
        "customer_id",
        "default_next_month",
        "split",
        "n_statements",
        "total_positive_bill",
    }

    engineered_cols = [
        col for col in df.columns
        if col not in excluded
    ]

    if len(engineered_cols) != 12:
        raise ValueError(
            f"Expected 12 engineered features, "
            f"found {len(engineered_cols)}."
        )

    # Reconstruct individual monthly inputs in wide format.
    history = history.copy()
    history["month"] = (
        pd.to_datetime(history["statement_date"])
        .dt.strftime("%Y-%m")
    )

    if history.duplicated(["customer_id", "month"]).any():
        raise ValueError("Duplicate customer-month observations.")

    if not history["month"].isin(MONTHS).all():
        raise ValueError("Unexpected statement month.")

    counts = history.groupby("customer_id")["month"].nunique()

    if not counts.eq(6).all():
        raise ValueError("Every customer must have six months.")

    wide = history.pivot(
        index="customer_id",
        columns="month",
        values=MONTHLY_COLUMNS,
    )

    expected_index = pd.MultiIndex.from_product(
        [MONTHLY_COLUMNS, MONTHS]
    )

    wide = wide.reindex(columns=expected_index)

    if wide.isna().any().any():
        raise ValueError("Missing monthly observations.")

    raw_cols = [
        f"{field}_{month.replace('-', '_')}"
        for field, month in wide.columns
    ]

    wide.columns = raw_cols
    wide = wide.reset_index()

    if len(raw_cols) != 18:
        raise ValueError("Expected 18 raw monthly features.")

    df = df.merge(
        wide,
        on="customer_id",
        validate="one_to_one",
    )

    feature_cols = engineered_cols + raw_cols

    if len(df) != 30000 or len(feature_cols) != 30:
        raise ValueError("Unexpected dataset or feature size.")

    prohibited = {
        "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "default_next_month", "customer_id", "split",
    }

    if prohibited.intersection(feature_cols):
        raise ValueError("Forbidden columns in model inputs.")

    return df, feature_cols


def evaluate(y, probabilities):
    return {
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
        "brier_score": brier_score_loss(y, probabilities),
    }


def main():
    df, feature_cols = load_dataset()

    train = df.loc[df["split"] == "train"].copy()
    val = df.loc[df["split"] == "validation"].copy()

    assert len(train) == 21000
    assert len(val) == 4500
    assert train["customer_id"].is_unique
    assert val["customer_id"].is_unique
    assert set(train["customer_id"]).isdisjoint(
        set(val["customer_id"])
    )

    X = train[feature_cols]
    y = train["default_next_month"].astype(int)

    X_val = val[feature_cols]
    y_val = val["default_next_month"].astype(int)

    print("Training customers:", len(train))
    print("Validation customers:", len(val))
    print("Engineered features: 12")
    print("Raw monthly features: 18")
    print("Total features:", len(feature_cols))

    # Hyperparameter selection happens inside the train split.
    X_fit, X_stop, y_fit, y_stop = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )

    params = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": -1,
        "random_state": SEED,
        "n_jobs": 2,
        "verbosity": -1,
        "metric": "auc",
    }

    selection_model = lgb.LGBMClassifier(**params)

    selection_model.fit(
        X_fit,
        y_fit,
        eval_X=X_stop,
        eval_y=y_stop,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=30,
                first_metric_only=True,
                verbose=False,
            )
        ],
    )

    best_iteration = int(selection_model.best_iteration_)

    if best_iteration < 1:
        raise ValueError("Invalid early-stopping iteration.")

    print("Internal best iteration:", best_iteration)

    # Train final candidate on all 21,000 training customers.
    # The original validation set is not used for fitting
    # or choosing the number of iterations.
    final_params = {
        **params,
        "n_estimators": best_iteration,
    }

    model = lgb.LGBMClassifier(**final_params)
    model.fit(X, y)

    probabilities = model.predict_proba(X_val)[:, 1]
    metrics = evaluate(y_val, probabilities)

    print("\nRAW + ENGINEERED LIGHTGBM")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    old_path = Path(
        "outputs/uci/stacking/"
        "lightgbm_full_control_predictions.parquet"
    )

    if old_path.is_file():
        old = pd.read_parquet(old_path)

        comparison = val[
            ["customer_id", "default_next_month"]
        ].copy()
        comparison["new_probability"] = probabilities

        comparison = comparison.merge(
            old[["customer_id", "probability"]],
            on="customer_id",
            how="inner",
            validate="one_to_one",
        )

        if len(comparison) != len(val):
            raise ValueError("Baseline comparison mismatch.")

        baseline = evaluate(
            comparison["default_next_month"],
            comparison["probability"],
        )

        print("\nNEW MINUS PREVIOUS LIGHTGBM")
        for name in metrics:
            print(f"{name}: {metrics[name] - baseline[name]:+.6f}")

    model_path = OUT / "model.joblib"
    joblib.dump(model, model_path)

    predictions = val[
        ["customer_id", "default_next_month"]
    ].copy()
    predictions["probability"] = probabilities

    predictions.to_parquet(
        OUT / "validation_predictions.parquet",
        index=False,
    )

    joblib.dump(
        feature_cols,
        OUT / "feature_columns.joblib",
    )

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        OUT / "feature_importance.csv",
        index=False,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit-seq-risk-uci")

    with mlflow.start_run(
        run_name="lightgbm_raw_30_features"
    ):
        mlflow.log_params({
            **final_params,
            "dataset": "uci",
            "n_features": len(feature_cols),
            "internal_stop_fraction": 0.20,
            "best_iteration": best_iteration,
            "validation_used_for_early_stopping": False,
            "test_evaluated": False,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(
            str(OUT / "feature_importance.csv")
        )

    print("\nSaved results to:", OUT)
    print("Held-out test set: NOT evaluated")


if __name__ == "__main__":
    main()
