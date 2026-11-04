from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

DATA = Path("data/processed/uci")
OOF = Path("outputs/uci/gru_oof")
OUT = Path("outputs/uci/stacking")
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load engineered features and labels
# --------------------------------------------------

features = pd.read_parquet(DATA / "customer_features.parquet")
customers = pd.read_parquet(DATA / "customers.parquet")
splits = pd.read_parquet(DATA / "splits.parquet")

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

# Same 12 features as our previous hybrid.
old_features = joblib.load(
    "outputs/uci/hybrid/feature_columns.joblib"
)

feature_cols = [
    col for col in old_features
    if not col.startswith("gru_embedding_")
]

assert len(feature_cols) == 12


# --------------------------------------------------
# 3. Load out-of-fold GRU predictions
# --------------------------------------------------

train_logits = pd.read_parquet(
    OOF / "train_oof_logits.parquet"
)

val_logits = pd.read_parquet(
    OOF / "validation_logits.parquet"
)

assert train_logits["customer_id"].is_unique
assert val_logits["customer_id"].is_unique

train = train_logits.merge(
    df, on="customer_id", validate="one_to_one"
)

val = val_logits.merge(
    df, on="customer_id", validate="one_to_one"
)

assert len(train) == 21000
assert len(val) == 4500
assert train["split"].eq("train").all()
assert val["split"].eq("validation").all()
assert np.isfinite(train["gru_oof_logit"]).all()
assert np.isfinite(val["gru_oof_logit"]).all()

assert set(train["customer_id"]).isdisjoint(
    set(val["customer_id"])
)

y_train = train["default_next_month"]
y_val = val["default_next_month"]

print("Training customers:", len(train))
print("Validation customers:", len(val))


# --------------------------------------------------
# 4. Evaluation
# --------------------------------------------------

def evaluate(y_true, probabilities):
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
    }


# --------------------------------------------------
# 5. Train one LightGBM model
# --------------------------------------------------

def fit_model(name, columns):

    params = {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "random_state": 42,
        "n_jobs": 2,
        "verbosity": -1,
        "metric": "auc",
    }

    model = lgb.LGBMClassifier(**params)

    model.fit(
        train[columns],
        y_train,
        eval_set=[(val[columns], y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=30,
                first_metric_only=True,
                verbose=False,
            )
        ],
    )

    probabilities = model.predict_proba(
        val[columns]
    )[:, 1]

    metrics = evaluate(y_val, probabilities)

    print(f"\n{name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("Best iteration:", model.best_iteration_)

    joblib.dump(model, OUT / f"{name}_model.joblib")

    predictions = val[
        ["customer_id", "default_next_month"]
    ].copy()

    predictions["probability"] = probabilities

    predictions.to_parquet(
        OUT / f"{name}_predictions.parquet",
        index=False,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit-seq-risk-uci")

    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            **params,
            "n_features": len(columns),
            "train_customers": len(train),
            "best_iteration": model.best_iteration_,
            "model_type": name,
        })

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(
            str(OUT / f"{name}_model.joblib")
        )

    return metrics


# --------------------------------------------------
# 6. Train both models on the same customers
# --------------------------------------------------

baseline_metrics = fit_model(
    "lightgbm_full_control",
    feature_cols,
)

stacked_metrics = fit_model(
    "gru_oof_lightgbm_stack",
    feature_cols + ["gru_oof_logit"],
)


# --------------------------------------------------
# 7. Compare
# --------------------------------------------------

comparison = pd.DataFrame({
    "LightGBM": baseline_metrics,
    "GRU + LightGBM": stacked_metrics,
}).T

print("\n========== FINAL COMPARISON ==========")
print(comparison.round(5).to_string())

print("\nStacked minus baseline:")
for metric in baseline_metrics:
    difference = (
        stacked_metrics[metric] - baseline_metrics[metric]
    )
    print(f"{metric}: {difference:+.5f}")

comparison.to_csv(OUT / "comparison.csv")

print("\nSaved to:", OUT)
