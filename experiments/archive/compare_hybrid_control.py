
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

DATA = Path("data/processed/uci")
OUT = Path("outputs/uci/hybrid")

# Use the exact customers selected for hybrid stage 2.
train_ids = pd.read_parquet(
    OUT / "train_embeddings.parquet",
    columns=["customer_id"],
)

val_ids = pd.read_parquet(
    OUT / "validation_embeddings.parquet",
    columns=["customer_id"],
)

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

# Reuse exactly the 12 engineered features from the hybrid.
all_features = joblib.load(OUT / "feature_columns.joblib")
feature_cols = [
    c for c in all_features
    if not c.startswith("gru_embedding_")
]

train = train_ids.merge(df, on="customer_id", validate="one_to_one")
val = val_ids.merge(df, on="customer_id", validate="one_to_one")

assert len(train) == 5250
assert len(val) == 4500
assert train["split"].eq("train").all()
assert val["split"].eq("validation").all()
assert len(feature_cols) == 12

X_train = train[feature_cols]
y_train = train["default_next_month"]

X_val = val[feature_cols]
y_val = val["default_next_month"]

print("Control training:", X_train.shape)
print("Control validation:", X_val.shape)

# Identical LightGBM configuration to the hybrid.
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    random_state=42,
    n_jobs=2,
    verbosity=-1,
    metric="auc",
)

model.fit(
    X_train,
    y_train,
    eval_X=X_val,
    eval_y=y_val,
    callbacks=[
        lgb.early_stopping(
            stopping_rounds=30,
            first_metric_only=True,
            verbose=False,
        )
    ],
)

prob = model.predict_proba(X_val)[:, 1]

metrics = {
    "roc_auc": roc_auc_score(y_val, prob),
    "pr_auc": average_precision_score(y_val, prob),
    "brier_score": brier_score_loss(y_val, prob),
}

print("\nLightGBM control (5,250 customers):")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

print("Best iteration:", model.best_iteration_)

# Compare against the existing hybrid on identical rows.
hybrid_pred = pd.read_parquet(
    OUT / "validation_predictions.parquet"
)

comparison = val[
    ["customer_id", "default_next_month"]
].merge(
    hybrid_pred[["customer_id", "probability"]],
    on="customer_id",
    validate="one_to_one",
)

assert len(comparison) == 4500
assert np.array_equal(
    comparison["default_next_month"].to_numpy(),
    y_val.to_numpy(),
)

print("\nHybrid ROC-AUC:",
      round(roc_auc_score(
          comparison["default_next_month"],
          comparison["probability"]
      ), 4))

print("Control ROC-AUC:", round(metrics["roc_auc"], 4))

# Save results.
joblib.dump(model, OUT / "control_lightgbm.joblib")

pred = val[["customer_id", "default_next_month"]].copy()
pred["probability"] = prob
pred.to_parquet(
    OUT / "control_validation_predictions.parquet",
    index=False,
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit-seq-risk-uci")

with mlflow.start_run(run_name="lightgbm_5250_control"):
    mlflow.log_params({
        "training_customers": len(train),
        "n_features": len(feature_cols),
        "comparison": "hybrid_without_embeddings",
        "best_iteration": model.best_iteration_,
    })
    mlflow.log_metrics(metrics)

print("\nControl experiment completed.")
