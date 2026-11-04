
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


# 1. Directories
data_dir = Path("data/processed/uci")
output_dir = Path("outputs/uci/hybrid")

# 2. Load embeddings
train_embeddings = pd.read_parquet(
    output_dir / "train_embeddings.parquet"
)
val_embeddings = pd.read_parquet(
    output_dir / "validation_embeddings.parquet"
)

# 3. Load customer features and labels
features = pd.read_parquet(
    data_dir / "customer_features.parquet"
)
customers = pd.read_parquet(
    data_dir / "customers.parquet"
)
splits = pd.read_parquet(
    data_dir / "splits.parquet"
)

df = (
    features
    .merge(customers, on="customer_id", validate="one_to_one")
    .merge(splits, on="customer_id", validate="one_to_one")
)

# 4. Same features as our improved LightGBM
df["credit_utilization"] = (
    df["latest_bill"].clip(lower=0)
    / df["credit_limit"].replace(0, np.nan)
)

df["payment_ratio"] = (
    df["total_payment"]
    / df["total_positive_bill"].replace(0, np.nan)
)

exclude = {
    "customer_id",
    "default_next_month",
    "split",
    "n_statements",
    "total_positive_bill",
}

feature_cols = [
    col for col in df.columns
    if col not in exclude
]

embedding_cols = [
    col for col in train_embeddings.columns
    if col.startswith("gru_embedding_")
]

assert len(embedding_cols) == 32

train = train_embeddings.merge(
    df, on="customer_id", validate="one_to_one"
)
val = val_embeddings.merge(
    df, on="customer_id", validate="one_to_one"
)

assert len(train) == 5250
assert len(val) == 4500
assert train["split"].eq("train").all()
assert val["split"].eq("validation").all()

all_features = feature_cols + embedding_cols

X_train = train[all_features]
y_train = train["default_next_month"]

X_val = val[all_features]
y_val = val["default_next_month"]

print("Hybrid training:", X_train.shape)
print("Hybrid validation:", X_val.shape)


# 5. Train LightGBM
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

probabilities = model.predict_proba(X_val)[:, 1]

# 6. Evaluate
metrics = {
    "roc_auc": roc_auc_score(y_val, probabilities),
    "pr_auc": average_precision_score(y_val, probabilities),
    "brier_score": brier_score_loss(y_val, probabilities),
}

print("\nHybrid validation results:")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("Best iteration:", model.best_iteration_)

# 7. Save
joblib.dump(
    model, output_dir / "hybrid_lightgbm.joblib"
)
joblib.dump(
    all_features, output_dir / "feature_columns.joblib"
)

predictions = val[
    ["customer_id", "default_next_month"]
].copy()
predictions["probability"] = probabilities

predictions.to_parquet(
    output_dir / "validation_predictions.parquet",
    index=False,
)

# 8. MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit-seq-risk-uci")

with mlflow.start_run(
    run_name="gru_lightgbm_disjoint_hybrid"
):
    mlflow.log_params({
        **params,
        "encoder": "GRU",
        "hidden_size": 32,
        "encoder_epochs": 20,
        "training_strategy": "disjoint_stage_split",
        "encoder_train_customers": 15750,
        "tree_train_customers": len(train),
    })

    mlflow.log_metrics(metrics)
    mlflow.log_artifact(
        str(output_dir / "hybrid_lightgbm.joblib")
    )
    mlflow.log_artifact(
        str(output_dir / "encoder.pt")
    )

print("\nSaved to:", output_dir)
