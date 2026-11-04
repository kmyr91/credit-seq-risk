
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


# 1. Directories
data_dir = Path("data/processed/uci")
output_dir = Path("outputs/uci/lightgbm_improved")
old_dir = Path("outputs/uci/lightgbm")

output_dir.mkdir(parents=True, exist_ok=True)


# 2. Load and join
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

assert len(df) == 30000
assert df["customer_id"].is_unique


# 3. Financial ratios
# Use NaN for undefined ratios. LightGBM handles missing values.

df["credit_utilization"] = (
    df["latest_bill"].clip(lower=0)
    / df["credit_limit"].replace(0, np.nan)
)

df["payment_ratio"] = (
    df["total_payment"]
    / df["total_positive_bill"].replace(0, np.nan)
)

# Remove helper columns and constant feature
exclude = [
    "customer_id",
    "default_next_month",
    "split",
    "n_statements",
    "total_positive_bill",
]

feature_cols = [
    col for col in df.columns
    if col not in exclude
]

# 4. Fixed customer splits
train = df[df["split"] == "train"]
val = df[df["split"] == "validation"]

X_train = train[feature_cols]
y_train = train["default_next_month"]

X_val = val[feature_cols]
y_val = val["default_next_month"]

print("Training customers:", len(train))
print("Validation customers:", len(val))
print("Features:", feature_cols)


# 5. Model
params = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": -1,
    "random_state": 42,
    "n_jobs": 2,
    "verbosity": -1,
    "metric": "auc",
}

model = lgb.LGBMClassifier(**params)


# 6. MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit-seq-risk-uci")

with mlflow.start_run(run_name="lightgbm_improved"):

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

    metrics = {
        "roc_auc": roc_auc_score(y_val, probabilities),
        "pr_auc": average_precision_score(y_val, probabilities),
        "brier_score": brier_score_loss(y_val, probabilities),
    }

    print("\nImproved validation results:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("Best iteration:", model.best_iteration_)

    # 7. Save model and validation predictions
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)

    predictions = val[
        ["customer_id", "default_next_month"]
    ].copy()

    predictions["probability"] = probabilities

    predictions.to_parquet(
        output_dir / "validation_predictions.parquet",
        index=False,
    )

    # 8. Calibration plot
    fig, ax = plt.subplots(figsize=(6, 5))

    CalibrationDisplay.from_predictions(
        y_val,
        probabilities,
        n_bins=10,
        strategy="quantile",
        ax=ax,
    )

    ax.set_title("Improved LightGBM Calibration")
    fig.tight_layout()

    calibration_path = output_dir / "calibration.png"
    fig.savefig(calibration_path, dpi=150)
    plt.close(fig)

    # 9. Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        output_dir / "feature_importance.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        importance["feature"][::-1],
        importance["importance"][::-1],
    )

    ax.set_xlabel("Split importance")
    ax.set_title("LightGBM Feature Importance")
    fig.tight_layout()

    importance_path = output_dir / "feature_importance.png"
    fig.savefig(importance_path, dpi=150)
    plt.close(fig)

    # 10. Compare against previous baseline
    old_path = old_dir / "validation_predictions.parquet"

    if old_path.exists():
        old = pd.read_parquet(old_path)

        comparison = predictions.merge(
            old,
            on=["customer_id", "default_next_month"],
            how="inner",
            validate="one_to_one",
            suffixes=("_new", "_old"),
        )

        if len(comparison) != len(val):
            raise ValueError(
                "Old and new validation customers do not match"
            )

        y = comparison["default_next_month"]

        old_metrics = {
            "roc_auc": roc_auc_score(
                y, comparison["probability_old"]
            ),
            "pr_auc": average_precision_score(
                y, comparison["probability_old"]
            ),
            "brier_score": brier_score_loss(
                y, comparison["probability_old"]
            ),
        }

        results = pd.DataFrame({
            "metric": list(metrics.keys()),
            "baseline": list(old_metrics.values()),
            "improved": list(metrics.values()),
        })

        results["change"] = (
            results["improved"] - results["baseline"]
        )

        print("\nModel comparison:")
        print(results.to_string(index=False))

        results.to_csv(
            output_dir / "model_comparison.csv",
            index=False,
        )

        mlflow.log_artifact(
            str(output_dir / "model_comparison.csv")
        )

    # 11. Log experiment
    mlflow.log_params(params)
    mlflow.log_param("dataset", "uci")
    mlflow.log_param("best_iteration", model.best_iteration_)
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_metrics(metrics)

    mlflow.log_artifact(str(model_path))
    mlflow.log_artifact(str(calibration_path))
    mlflow.log_artifact(str(importance_path))
    mlflow.log_artifact(
        str(output_dir / "feature_importance.csv")
    )

    print("\nSaved to:", output_dir)
