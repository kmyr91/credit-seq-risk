"""Retrain the 12-feature LightGBM with internal early stopping."""

from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from src.models.train_lightgbm_raw import load_dataset


OUTPUT = Path("outputs/uci/lightgbm_fair")
SEED = 42


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    df, all_features = load_dataset()
    features = all_features[:12]

    train = df[df["split"] == "train"]
    val = df[df["split"] == "validation"]

    X_train = train[features]
    y_train = train["default_next_month"].astype(int)
    X_val = val[features]
    y_val = val["default_next_month"].astype(int)

    X_fit, X_stop, y_fit, y_stop = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        stratify=y_train,
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

    selector = lgb.LGBMClassifier(**params)

    selector.fit(
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

    best_iteration = int(selector.best_iteration_)

    model = lgb.LGBMClassifier(
        **{**params, "n_estimators": best_iteration}
    )
    model.fit(X_train, y_train)

    p = model.predict_proba(X_val)[:, 1]

    print("\nFAIR 12-FEATURE LIGHTGBM")
    print("Best iteration:", best_iteration)
    print("ROC-AUC:", round(roc_auc_score(y_val, p), 6))
    print("PR-AUC:", round(average_precision_score(y_val, p), 6))
    print("Brier:", round(brier_score_loss(y_val, p), 6))

    predictions = val[
        ["customer_id", "default_next_month"]
    ].copy()
    predictions["probability"] = p

    predictions.to_parquet(
        OUTPUT / "validation_predictions.parquet",
        index=False,
    )

    joblib.dump(model, OUTPUT / "model.joblib")

    print("\nSaved to:", OUTPUT)
    print("Test set remains untouched.")


if __name__ == "__main__":
    main()
