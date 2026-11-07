"""Credit-risk data and prediction drift monitoring."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DATA = Path("data/processed/uci")
OUT = Path("outputs/uci/monitoring")
PREDICTIONS = Path(
    "outputs/uci/stacking/"
    "lightgbm_full_control_predictions.parquet"
)

PSI_WARNING = 0.10
PSI_ALERT = 0.25


def population_stability_index(reference, current, bins=10):
    """Compute PSI using bins determined by the reference sample.

    Missing values are treated as an additional category.
    Larger PSI means greater distribution change.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if reference.size == 0 or current.size == 0:
        raise ValueError("Both samples must contain observations.")

    if not (np.isfinite(reference) | np.isnan(reference)).all():
        raise ValueError("Reference contains infinity.")

    if not (np.isfinite(current) | np.isnan(current)).all():
        raise ValueError("Current sample contains infinity.")

    ref_valid = reference[~np.isnan(reference)]
    cur_valid = current[~np.isnan(current)]

    if ref_valid.size == 0:
        edges = np.array([-np.inf, np.inf])
    else:
        quantiles = np.quantile(
            ref_valid, np.linspace(0, 1, bins + 1)
        )
        internal = np.unique(quantiles[1:-1])

        # Include boundaries around a constant reference value.
        if np.unique(ref_valid).size == 1:
            value = ref_valid[0]
            internal = np.array([
                np.nextafter(value, -np.inf),
                np.nextafter(value, np.inf),
            ])

        edges = np.concatenate(
            ([-np.inf], internal, [np.inf])
        )

    ref_counts = np.histogram(ref_valid, bins=edges)[0]
    cur_counts = np.histogram(cur_valid, bins=edges)[0]

    # Explicit missing-value category.
    ref_counts = np.append(
        ref_counts, np.isnan(reference).sum()
    )
    cur_counts = np.append(
        cur_counts, np.isnan(current).sum()
    )

    # Small smoothing prevents log(0).
    smoothing = 1e-6
    ref_share = ref_counts / reference.size
    cur_share = cur_counts / current.size

    ref_share = np.maximum(ref_share, smoothing)
    cur_share = np.maximum(cur_share, smoothing)

    return float(
        np.sum(
            (cur_share - ref_share)
            * np.log(cur_share / ref_share)
        )
    )


def drift_report(reference, current, columns):
    """Compare numerical distributions for selected columns."""
    if not columns:
        raise ValueError("At least one feature is required.")

    rows = []

    for name in columns:
        if name not in reference or name not in current:
            raise ValueError(f"Missing monitoring feature: {name}")

        ref = pd.to_numeric(
            reference[name], errors="raise"
        ).to_numpy(dtype=float)

        cur = pd.to_numeric(
            current[name], errors="raise"
        ).to_numpy(dtype=float)

        psi = population_stability_index(ref, cur)

        if psi >= PSI_ALERT:
            status = "alert"
        elif psi >= PSI_WARNING:
            status = "warning"
        else:
            status = "stable"

        rows.append({
            "feature": name,
            "psi": psi,
            "reference_missing_rate": float(
                np.isnan(ref).mean()
            ),
            "current_missing_rate": float(
                np.isnan(cur).mean()
            ),
            "reference_mean": (
                float(np.nanmean(ref))
                if np.isfinite(ref).any() else np.nan
            ),
            "current_mean": (
                float(np.nanmean(cur))
                if np.isfinite(cur).any() else np.nan
            ),
            "status": status,
        })

    return pd.DataFrame(rows).sort_values(
        "psi", ascending=False
    ).reset_index(drop=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(
        DATA / "customer_features.parquet"
    )
    customers = pd.read_parquet(
        DATA / "customers.parquet"
    )
    splits = pd.read_parquet(
        DATA / "splits.parquet"
    )

    df = (
        features
        .merge(
            customers[["customer_id", "credit_limit"]],
            on="customer_id",
            validate="one_to_one",
        )
        .merge(
            splits,
            on="customer_id",
            validate="one_to_one",
        )
    )

    df["credit_utilization"] = (
        df["latest_bill"].clip(lower=0)
        / df["credit_limit"].replace(0, np.nan)
    )
    df["payment_ratio"] = (
        df["total_payment"]
        / df["total_positive_bill"].replace(0, np.nan)
    )

    all_columns = joblib.load(
        "outputs/uci/hybrid/feature_columns.joblib"
    )
    columns = [
        x for x in all_columns
        if not x.startswith("gru_embedding_")
    ]

    reference = df[df["split"] == "train"]
    current = df[df["split"] == "validation"]

    assert len(reference) == 21000
    assert len(current) == 4500
    assert len(columns) == 12

    feature_report = drift_report(
        reference, current, columns
    )

    feature_report.to_csv(
        OUT / "feature_drift.csv",
        index=False,
    )

    print("\nFEATURE DRIFT: TRAIN VS VALIDATION")
    print(
        feature_report[
            ["feature", "psi", "status"]
        ].round(4).to_string(index=False)
    )

    # No live production scores exist yet.
    # Demonstrate score monitoring with an artificial shift.
    predictions = pd.read_parquet(PREDICTIONS)
    scores = predictions["probability"].to_numpy()

    assert len(scores) == 4500

    rng = np.random.default_rng(42)
    order = rng.permutation(len(scores))
    midpoint = len(order) // 2

    reference_scores = scores[order[:midpoint]]
    current_scores = scores[order[midpoint:]]

    shifted_scores = np.clip(
        current_scores + 0.10, 0, 1
    )

    score_report = drift_report(
        pd.DataFrame({"score": reference_scores}),
        pd.DataFrame({"score": current_scores}),
        ["score"],
    )

    shifted_report = drift_report(
        pd.DataFrame({"score": reference_scores}),
        pd.DataFrame({"score": shifted_scores}),
        ["score"],
    )

    print("\nSCORE DRIFT DEMONSTRATION")
    print(
        "Random validation halves PSI:",
        round(score_report.loc[0, "psi"], 4),
    )
    print(
        "Artificial +0.10 score shift PSI:",
        round(shifted_report.loc[0, "psi"], 4),
    )

    pd.DataFrame([
        {
            "scenario": "random_validation_halves",
            "psi": score_report.loc[0, "psi"],
            "status": score_report.loc[0, "status"],
        },
        {
            "scenario": "synthetic_score_shift",
            "psi": shifted_report.loc[0, "psi"],
            "status": shifted_report.loc[0, "status"],
        },
    ]).to_csv(
        OUT / "score_drift_demo.csv",
        index=False,
    )

    print("\nSaved reports to:", OUT)


if __name__ == "__main__":
    main()
