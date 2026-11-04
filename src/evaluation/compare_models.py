from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


# --------------------------------------------------
# 1. Settings
# --------------------------------------------------

SEED = 42
N_BOOTSTRAP = 2000

INPUT = Path("outputs/uci/stacking")
OUTPUT = Path("outputs/uci/evaluation")
OUTPUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)


# --------------------------------------------------
# 2. Load predictions and match customers
# --------------------------------------------------

baseline = pd.read_parquet(
    INPUT / "lightgbm_full_control_predictions.parquet"
)

hybrid = pd.read_parquet(
    INPUT / "gru_oof_lightgbm_stack_predictions.parquet"
)

assert baseline["customer_id"].is_unique
assert hybrid["customer_id"].is_unique

df = baseline.merge(
    hybrid,
    on=["customer_id", "default_next_month"],
    how="inner",
    suffixes=("_baseline", "_hybrid"),
    validate="one_to_one",
)

assert len(df) == len(baseline) == len(hybrid) == 4500

y = df["default_next_month"].to_numpy()
p_base = df["probability_baseline"].to_numpy()
p_hybrid = df["probability_hybrid"].to_numpy()

assert set(np.unique(y)).issubset({0, 1})
assert len(np.unique(y)) == 2
assert np.isfinite(p_base).all()
assert np.isfinite(p_hybrid).all()

print("Matched validation customers:", len(df))


# --------------------------------------------------
# 3. Metrics
# --------------------------------------------------

METRICS = {
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
    "brier_score": brier_score_loss,
}

point_estimates = {}

for name, metric in METRICS.items():
    base_value = metric(y, p_base)
    hybrid_value = metric(y, p_hybrid)

    point_estimates[name] = {
        "baseline": base_value,
        "hybrid": hybrid_value,
        "difference": hybrid_value - base_value,
    }


# --------------------------------------------------
# 4. Paired, stratified bootstrap
# --------------------------------------------------

# Resample the same customer indices for both models.
# Stratifying ensures both classes remain represented.
positive_idx = np.flatnonzero(y == 1)
negative_idx = np.flatnonzero(y == 0)

bootstrap_differences = {
    name: np.empty(N_BOOTSTRAP)
    for name in METRICS
}

for iteration in range(N_BOOTSTRAP):

    sampled_positive = rng.choice(
        positive_idx,
        size=len(positive_idx),
        replace=True,
    )

    sampled_negative = rng.choice(
        negative_idx,
        size=len(negative_idx),
        replace=True,
    )

    idx = np.concatenate([
        sampled_positive,
        sampled_negative,
    ])

    yy = y[idx]
    bb = p_base[idx]
    hh = p_hybrid[idx]

    for name, metric in METRICS.items():
        bootstrap_differences[name][iteration] = (
            metric(yy, hh) - metric(yy, bb)
        )


# --------------------------------------------------
# 5. Confidence intervals
# --------------------------------------------------

results = []

for name in METRICS:

    diffs = bootstrap_differences[name]

    lower, upper = np.quantile(
        diffs, [0.025, 0.975]
    )

    values = point_estimates[name]

    results.append({
        "metric": name,
        "baseline": values["baseline"],
        "hybrid": values["hybrid"],
        "difference": values["difference"],
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_excludes_zero": bool(
            lower > 0 or upper < 0
        ),
    })

summary = pd.DataFrame(results)

print("\n========== PAIRED BOOTSTRAP ==========")

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

summary.to_csv(
    OUTPUT / "bootstrap_comparison.csv",
    index=False,
)


# --------------------------------------------------
# 6. Visualization
# --------------------------------------------------

fig, axes = plt.subplots(
    1, 3, figsize=(13, 4)
)

for ax, name in zip(axes, METRICS):

    diffs = bootstrap_differences[name]

    ax.hist(
        diffs,
        bins=35,
        alpha=0.8,
    )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="No difference",
    )

    ax.axvline(
        point_estimates[name]["difference"],
        linestyle="-",
        linewidth=1.5,
        label="Observed difference",
    )

    ax.set_title(name)
    ax.set_xlabel("Hybrid minus LightGBM")
    ax.set_ylabel("Bootstrap samples")
    ax.legend(fontsize=8)

fig.suptitle(
    "Paired bootstrap: GRU stacking vs LightGBM"
)

fig.tight_layout()

fig.savefig(
    OUTPUT / "bootstrap_differences.png",
    dpi=180,
)

plt.close(fig)


# --------------------------------------------------
# 7. Simple prediction diagnostics
# --------------------------------------------------

prob_correlation = np.corrcoef(
    p_base, p_hybrid
)[0, 1]

print(
    "\nCorrelation between final validation predictions:",
    round(float(prob_correlation), 5),
)

print(
    "Note: this compares final LightGBM predictions, "
    "not GRU-vs-LightGBM OOF logits."
)

print("\nSaved results to:", OUTPUT)
