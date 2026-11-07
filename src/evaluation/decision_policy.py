from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/uci/stacking/"
    "lightgbm_full_control_predictions.parquet"
)

OUTPUT = Path("outputs/uci/decisioning")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INPUT)

required = {
    "customer_id",
    "default_next_month",
    "probability",
}

assert required.issubset(df.columns)
assert df["customer_id"].is_unique
assert df["probability"].between(0, 1).all()

y = df["default_next_month"].to_numpy()
p = df["probability"].to_numpy()

thresholds = [
    0.05, 0.10, 0.15, 0.20,
    0.25, 0.30, 0.35, 0.40, 0.50,
]

results = []

for threshold in thresholds:

    approved = p < threshold
    declined = ~approved

    n_approved = int(approved.sum())
    n_declined = int(declined.sum())

    approval_rate = approved.mean()

    default_rate_approved = (
        y[approved].mean()
        if n_approved > 0 else np.nan
    )

    default_rate_declined = (
        y[declined].mean()
        if n_declined > 0 else np.nan
    )

    defaults_approved = int(y[approved].sum())

    results.append({
        "threshold": threshold,
        "approval_rate": approval_rate,
        "customers_approved": n_approved,
        "customers_declined": n_declined,
        "default_rate_approved": default_rate_approved,
        "default_rate_declined": default_rate_declined,
        "defaults_among_approved": defaults_approved,
    })

summary = pd.DataFrame(results)

print("\nCREDIT-RISK DECISION POLICY")
print("-" * 65)

print(
    summary[
        [
            "threshold",
            "approval_rate",
            "default_rate_approved",
            "customers_approved",
            "defaults_among_approved",
        ]
    ].round(4).to_string(index=False)
)

summary.to_csv(
    OUTPUT / "threshold_analysis.csv",
    index=False,
)

print("\nSaved to:", OUTPUT / "threshold_analysis.csv")

# --------------------------------------------------
# Visualization and illustrative policy constraint
# --------------------------------------------------

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5))

valid = summary.dropna(
    subset=["default_rate_approved"]
)

ax.plot(
    valid["approval_rate"] * 100,
    valid["default_rate_approved"] * 100,
    marker="o",
)

for _, row in valid.iterrows():
    ax.annotate(
        f'{row["threshold"]:.2f}',
        (
            row["approval_rate"] * 100,
            row["default_rate_approved"] * 100,
        ),
        xytext=(4, 5),
        textcoords="offset points",
        fontsize=8,
    )

ax.set_xlabel("Approval rate (%)")
ax.set_ylabel("Observed default rate among approved (%)")
ax.set_title("Approval–Default Trade-off")
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUTPUT / "approval_default_tradeoff.png",
    dpi=180,
)
plt.close(fig)

MAX_OBSERVED_DEFAULT_RATE = 0.12

eligible = summary[
    summary["default_rate_approved"].notna()
    & (
        summary["default_rate_approved"]
        <= MAX_OBSERVED_DEFAULT_RATE
    )
]

if not eligible.empty:
    selected = eligible.sort_values(
        "approval_rate", ascending=False
    ).iloc[0]

    print("\nILLUSTRATIVE POLICY CONSTRAINT")
    print("Maximum observed default rate: 12%")
    print(f'Highest qualifying tested threshold: {selected["threshold"]:.2f}')
    print(f'Approval rate: {selected["approval_rate"]:.2%}')
    print(f'Observed default rate: {selected["default_rate_approved"]:.2%}')
else:
    print("\nNo tested threshold meets the illustrative constraint.")

print("Saved visualization:", OUTPUT / "approval_default_tradeoff.png")
