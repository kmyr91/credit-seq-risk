
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.audit import audit_statements



raw_dir = Path("data/raw")
output_dir = Path("data/processed/uci")

output_dir.mkdir(parents=True, exist_ok=True)

file = next(raw_dir.glob("*.xls"))

df = pd.read_excel(file, header=1, engine="calamine")

print("Original dataset:", df.shape)



required_columns = (
    ["ID", "LIMIT_BAL", "default payment next month"]
    + ["PAY_6", "PAY_5", "PAY_4", "PAY_3", "PAY_2", "PAY_0"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)

if not set(required_columns).issubset(df.columns):
    raise ValueError("Missing required UCI columns")

if df["ID"].isna().any() or df["ID"].duplicated().any():
    raise ValueError("Customer IDs must be unique and non-null")

if df["default payment next month"].isna().any():
    raise ValueError("Missing default labels")

if not df["default payment next month"].isin([0, 1]).all():
    raise ValueError("Default labels must be 0 or 1")



# Original columns are reverse chronological.
# We convert them to April -> September.

months = [
    ("2005-04-01", "PAY_6", "BILL_AMT6", "PAY_AMT6"),
    ("2005-05-01", "PAY_5", "BILL_AMT5", "PAY_AMT5"),
    ("2005-06-01", "PAY_4", "BILL_AMT4", "PAY_AMT4"),
    ("2005-07-01", "PAY_3", "BILL_AMT3", "PAY_AMT3"),
    ("2005-08-01", "PAY_2", "BILL_AMT2", "PAY_AMT2"),
    ("2005-09-01", "PAY_0", "BILL_AMT1", "PAY_AMT1"),
]

monthly_tables = []

for date, status_col, bill_col, payment_col in months:

    monthly = pd.DataFrame({
        "customer_id": df["ID"],
        "statement_date": pd.to_datetime(date),
        "repayment_status": df[status_col],
        "bill_amount": df[bill_col],
        "payment_amount": df[payment_col],
    })

    monthly_tables.append(monthly)

history = pd.concat(monthly_tables, ignore_index=True)

history = history.sort_values(
    ["customer_id", "statement_date"]
).reset_index(drop=True)


# --------------------------------------------------
# 4. Build customer-level dataset
# --------------------------------------------------

customers = pd.DataFrame({
    "customer_id": df["ID"],
    "credit_limit": df["LIMIT_BAL"],
    "default_next_month": df["default payment next month"],
})


# --------------------------------------------------
# 5. Create train / validation / test splits
# --------------------------------------------------

train, temp = train_test_split(
    customers,
    test_size=0.30,
    random_state=42,
    stratify=customers["default_next_month"],
)

validation, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=temp["default_next_month"],
)

splits = pd.concat([
    train[["customer_id"]].assign(split="train"),
    validation[["customer_id"]].assign(split="validation"),
    test[["customer_id"]].assign(split="test"),
], ignore_index=True)


# --------------------------------------------------
# 6. Validate the processed datasets
# --------------------------------------------------

audit = audit_statements(history)

print("\nHistory audit:")
print(audit)

assert len(history) == len(customers) * 6
assert audit["n_duplicated_combinations"] == 0
assert audit["n_unique_customers"] == len(customers)

assert len(splits) == len(customers)
assert not splits["customer_id"].duplicated().any()
assert set(splits["customer_id"]) == set(customers["customer_id"])

print("\nSplit sizes:")
print(splits["split"].value_counts())

print("\nDefault rates by split:")
print(
    customers
    .merge(splits, on="customer_id")
    .groupby("split")["default_next_month"]
    .mean()
)


# --------------------------------------------------
# 7. Save the processed datasets
# --------------------------------------------------

history.to_parquet(
    output_dir / "history.parquet",
    index=False
)

customers.to_parquet(
    output_dir / "customers.parquet",
    index=False
)

splits.to_parquet(
    output_dir / "splits.parquet",
    index=False
)

print("\nProcessed data:")
print("History:", history.shape)
print("Customers:", customers.shape)
print("Splits:", splits.shape)

print("\nFirst customer:")
print(history.head(6))

print("\nFiles saved to:", output_dir)
