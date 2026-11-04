
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler


# 1. Load data
data_dir = Path("data/processed/uci")

history = pd.read_parquet(data_dir / "history.parquet")
customers = pd.read_parquet(data_dir / "customers.parquet")
splits = pd.read_parquet(data_dir / "splits.parquet")

feature_cols = [
    "repayment_status",
    "bill_amount",
    "payment_amount",
]


# 2. Order observations by customer and month
history = history.sort_values(
    ["customer_id", "statement_date"]
).reset_index(drop=True)

counts = history.groupby("customer_id").size()

assert counts.eq(6).all()
assert not history.duplicated(
    ["customer_id", "statement_date"]
).any()

customer_ids = history["customer_id"].drop_duplicates().to_numpy()

assert len(customer_ids) == 30000


# 3. Build sequence arrays
X = history[feature_cols].to_numpy(
    dtype=np.float32
).reshape(-1, 6, 3)

customer_info = (
    customers
    .merge(splits, on="customer_id", validate="one_to_one")
    .set_index("customer_id")
    .reindex(customer_ids)
)

assert customer_info["split"].notna().all()
assert customer_info["default_next_month"].notna().all()

y = customer_info["default_next_month"].to_numpy(
    dtype=np.float32
)

split_labels = customer_info["split"].to_numpy()

train_mask = split_labels == "train"
val_mask = split_labels == "validation"
test_mask = split_labels == "test"

assert train_mask.sum() == 21000
assert val_mask.sum() == 4500
assert test_mask.sum() == 4500


# 4. Normalize using training data only
scaler = StandardScaler()

scaler.fit(
    X[train_mask].reshape(-1, 3)
)

X_scaled = scaler.transform(
    X.reshape(-1, 3)
).reshape(-1, 6, 3).astype(np.float32)


# 5. Save arrays and scaler
np.savez_compressed(
    data_dir / "sequences.npz",
    X_train=X_scaled[train_mask],
    y_train=y[train_mask],
    ids_train=customer_ids[train_mask],
    X_val=X_scaled[val_mask],
    y_val=y[val_mask],
    ids_val=customer_ids[val_mask],
    X_test=X_scaled[test_mask],
    y_test=y[test_mask],
    ids_test=customer_ids[test_mask],
)

joblib.dump(
    scaler,
    data_dir / "sequence_scaler.joblib"
)


# 6. Test PyTorch and GPU
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

sample = torch.tensor(
    X_scaled[train_mask][:32],
    dtype=torch.float32,
    device=device
)

print("Training sequences:", X_scaled[train_mask].shape)
print("Validation sequences:", X_scaled[val_mask].shape)
print("Test sequences:", X_scaled[test_mask].shape)
print("PyTorch batch:", sample.shape)
print("Device:", device)
print("Scaler mean:", scaler.mean_)
print("Saved sequence data successfully.")
