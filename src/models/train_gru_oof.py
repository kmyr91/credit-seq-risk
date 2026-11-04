
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------
# 1. Settings
# --------------------------------------------------

SEED = 42
N_FOLDS = 5
EPOCHS = 20
BATCH_SIZE = 256
HIDDEN_SIZE = 32
LEARNING_RATE = 0.001

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)

DATA = Path("data/processed/uci")
OUT = Path("outputs/uci/gru_oof")
OUT.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device, flush=True)


# --------------------------------------------------
# 2. Load customer IDs and targets
# --------------------------------------------------

with np.load(DATA / "sequences.npz") as data:
    ids_train = data["ids_train"].copy()
    y_train = data["y_train"].astype(np.int64)
    ids_val = data["ids_val"].copy()
    y_val = data["y_val"].astype(np.int64)

assert len(ids_train) == 21000
assert len(ids_val) == 4500
assert len(np.unique(ids_train)) == len(ids_train)
assert len(np.unique(ids_val)) == len(ids_val)
assert set(ids_train).isdisjoint(set(ids_val))


# --------------------------------------------------
# 3. Rebuild raw sequences from monthly history
# --------------------------------------------------

history = pd.read_parquet(DATA / "history.parquet")

columns = [
    "repayment_status",
    "bill_amount",
    "payment_amount",
]

history = history.sort_values(
    ["customer_id", "statement_date"]
).reset_index(drop=True)

counts = history.groupby("customer_id").size()
assert counts.eq(6).all()
assert not history.duplicated(
    ["customer_id", "statement_date"]
).any()

history_ids = history["customer_id"].drop_duplicates().to_numpy()

X_all = (
    history[columns]
    .to_numpy(dtype=np.float32)
    .reshape(-1, 6, 3)
)

lookup = pd.Index(history_ids)

train_positions = lookup.get_indexer(ids_train)
val_positions = lookup.get_indexer(ids_val)

assert (train_positions >= 0).all()
assert (val_positions >= 0).all()

X_train_raw = X_all[train_positions]
X_val_raw = X_all[val_positions]

print("Train sequences:", X_train_raw.shape)
print("Validation sequences:", X_val_raw.shape)


# --------------------------------------------------
# 4. GRU model
# --------------------------------------------------

class CreditGRU(nn.Module):

    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=3,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
        )

        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.fc(hidden[-1]).squeeze(-1)


def transform_sequences(scaler, X):
    shape = X.shape

    transformed = scaler.transform(
        X.reshape(-1, 3)
    ).reshape(shape)

    return np.ascontiguousarray(
        transformed,
        dtype=np.float32,
    )


def predict_logits(model, X):
    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):

            batch = torch.from_numpy(
                X[start:start + BATCH_SIZE].copy()
            ).to(device)

            logits = model(batch)

            predictions.append(
                logits.cpu().numpy().reshape(-1)
            )

    return np.concatenate(predictions)


# --------------------------------------------------
# 5. Out-of-fold training
# --------------------------------------------------

folds = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED,
)

oof_logits = np.full(
    len(ids_train),
    np.nan,
    dtype=np.float64,
)

val_logits_by_fold = []

for fold, (train_idx, holdout_idx) in enumerate(
    folds.split(X_train_raw, y_train),
    start=1,
):

    print(f"\n========== Fold {fold}/{N_FOLDS} ==========")
    print("GRU training:", len(train_idx))
    print("OOF predictions:", len(holdout_idx))

    assert np.isnan(oof_logits[holdout_idx]).all()

    # Only fit preprocessing on the fold's GRU training data.
    scaler = StandardScaler()

    scaler.fit(
        X_train_raw[train_idx].reshape(-1, 3)
    )

    X_fold_train = transform_sequences(
        scaler,
        X_train_raw[train_idx],
    )

    X_fold_holdout = transform_sequences(
        scaler,
        X_train_raw[holdout_idx],
    )

    X_fold_val = transform_sequences(
        scaler,
        X_val_raw,
    )

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_fold_train),
            torch.tensor(
                y_train[train_idx],
                dtype=torch.float32,
            ),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    torch.manual_seed(SEED + fold)

    model = CreditGRU().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0.0

        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:02d} | "
                f"Loss: {total_loss / len(loader.dataset):.4f}",
                flush=True,
            )

    # Predict only customers excluded from GRU training.
    fold_logits = predict_logits(
        model,
        X_fold_holdout,
    )

    oof_logits[holdout_idx] = fold_logits

    # Each fold model also scores the untouched validation set.
    fold_val_logits = predict_logits(
        model,
        X_fold_val,
    )

    val_logits_by_fold.append(fold_val_logits)

    # Diagnostics: predictions transformed to probabilities.
    fold_prob = torch.sigmoid(
        torch.from_numpy(fold_logits)
    ).numpy()

    fold_auc = roc_auc_score(
        y_train[holdout_idx],
        fold_prob,
    )

    print(f"Fold OOF ROC-AUC: {fold_auc:.4f}")

    # Save every encoder and its own preprocessing scaler.
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_size": 3,
            "hidden_size": HIDDEN_SIZE,
            "fold": fold,
        },
        OUT / f"gru_fold_{fold}.pt",
    )

    joblib.dump(
        scaler,
        OUT / f"scaler_fold_{fold}.joblib",
    )


# --------------------------------------------------
# 6. Combine out-of-fold predictions
# --------------------------------------------------

assert np.isfinite(oof_logits).all()

val_logits = np.mean(
    np.stack(val_logits_by_fold),
    axis=0,
)

assert len(val_logits) == len(ids_val)

train_output = pd.DataFrame({
    "customer_id": ids_train,
    "gru_oof_logit": oof_logits,
})

val_output = pd.DataFrame({
    "customer_id": ids_val,
    "gru_oof_logit": val_logits,
})

train_output.to_parquet(
    OUT / "train_oof_logits.parquet",
    index=False,
)

val_output.to_parquet(
    OUT / "validation_logits.parquet",
    index=False,
)

overall_prob = torch.sigmoid(
    torch.from_numpy(oof_logits)
).numpy()

print("\nOverall OOF GRU ROC-AUC:",
      round(roc_auc_score(y_train, overall_prob), 4))

print("Saved training:", train_output.shape)
print("Saved validation:", val_output.shape)
print("Saved to:", OUT)
