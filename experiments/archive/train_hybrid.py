
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# 1. Settings
SEED = 42
HIDDEN_SIZE = 32
BATCH_SIZE = 256
EPOCHS = 20

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(1)

data_dir = Path("data/processed/uci")
output_dir = Path("outputs/uci/hybrid")
output_dir.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Device:", device, flush=True)


# 2. Load existing sequences
data = np.load(data_dir / "sequences.npz")

X_train = data["X_train"]
y_train = data["y_train"]
ids_train = data["ids_train"]

X_val = data["X_val"]
y_val = data["y_val"]
ids_val = data["ids_val"]

encoder_idx, hybrid_idx = train_test_split(
    np.arange(len(y_train)),
    test_size=0.25,
    random_state=SEED,
    stratify=y_train,
)

assert set(ids_train[encoder_idx]).isdisjoint(
    set(ids_train[hybrid_idx])
)

print("Encoder customers:", len(encoder_idx))
print("Hybrid customers:", len(hybrid_idx))


# 3. GRU
class CreditGRU(nn.Module):

    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=3,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
        )
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def encode(self, x):
        _, hidden = self.gru(x)
        return hidden[-1]

    def forward(self, x):
        return self.fc(self.encode(x)).squeeze(-1)


model = CreditGRU().to(device)

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(
            X_train[encoder_idx].copy(),
            dtype=torch.float32,
        ),
        torch.tensor(
            y_train[encoder_idx].copy(),
            dtype=torch.float32,
        ),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)
criterion = nn.BCEWithLogitsLoss()


# 4. Train GRU
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(xb)

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Loss: {total_loss / len(train_loader.dataset):.4f}",
        flush=True,
    )


# 5. Extract embeddings
def extract_embeddings(X):
    model.eval()
    result = []

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):

            batch = torch.tensor(
                X[start:start + BATCH_SIZE].copy(),
                dtype=torch.float32,
                device=device,
            )

            result.append(
                model.encode(batch).cpu().numpy()
            )

    return np.concatenate(result, axis=0)


embedding_columns = [
    f"gru_embedding_{i}" for i in range(HIDDEN_SIZE)
]

for name, X, ids in [
    ("train", X_train[hybrid_idx], ids_train[hybrid_idx]),
    ("validation", X_val, ids_val),
]:
    embeddings = extract_embeddings(X)

    result = pd.DataFrame(
        embeddings,
        columns=embedding_columns,
    )
    result.insert(0, "customer_id", ids)

    result.to_parquet(
        output_dir / f"{name}_embeddings.parquet",
        index=False,
    )

    print(f"{name} embeddings:", result.shape)


# 6. Save encoder
torch.save(
    {
        "model_state": model.state_dict(),
        "hidden_size": HIDDEN_SIZE,
        "input_size": 3,
        "seed": SEED,
        "encoder_customer_ids": ids_train[encoder_idx].tolist(),
    },
    output_dir / "encoder.pt",
)

print("Encoder and embeddings saved.")
