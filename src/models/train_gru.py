
from pathlib import Path
from copy import deepcopy

import mlflow
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


# 1. Settings
torch.manual_seed(42)
np.random.seed(42)

data_dir = Path("data/processed/uci")
output_dir = Path("outputs/uci/gru")
output_dir.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

batch_size = 256
hidden_size = 32
learning_rate = 0.001
max_epochs = 30
patience = 5

print("Device:", device)


# 2. Load sequences
data = np.load(data_dir / "sequences.npz")

X_train = torch.tensor(
    data["X_train"], dtype=torch.float32
)
y_train = torch.tensor(
    data["y_train"], dtype=torch.float32
)

X_val = torch.tensor(
    data["X_val"], dtype=torch.float32
)
y_val = torch.tensor(
    data["y_val"], dtype=torch.float32
)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=batch_size,
    shuffle=False,
)


# 3. GRU model
class CreditGRU(nn.Module):

    def __init__(self, input_size=3, hidden_size=32):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, hidden = self.gru(x)

        embedding = hidden[-1]
        logits = self.fc(embedding).squeeze(-1)

        return logits


model = CreditGRU(
    input_size=3,
    hidden_size=hidden_size,
).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)


# 4. Training
best_loss = float("inf")
best_state = None
best_epoch = 0
epochs_without_improvement = 0

for epoch in range(max_epochs):

    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(X_batch)

    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            val_loss += loss.item() * len(X_batch)

    val_loss /= len(val_loader.dataset)

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Train loss: {train_loss:.4f} | "
        f"Val loss: {val_loss:.4f}"
    )

    if val_loss < best_loss - 1e-5:
        best_loss = val_loss
        best_epoch = epoch + 1
        best_state = deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping.")
        break


# 5. Restore best model
model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    val_logits = model(X_val.to(device))
    val_prob = torch.sigmoid(val_logits).cpu().numpy()


# 6. Evaluate
y_true = y_val.numpy()

metrics = {
    "roc_auc": roc_auc_score(y_true, val_prob),
    "pr_auc": average_precision_score(y_true, val_prob),
    "brier_score": brier_score_loss(y_true, val_prob),
}

print("\nGRU validation results:")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("Best epoch:", best_epoch)


# 7. Save model and predictions
model_path = output_dir / "model.pt"

torch.save(
    {
        "model_state": model.state_dict(),
        "input_size": 3,
        "hidden_size": hidden_size,
        "best_epoch": best_epoch,
    },
    model_path,
)

predictions = pd.DataFrame({
    "customer_id": data["ids_val"],
    "default_next_month": y_true.astype(int),
    "probability": val_prob,
})

predictions.to_parquet(
    output_dir / "validation_predictions.parquet",
    index=False,
)


# 8. Log in MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit-seq-risk-uci")

with mlflow.start_run(run_name="gru_baseline"):
    mlflow.log_params({
        "model": "GRU",
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "best_epoch": best_epoch,
        "dataset": "uci",
    })

    mlflow.log_metrics(metrics)
    mlflow.log_artifact(str(model_path))

print("\nSaved results to:", output_dir)
