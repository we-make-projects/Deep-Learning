import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# LOAD DATA (ONCE)
# ==============================
df = pd.read_csv("Combine_Dengue.csv")

X = df.drop("Class", axis=1).values
y = df["Class"].values

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# reshape for CNN → (batch, 1, features)
X = np.expand_dims(X, axis=1)

# train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# convert to tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# ==============================
# MODEL
# ==============================
class CNN_BiLSTM_Model(nn.Module):
    def __init__(self, cnn_filters, lstm_units, dropout):
        super().__init__()

        self.conv = nn.Conv1d(1, cnn_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_units * 2, 1)

    def forward(self, x):
        x = self.relu(self.conv(x))   # (B, filters, L)
        x = x.permute(0, 2, 1)        # (B, L, filters)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)

# ==============================
# DECODE FUNCTION
# ==============================
def decode_solution(sol):
    return {
        "lr": 10**(-4 + sol[0] * 2),
        "cnn_filters": int(32 + sol[1] * 64),
        "lstm_units": int(32 + sol[2] * 64),
        "dropout": sol[3] * 0.5,
        "batch_size": int(16 + sol[4] * 32)
    }

# ==============================
# OBJECTIVE FUNCTION (FIXED)
# ==============================
def objective_function(sol):
    params = decode_solution(sol)

    model = CNN_BiLSTM_Model(
        params["cnn_filters"],
        params["lstm_units"],
        params["dropout"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=params["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=params["batch_size"]
    )

    # 🔥 Debug mode (optional)
    torch.autograd.set_detect_anomaly(True)

    # TRAIN
    for epoch in range(2):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            out = model(x)
            out = out.view(-1)        # ✅ FIXED
            y = y.view(-1).float()   # ✅ FIXED

            loss = criterion(out, y)

            loss.backward()

            # ✅ prevents gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

    # EVALUATION
    model.eval()
    preds = []

    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)

            out = torch.sigmoid(model(x))
            out = out.view(-1)   # ✅ FIXED

            preds.extend(out.cpu().numpy())

    preds = np.array(preds)

    try:
        auc = roc_auc_score(y_val.numpy(), preds)
    except:
        auc = 0.5

    return auc

# ==============================
# DANDELION OPTIMIZER
# ==============================
class DandelionOptimizer:
    def __init__(self, obj_func, bounds, pop_size=5, max_iter=5):
        self.obj_func = obj_func
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = len(bounds)

    def optimize(self):
        population = np.random.uniform(0, 1, (self.pop_size, self.dim))
        fitness = np.array([self.obj_func(p) for p in population])

        best_idx = np.argmax(fitness)
        best_sol = population[best_idx]
        best_fit = fitness[best_idx]

        for t in range(self.max_iter):
            new_pop = []

            for i in range(self.pop_size):
                new = population[i] + np.random.uniform(-0.1, 0.1, self.dim)
                new += np.random.rand(self.dim) * (best_sol - population[i])
                new += np.random.normal(0, 0.01, self.dim)

                new = np.clip(new, 0, 1)
                new_pop.append(new)

            population = np.array(new_pop)
            fitness = np.array([self.obj_func(p) for p in population])

            idx = np.argmax(fitness)
            if fitness[idx] > best_fit:
                best_fit = fitness[idx]
                best_sol = population[idx]

            print(f"Iter {t+1} | Best AUC: {best_fit:.4f}")

        return best_sol, best_fit

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    bounds = [(0, 1)] * 5

    optimizer = DandelionOptimizer(
        obj_func=objective_function,
        bounds=bounds,
        pop_size=5,
        max_iter=5
    )

    best_sol, best_auc = optimizer.optimize()

    print("\n🔥 FINAL RESULTS")
    print("Best Params:", decode_solution(best_sol))
    print("Best AUC:", best_auc)