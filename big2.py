import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import sys

# -----------------------------
# 1. LOAD & AUTO-DETECT COLUMNS
# -----------------------------
print("Loading dataset...")
try:
    # Try reading without skipping first
    df = pd.read_csv('bcell_full_v3.csv', low_memory=False)
    
    # Check if the first row is a "Section" header (common in IEDB)
    # If the first column name is 'Receptor' or 'Epitope', we probably need to skip row 0
    if 'Receptor' in df.columns or 'Epitope' in df.columns:
        print("Detected IEDB section headers. Re-loading with correct header row...")
        df = pd.read_csv('bcell_full_v3.csv', skipinitialspace=True, header=1, low_memory=False)
except FileNotFoundError:
    print("Error: 'bcell_full_v3.csv' not found.")
    sys.exit()

# Auto-detect Label Column
possible_label_cols = ['Assay Qualitative Measure', 'Qualitative Measure', 'Assay Result', 'Label', 'Result']
label_col = next((c for c in possible_label_cols if c in df.columns), None)

# Auto-detect Sequence Column
possible_seq_cols = ['Sequence', 'Name', 'Epitope Name', 'Peptide Sequence']
seq_col = next((c for c in possible_seq_cols if c in df.columns), None)

if not label_col or not seq_col:
    print(f"Error: Required columns not found.\nFound: {df.columns.tolist()}")
    print("Please check if your CSV uses names like 'Name' for sequences and 'Assay Qualitative Measure' for labels.")
    sys.exit()

print(f"Using '{seq_col}' for sequences and '{label_col}' for labels.")

# -----------------------------
# 2. CLEANING & MAPPING
# -----------------------------
# Drop empty rows and standardize
df = df.dropna(subset=[label_col, seq_col])
df[seq_col] = df[seq_col].astype(str)
df[label_col] = df[label_col].astype(str).str.lower().str.strip()

# Flexible mapping for various IEDB label styles
label_map = {
    "negative": 0,
    "positive": 1,
    "positive low": 1,
    "positive high": 1,
    "positive-low": 1,
    "positive-high": 1,
    "intermediate": 1
}

df['label'] = df[label_col].map(label_map)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# -----------------------------
# 3. CLASS BALANCING
# -----------------------------
neg_count = len(df[df['label'] == 0])
pos_count = len(df[df['label'] == 1])
print(f"Counts - Negative: {neg_count}, Positive: {pos_count}")

if neg_count == 0 or pos_count == 0:
    print("Error: Dataset must contain both Positive and Negative samples.")
    sys.exit()

min_count = min(neg_count, pos_count)
df_balanced = pd.concat([
    df[df['label'] == 0].sample(min_count, random_state=42),
    df[df['label'] == 1].sample(min_count, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Balanced to {len(df_balanced)} samples.")

# -----------------------------
# 4. SEQUENCE ENCODING (NEW)
# -----------------------------
print("Encoding sequences...")

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_idx = {aa: i+1 for i, aa in enumerate(amino_acids)}  # 0 = padding

MAX_LEN =  50 # you can tune this

def encode_sequence(seq):
    seq = seq.upper()
    encoded = [aa_to_idx.get(aa, 0) for aa in seq if aa in aa_to_idx]
    
    # Padding / truncation
    if len(encoded) < MAX_LEN:
        encoded += [0] * (MAX_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_LEN]
    
    return encoded

X_data = np.array([encode_sequence(s) for s in df_balanced[seq_col]])
y_data = df_balanced['label'].values


# -----------------------------
# DATASET (UPDATED)
# -----------------------------
class EpitopeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# CNN MODEL (NEW 🔥)
# -----------------------------
class CNNModel(nn.Module):
    def __init__(self, vocab_size=21, embed_dim=64):
        super(CNNModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.embedding(x)          # (batch, seq, embed)
        x = x.permute(0, 2, 1)         # (batch, embed, seq)
        
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        x = self.pool(x).squeeze(-1)   # (batch, 128)
        x = self.fc(x)
        
        return x


model = CNNModel()

# 🔥 IMPORTANT CHANGE
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Split dataset
full_dataset = EpitopeDataset(X_data, y_data)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_set, val_set = random_split(full_dataset, [train_size, val_size])

# DataLoaders (THIS WAS MISSING ❗)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128)

# -----------------------------
# 6. TRAINING LOOP
# -----------------------------
print("\nStarting Training...")
epochs = 30 # AAC usually converges reasonably well within 30 epochs on large data

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Evaluation every 5 epochs and on the last epoch
    if epoch % 5 == 0 or epoch == epochs - 1:
        model.eval()
        correct = 0
        val_loss = 0
        with torch.no_grad():
            for v_X, v_y in val_loader:
                v_out = model(v_X)
                val_loss += criterion(v_out, v_y).item()
                probs = torch.sigmoid(v_out)
                preds = (probs > 0.5).float()
                correct += (preds == v_y).sum().item()
        
        acc = correct / len(val_set)
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {acc:.2%}")

# Save the model
torch.save(model.state_dict(), "epitope_model_final.pth")
print("\nSuccess! Final model saved as epitope_model_final.pth")