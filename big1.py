import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import sys

# -----------------------------
# 1. ADVANCED FEATURE EXTRACTION
# -----------------------------
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
# Hydrophobicity scales (Kyte-Doolittle)
HYDRO_MAP = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5, 
             'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 
             'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}

def extract_advanced_features(seq):
    seq = str(seq).upper()
    valid_seq = "".join([aa for aa in seq if aa in AMINO_ACIDS])
    if len(valid_seq) < 2: return np.zeros(20 + 400 + 2) # AAC + DPC + Properties

    # 1. AAC (20 features)
    aac = [valid_seq.count(aa) / len(valid_seq) for aa in AMINO_ACIDS]
    
    # 2. DPC (400 features - captures sequence order)
    dpc = []
    for aa1 in AMINO_ACIDS:
        for aa2 in AMINO_ACIDS:
            dipeptide = aa1 + aa2
            count = valid_seq.count(dipeptide)
            dpc.append(count / (len(valid_seq) - 1))
            
    # 3. Physicochemical Properties (2 features)
    avg_hydro = np.mean([HYDRO_MAP[aa] for aa in valid_seq])
    # Basic sequence length normalization
    norm_len = len(valid_seq) / 50 
    
    return np.array(aac + dpc + [avg_hydro, norm_len])

# -----------------------------
# 2. DATA PREPARATION
# -----------------------------
print("Loading and Balancing Dataset...")
df = pd.read_csv('bcell_full_v3.csv', low_memory=False)

# Auto-detect label column
possible_label_cols = ['Assay Qualitative Measure', 'Qualitative Measure', 'Assay Result', 'Label']
label_col = next((c for c in possible_label_cols if c in df.columns), None)
if not label_col: sys.exit("Label column not found.")

df = df.dropna(subset=[label_col, 'Sequence'])
df[label_col] = df[label_col].astype(str).str.lower().str.strip()

label_map = {"negative": 0, "positive": 1, "positive low": 1, "positive high": 1}
df['label'] = df[label_col].map(label_map)
df = df.dropna(subset=['label'])

# Balancing classes
neg_df = df[df['label'] == 0]
pos_df = df[df['label'] == 1]
min_count = min(len(neg_df), len(pos_df))
# Sampling a significant subset for speed, or use all if your RAM allows
sample_size = min(min_count, 50000) 
df_balanced = pd.concat([neg_df.sample(sample_size), pos_df.sample(sample_size)]).sample(frac=1)

print(f"Extracting 422 features for {len(df_balanced)} sequences...")
X_data = np.array([extract_advanced_features(s) for s in df_balanced['Sequence']])
y_data = df_balanced['label'].values

# -----------------------------
# 3. ADVANCED MODEL ARCHITECTURE
# -----------------------------
class EpitopeDeepNet(nn.Module):
    def __init__(self, input_size=422):
        super(EpitopeDeepNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256), # Stabilizes learning
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# -----------------------------
# 4. TRAINING
# -----------------------------
dataset = torch.utils.data.TensorDataset(torch.tensor(X_data, dtype=torch.float32), 
                                         torch.tensor(y_data, dtype=torch.float32).view(-1, 1))
train_size = int(0.8 * len(dataset))
train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
val_loader = DataLoader(val_set, batch_size=256)

model = EpitopeDeepNet()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4) # AdamW for better regularization
criterion = nn.BCELoss()

print("\nTraining Deep Model...")
for epoch in range(40):
    model.train()
    for b_x, b_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(b_x), b_y)
        loss.backward()
        optimizer.step()
    
    if epoch % 5 == 0 or epoch == 39:
        model.eval()
        correct = 0
        with torch.no_grad():
            for v_x, v_y in val_loader:
                preds = (model(v_x) > 0.5).float()
                correct += (preds == v_y).sum().item()
        print(f"Epoch {epoch:2d} | Val Accuracy: {correct/len(val_set):.2%}")

torch.save(model.state_dict(), "advanced_epitope_model.pth")