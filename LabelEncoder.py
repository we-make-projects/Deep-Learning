# =========================
# 📦 IMPORTS
# =========================
import pandas as pd
import numpy as np

# =========================
# ⚙️ CONFIG
# =========================
FILE_PATH = "short_dl_dataset_mew.csv"
SEQ_COL = 11        # ✅ confirmed sequence column
LABEL_COL = 25      # 🔴 replace with your detected label column
MAX_LEN = 50        # padding length (can tune later)

# =========================
# 📥 LOAD DATA
# =========================
df = pd.read_csv(FILE_PATH, header=None)

# Extract only needed columns
data = df[[SEQ_COL, LABEL_COL]].copy()
data.columns = ["sequence", "label"]

# Clean
data.dropna(inplace=True)
data["sequence"] = data["sequence"].astype(str)

print("Data shape:", data.shape)

# =========================
# 🧬 AMINO ACID SETUP
# =========================
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

# =========================
# 🔢 ONE-HOT ENCODING
# =========================
def one_hot_encode(sequence, max_len=MAX_LEN):
    matrix = np.zeros((max_len, 20))  # 20 amino acids
    
    for i, aa in enumerate(sequence[:max_len]):
        if aa in aa_to_index:
            matrix[i, aa_to_index[aa]] = 1
            
    return matrix

# =========================
# 🔁 FEATURE VECTOR CREATION
# =========================
print("Converting sequences to feature vectors...")

X = np.array([one_hot_encode(seq) for seq in data["sequence"]])

print("Feature matrix shape:", X.shape)
# (samples, 50, 20)

# =========================
# 🎯 LABEL ENCODING
# =========================
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(data["label"].astype(str))

print("Label shape:", y.shape)
print("Classes:", le.classes_)

# =========================
# 💾 SAVE OUTPUT
# =========================
np.save("X.npy", X)
np.save("y.npy", y)

print("Saved X.npy and y.npy successfully!")