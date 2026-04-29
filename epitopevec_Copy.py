# =========================
# 📦 IMPORTS
# =========================
import pandas as pd
import numpy as np
import itertools

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# =========================
# 📂 LOAD FINAL DATASET
# =========================
df = pd.read_csv("final_balanced_dataset.csv")

sequences = df["sequence"].astype(str)
labels = df["label"]

# =========================
# 🧬 AMINO ACIDS
# =========================
amino_acids = "ACDEFGHIKLMNPQRSTVWY"

# =========================
# 🔢 AAC
# =========================
def compute_aac(seq):
    length = len(seq)
    if length == 0:
        return [0]*20
    return [seq.count(aa)/length for aa in amino_acids]

# =========================
# 🔗 DPC
# =========================
dipeptides = [a+b for a in amino_acids for b in amino_acids]

def compute_dpc(seq):
    length = len(seq) - 1
    dpc_dict = {dp: 0 for dp in dipeptides}
    
    for i in range(len(seq)-1):
        dp = seq[i:i+2]
        if dp in dpc_dict:
            dpc_dict[dp] += 1
    
    if length > 0:
        return [dpc_dict[dp]/length for dp in dipeptides]
    return [0]*400

# =========================
# ⚗️ PHYSICOCHEMICAL
# =========================
hydrophobicity = {
    'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,
    'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,
    'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3
}

def compute_phys(seq):
    values = [hydrophobicity.get(aa, 0) for aa in seq]
    return [np.mean(values), np.max(values), np.min(values)]

# =========================
# 🧠 FEATURE EXTRACTION
# =========================
features = []

for seq in sequences:
    features.append(
        compute_aac(seq) +
        compute_dpc(seq) +
        compute_phys(seq)
    )

X = pd.DataFrame(features)
y = labels

print("Feature shape:", X.shape)

# =========================
# ⚖️ SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 🔀 SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 🌲 MODEL
# =========================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# =========================
# 📊 EVALUATION
# =========================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))