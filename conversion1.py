import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler

# =============================
# Amino acids
# =============================
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA)}

# =============================
# 1. AAC (20 features)
# =============================
def compute_aac(seq):
    seq = seq.upper()
    length = len(seq)
    count = Counter(seq)
    return np.array([count[aa] / length for aa in AA])


# =============================
# 2. Dipeptide Composition (400)
# =============================
def compute_dpc(seq):
    seq = seq.upper()
    dpc = np.zeros(400)

    for i in range(len(seq) - 1):
        if seq[i] in AA and seq[i+1] in AA:
            idx = AA_INDEX[seq[i]] * 20 + AA_INDEX[seq[i+1]]
            dpc[idx] += 1

    if len(seq) > 1:
        dpc /= (len(seq) - 1)

    return dpc


# =============================
# 3. Hydrophobicity (CTD-like)
# =============================
hydrophobic = set("AILMFWYV")
hydrophilic = set("RNDQEK")
neutral = set("CGHSTP")

def compute_ctd(seq):
    seq = seq.upper()
    length = len(seq)

    # Composition
    comp = [
        sum(aa in hydrophobic for aa in seq) / length,
        sum(aa in hydrophilic for aa in seq) / length,
        sum(aa in neutral for aa in seq) / length
    ]

    # Transition
    transitions = [0, 0, 0]
    for i in range(len(seq)-1):
        a, b = seq[i], seq[i+1]

        if (a in hydrophobic and b in hydrophilic) or (a in hydrophilic and b in hydrophobic):
            transitions[0] += 1
        elif (a in hydrophobic and b in neutral) or (a in neutral and b in hydrophobic):
            transitions[1] += 1
        elif (a in hydrophilic and b in neutral) or (a in neutral and b in hydrophilic):
            transitions[2] += 1

    transitions = [t / (length - 1) if length > 1 else 0 for t in transitions]

    # Distribution
    def distribution(group):
        positions = [i+1 for i, aa in enumerate(seq) if aa in group]
        if not positions:
            return [0]*5
        return [
            positions[int(len(positions)*p/100)] / length
            for p in [0, 25, 50, 75, 99]
        ]

    dist = distribution(hydrophobic) + \
           distribution(hydrophilic) + \
           distribution(neutral)

    return np.array(comp + transitions + dist)


# =============================
# 4. Physicochemical (9)
# =============================
positive = set("KRH")
negative = set("DE")
polar = set("STNQ")
nonpolar = set("AVLIMFWY")

def compute_physicochemical(seq):
    length = len(seq)

    features = [
        sum(aa in positive for aa in seq) / length,
        sum(aa in negative for aa in seq) / length,
        sum(aa in polar for aa in seq) / length,
        sum(aa in nonpolar for aa in seq) / length,
        sum(aa in hydrophobic for aa in seq) / length,
        sum(aa in hydrophilic for aa in seq) / length,
        sum(aa in neutral for aa in seq) / length,
        length,
        np.mean([AA_INDEX.get(aa, 0) for aa in seq])
    ]

    return np.array(features)


# =============================
# MASTER FEATURE FUNCTION
# =============================
def featurize_sequence(seq):
    return np.concatenate([
        compute_aac(seq),
        compute_dpc(seq),
        compute_ctd(seq),
        compute_physicochemical(seq)
    ])


# =============================
# BUILD FEATURE MATRIX
# =============================
def build_feature_matrix(csv_path, save_csv=True):
    df = pd.read_csv(csv_path)

    X = np.array([featurize_sequence(seq) for seq in df['sequence']])
    y = df['label'].values

    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ---- PRINT INFO ----
    print("\n🔥 DATASET SUMMARY")
    print("Number of sequences (rows):", X.shape[0])
    print("Number of features (columns):", X.shape[1])
    print("Each sequence → vector length:", X.shape[1])

    # ---- COLUMN NAMES ----
    feature_names = (
        [f"AAC_{aa}" for aa in AA] +
        [f"DPC_{a1}{a2}" for a1 in AA for a2 in AA] +
        [f"CTD_{i}" for i in range(21)] +
        [f"PHY_{i}" for i in range(9)]
    )

    # ---- SAVE TO CSV ----
    if save_csv:
        feature_df = pd.DataFrame(X, columns=feature_names)
        feature_df['label'] = y

        file_name = "feature_matrix.csv"
        feature_df.to_csv(file_name, index=False)

        print(f"\n✅ Saved as: {file_name}")
        print("Total columns in CSV:", len(feature_df.columns))

    return X, y


# =============================
# RUN
# =============================
if __name__ == "__main__":
    X, y = build_feature_matrix("dataset_epiandnonepi_balanced.csv")