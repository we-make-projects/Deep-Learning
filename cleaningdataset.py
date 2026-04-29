import pandas as pd
import re

# =========================
# 📂 LOAD DATA (IMPORTANT FIX)
# =========================
df = pd.read_csv("epitope_table_export_1777011535.csv", header=None)

# =========================
# 🔍 EXTRACT ALL CELLS
# =========================
# Flatten entire dataframe into one column
all_values = df.values.flatten()

# =========================
# 🧬 FILTER VALID SEQUENCES
# =========================
def is_valid_sequence(seq):
    if not isinstance(seq, str):
        return False
    seq = seq.strip().upper()
    
    # Only amino acids allowed
    return bool(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", seq))

sequences = [s.strip().upper() for s in all_values if is_valid_sequence(s)]

# =========================
# 🧹 REMOVE DUPLICATES
# =========================
sequences = list(set(sequences))

print("Total clean sequences:", len(sequences))

# =========================
# 🏷️ ADD LABELS
# =========================
# Your dataset = positive epitopes
df_clean = pd.DataFrame({
    "sequence": sequences,
    "label": [1]*len(sequences)
})

# =========================
# 💾 SAVE CLEAN DATASET
# =========================
df_clean.to_csv("1clean_epitope_dataset.csv", index=False)

print("✅ Clean dataset saved!")
print(df_clean.head())
