import random
import pandas as pd

amino_acids = "ACDEFGHIKLMNPQRSTVWY"

# Load your clean dataset
df = pd.read_csv("1clean_epitope_dataset.csv")

sequences = df["sequence"]

# Match lengths (IMPORTANT)
neg_sequences = [
    ''.join(random.choice(amino_acids) for _ in range(len(seq)))
    for seq in sequences
]

df_neg = pd.DataFrame({
    "sequence": neg_sequences,
    "label": [0]*len(neg_sequences)
})

# Combine
final_df = pd.concat([df, df_neg], ignore_index=True)

# Shuffle dataset
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
final_df.to_csv("2final_balanced_dataset.csv", index=False)

print("🔥 Final dataset ready:", final_df.shape)
print(final_df.head())