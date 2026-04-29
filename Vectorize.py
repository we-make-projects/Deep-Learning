import numpy as np
import pandas as pd

# Load
X = np.load("X.npy")
y = np.load("y.npy")

# Take SAME sample size
sample_size = min(len(X), len(y))

X_sample = X[:sample_size]
y_sample = y[:sample_size]

X_flat = X_sample.reshape(sample_size, -1)

# Flatten correctly
X_flat = X_sample.reshape(sample_size, -1)

# Create DataFrame
df_out = pd.DataFrame(X_flat)

# Add labels (same length now ✅)
df_out["label"] = y_sample

# Save
df_out.to_excel("feature_vectors_sample.xlsx", index=False)

print("Saved successfully!")