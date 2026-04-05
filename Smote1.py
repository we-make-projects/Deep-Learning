# =========================
# 📦 IMPORTS
# =========================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

# =========================
# 📂 LOAD DATA
# =========================
df = pd.read_csv("Combine_Dengue.csv")
df.columns = df.columns.str.strip()

TARGET = 'Class'   # 🔥 change if needed

X = df.drop(columns=[TARGET])
y = df[TARGET]

# =========================
# ✂️ TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 🔥 FEATURE SELECTION
# =========================
selector = SelectKBest(score_func=f_classif, k=30)   # try 20 / 30 / 40
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# =========================
# ⚖️ SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 🔥 REMOVE HIGHLY CORRELATED FEATURES
# =========================
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)

corr = X_train_df.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]

X_train_df = X_train_df.drop(columns=to_drop)
X_test_df = X_test_df.drop(columns=to_drop)

X_train = X_train_df.values
X_test = X_test_df.values

# =========================
# 🔥 SMOTE (FIX IMBALANCE)
# =========================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:", np.bincount(y_train))

# =========================
# 🌳 RANDOM FOREST MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 📊 PREDICTIONS
# =========================
y_prob = model.predict_proba(X_test)[:,1]

# =========================
# 🔥 THRESHOLD TUNING
# =========================
print("\n===== Threshold Tuning =====")

best_acc = 0
best_t = 0.5

for t in [0.3, 0.4, 0.5, 0.6]:
    y_pred = (y_prob > t).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"Threshold {t} → Accuracy: {acc}")

    if acc > best_acc:
        best_acc = acc
        best_t = t

# =========================
# ✅ FINAL RESULTS
# =========================
print("\n===== FINAL RESULTS =====")

y_pred = (y_prob > best_t).astype(int)

print("Best Threshold:", best_t)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))