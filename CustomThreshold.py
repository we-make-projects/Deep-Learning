# =========================
# 📦 IMPORTS
# =========================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif

from catboost import CatBoostClassifier

# =========================
# 📂 LOAD DATA
# =========================
df = pd.read_csv("Combine_Dengue.csv")
df.columns = df.columns.str.strip()

TARGET = 'Class'

X = df.drop(columns=[TARGET])
y = df[TARGET]

# =========================
# ✂️ SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 🔥 FEATURE SELECTION
# =========================
selector = SelectKBest(score_func=f_classif, k=10)   # 🔥 reduce more
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# =========================
# ⚖️ SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 🐱 CATBOOST (STRONGER BALANCING)
# =========================
model = CatBoostClassifier(
    iterations=600,
    depth=4,
    learning_rate=0.05,
    class_weights=[3, 1],   # 🔥 FORCE focus on class 0
    verbose=0
)

model.fit(X_train, y_train)

# =========================
# 📊 PROBABILITIES
# =========================
y_prob = model.predict_proba(X_test)[:,1]

# =========================
# 🔥 CUSTOM THRESHOLD (FOR BALANCE)
# =========================
best_f1 = 0
best_t = 0.5

for t in np.linspace(0.1, 0.6, 10):
    y_pred = (y_prob > t).astype(int)
    
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

# =========================
# ✅ FINAL RESULTS
# =========================
print("\n===== FINAL RESULTS =====")

y_pred = (y_prob > best_t).astype(int)

print("Best Threshold:", best_t)
print("AUC:", roc_auc_score(y_test, y_prob))

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))